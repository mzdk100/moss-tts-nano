mod builder;
mod codec;
mod config;
mod error;
mod generation;
mod models;
mod sampling;
mod sp_model;
mod tokenizer;

pub use {builder::MossTtsNanoBuilder, error::TtsError};
use {
    codec::{StreamingDecodeSession, resolve_stream_decode_frame_budget},
    config::{AppConfig, GenerationConfig, SampleMode},
    futures_util::{Stream, StreamExt},
    generation::{
        EndOfSpeechDetector, build_next_row, create_numpy_compatible_rng, generate_frame_fixed,
        generate_frame_full,
    },
    models::{Sessions, encode_ref, run_decode_step, run_prefill},
    ndarray::{Array2, Array3},
    sampling::SamplingConfig,
    std::{collections::HashSet, path::Path, pin::Pin},
    tokenizer::Tokenizer,
};

/// MOSS-TTS-Nano text-to-speech engine.
///
/// Use [`MossTtsNano::builder()`] to construct an instance, then call
/// [`synth()`](MossTtsNano::synth) for batch synthesis or
/// [`synth_stream()`](MossTtsNano::synth_stream) for streaming audio chunks.
pub struct MossTtsNano {
    app_config: AppConfig,
    sessions: Sessions,
    tokenizer: Tokenizer,
    gen_config: GenerationConfig,
    ref_codes: Option<Array3<i32>>,
    seed: u64,
    voice_clone_max_text_tokens: usize,
    sample_rate: u32,
    channels: usize,
}

impl MossTtsNano {
    /// Create a builder with default settings.
    pub fn builder() -> MossTtsNanoBuilder {
        MossTtsNanoBuilder::new()
    }

    /// Switch to a built-in voice preset **without reloading any models**.
    ///
    /// A voice is fully determined by its reference codes; for a preset these
    /// are copied from the already-loaded config, so this is effectively
    /// instantaneous. Returns [`TtsError::Config`] if `name` is not a known
    /// built-in voice (the current voice is left unchanged in that case).
    pub async fn set_voice<S>(&mut self, name: S) -> Result<(), TtsError>
    where
        S: AsRef<str>,
    {
        let name = name.as_ref();
        match builder::resolve_preset_codes(&self.app_config, name) {
            Some(codes) => {
                self.ref_codes = Some(codes);
                Ok(())
            }
            None => Err(TtsError::Config(format!(
                "built-in voice '{name}' not found"
            ))),
        }
    }

    /// Switch to a cloned voice from reference audio **without reloading the
    /// transformer/codec models**.
    ///
    /// Only the reference codes are recomputed, by running the already-loaded
    /// codec encoder over `path`. This is far cheaper than rebuilding the engine
    /// (a single short encode pass rather than loading every ONNX session).
    pub async fn set_prompt_audio<P>(&mut self, path: P) -> Result<(), TtsError>
    where
        P: AsRef<Path>,
    {
        let waveform =
            builder::load_wav_stereo(path, self.app_config.codec_config.sample_rate).await?;
        let codes = encode_ref(&mut self.sessions.codec_encode, &waveform)
            .await
            .map_err(|e| TtsError::Config(e.to_string()))?;
        self.ref_codes = Some(codes);
        Ok(())
    }

    /// Synthesize speech from text, returning interleaved f32 samples.
    ///
    /// The returned `Vec<f32>` contains samples in the format expected by most
    /// audio libraries: for stereo, samples alternate `[L, R, L, R, ...]`.
    pub async fn synth(&mut self, text: &str) -> Result<Vec<f32>, TtsError> {
        let mut stream = self.synth_stream(text);
        let mut all_samples = Vec::new();

        while let Some(chunk) = stream.next().await {
            all_samples.extend(chunk?);
        }

        if all_samples.is_empty() {
            return Err(TtsError::NoAudioGenerated);
        }

        Ok(all_samples)
    }

    /// Synthesize speech from text, streaming audio chunks as they are decoded.
    ///
    /// Each `Vec<f32>` item in the stream is a chunk of interleaved audio samples.
    /// Frame generation and codec decode are interleaved for minimum first-packet latency:
    /// each frame is decoded to audio immediately after it is generated.
    pub fn synth_stream<'a>(
        &'a mut self,
        text: &str,
    ) -> Pin<Box<dyn Stream<Item = Result<Vec<f32>, TtsError>> + Send + 'a>> {
        let normalized = self.tokenizer.normalize_for_tts(text);
        let chunks = self
            .tokenizer
            .split_voice_clone_text(&normalized, self.voice_clone_max_text_tokens);

        if chunks.is_empty() {
            return Box::pin(async_stream::stream! {
                yield Err(TtsError::Config("empty text after normalization".into()));
            });
        }

        let sample_rate = self.sample_rate;
        let channels = self.channels;

        Box::pin(async_stream::stream! {
            let mut first_audio_emitted_at: Option<std::time::Instant> = None;
            let mut emitted_samples_total = 0usize;

            for (i, chunk_text) in chunks.iter().enumerate() {
                // --- Phase 1: Prefill (encode text + prompt into hidden states) ---
                let token_ids = self.tokenizer.encode(chunk_text);
                let input_ids = models::build_input(
                    &token_ids,
                    self.ref_codes.as_ref(),
                    &self.app_config.tts_config,
                    &self.app_config.prompt_templates,
                );

                let (hidden, mut kv) = match run_prefill(
                    &mut self.sessions.prefill,
                    &input_ids,
                    self.app_config.tts_config.global_layers,
                ).await {
                    Ok(r) => r,
                    Err(e) => { yield Err(e); return; }
                };

                // --- Phase 2: Interleaved frame generation + codec decode ---
                let n_vq = self.app_config.tts_config.n_vq;
                let mut streaming_session = StreamingDecodeSession::new(
                    &mut self.sessions.codec_decode_step,
                    self.app_config.transformer_offset_specs.clone(),
                    self.app_config.attention_cache_specs.clone(),
                    n_vq,
                );

                let seq_len = hidden.shape()[1];
                let mut cur_hidden_2d = hidden
                    .index_axis(ndarray::Axis(0), 0)
                    .index_axis(ndarray::Axis(0), seq_len - 1)
                    .insert_axis(ndarray::Axis(0))
                    .to_owned();
                let mut past_valid_length = seq_len as i32;

                let mut rng = create_numpy_compatible_rng(self.seed);
                let mut eos_detector = EndOfSpeechDetector::new(8);
                let mut previous_tokens_by_channel: Vec<Vec<i32>> = vec![Vec::new(); n_vq];
                let mut previous_token_sets_by_channel: Vec<HashSet<i32>> =
                    vec![HashSet::new(); n_vq];
                let mut pending_frames: Vec<Vec<i32>> = Vec::new();
                let audio_codebook_size = self.app_config.tts_config.audio_codebook_sizes[0];
                let max_new_frames = self.gen_config.max_new_frames;

                // Destructure sessions to avoid borrowing all of &mut self.sessions
                let Sessions {
                    ref mut decode_step,
                    ref mut local_fixed_sampled_frame,
                    ref mut local_cached_step,
                    ..
                } = self.sessions;

                #[allow(clippy::explicit_counter_loop)]
                for step in 0..max_new_frames {
                    // Generate one frame
                    let frame = match self.gen_config.sample_mode {
                        SampleMode::Fixed | SampleMode::Greedy => {
                            generate_frame_fixed(
                                local_fixed_sampled_frame,
                                &cur_hidden_2d,
                                &previous_token_sets_by_channel,
                                n_vq,
                                audio_codebook_size,
                                &mut rng,
                            ).await?
                        }
                        SampleMode::Full => {
                            generate_frame_full(
                                local_cached_step,
                                &cur_hidden_2d,
                                &mut previous_tokens_by_channel,
                                &mut previous_token_sets_by_channel,
                                &self.app_config.tts_config,
                                &sampling_config_from_gen_config(&self.gen_config),
                                &mut rng,
                            ).await?
                        }
                    };

                    let frame_tokens = match frame {
                        Some(f) => f,
                        None => break,
                    };

                    // Update token tracking
                    for (ch, &token) in frame_tokens.iter().enumerate() {
                        previous_tokens_by_channel[ch].push(token);
                        previous_token_sets_by_channel[ch].insert(token);
                    }

                    // End-of-speech check
                    if eos_detector.check(&frame_tokens, step) {
                        break;
                    }

                    pending_frames.push(frame_tokens.clone());

                    // Interleaved codec decode: try to decode when budget allows
                    let budget = resolve_stream_decode_frame_budget(
                        emitted_samples_total,
                        sample_rate,
                        first_audio_emitted_at,
                    );
                    if pending_frames.len() >= budget {
                        let frame_chunk: Vec<Vec<i32>> =
                            pending_frames.drain(..budget).collect();
                        match streaming_session.run_frames(&frame_chunk).await {
                            Ok(Some((audio, audio_length))) => {
                                if audio_length > 0 {
                                    if first_audio_emitted_at.is_none() {
                                        first_audio_emitted_at = Some(std::time::Instant::now());
                                    }
                                    emitted_samples_total += audio_length;

                                    let ch_count = audio.shape()[1];
                                    let mut channel_audio =
                                        Array2::<f32>::zeros((ch_count, audio_length));
                                    for ch in 0..ch_count {
                                        for s in 0..audio_length {
                                            channel_audio[[ch, s]] = audio[[0, ch, s]];
                                        }
                                    }
                                    yield Ok(interleave_channels(&channel_audio));
                                }
                            }
                            Ok(None) => {}
                            Err(e) => {
                                yield Err(TtsError::Config(e.to_string()));
                                return;
                            }
                        }
                    }

                    // Advance global hidden state via decode step
                    let next_row = build_next_row(&frame_tokens, &self.app_config.tts_config);
                    let (new_hidden, new_kv) = run_decode_step(
                        decode_step,
                        &next_row,
                        past_valid_length,
                        &kv,
                    ).await?;
                    let h_seq = new_hidden.shape()[1];
                    cur_hidden_2d = new_hidden
                        .index_axis(ndarray::Axis(0), 0)
                        .index_axis(ndarray::Axis(0), h_seq - 1)
                        .insert_axis(ndarray::Axis(0))
                        .to_owned();
                    kv = new_kv;
                    past_valid_length += 1;
                }

                // Flush remaining pending frames
                if !pending_frames.is_empty() {
                    match streaming_session.run_frames(&pending_frames).await {
                        Ok(Some((audio, audio_length))) => {
                            if audio_length > 0 {
                                if first_audio_emitted_at.is_none() {
                                    first_audio_emitted_at = Some(std::time::Instant::now());
                                }
                                emitted_samples_total += audio_length;

                                let ch_count = audio.shape()[1];
                                let mut channel_audio =
                                    Array2::<f32>::zeros((ch_count, audio_length));
                                for ch in 0..ch_count {
                                    for s in 0..audio_length {
                                        channel_audio[[ch, s]] = audio[[0, ch, s]];
                                    }
                                }
                                yield Ok(interleave_channels(&channel_audio));
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            yield Err(TtsError::Config(e.to_string()));
                            return;
                        }
                    }
                }

                // Insert pause between chunks
                if i < chunks.len() - 1 {
                    let pause_sec = estimate_inter_chunk_pause(chunk_text);
                    let pause_samples = (sample_rate as f64 * pause_sec).round() as usize;
                    yield Ok(vec![0.0f32; pause_samples * channels]);
                }
            }
        })
    }
}

/// Interleave multi-channel audio into a single sample vec: [L, R, L, R, ...]
fn interleave_channels(audio: &Array2<f32>) -> Vec<f32> {
    let channels = audio.shape()[0];
    let samples = audio.shape()[1];
    let mut out = Vec::with_capacity(channels * samples);
    for s in 0..samples {
        for ch in 0..channels {
            out.push(audio[[ch, s]]);
        }
    }
    out
}

fn estimate_inter_chunk_pause(text: &str) -> f64 {
    let word_count = text.split_whitespace().filter(|w| !w.is_empty()).count();
    if word_count <= 4 { 0.40 } else { 0.24 }
}

fn sampling_config_from_gen_config(gen_config: &GenerationConfig) -> SamplingConfig {
    SamplingConfig {
        text_temperature: gen_config.text_temperature,
        text_top_p: gen_config.text_top_p,
        text_top_k: gen_config.text_top_k,
        audio_temperature: gen_config.audio_temperature,
        audio_top_p: gen_config.audio_top_p,
        audio_top_k: gen_config.audio_top_k,
        audio_repetition_penalty: gen_config.audio_repetition_penalty,
        do_sample: gen_config.do_sample,
    }
}
