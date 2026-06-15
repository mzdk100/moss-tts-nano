use {
    super::{
        TtsError,
        config::{AppConfig, GenerationConfig, SampleMode, TtsModelConfig},
        models::{
            KvCache, Sessions, create_empty_local_cached_past, run_decode_step,
            run_local_cached_step, run_local_fixed_sampled_frame,
        },
        sampling::{self, SamplingConfig},
    },
    ndarray::{Array2, Array3},
    ort::session::Session,
    rand::{Rng, RngExt},
    rand_pcg::Pcg64,
    std::collections::HashSet,
};

/// Create a PCG64 RNG compatible with numpy.random.default_rng(seed).
/// Uses pre-computed values from Python for common seeds.
pub(super) fn create_numpy_compatible_rng(seed: u64) -> Pcg64 {
    let s = seed as u128;
    let state = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let inc = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
        | 1;
    let stream = (inc - 1) >> 1;

    Pcg64::new(state, stream)
}

/// End-of-speech detector: tracks recent frames and detects repetition patterns.
pub(super) struct EndOfSpeechDetector {
    recent_frames: Vec<Vec<i32>>,
    repeat_window: usize,
}

impl EndOfSpeechDetector {
    pub fn new(repeat_window: usize) -> Self {
        Self {
            recent_frames: Vec::new(),
            repeat_window,
        }
    }

    /// Check if the current frame indicates end of speech.
    /// Returns true if speech should stop.
    pub fn check(&mut self, frame: &[i32], step: usize) -> bool {
        self.recent_frames.push(frame.to_vec());
        if self.recent_frames.len() > self.repeat_window {
            self.recent_frames.remove(0);
        }

        if self.recent_frames.len() < self.repeat_window {
            return false;
        }

        // Check 1: all frames identical
        let all_identical = self.recent_frames.windows(2).all(|w| w[0] == w[1]);
        if all_identical {
            for _ in 0..self.repeat_window {
                self.recent_frames.pop();
            }
            return true;
        }

        // Check 2: first channel tokens repeating (dominant token pattern)
        let first_channel_tokens: Vec<i32> = self.recent_frames.iter().map(|f| f[0]).collect();
        let dominant_token = first_channel_tokens[0];
        let dominant_count = first_channel_tokens
            .iter()
            .filter(|&&t| t == dominant_token)
            .count();
        let dominant_ratio = dominant_count as f64 / first_channel_tokens.len() as f64;

        // Check 3: small set of unique tokens (cycling pattern)
        let unique_first: HashSet<i32> = first_channel_tokens.iter().copied().collect();

        if dominant_ratio > 0.75 && step > 30 {
            for _ in 0..self.repeat_window {
                self.recent_frames.pop();
            }
            return true;
        }

        if unique_first.len() <= 2 && step > 30 {
            for _ in 0..self.repeat_window {
                self.recent_frames.pop();
            }
            return true;
        }

        false
    }
}

/// Generate audio frames using autoregressive decoding.
/// Returns a vector of frames, each frame is a vector of n_vq token IDs.
#[allow(dead_code)]
pub(super) async fn generate_frames(
    sessions: &mut Sessions,
    hidden: &Array3<f32>,
    mut kv: KvCache,
    config: &AppConfig,
    gen_config: &GenerationConfig,
    seed: u64,
) -> Result<Vec<Vec<i32>>, TtsError> {
    // Use PCG64 with numpy-compatible seed derivation
    // numpy.random.default_rng(seed) uses SeedSequence to derive PCG64 state
    let mut rng = create_numpy_compatible_rng(seed);
    let n_vq = config.tts_config.n_vq;
    let audio_codebook_size = config.tts_config.audio_codebook_sizes[0];

    let sampling_config = SamplingConfig {
        text_temperature: gen_config.text_temperature,
        text_top_p: gen_config.text_top_p,
        text_top_k: gen_config.text_top_k,
        audio_temperature: gen_config.audio_temperature,
        audio_top_p: gen_config.audio_top_p,
        audio_top_k: gen_config.audio_top_k,
        audio_repetition_penalty: gen_config.audio_repetition_penalty,
        do_sample: gen_config.do_sample,
    };

    // Use max_new_frames directly (matching Python behavior).
    // The model's should_continue signal handles end-of-speech detection.
    let dynamic_limit = gen_config.max_new_frames;

    let mut frames = Vec::new();
    let mut previous_tokens_by_channel: Vec<Vec<i32>> = vec![Vec::new(); n_vq];
    let mut previous_token_sets_by_channel: Vec<HashSet<i32>> = vec![HashSet::new(); n_vq];

    let mut eos_detector = EndOfSpeechDetector::new(8);

    // Extract last hidden state: hidden shape [batch, seq, hidden_size]
    // We want [1, hidden_size] for the local models
    let seq_len = hidden.shape()[1];
    let mut cur_hidden_2d = hidden
        .index_axis(ndarray::Axis(0), 0)
        .index_axis(ndarray::Axis(0), seq_len - 1)
        .insert_axis(ndarray::Axis(0))
        .to_owned();
    let mut past_valid_length = seq_len as i32;

    for step in 0..dynamic_limit {
        let frame = match gen_config.sample_mode {
            SampleMode::Fixed | SampleMode::Greedy => {
                generate_frame_fixed(
                    &mut sessions.local_fixed_sampled_frame,
                    &cur_hidden_2d,
                    &previous_token_sets_by_channel,
                    n_vq,
                    audio_codebook_size,
                    &mut rng,
                )
                .await?
            }
            SampleMode::Full => {
                generate_frame_full(
                    &mut sessions.local_cached_step,
                    &cur_hidden_2d,
                    &mut previous_tokens_by_channel,
                    &mut previous_token_sets_by_channel,
                    &config.tts_config,
                    &sampling_config,
                    &mut rng,
                )
                .await?
            }
        };

        match frame {
            Some(frame_tokens) => {
                // Update previous tokens tracking
                for (ch, &token) in frame_tokens.iter().enumerate() {
                    previous_tokens_by_channel[ch].push(token);
                    previous_token_sets_by_channel[ch].insert(token);
                }

                // Check for end of speech
                if eos_detector.check(&frame_tokens, step) {
                    break;
                }

                frames.push(frame_tokens);

                // Run decode step to get next hidden state
                let next_row = build_next_row(frames.last().unwrap(), &config.tts_config);
                let (new_hidden, new_kv) =
                    run_decode_step(&mut sessions.decode_step, &next_row, past_valid_length, &kv)
                        .await?;
                // new_hidden shape: [batch=1, seq=1, hidden_size]
                // Extract as [1, hidden_size] = Array2
                let hidden_3d = new_hidden;
                let seq_len = hidden_3d.shape()[1];
                cur_hidden_2d = hidden_3d
                    .index_axis(ndarray::Axis(0), 0)
                    .index_axis(ndarray::Axis(0), seq_len - 1)
                    .insert_axis(ndarray::Axis(0))
                    .to_owned();
                kv = new_kv;
                past_valid_length += 1;
            }
            None => break,
        }
    }

    Ok(frames)
}

/// Generate a single frame using fixed sampling mode.
pub(super) async fn generate_frame_fixed<R: Rng>(
    session: &mut Session,
    global_hidden: &Array2<f32>,
    previous_token_sets_by_channel: &[HashSet<i32>],
    n_vq: usize,
    audio_codebook_size: usize,
    rng: &mut R,
) -> Result<Option<Vec<i32>>, TtsError> {
    // Build repetition seen mask
    let mut repetition_seen_mask = Array3::<i32>::zeros((1, n_vq, audio_codebook_size));
    for (ch, token_set) in previous_token_sets_by_channel.iter().enumerate() {
        for &token_id in token_set {
            if token_id >= 0 && (token_id as usize) < audio_codebook_size {
                repetition_seen_mask[[0, ch, token_id as usize]] = 1;
            }
        }
    }

    // Generate random values matching numpy: f64 from rng, clamp, convert to f32
    let raw_asst: f32 = rng.random();
    let assistant_random_u: f32 = raw_asst.clamp(0.0, 0.99999994);
    let audio_random_u: Vec<f32> = (0..n_vq)
        .map(|_| {
            let raw: f64 = rng.random();
            raw.clamp(0.0, 0.99999994) as f32
        })
        .collect();

    let (should_continue, frame) = run_local_fixed_sampled_frame(
        session,
        global_hidden,
        &repetition_seen_mask,
        assistant_random_u,
        &audio_random_u,
    )
    .await?;

    if should_continue {
        Ok(Some(frame))
    } else {
        Ok(None)
    }
}

/// Generate a single frame using full sampling mode with local cached step.
pub(super) async fn generate_frame_full<R: Rng>(
    session: &mut Session,
    global_hidden: &Array2<f32>,
    previous_tokens_by_channel: &mut [Vec<i32>],
    previous_token_sets_by_channel: &mut [HashSet<i32>],
    tts_config: &TtsModelConfig,
    sampling_config: &SamplingConfig,
    rng: &mut R,
) -> Result<Option<Vec<i32>>, TtsError> {
    let n_vq = tts_config.n_vq;

    // Create empty local cached past
    let mut local_past = create_empty_local_cached_past(
        tts_config.local_layers,
        tts_config.local_heads,
        tts_config.local_head_dim,
    );
    let mut local_past_valid_length = 0i32;

    // Step 0: Get text logits and sample assistant token
    let (text_logits, _audio_logits, new_past) = run_local_cached_step(
        session,
        global_hidden,
        0, // text_token_id
        0, // audio_token_id
        0, // channel_index
        0, // step_type
        local_past_valid_length,
        &local_past,
    )
    .await?;
    local_past = new_past;
    local_past_valid_length += 1;

    let next_text_token = sampling::sample_assistant_text_token(
        &text_logits,
        tts_config.audio_assistant_slot_token_id as i32,
        tts_config.audio_end_token_id as i32,
        sampling_config,
        rng,
    );

    if next_text_token != tts_config.audio_assistant_slot_token_id as i32 {
        return Ok(None);
    }

    // Step 1: Get audio logits for first channel
    let (_text_logits, audio_logits, new_past) = run_local_cached_step(
        session,
        global_hidden,
        next_text_token,
        0, // audio_token_id
        0, // channel_index
        1, // step_type
        local_past_valid_length,
        &local_past,
    )
    .await?;
    local_past = new_past;
    local_past_valid_length += 1;

    let audio_logits_flat = audio_logits.iter().cloned().collect::<Vec<f32>>();
    let first_channel_logits = sampling::slice_audio_channel_logits(&audio_logits_flat, 0, n_vq);
    let sampled_token = sampling::sample_audio_token(
        &first_channel_logits,
        &previous_tokens_by_channel[0],
        &previous_token_sets_by_channel[0],
        sampling_config,
        rng,
    );

    let mut frame = vec![0i32; n_vq];
    frame[0] = sampled_token;
    previous_tokens_by_channel[0].push(sampled_token);
    previous_token_sets_by_channel[0].insert(sampled_token);

    // Steps 2..n_vq: Sample remaining channels
    let mut previous_token = sampled_token;
    for channel_index in 1..n_vq {
        let (_text_logits, audio_logits, new_past) = run_local_cached_step(
            session,
            global_hidden,
            0, // text_token_id
            previous_token,
            (channel_index - 1) as i32,
            2, // step_type
            local_past_valid_length,
            &local_past,
        )
        .await?;
        local_past = new_past;
        local_past_valid_length += 1;

        let audio_logits_flat = audio_logits.iter().cloned().collect::<Vec<f32>>();
        let channel_logits =
            sampling::slice_audio_channel_logits(&audio_logits_flat, channel_index, n_vq);
        let sampled_token = sampling::sample_audio_token(
            &channel_logits,
            &previous_tokens_by_channel[channel_index],
            &previous_token_sets_by_channel[channel_index],
            sampling_config,
            rng,
        );

        frame[channel_index] = sampled_token;
        previous_tokens_by_channel[channel_index].push(sampled_token);
        previous_token_sets_by_channel[channel_index].insert(sampled_token);
        previous_token = sampled_token;
    }

    Ok(Some(frame))
}

/// Build the next input row for decode step.
pub(super) fn build_next_row(frame: &[i32], config: &TtsModelConfig) -> Array3<i32> {
    let n = config.n_vq;
    let mut ids = Array3::<i32>::zeros((1, 1, n + 1));
    ids[[0, 0, 0]] = config.audio_assistant_slot_token_id as i32;
    for (i, &t) in frame.iter().enumerate() {
        ids[[0, 0, i + 1]] = t;
    }
    ids
}
