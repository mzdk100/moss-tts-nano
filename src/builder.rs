use {
    super::{
        MossTtsNano, TtsError,
        config::{AppConfig, GenerationConfig},
        models::{Sessions, encode_ref},
        tokenizer::Tokenizer,
    },
    ndarray::{Array1, Array2, Array3},
    std::path::{Path, PathBuf},
    voxudio::load_audio,
};

/// Builder for [`MossTtsNano`].
pub struct MossTtsNanoBuilder {
    model_dir: PathBuf,
    codec_dir: PathBuf,
    voice: String,
    prompt_audio_path: Option<PathBuf>,
    sample_mode: String,
    seed: u64,
    audio_repetition_penalty: f32,
    voice_clone_max_text_tokens: usize,
}

impl MossTtsNanoBuilder {
    //noinspection SpellCheckingInspection
    pub(super) fn new() -> Self {
        Self {
            model_dir: PathBuf::from("models/MOSS-TTS-Nano-100M-ONNX"),
            codec_dir: PathBuf::from("models/MOSS-Audio-Tokenizer-Nano-ONNX"),
            voice: "Junhao".into(),
            prompt_audio_path: None,
            sample_mode: "fixed".into(),
            seed: 1234,
            audio_repetition_penalty: 1.2,
            voice_clone_max_text_tokens: 75,
        }
    }

    /// Path to the TTS ONNX model directory.
    pub fn model_dir<P>(mut self, path: P) -> Self
    where
        P: AsRef<Path>,
    {
        self.model_dir = path.as_ref().into();
        self
    }

    /// Path to the codec ONNX model directory.
    pub fn codec_dir<P>(mut self, path: P) -> Self
    where
        P: AsRef<Path>,
    {
        self.codec_dir = path.as_ref().into();
        self
    }

    //noinspection SpellCheckingInspection
    /// Built-in voice preset name (default: "Junhao").
    pub fn voice<S>(mut self, name: S) -> Self
    where
        S: AsRef<str>,
    {
        self.voice = name.as_ref().into();
        self
    }

    /// Path to reference audio for voice cloning (overrides built-in voice).
    pub fn prompt_audio_path<P>(mut self, path: P) -> Self
    where
        P: AsRef<Path>,
    {
        self.prompt_audio_path = Some(path.as_ref().into());
        self
    }

    /// Sample mode: "greedy", "fixed", or "full" (default: "fixed").
    pub fn sample_mode<S>(mut self, mode: S) -> Self
    where
        S: AsRef<str>,
    {
        self.sample_mode = mode.as_ref().into();
        self
    }

    /// Random seed for reproducibility (default: 1234).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Audio-layer repetition penalty (default: 1.2).
    pub fn audio_repetition_penalty(mut self, penalty: f32) -> Self {
        self.audio_repetition_penalty = penalty;
        self
    }

    /// Max text tokens per chunk for voice cloning (default: 75).
    pub fn voice_clone_max_text_tokens(mut self, max: usize) -> Self {
        self.voice_clone_max_text_tokens = max;
        self
    }

    /// Build the [`MossTtsNano`] instance. Loads all ONNX models and config files.
    pub async fn build(self) -> Result<MossTtsNano, TtsError> {
        let model_dir = &self.model_dir;
        let codec_dir = &self.codec_dir;

        let app_config = AppConfig::load(model_dir, codec_dir).await?;
        let mut gen_config = GenerationConfig::with_sample_mode(&self.sample_mode);
        gen_config.audio_repetition_penalty = self.audio_repetition_penalty;

        let mut sessions = Sessions::load(model_dir, codec_dir)?;

        let tokenizer_path = model_dir.join("tokenizer.model");
        let tokenizer = Tokenizer::open(&tokenizer_path).await?;

        // Resolve reference audio codes. Both paths are cheap relative to model
        // loading and are shared with the live setters on `MossTtsNano`.
        let ref_codes: Option<Array3<i32>> = if let Some(ref path) = self.prompt_audio_path {
            let waveform = load_wav_stereo(path, app_config.codec_config.sample_rate).await?;
            let codes = encode_ref(&mut sessions.codec_encode, &waveform)
                .await
                .map_err(|e| TtsError::Config(e.to_string()))?;
            Some(codes)
        } else {
            let codes = resolve_preset_codes(&app_config, &self.voice);
            if codes.is_none() {
                let available: Vec<&str> = app_config
                    .builtin_voices
                    .iter()
                    .map(|v| v.name.as_str())
                    .collect();
                eprintln!(
                    "Warning: voice '{}' not found. Available: {:?}",
                    self.voice, available
                );
            }
            codes
        };

        let sample_rate = app_config.codec_config.sample_rate;
        let channels = app_config.codec_config.channels;

        Ok(MossTtsNano {
            app_config,
            sessions,
            tokenizer,
            gen_config,
            ref_codes,
            seed: self.seed,
            voice_clone_max_text_tokens: self.voice_clone_max_text_tokens,
            sample_rate,
            channels,
        })
    }
}

/// Resolve a built-in voice preset name to its reference codes.
///
/// This is just an array copy from the loaded config — no ONNX inference — so it
/// is reused by [`MossTtsNano::set_voice`](super::MossTtsNano::set_voice) to
/// switch presets without rebuilding the engine. Returns `None` if no preset
/// matches `voice`.
pub(super) fn resolve_preset_codes(app_config: &AppConfig, voice: &str) -> Option<Array3<i32>> {
    let preset = app_config.builtin_voices.iter().find(|v| v.name == voice)?;
    let n_vq = app_config.tts_config.n_vq;
    let nf = preset.codes.len();
    let mut codes = Array3::<i32>::zeros((1, nf, n_vq));
    for (f, frame) in preset.codes.iter().enumerate() {
        for (ch, &val) in frame.iter().enumerate().take(n_vq) {
            codes[[0, f, ch]] = val;
        }
    }
    Some(codes)
}

/// Load reference audio as a 2-channel `[2, samples]` waveform at `target_sr`.
///
/// Shared by the builder and by
/// [`MossTtsNano::set_prompt_audio`](super::MossTtsNano::set_prompt_audio).
pub(super) async fn load_wav_stereo<P>(path: P, target_sr: u32) -> Result<Array2<f32>, TtsError>
where
    P: AsRef<Path>,
{
    // Load stereo (mono=false) at 48kHz
    let (interleaved, channels) = match target_sr {
        16000 => load_audio::<16000, f32, _>(path, false).await,
        22050 => load_audio::<22050, f32, _>(path, false).await,
        24000 => load_audio::<24000, f32, _>(path, false).await,
        32000 => load_audio::<32000, f32, _>(path, false).await,
        44100 => load_audio::<44100, f32, _>(path, false).await,
        48000 => load_audio::<48000, f32, _>(path, false).await,
        _ => return Err(TtsError::Config("Unsupported sample rate".into())),
    }?;

    let num_samples = interleaved.len() / channels;

    // Deinterleave to [channels, samples]
    let mut ch_audio: Vec<Vec<f32>> = (0..channels)
        .map(|_| Vec::with_capacity(num_samples))
        .collect();
    for (i, &s) in interleaved.iter().enumerate() {
        ch_audio[i % channels].push(s);
    }

    // Take first two channels (or duplicate mono)
    let (left, right) = if channels >= 2 {
        (ch_audio[0].clone(), ch_audio[1].clone())
    } else {
        (ch_audio[0].clone(), ch_audio[0].clone())
    };

    let (left, right) = (Array1::from_vec(left), Array1::from_vec(right));

    let len = left.len();
    let mut wf = Array2::<f32>::zeros((2, len));
    wf.row_mut(0).assign(&left);
    wf.row_mut(1).assign(&right);
    Ok(wf)
}
