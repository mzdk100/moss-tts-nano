use {
    super::TtsError,
    serde::Deserialize,
    serde_json::{Value, from_str},
    std::path::Path,
    tokio::fs::read_to_string,
};

/// Sample mode enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleMode {
    Greedy,
    Fixed,
    Full,
}

impl SampleMode {
    pub fn from_str(s: &str, do_sample: bool) -> Self {
        match s.trim().to_lowercase().as_str() {
            "greedy" => Self::Greedy,
            "fixed" => Self::Fixed,
            "full" => Self::Full,
            "mixed3" => {
                if do_sample {
                    Self::Fixed
                } else {
                    Self::Greedy
                }
            }
            _ => {
                if do_sample {
                    Self::Fixed
                } else {
                    Self::Greedy
                }
            }
        }
    }
}

/// Generation configuration
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_frames: usize,
    pub do_sample: bool,
    pub sample_mode: SampleMode,
    pub text_temperature: f32,
    pub text_top_p: f32,
    pub text_top_k: usize,
    pub audio_temperature: f32,
    pub audio_top_p: f32,
    pub audio_top_k: usize,
    pub audio_repetition_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_frames: 375,
            do_sample: true,
            sample_mode: SampleMode::Fixed,
            text_temperature: 1.0,
            text_top_p: 1.0,
            text_top_k: 50,
            audio_temperature: 0.8,
            audio_top_p: 0.95,
            audio_top_k: 25,
            audio_repetition_penalty: 1.2,
        }
    }
}

impl GenerationConfig {
    /// Create a config with the given sample mode string ("greedy", "fixed", "full").
    pub fn with_sample_mode(sample_mode: &str) -> Self {
        let do_sample = sample_mode != "greedy";
        let mode = SampleMode::from_str(sample_mode, do_sample);
        Self {
            do_sample: mode != SampleMode::Greedy,
            sample_mode: mode,
            ..Self::default()
        }
    }
}

/// TTS model configuration from tts_browser_onnx_meta.json
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TtsModelConfig {
    pub n_vq: usize,
    pub hidden_size: usize,
    pub global_layers: usize,
    pub global_heads: usize,
    pub head_dim: usize,
    pub local_layers: usize,
    pub local_heads: usize,
    pub local_head_dim: usize,
    pub vocab_size: usize,
    pub audio_codebook_sizes: Vec<usize>,
    pub audio_pad_token_id: i64,
    pub im_start_token_id: i64,
    pub im_end_token_id: i64,
    pub audio_start_token_id: i64,
    pub audio_end_token_id: i64,
    pub audio_user_slot_token_id: i64,
    pub audio_assistant_slot_token_id: i64,
}

/// Codec configuration from codec_browser_onnx_meta.json
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CodecConfig {
    pub sample_rate: u32,
    pub channels: usize,
    pub downsample_rate: u32,
    pub num_quantizers: usize,
}

/// Manifest prompt templates
#[derive(Debug, Clone)]
pub struct PromptTemplates {
    pub user_prompt_prefix_token_ids: Vec<i32>,
    pub user_prompt_after_reference_token_ids: Vec<i32>,
    pub assistant_prompt_prefix_token_ids: Vec<i32>,
}

/// Streaming decode specification for a transformer offset
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct TransformerOffsetSpec {
    pub index: usize,
    pub input_name: String,
    pub output_name: String,
    pub shape: Vec<usize>,
}

/// Streaming decode specification for an attention cache
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct AttentionCacheSpec {
    pub index: usize,
    pub layer_index: usize,
    pub offset_input_name: String,
    pub offset_output_name: String,
    pub cached_keys_input_name: String,
    pub cached_keys_output_name: String,
    pub cached_values_input_name: String,
    pub cached_values_output_name: String,
    pub cached_positions_input_name: String,
    pub cached_positions_output_name: String,
    pub offset_shape: Vec<usize>,
    pub cache_shape: Vec<usize>,
    pub positions_shape: Vec<usize>,
}

/// A built-in voice preset with pre-computed reference audio codes.
#[derive(Debug, Clone)]
pub struct BuiltinVoice {
    pub name: String,
    pub codes: Vec<Vec<i32>>, // [frames][channels]
}

/// Full configuration combining all model configs
pub(super) struct AppConfig {
    pub tts_config: TtsModelConfig,
    pub codec_config: CodecConfig,
    pub prompt_templates: PromptTemplates,
    pub transformer_offset_specs: Vec<TransformerOffsetSpec>,
    pub attention_cache_specs: Vec<AttentionCacheSpec>,
    pub builtin_voices: Vec<BuiltinVoice>,
}

impl AppConfig {
    pub(super) async fn load(model_dir: &Path, codec_dir: &Path) -> Result<Self, TtsError> {
        // Load TTS meta
        let tts_meta_path = model_dir.join("tts_browser_onnx_meta.json");
        let tts_meta = from_str::<Value>(&read_to_string(&tts_meta_path).await?)?;

        let mc = &tts_meta["model_config"];
        let tts_config = TtsModelConfig {
            n_vq: mc["n_vq"].as_u64().unwrap() as usize,
            hidden_size: mc["hidden_size"].as_u64().unwrap() as usize,
            global_layers: mc["global_layers"].as_u64().unwrap() as usize,
            global_heads: mc["global_heads"].as_u64().unwrap() as usize,
            head_dim: mc["head_dim"].as_u64().unwrap() as usize,
            local_layers: mc["local_layers"].as_u64().unwrap() as usize,
            local_heads: mc["local_heads"].as_u64().unwrap() as usize,
            local_head_dim: mc["local_head_dim"].as_u64().unwrap() as usize,
            vocab_size: mc["vocab_size"].as_u64().unwrap() as usize,
            audio_codebook_sizes: mc["audio_codebook_sizes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect(),
            audio_pad_token_id: mc["audio_pad_token_id"].as_i64().unwrap(),
            im_start_token_id: mc["im_start_token_id"].as_i64().unwrap(),
            im_end_token_id: mc["im_end_token_id"].as_i64().unwrap(),
            audio_start_token_id: mc["audio_start_token_id"].as_i64().unwrap(),
            audio_end_token_id: mc["audio_end_token_id"].as_i64().unwrap(),
            audio_user_slot_token_id: mc["audio_user_slot_token_id"].as_i64().unwrap(),
            audio_assistant_slot_token_id: mc["audio_assistant_slot_token_id"].as_i64().unwrap(),
        };

        // Load codec meta
        let codec_meta_path = codec_dir.join("codec_browser_onnx_meta.json");
        let codec_meta = from_str::<Value>(&read_to_string(&codec_meta_path).await?)?;

        let cc = &codec_meta["codec_config"];
        let codec_config = CodecConfig {
            sample_rate: cc["sample_rate"].as_u64().unwrap() as u32,
            channels: cc["channels"].as_u64().unwrap() as usize,
            downsample_rate: cc["downsample_rate"].as_u64().unwrap() as u32,
            num_quantizers: cc["num_quantizers"].as_u64().unwrap() as usize,
        };

        // Load manifest for prompt templates
        let manifest_path = model_dir.join("browser_poc_manifest.json");
        let manifest = from_str::<Value>(&read_to_string(&manifest_path).await?)?;

        let pt = &manifest["prompt_templates"];
        let prompt_templates = PromptTemplates {
            user_prompt_prefix_token_ids: pt["user_prompt_prefix_token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect(),
            user_prompt_after_reference_token_ids: pt["user_prompt_after_reference_token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect(),
            assistant_prompt_prefix_token_ids: pt["assistant_prompt_prefix_token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect(),
        };

        // Load streaming decode specs from codec meta
        let streaming = &codec_meta["streaming_decode"];
        let transformer_offset_specs: Vec<TransformerOffsetSpec> = streaming["transformer_offsets"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|v| TransformerOffsetSpec {
                index: v["index"].as_u64().unwrap() as usize,
                input_name: v["input_name"].as_str().unwrap().to_string(),
                output_name: v["output_name"].as_str().unwrap().to_string(),
                shape: v["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect(),
            })
            .collect();

        let attention_cache_specs: Vec<AttentionCacheSpec> = streaming["attention_caches"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|v| AttentionCacheSpec {
                index: v["index"].as_u64().unwrap() as usize,
                layer_index: v["layer_index"].as_u64().unwrap() as usize,
                offset_input_name: v["offset_input_name"].as_str().unwrap().to_string(),
                offset_output_name: v["offset_output_name"].as_str().unwrap().to_string(),
                cached_keys_input_name: v["cached_keys_input_name"].as_str().unwrap().to_string(),
                cached_keys_output_name: v["cached_keys_output_name"].as_str().unwrap().to_string(),
                cached_values_input_name: v["cached_values_input_name"]
                    .as_str()
                    .unwrap()
                    .to_string(),
                cached_values_output_name: v["cached_values_output_name"]
                    .as_str()
                    .unwrap()
                    .to_string(),
                cached_positions_input_name: v["cached_positions_input_name"]
                    .as_str()
                    .unwrap()
                    .to_string(),
                cached_positions_output_name: v["cached_positions_output_name"]
                    .as_str()
                    .unwrap()
                    .to_string(),
                offset_shape: v["offset_shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect(),
                cache_shape: v["cache_shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect(),
                positions_shape: v["positions_shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect(),
            })
            .collect();

        // Load built-in voices from manifest
        let builtin_voices: Vec<BuiltinVoice> = manifest["builtin_voices"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|v| {
                let name = v["voice"].as_str().unwrap_or("").to_string();
                let codes = v["prompt_audio_codes"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|frame| {
                        frame
                            .as_array()
                            .unwrap_or(&vec![])
                            .iter()
                            .map(|x| x.as_i64().unwrap() as i32)
                            .collect()
                    })
                    .collect();
                BuiltinVoice { name, codes }
            })
            .collect();

        Ok(AppConfig {
            tts_config,
            codec_config,
            prompt_templates,
            transformer_offset_specs,
            attention_cache_specs,
            builtin_voices,
        })
    }
}
