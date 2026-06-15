use {
    super::{
        TtsError,
        config::{PromptTemplates, TtsModelConfig},
    },
    ndarray::{Array1, Array2, Array3, Array4},
    ort::{
        inputs,
        session::{RunOptions, Session},
        value::{Tensor, TensorRef, Value},
    },
    std::path::Path,
};

/// KV cache: one (key, value) pair per transformer layer.
pub(super) type KvCache = Vec<(Array4<f32>, Array4<f32>)>;

/// Groups all ONNX inference sessions.
pub(super) struct Sessions {
    pub prefill: Session,
    pub decode_step: Session,
    pub local_fixed_sampled_frame: Session,
    pub local_cached_step: Session,
    pub codec_encode: Session,
    pub codec_decode_step: Session,
}

impl Sessions {
    pub(super) fn load<P>(model_dir: P, codec_dir: P) -> Result<Self, TtsError>
    where
        P: AsRef<Path>,
    {
        let prefill = Session::builder()?
            .commit_from_file(model_dir.as_ref().join("moss_tts_prefill.onnx"))?;
        let decode_step = Session::builder()?
            .commit_from_file(model_dir.as_ref().join("moss_tts_decode_step.onnx"))?;
        let local_fixed_sampled_frame = Session::builder()?.commit_from_file(
            model_dir
                .as_ref()
                .join("moss_tts_local_fixed_sampled_frame.onnx"),
        )?;
        let local_cached_step = Session::builder()?
            .commit_from_file(model_dir.as_ref().join("moss_tts_local_cached_step.onnx"))?;

        let codec_encode = Session::builder()?
            .commit_from_file(codec_dir.as_ref().join("moss_audio_tokenizer_encode.onnx"))?;

        let codec_decode_step = Session::builder()?.commit_from_file(
            codec_dir
                .as_ref()
                .join("moss_audio_tokenizer_decode_step.onnx"),
        )?;

        Ok(Self {
            prefill,
            decode_step,
            local_fixed_sampled_frame,
            local_cached_step,
            codec_encode,
            codec_decode_step,
        })
    }
}

/// Build input tensor for prefill.
/// Constructs the multi-row input: special tokens + reference audio codes + text tokens + assistant prompt.
pub(super) fn build_input(
    token_ids: &[i32],
    ref_codes: Option<&Array3<i32>>,
    config: &TtsModelConfig,
    prompt_templates: &PromptTemplates,
) -> Array3<i32> {
    let n = config.n_vq;
    let pad = config.audio_pad_token_id as i32;

    let mut rows = Vec::new();

    // User prompt prefix tokens
    for &t in &prompt_templates.user_prompt_prefix_token_ids {
        rows.push(row_i32(t, pad, n));
    }
    // Audio start token
    rows.push(row_i32(config.audio_start_token_id as i32, pad, n));

    // Reference audio codes: shape [batch=1, frames, channels]
    if let Some(codes) = ref_codes {
        let num_frames = codes.shape()[1];
        for f in 0..num_frames {
            let mut r = vec![config.audio_user_slot_token_id as i32];
            for ch in 0..n {
                r.push(codes[[0, f, ch]]);
            }
            rows.push(r);
        }
    }

    // Audio end token
    rows.push(row_i32(config.audio_end_token_id as i32, pad, n));

    // User prompt after reference tokens
    for &t in &prompt_templates.user_prompt_after_reference_token_ids {
        rows.push(row_i32(t, pad, n));
    }

    // Text tokens
    for &t in token_ids {
        rows.push(row_i32(t, pad, n));
    }

    // Assistant prompt prefix (matching Python: no im_end/im_start between text and assistant)
    for &t in &prompt_templates.assistant_prompt_prefix_token_ids {
        rows.push(row_i32(t, pad, n));
    }
    rows.push(row_i32(config.audio_start_token_id as i32, pad, n));

    // Build 3D array
    let sl = rows.len();
    let w = n + 1;
    let mut arr = Array3::<i32>::zeros((1, sl, w));
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            arr[[0, i, j]] = v;
        }
    }

    arr
}

/// Create a single row with token + padding.
fn row_i32(t: i32, pad: i32, n: usize) -> Vec<i32> {
    let mut r = vec![t];
    r.extend(vec![pad; n]);
    r
}

/// Run prefill and return hidden states and KV cache.
pub(super) async fn run_prefill(
    session: &mut Session,
    ids: &Array3<i32>,
    global_layers: usize,
) -> Result<(Array3<f32>, KvCache), TtsError> {
    let run_options = RunOptions::new()?;
    let (b, s) = (ids.shape()[0], ids.shape()[1]);
    let mask = Array2::<i32>::ones((b, s));
    let out = session
        .run_async(
            inputs![
                TensorRef::from_array_view(ids)?,
                TensorRef::from_array_view(&mask)?
            ],
            &run_options,
        )?
        .await?;

    let h = extract_tensor_f32_3d(&out["global_hidden"]);
    let mut kv = Vec::new();
    for i in 0..global_layers {
        let k = extract_tensor_f32_4d(&out[format!("present_key_{}", i).as_str()]);
        let v = extract_tensor_f32_4d(&out[format!("present_value_{}", i).as_str()]);
        kv.push((k, v));
    }
    Ok((h, kv))
}

/// Run decode step and return updated hidden states and KV cache.
pub(super) async fn run_decode_step(
    session: &mut Session,
    ids: &Array3<i32>,
    past_valid_length: i32,
    kv: &KvCache,
) -> Result<(Array3<f32>, KvCache), TtsError> {
    let run_options = RunOptions::new()?;
    let pvl = Array1::<i32>::from_vec(vec![past_valid_length]);
    let mut inputs = Vec::new();
    inputs.push(TensorRef::from_array_view(ids)?.into());
    inputs.push(TensorRef::from_array_view(&pvl)?.into());
    for (k, v) in kv {
        inputs.push(TensorRef::from_array_view(k)?.into());
        inputs.push(TensorRef::from_array_view(v)?.into());
    }

    let out = session.run_async(inputs.as_slice(), &run_options)?.await?;
    let h = extract_tensor_f32_3d(&out["global_hidden"]);
    let mut new_kv = Vec::new();
    for i in 0..kv.len() {
        let k = extract_tensor_f32_4d(&out[format!("present_key_{}", i).as_str()]);
        let v = extract_tensor_f32_4d(&out[format!("present_value_{}", i).as_str()]);
        new_kv.push((k, v));
    }

    Ok((h, new_kv))
}

/// Run local_fixed_sampled_frame model.
pub async fn run_local_fixed_sampled_frame(
    session: &mut Session,
    global_hidden: &Array2<f32>,
    repetition_seen_mask: &Array3<i32>,
    assistant_random_u: f32,
    audio_random_u: &[f32],
) -> Result<(bool, Vec<i32>), TtsError> {
    let run_options = RunOptions::new()?;
    let n_vq = audio_random_u.len();
    let asst_rand = Array1::<f32>::from_vec(vec![assistant_random_u]);
    let audio_rand = Array2::<f32>::from_shape_fn((1, n_vq), |(_, j)| audio_random_u[j]);
    let out = session
        .run_async(
            inputs![
                TensorRef::from_array_view(global_hidden)?,
                TensorRef::from_array_view(repetition_seen_mask)?,
                TensorRef::from_array_view(&asst_rand)?,
                TensorRef::from_array_view(&audio_rand)?
            ],
            &run_options,
        )?
        .await?;

    let sc = extract_tensor_i32_2d(&out["should_continue"]);
    let ft = extract_tensor_i32_2d(&out["frame_token_ids"]);
    let should_continue = sc[[0, 0]] != 0;
    let frame: Vec<i32> = ft.iter().copied().collect();

    Ok((should_continue, frame))
}

/// Run local_cached_step model (for full sample mode with caching).
#[allow(clippy::too_many_arguments, clippy::vec_init_then_push)]
pub async fn run_local_cached_step(
    session: &mut Session,
    global_hidden: &Array2<f32>,
    text_token_id: i32,
    audio_token_id: i32,
    channel_index: i32,
    step_type: i32,
    past_valid_lengths: i32,
    local_past: &KvCache,
) -> Result<(Vec<f32>, Array2<f32>, KvCache), TtsError> {
    let run_options = RunOptions::new()?;
    let text_tok = Array1::<i32>::from_vec(vec![text_token_id]);
    let audio_tok = Array1::<i32>::from_vec(vec![audio_token_id]);
    let ch_idx = Array1::<i32>::from_vec(vec![channel_index]);
    let stype = Array1::<i32>::from_vec(vec![step_type]);
    let pvl = Array1::<i32>::from_vec(vec![past_valid_lengths]);
    let mut inputs: Vec<ort::session::SessionInputValue> = Vec::new();
    inputs.push(TensorRef::from_array_view(global_hidden)?.into());
    inputs.push(TensorRef::from_array_view(&text_tok)?.into());
    inputs.push(TensorRef::from_array_view(&audio_tok)?.into());
    inputs.push(TensorRef::from_array_view(&ch_idx)?.into());
    inputs.push(TensorRef::from_array_view(&stype)?.into());
    inputs.push(TensorRef::from_array_view(&pvl)?.into());
    for (k, v) in local_past {
        inputs.push(TensorRef::from_array_view(k)?.into());
        inputs.push(TensorRef::from_array_view(v)?.into());
    }

    let out = session.run_async(inputs.as_slice(), &run_options)?.await?;
    let text_logits = extract_tensor_f32_1d(&out["text_logits"]);
    let audio_logits = extract_tensor_f32_2d(&out["audio_logits"]);
    let mut new_past = Vec::new();
    for i in 0..local_past.len() {
        let k = extract_tensor_f32_4d(&out[format!("local_present_key_{}", i).as_str()]);
        let v = extract_tensor_f32_4d(&out[format!("local_present_value_{}", i).as_str()]);
        new_past.push((k, v));
    }

    Ok((text_logits, audio_logits, new_past))
}

/// Encode reference audio to codes.
pub async fn encode_ref(
    session: &mut Session,
    waveform: &Array2<f32>,
) -> Result<Array3<i32>, TtsError> {
    let run_options = RunOptions::new()?;
    let (ch, len) = (waveform.shape()[0], waveform.shape()[1]);
    let arr = Array3::from_shape_vec((1, ch, len), waveform.iter().cloned().collect())?;
    let out = session
        .run_async(
            inputs![
                Tensor::from_array(arr)?,
                Tensor::from_array(Array1::from_vec(vec![len as i32]))?
            ],
            &run_options,
        )?
        .await?;

    // Get actual code length (maybe shorter than the allocated tensor)
    let code_length = {
        let (_, data) = out["audio_code_lengths"].try_extract_tensor::<i32>()?;
        data[0] as usize
    };

    let codes = &out["audio_codes"];
    let (shape, data) = codes.try_extract_tensor::<i32>()?;
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let n_vq = dims[2];
    // Trim to actual code length
    let actual_len = code_length.min(dims[1]);
    let mut trimmed = Array3::<i32>::zeros((1, actual_len, n_vq));
    for f in 0..actual_len {
        for ch in 0..n_vq {
            trimmed[[0, f, ch]] = data[f * n_vq + ch];
        }
    }

    Ok(trimmed)
}

/// Create empty local cached past tensors.
pub(super) fn create_empty_local_cached_past(
    local_layers: usize,
    local_heads: usize,
    local_head_dim: usize,
) -> KvCache {
    (0..local_layers)
        .map(|_| {
            (
                Array4::<f32>::zeros((1, 0, local_heads, local_head_dim)),
                Array4::<f32>::zeros((1, 0, local_heads, local_head_dim)),
            )
        })
        .collect()
}

// Helper functions to extract tensors from ORT outputs

pub(super) fn extract_tensor_f32_1d(v: &Value) -> Vec<f32> {
    let (shape, data) = v.try_extract_tensor::<f32>().unwrap();
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let total: usize = dims.iter().product();
    data[..total].to_vec()
}

pub(super) fn extract_tensor_f32_2d(v: &Value) -> Array2<f32> {
    let (shape, data) = v.try_extract_tensor::<f32>().unwrap();
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    Array2::from_shape_vec((dims[0], dims[1]), data.to_vec()).unwrap()
}

pub(super) fn extract_tensor_f32_3d(v: &Value) -> Array3<f32> {
    let (shape, data) = v.try_extract_tensor::<f32>().unwrap();
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    match dims.len() {
        3 => Array3::from_shape_vec((dims[0], dims[1], dims[2]), data.to_vec()).unwrap(),
        4 => {
            // [batch, seq, heads, head_dim] -> [batch, seq, heads*head_dim]
            let merged = dims[2] * dims[3];
            Array3::from_shape_vec((dims[0], dims[1], merged), data.to_vec()).unwrap()
        }
        _ => panic!("Expected 3D or 4D tensor, got {}D: {:?}", dims.len(), dims),
    }
}

pub(super) fn extract_tensor_i32_2d(v: &Value) -> Array2<i32> {
    let (shape, data) = v.try_extract_tensor::<i32>().unwrap();
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    Array2::from_shape_vec((dims[0], dims[1]), data.to_vec()).unwrap()
}

pub(super) fn extract_tensor_f32_4d(v: &Value) -> Array4<f32> {
    let (shape, data) = v.try_extract_tensor::<f32>().unwrap();
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    assert_eq!(
        dims.len(),
        4,
        "Expected 4D tensor, got {}D: {:?}",
        dims.len(),
        dims
    );
    Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), data.to_vec()).unwrap()
}
