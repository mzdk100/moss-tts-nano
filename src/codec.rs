use {
    super::{
        TtsError,
        config::{AttentionCacheSpec, TransformerOffsetSpec},
    },
    ndarray::{Array1, Array2, Array3, Array4},
    ort::{
        session::{RunOptions, Session, SessionInputValue},
        value::{Tensor, TensorRef},
    },
    std::time::Instant,
};

/// Streaming codec decode session.
/// Maintains state across multiple decode steps for real-time audio generation.
pub(super) struct StreamingDecodeSession {
    session: *mut Session,
    transformer_specs: Vec<TransformerOffsetSpec>,
    attention_specs: Vec<AttentionCacheSpec>,
    n_vq: usize,
    // State tensors
    transformer_offsets: Vec<Array1<i32>>,
    attn_offsets: Vec<Array1<i32>>,
    attn_cached_keys: Vec<Array4<f32>>,
    attn_cached_values: Vec<Array4<f32>>,
    attn_cached_positions: Vec<Array2<i32>>,
}

// Safety: We ensure the session pointer is valid for the lifetime of this struct
unsafe impl Send for StreamingDecodeSession {}
unsafe impl Sync for StreamingDecodeSession {}

impl StreamingDecodeSession {
    pub fn new(
        session: &mut Session,
        transformer_specs: Vec<TransformerOffsetSpec>,
        attention_specs: Vec<AttentionCacheSpec>,
        n_vq: usize,
    ) -> Self {
        let session_ptr = session as *mut Session;

        let transformer_offsets: Vec<Array1<i32>> = transformer_specs
            .iter()
            .map(|spec| Array1::<i32>::zeros(spec.shape[0]))
            .collect();

        let attn_offsets: Vec<Array1<i32>> = attention_specs
            .iter()
            .map(|spec| Array1::<i32>::zeros(spec.offset_shape[0]))
            .collect();

        let attn_cached_keys: Vec<Array4<f32>> = attention_specs
            .iter()
            .map(|spec| {
                Array4::<f32>::zeros((
                    spec.cache_shape[0],
                    spec.cache_shape[1],
                    spec.cache_shape[2],
                    spec.cache_shape[3],
                ))
            })
            .collect();

        let attn_cached_values: Vec<Array4<f32>> = attention_specs
            .iter()
            .map(|spec| {
                Array4::<f32>::zeros((
                    spec.cache_shape[0],
                    spec.cache_shape[1],
                    spec.cache_shape[2],
                    spec.cache_shape[3],
                ))
            })
            .collect();

        let attn_cached_positions: Vec<Array2<i32>> = attention_specs
            .iter()
            .map(|spec| {
                Array2::<i32>::from_elem((spec.positions_shape[0], spec.positions_shape[1]), -1)
            })
            .collect();

        Self {
            session: session_ptr,
            transformer_specs,
            attention_specs,
            n_vq,
            transformer_offsets,
            attn_offsets,
            attn_cached_keys,
            attn_cached_values,
            attn_cached_positions,
        }
    }

    /// Run streaming decode on a chunk of frames.
    /// Returns (audio, audio_length) or None if no frames.
    pub(super) async fn run_frames(
        &mut self,
        frame_rows: &[Vec<i32>],
    ) -> Result<Option<(Array3<f32>, usize)>, TtsError> {
        if frame_rows.is_empty() {
            return Ok(None);
        }

        let run_options = RunOptions::new()?;
        let frame_count = frame_rows.len();
        let mut audio_codes = Array3::<i32>::zeros((1, frame_count, self.n_vq));
        for (f, frame) in frame_rows.iter().enumerate() {
            for ch in 0..self.n_vq {
                audio_codes[[0, f, ch]] = if ch < frame.len() { frame[ch] } else { 0 };
            }
        }

        // Run inference and extract all output data as owned values.
        // TensorRef borrows self state tensors; we must extract outputs before
        // we can mutably update self, so everything happens in one block.
        let (
            new_transformer_offsets,
            new_attn_offsets,
            new_attn_keys,
            new_attn_values,
            new_attn_positions,
            audio,
            audio_length,
        ) = {
            let session = unsafe { &mut *self.session };
            let fc = Array1::<i32>::from_vec(vec![frame_count as i32]);
            let mut inputs: Vec<SessionInputValue> = Vec::new();

            inputs.push(Tensor::from_array(audio_codes)?.into());
            inputs.push(TensorRef::from_array_view(&fc)?.into());

            for (i, _) in self.transformer_specs.iter().enumerate() {
                inputs.push(TensorRef::from_array_view(&self.transformer_offsets[i])?.into());
            }
            for (i, _) in self.attention_specs.iter().enumerate() {
                inputs.push(TensorRef::from_array_view(&self.attn_offsets[i])?.into());
                inputs.push(TensorRef::from_array_view(&self.attn_cached_keys[i])?.into());
                inputs.push(TensorRef::from_array_view(&self.attn_cached_values[i])?.into());
                inputs.push(TensorRef::from_array_view(&self.attn_cached_positions[i])?.into());
            }

            let out = session.run_async(inputs.as_slice(), &run_options)?.await?;

            // Extract all output data as owned values before dropping out/inputs
            let mut new_t_off = Vec::with_capacity(self.transformer_specs.len());
            for spec in &self.transformer_specs {
                let (_, d) = out[spec.output_name.as_str()].try_extract_tensor::<i32>()?;
                new_t_off.push(Array1::from_vec(d.to_vec()));
            }

            let n_attn = self.attention_specs.len();
            let mut new_a_off = Vec::with_capacity(n_attn);
            let mut new_a_keys = Vec::with_capacity(n_attn);
            let mut new_a_vals = Vec::with_capacity(n_attn);
            let mut new_a_pos = Vec::with_capacity(n_attn);
            for (i, spec) in self.attention_specs.iter().enumerate() {
                let (_, od) = out[spec.offset_output_name.as_str()].try_extract_tensor::<i32>()?;
                new_a_off.push(Array1::from_vec(od.to_vec()));

                let ks = self.attn_cached_keys[i].shape();
                let (_, kd) =
                    out[spec.cached_keys_output_name.as_str()].try_extract_tensor::<f32>()?;
                new_a_keys.push(Array4::from_shape_vec(
                    (ks[0], ks[1], ks[2], ks[3]),
                    kd.to_vec(),
                )?);

                let (_, vd) =
                    out[spec.cached_values_output_name.as_str()].try_extract_tensor::<f32>()?;
                new_a_vals.push(Array4::from_shape_vec(
                    (ks[0], ks[1], ks[2], ks[3]),
                    vd.to_vec(),
                )?);

                let ps = self.attn_cached_positions[i].shape();
                let (_, pd) =
                    out[spec.cached_positions_output_name.as_str()].try_extract_tensor::<i32>()?;
                new_a_pos.push(Array2::from_shape_vec((ps[0], ps[1]), pd.to_vec())?);
            }

            let (a_shape, a_data) = out["audio"].try_extract_tensor::<f32>()?;
            let dims: Vec<usize> = a_shape.iter().map(|&d| d as usize).collect();
            let a = Array3::from_shape_vec((dims[0], dims[1], dims[2]), a_data.to_vec())?;

            let (_, al_data) = out["audio_lengths"].try_extract_tensor::<i32>()?;
            let al = al_data[0] as usize;

            (
                new_t_off, new_a_off, new_a_keys, new_a_vals, new_a_pos, a, al,
            )
        };

        // Now update self state with owned data — no borrows active
        self.transformer_offsets = new_transformer_offsets;
        for (i, (((off, keys), vals), pos)) in new_attn_offsets
            .into_iter()
            .zip(new_attn_keys)
            .zip(new_attn_values)
            .zip(new_attn_positions)
            .enumerate()
        {
            self.attn_offsets[i] = off;
            self.attn_cached_keys[i] = keys;
            self.attn_cached_values[i] = vals;
            self.attn_cached_positions[i] = pos;
        }

        Ok(Some((audio, audio_length)))
    }
}

/// Compute streaming decode frame budget based on lead time.
pub fn resolve_stream_decode_frame_budget(
    emitted_samples_total: usize,
    sample_rate: u32,
    first_audio_emitted_at: Option<Instant>,
) -> usize {
    let lead_seconds =
        compute_stream_lead_seconds(emitted_samples_total, sample_rate, first_audio_emitted_at);

    if first_audio_emitted_at.is_none() || lead_seconds < 0.20 {
        1
    } else if lead_seconds < 0.55 {
        2
    } else if lead_seconds < 1.10 {
        4
    } else {
        8
    }
}

/// Compute how far ahead the decoded audio is from real-time playback.
fn compute_stream_lead_seconds(
    emitted_samples_total: usize,
    sample_rate: u32,
    first_audio_emitted_at: Option<Instant>,
) -> f64 {
    match first_audio_emitted_at {
        None => 0.0,
        Some(start) => {
            let elapsed = start.elapsed().as_secs_f64();
            let emitted = emitted_samples_total as f64 / sample_rate as f64;
            emitted - elapsed
        }
    }
}
