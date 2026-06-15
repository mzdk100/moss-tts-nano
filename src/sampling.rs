use {
    rand::{Rng, RngExt},
    std::collections::HashSet,
};

/// Sampling configuration for text and audio tokens.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub text_temperature: f32,
    pub text_top_p: f32,
    pub text_top_k: usize,
    pub audio_temperature: f32,
    pub audio_top_p: f32,
    pub audio_top_k: usize,
    pub audio_repetition_penalty: f32,
    pub do_sample: bool,
}

/// Compute softmax over a slice of f32 values.
fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return vec![];
    }
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = values
        .iter()
        .map(|&v| ((v - max_val) as f64).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / values.len() as f32; values.len()];
    }
    exps.iter().map(|&e| (e / sum) as f32).collect()
}

/// Apply repetition penalty to logits.
fn apply_repetition_penalty(values: &[f32], previous_token_ids: &[i32], penalty: f32) -> Vec<f32> {
    if previous_token_ids.is_empty() || penalty == 1.0 {
        return values.to_vec();
    }
    let mut result = values.to_vec();
    let mut seen = HashSet::new();
    for &token_id in previous_token_ids {
        let id = token_id as usize;
        if seen.contains(&id) || id >= result.len() {
            continue;
        }
        seen.insert(id);
        if result[id] < 0.0 {
            result[id] *= penalty;
        } else {
            result[id] /= penalty;
        }
    }
    result
}

/// Argmax with repetition penalty (greedy decoding).
pub fn argmax_with_repetition_penalty(
    values: &[f32],
    previous_token_set: &HashSet<i32>,
    penalty: f32,
) -> usize {
    let mut best_index = 0;
    let mut best_value = f32::NEG_INFINITY;
    let apply_penalty = !previous_token_set.is_empty() && penalty != 1.0;

    for (index, &value) in values.iter().enumerate() {
        let mut score = value;
        if apply_penalty && previous_token_set.contains(&(index as i32)) {
            if score < 0.0 {
                score *= penalty;
            } else {
                score /= penalty;
            }
        }
        if score > best_value {
            best_value = score;
            best_index = index;
        }
    }
    best_index
}

/// Sample from logits with temperature, top-k, and top-p filtering.
pub fn sample_from_scores<R: Rng>(
    values: &[f32],
    do_sample: bool,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut R,
) -> usize {
    if !do_sample {
        return values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
    }
    if temperature <= 0.0 {
        panic!("temperature must be positive when do_sample=true");
    }

    let mut scores: Vec<f32> = values.iter().map(|&v| v / temperature).collect();

    // Top-k filtering
    if top_k > 0 && top_k < scores.len() {
        let mut sorted: Vec<f32> = scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted[top_k - 1];
        for s in &mut scores {
            if *s < threshold {
                *s = f32::NEG_INFINITY;
            }
        }
    }

    // Top-p (nucleus) filtering
    if top_p > 0.0 && top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> =
            scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let sorted_scores: Vec<f32> = indexed.iter().map(|(_, s)| *s).collect();
        let sorted_probs = softmax(&sorted_scores);

        let mut cumulative = 0.0;
        let mut remove_mask = vec![false; indexed.len()];
        for (i, prob) in sorted_probs.iter().enumerate() {
            cumulative += prob;
            if cumulative > top_p {
                remove_mask[i] = true;
            }
        }
        // Shift mask: the token that crosses the threshold is kept (matching Python)
        for i in (1..remove_mask.len()).rev() {
            remove_mask[i] = remove_mask[i - 1];
        }
        if !remove_mask.is_empty() {
            remove_mask[0] = false;
        }

        for (i, should_remove) in remove_mask.iter().enumerate() {
            if *should_remove {
                scores[indexed[i].0] = f32::NEG_INFINITY;
            }
        }
    }

    // Sample from the distribution
    let probs = softmax(&scores);
    let random_value: f32 = rng.random_range(0.0..1.0);
    let mut cumulative = 0.0;
    for (i, prob) in probs.iter().enumerate() {
        cumulative += prob;
        if cumulative >= random_value {
            return i;
        }
    }
    // Fallback to argmax
    scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Sample a text token for the assistant (choosing between assistant_slot and audio_end).
pub fn sample_assistant_text_token<R: Rng>(
    text_logits: &[f32],
    assistant_slot_token_id: i32,
    audio_end_token_id: i32,
    config: &SamplingConfig,
    rng: &mut R,
) -> i32 {
    let candidate_ids = [assistant_slot_token_id, audio_end_token_id];
    let candidate_scores: Vec<f32> = candidate_ids
        .iter()
        .map(|&id| text_logits[id as usize])
        .collect();

    let sampled_index = sample_from_scores(
        &candidate_scores,
        config.do_sample,
        config.text_temperature,
        config.text_top_k.min(candidate_scores.len()),
        config.text_top_p,
        rng,
    );
    candidate_ids[sampled_index]
}

/// Sample an audio token with repetition penalty.
pub fn sample_audio_token<R: Rng>(
    audio_logits: &[f32],
    previous_token_ids: &[i32],
    previous_token_set: &HashSet<i32>,
    config: &SamplingConfig,
    rng: &mut R,
) -> i32 {
    if !config.do_sample {
        return argmax_with_repetition_penalty(
            audio_logits,
            previous_token_set,
            config.audio_repetition_penalty,
        ) as i32;
    }

    let penalized = apply_repetition_penalty(
        audio_logits,
        previous_token_ids,
        config.audio_repetition_penalty,
    );
    sample_from_scores(
        &penalized,
        true,
        config.audio_temperature,
        config.audio_top_k,
        config.audio_top_p,
        rng,
    ) as i32
}

/// Slice audio channel logits from a flat audio_logits array.
/// audio_logits shape: [1, n_vq * codebook_size] or [n_vq, codebook_size]
pub fn slice_audio_channel_logits(
    audio_logits: &[f32],
    channel_index: usize,
    n_vq: usize,
) -> Vec<f32> {
    let per_channel = audio_logits.len() / n_vq;
    let start = channel_index * per_channel;
    let end = start + per_channel;
    audio_logits[start..end].to_vec()
}
