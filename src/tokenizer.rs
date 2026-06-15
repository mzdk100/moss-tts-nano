use {
    super::{TtsError, sp_model::SentencePieceModel},
    std::{iter::once, path::Path},
};

/// Tokenizer wrapper with text chunking support.
pub(super) struct Tokenizer {
    sp: SentencePieceModel,
}

/// Characters that indicate sentence endings.
const SENTENCE_END_PUNCTUATION: &[char] = &['.', '!', '?', '。', '！', '？', '；', ';'];

/// Characters that indicate clause boundaries.
const CLAUSE_SPLIT_PUNCTUATION: &[char] = &[',', '，', '、', '；', ';', '：', ':'];

/// Closing punctuation that should not be split from preceding punctuation.
const CLOSING_PUNCTUATION: &[char] = &[
    '"', '\'', '"', '\'', ')', ']', '}', '）', '】', '》', '」', '』',
];

impl Tokenizer {
    pub(super) async fn open<P>(path: P) -> Result<Self, TtsError>
    where
        P: AsRef<Path>,
    {
        let sp = SentencePieceModel::open(path).await?;

        Ok(Self { sp })
    }

    /// Encode text to token IDs.
    pub(super) fn encode(&self, text: &str) -> Vec<i32> {
        self.sp
            .encode(text)
            .unwrap_or_default()
            .iter()
            .map(|t| t.id)
            .collect()
    }

    /// Count tokens in text.
    pub(super) fn count_tokens(&self, text: &str) -> usize {
        self.encode(text).len()
    }

    /// Normalize text for TTS: ensure terminal punctuation, handle whitespace.
    pub(super) fn normalize_for_tts(&self, text: &str) -> String {
        let mut s = text.trim().to_string();
        if s.is_empty() {
            return s;
        }
        // Replace newlines and tabs with spaces
        s = s.replace(['\r', '\n', '\t'], " ");
        // Collapse multiple spaces
        while s.contains("  ") {
            s = s.replace("  ", " ");
        }
        s = s.trim().to_string();

        // Ensure terminal punctuation
        if let Some(last) = s.chars().last()
            && !last.is_ascii_punctuation()
            && !"。！？；：、，.!?:;,）】》」』\"'".contains(last)
        {
            if Self::contains_cjk(&s) {
                s.push('。');
            } else {
                s.push('.');
            }
        }
        s
    }

    /// Check if text contains CJK characters.
    fn contains_cjk(text: &str) -> bool {
        text.chars().any(|c| {
            ('\u{4e00}'..='\u{9fff}').contains(&c)
                || ('\u{3400}'..='\u{4dbf}').contains(&c)
                || ('\u{3040}'..='\u{30ff}').contains(&c)
                || ('\u{ac00}'..='\u{d7af}').contains(&c)
        })
    }

    /// Prepare text for sentence chunking by normalizing whitespace and adding terminal punctuation.
    fn prepare_text_for_sentence_chunking(text: &str) -> String {
        let mut normalized = text.trim().to_string();
        if normalized.is_empty() {
            return normalized;
        }
        normalized = normalized.replace(['\r', '\n'], " ");
        while normalized.contains("  ") {
            normalized = normalized.replace("  ", " ");
        }
        if Self::contains_cjk(&normalized) {
            if !SENTENCE_END_PUNCTUATION.contains(&normalized.chars().last().unwrap_or(' ')) {
                normalized.push('。');
            }
            return normalized;
        }
        // Capitalize first letter for non-CJK
        if let Some(first) = normalized.chars().next()
            && first.is_ascii_lowercase()
        {
            normalized = format!(
                "{}{}",
                first.to_uppercase(),
                &normalized[first.len_utf8()..]
            );
        }
        // Add terminal punctuation if missing
        if let Some(last) = normalized.chars().last()
            && last.is_alphanumeric()
        {
            normalized.push('.');
        }
        // Pad short texts
        let word_count = normalized
            .split_whitespace()
            .filter(|w| !w.is_empty())
            .count();
        if word_count < 5 {
            normalized = format!("        {}", normalized);
        }
        normalized
    }

    /// Split text by punctuation characters.
    fn split_text_by_punctuation(text: &str, punctuation: &[char]) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current_chars = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut index = 0;

        while index < chars.len() {
            let c = chars[index];
            current_chars.push(c);

            if punctuation.contains(&c) {
                // Look ahead for closing punctuation
                let mut lookahead = index + 1;
                while lookahead < chars.len() && CLOSING_PUNCTUATION.contains(&chars[lookahead]) {
                    current_chars.push(chars[lookahead]);
                    lookahead += 1;
                }
                let sentence: String = current_chars.iter().collect();
                let trimmed = sentence.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current_chars.clear();

                // Skip whitespace after punctuation
                while lookahead < chars.len() && chars[lookahead].is_whitespace() {
                    lookahead += 1;
                }
                index = lookahead;
                continue;
            }
            index += 1;
        }

        let tail: String = current_chars.iter().collect();
        let trimmed = tail.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }
        sentences
    }

    /// Join two sentence parts, adding space for non-CJK text.
    fn join_sentence_parts(left: &str, right: &str) -> String {
        if left.is_empty() {
            return right.to_string();
        }
        if right.is_empty() {
            return left.to_string();
        }
        if Self::contains_cjk(left) || Self::contains_cjk(right) {
            format!("{}{}", left, right)
        } else {
            format!("{} {}", left, right)
        }
    }

    /// Split text by token budget, trying to break at preferred boundary characters.
    fn split_text_by_token_budget(&self, text: &str, max_tokens: usize) -> Vec<String> {
        let mut remaining = text.trim().to_string();
        let mut pieces = Vec::new();
        let preferred_boundary: Vec<char> = CLAUSE_SPLIT_PUNCTUATION
            .iter()
            .chain(SENTENCE_END_PUNCTUATION.iter())
            .cloned()
            .chain(once(' '))
            .collect();

        while !remaining.is_empty() {
            if self.count_tokens(&remaining) <= max_tokens {
                pieces.push(remaining.clone());
                break;
            }

            // Binary search for the longest prefix within token budget
            let chars: Vec<char> = remaining.chars().collect();
            let mut low = 1usize;
            let mut high = chars.len();
            let mut best_prefix_len = 1usize;

            while low <= high {
                let mid = (low + high) / 2;
                let candidate: String = chars[..mid].iter().collect();
                let trimmed = candidate.trim().to_string();
                if trimmed.is_empty() {
                    low = mid + 1;
                    continue;
                }
                if self.count_tokens(&trimmed) <= max_tokens {
                    best_prefix_len = mid;
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }

            let mut cut_index = best_prefix_len;

            // Try to find a preferred boundary near the end
            let scan_min = best_prefix_len.saturating_sub(25);
            for scan in (scan_min..best_prefix_len).rev() {
                if preferred_boundary.contains(&chars[scan]) {
                    cut_index = scan + 1;
                    break;
                }
            }

            let piece = chars[..cut_index]
                .iter()
                .collect::<String>()
                .trim()
                .to_string();
            if piece.is_empty() {
                cut_index = best_prefix_len;
            }

            let piece = chars[..cut_index]
                .iter()
                .collect::<String>()
                .trim()
                .to_string();
            if !piece.is_empty() {
                pieces.push(piece);
            }
            remaining = chars[cut_index..]
                .iter()
                .collect::<String>()
                .trim()
                .to_string();
        }

        pieces
    }

    /// Split long text into chunks suitable for voice cloning.
    /// Each chunk will have at most `max_tokens` tokens.
    pub(super) fn split_voice_clone_text(&self, text: &str, max_tokens: usize) -> Vec<String> {
        let normalized = text.trim().to_string();
        if normalized.is_empty() {
            return vec![];
        }
        let safe_max = max_tokens.max(1);
        let prepared = Self::prepare_text_for_sentence_chunking(&normalized);

        // Split by sentence-ending punctuation
        let sentence_candidates =
            Self::split_text_by_punctuation(&prepared, SENTENCE_END_PUNCTUATION);
        let sentence_candidates = if sentence_candidates.is_empty() {
            vec![prepared.clone()]
        } else {
            sentence_candidates
        };

        // Process each sentence
        let mut sentence_slices: Vec<(usize, String)> = Vec::new();
        for sentence in &sentence_candidates {
            let trimmed = sentence.trim().to_string();
            if trimmed.is_empty() {
                continue;
            }
            let token_count = self.count_tokens(&trimmed);
            if token_count <= safe_max {
                sentence_slices.push((token_count, trimmed));
                continue;
            }
            // Split by clause punctuation
            let clause_candidates =
                Self::split_text_by_punctuation(&trimmed, CLAUSE_SPLIT_PUNCTUATION);
            let clause_candidates = if clause_candidates.len() <= 1 {
                vec![trimmed.clone()]
            } else {
                clause_candidates
            };

            for clause in &clause_candidates {
                let clause_trimmed = clause.trim().to_string();
                if clause_trimmed.is_empty() {
                    continue;
                }
                let clause_tokens = self.count_tokens(&clause_trimmed);
                if clause_tokens <= safe_max {
                    sentence_slices.push((clause_tokens, clause_trimmed));
                    continue;
                }
                // Further split by token budget
                for piece in self.split_text_by_token_budget(&clause_trimmed, safe_max) {
                    let piece_trimmed = piece.trim().to_string();
                    if !piece_trimmed.is_empty() {
                        let piece_tokens = self.count_tokens(&piece_trimmed);
                        sentence_slices.push((piece_tokens, piece_trimmed));
                    }
                }
            }
        }

        // Merge small pieces into chunks
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_tokens = 0usize;

        for (token_count, text_piece) in &sentence_slices {
            if current_chunk.is_empty() {
                current_chunk = text_piece.clone();
                current_tokens = *token_count;
                continue;
            }
            if current_tokens + token_count > safe_max {
                chunks.push(current_chunk.trim().to_string());
                current_chunk = text_piece.clone();
                current_tokens = *token_count;
            } else {
                current_chunk = Self::join_sentence_parts(&current_chunk, text_piece);
                current_tokens = self.count_tokens(&current_chunk);
            }
        }
        if !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }

        if chunks.len() > 1 {
            chunks
        } else {
            vec![normalized]
        }
    }
}
