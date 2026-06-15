//! Pure Rust SentencePiece Unigram tokenizer.
//!
//! Parses the `.model` protobuf file and implements the optimized Viterbi algorithm
//! for unigram tokenization, avoiding C++ protobuf/sentencepiece dependencies.

use {
    super::TtsError,
    std::{collections::HashMap, path::Path, str::from_utf8},
    tokio::fs::read,
};

/// A piece in the vocabulary.
#[derive(Debug, Clone)]
struct Piece {
    piece: String,
    score: f32,
    piece_type: PieceType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PieceType {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl PieceType {
    fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }
}

/// Token result from encoding.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(super) struct Token {
    pub id: i32,
    pub piece: String,
}

/// Pure Rust SentencePiece Unigram model.
pub(super) struct SentencePieceModel {
    pieces: Vec<Piece>,
    piece_to_id: HashMap<String, i32>,
    unk_id: i32,
    byte_fallback: bool,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    min_score: f32,
    model_type: i32, // 1=UNIGRAM, 2=BPE, 3=WORD, 4=CHAR
}

impl SentencePieceModel {
    /// Load a SentencePiece model from a `.model` file.
    pub(super) async fn open<P>(path: P) -> Result<Self, TtsError>
    where
        P: AsRef<Path>,
    {
        let data = read(path).await?;
        Self::from_bytes(&data)
    }

    /// Parse a SentencePiece model from bytes.
    pub(super) fn from_bytes(data: &[u8]) -> Result<Self, TtsError> {
        let model = Self::decode_model_proto(data)?;
        Ok(model)
    }

    fn decode_model_proto(data: &[u8]) -> Result<Self, TtsError> {
        let mut pieces = Vec::new();
        let mut unk_id = 0i32;
        let mut byte_fallback = false;
        let mut add_dummy_prefix = true;
        let mut remove_extra_whitespaces = true;
        let mut escape_whitespaces = true;
        let mut model_type = 1i32;

        // Parse using manual protobuf decoding
        let mut pos = 0;
        while pos < data.len() {
            let (tag, new_pos) = decode_varint(data, pos)?;
            pos = new_pos;
            let field_number = tag >> 3;
            let wire_type = tag & 0x7;

            match field_number {
                1 => {
                    // repeated SentencePiece pieces = 1
                    let (len, new_pos) = decode_varint(data, pos)?;
                    pos = new_pos;
                    let end = pos + len;
                    let piece = Self::decode_sentence_piece(&data[pos..end])?;
                    pieces.push(piece);
                    pos = end;
                }
                2 => {
                    // optional TrainerSpec trainer_spec = 2
                    let (len, new_pos) = decode_varint(data, pos)?;
                    pos = new_pos;
                    let end = pos + len;
                    let spec = Self::decode_trainer_spec(&data[pos..end])?;
                    unk_id = spec.unk_id;
                    byte_fallback = spec.byte_fallback;
                    model_type = spec.model_type;
                    pos = end;
                }
                3 => {
                    // optional NormalizerSpec normalizer_spec = 3
                    let (len, new_pos) = decode_varint(data, pos)?;
                    pos = new_pos;
                    let end = pos + len;
                    let spec = Self::decode_normalizer_spec(&data[pos..end])?;
                    add_dummy_prefix = spec.add_dummy_prefix;
                    remove_extra_whitespaces = spec.remove_extra_whitespaces;
                    escape_whitespaces = spec.escape_whitespaces;
                    pos = end;
                }
                _ => {
                    // Skip unknown fields
                    pos = skip_field(data, pos, wire_type)?;
                }
            }
        }

        let min_score = pieces.iter().map(|p| p.score).fold(f32::INFINITY, f32::min);

        let mut piece_to_id = HashMap::new();
        for (i, p) in pieces.iter().enumerate() {
            piece_to_id.insert(p.piece.clone(), i as i32);
        }

        Ok(Self {
            pieces,
            piece_to_id,
            unk_id,
            byte_fallback,
            add_dummy_prefix,
            remove_extra_whitespaces,
            escape_whitespaces,
            min_score,
            model_type,
        })
    }

    //noinspection GrazieInspection
    fn decode_sentence_piece(data: &[u8]) -> Result<Piece, TtsError> {
        let mut piece = String::new();
        let mut score = 0.0f32;
        let mut piece_type = PieceType::Normal as i32;

        let mut pos = 0;
        while pos < data.len() {
            let (tag, new_pos) = decode_varint(data, pos)?;
            pos = new_pos;
            let field_number = tag >> 3;
            let wire_type = tag & 0x7;

            match field_number {
                1 => {
                    // string piece
                    let (len, new_pos) = decode_varint(data, pos)?;
                    pos = new_pos;
                    piece = String::from_utf8(data[pos..pos + len].to_vec())?;
                    pos += len;
                }
                2 => {
                    // float score
                    if pos + 4 <= data.len() {
                        score = f32::from_le_bytes([
                            data[pos],
                            data[pos + 1],
                            data[pos + 2],
                            data[pos + 3],
                        ]);
                        pos += 4;
                    }
                }
                3 => {
                    // Type type
                    let (v, new_pos) = decode_varint(data, pos)?;
                    pos = new_pos;
                    piece_type = v as i32;
                }
                _ => {
                    pos = skip_field(data, pos, wire_type)?;
                }
            }
        }

        Ok(Piece {
            piece,
            score,
            piece_type: PieceType::from_i32(piece_type),
        })
    }

    fn decode_trainer_spec(data: &[u8]) -> Result<TrainerSpec, TtsError> {
        let mut unk_id = 0i32;
        let mut byte_fallback = false;
        let mut model_type = 1i32; // default UNIGRAM

        let mut pos = 0;
        while pos < data.len() {
            let (tag, new_pos) = decode_varint(data, pos)?;
            pos = new_pos;
            let field_number = tag >> 3;
            let wire_type = tag & 0x7;

            match field_number {
                3 => {
                    // ModelType model_type
                    let (v, new_pos) = decode_varint(data, pos)?;
                    pos = new_pos;
                    model_type = v as i32;
                }
                35 => {
                    // bool byte_fallback
                    let (v, new_pos) = decode_varint(data, pos)?;
                    pos = new_pos;
                    byte_fallback = v != 0;
                }
                40 => {
                    // int32 unk_id
                    let (v, new_pos) = decode_varint(data, pos)?;
                    pos = new_pos;
                    unk_id = v as i32;
                }
                _ => {
                    pos = skip_field(data, pos, wire_type)?;
                }
            }
        }

        Ok(TrainerSpec {
            unk_id,
            byte_fallback,
            model_type,
        })
    }

    //noinspection SpellCheckingInspection
    fn decode_normalizer_spec(data: &[u8]) -> Result<NormalizerSpec, TtsError> {
        let mut add_dummy_prefix = true;
        let mut remove_extra_whitespaces = true;
        let mut escape_whitespaces = true;

        let mut pos = 0;
        while pos < data.len() {
            let (tag, new_pos) = decode_varint(data, pos)?;
            pos = new_pos;
            let field_number = tag >> 3;
            let wire_type = tag & 0x7;

            match field_number {
                1 => {
                    // bytes precompiled_charsmap (skip, we use our own normalization)
                    let (len, new_pos) = decode_varint(data, pos)?;
                    pos = new_pos;
                    pos += len;
                }
                3 => {
                    // bool add_dummy_prefix
                    let (v, new_pos) = decode_varint(data, pos)?;
                    pos = new_pos;
                    add_dummy_prefix = v != 0;
                }
                4 => {
                    // bool remove_extra_whitespaces
                    let (v, new_pos) = decode_varint(data, pos)?;
                    pos = new_pos;
                    remove_extra_whitespaces = v != 0;
                }
                5 => {
                    // bool escape_whitespaces
                    let (v, new_pos) = decode_varint(data, pos)?;
                    pos = new_pos;
                    escape_whitespaces = v != 0;
                }
                _ => {
                    pos = skip_field(data, pos, wire_type)?;
                }
            }
        }

        Ok(NormalizerSpec {
            add_dummy_prefix,
            remove_extra_whitespaces,
            escape_whitespaces,
        })
    }

    //noinspection SpellCheckingInspection
    /// Normalize a character using NFKC-like normalization.
    /// This implements the common character mappings that SentencePiece's
    /// precompiled_charsmap would apply.
    /// Note: CJK punctuation (。，！？etc) is NOT normalized — they have their own
    /// vocabulary entries in the SentencePiece model.
    fn normalize_char(c: char) -> char {
        match c {
            // Fullwidth ASCII (U+FF01-U+FF5E) → ASCII (U+0021-U+007E)
            // This covers: ！＂＃＄％＆＇（）＊＋，－．／０-９：；＜＝＞？＠Ａ-Ｚ［＼］＾＿｀ａ-ｚ｛｜｝～
            '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFF01 + 0x0021).unwrap_or(c),
            // Fullwidth space → ASCII space
            '\u{3000}' => ' ',
            // Latin ligatures
            '\u{FB00}' => 'f', // ﬀ → f
            '\u{FB01}' => 'f', // ﬁ → f
            '\u{FB02}' => 'f', // ﬂ → f
            '\u{FB03}' => 'f', // ﬃ → f
            '\u{FB04}' => 'f', // ﬄ → f
            '\u{FB05}' => 'f', // ﬅ → f
            '\u{FB06}' => 'f', // ﬆ → f
            _ => c,
        }
    }

    /// Normalize text before tokenization.
    /// Applies NFKC-like character normalization and whitespace normalization.
    fn normalize(&self, text: &str) -> String {
        let mut result = String::new();

        if self.add_dummy_prefix {
            result.push('\u{2581}'); // ▁ (lower one eighth block)
        }

        let chars: Vec<char> = text.chars().collect();
        let mut prev_is_space = self.add_dummy_prefix;

        for &c in &chars {
            let normalized_c = Self::normalize_char(c);

            if normalized_c == ' '
                || normalized_c == '\t'
                || normalized_c == '\n'
                || normalized_c == '\r'
            {
                if self.escape_whitespaces {
                    if !prev_is_space {
                        result.push('\u{2581}');
                        prev_is_space = true;
                    }
                    if !self.remove_extra_whitespaces {
                        result.push('\u{2581}');
                    }
                } else {
                    if !self.remove_extra_whitespaces || !prev_is_space {
                        result.push(normalized_c);
                        prev_is_space = true;
                    }
                }
            } else {
                result.push(normalized_c);
                prev_is_space = false;
            }
        }

        result
    }

    /// Encode text to token IDs.
    /// Uses BPE algorithm for BPE models, Viterbi for Unigram models.
    pub(super) fn encode(&self, text: &str) -> Result<Vec<Token>, TtsError> {
        let normalized = self.normalize(text);

        if self.model_type == 2 {
            // BPE encoding
            self.encode_bpe(&normalized)
        } else {
            // Unigram Viterbi encoding
            self.encode_unigram(&normalized)
        }
    }

    /// BPE encoding: start with characters, repeatedly merge highest-priority pair.
    fn encode_bpe(&self, normalized: &str) -> Result<Vec<Token>, TtsError> {
        let chars: Vec<char> = normalized.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Build character byte offsets
        let mut char_to_byte = Vec::with_capacity(n + 1);
        let mut byte_offset = 0;
        for c in &chars {
            char_to_byte.push(byte_offset);
            byte_offset += c.len_utf8();
        }
        char_to_byte.push(byte_offset);

        let normalized_bytes = normalized.as_bytes();

        // Initialize tokens: each character is a token
        //  is (start_char, end_char, piece_id, score)
        #[derive(Clone, Debug)]
        #[allow(dead_code)]
        struct BpeToken {
            start: usize,
            end: usize,
            id: i32,
            score: f32,
        }

        let mut tokens: Vec<BpeToken> = Vec::new();

        // Find initial pieces for each character
        for pos in 0..n {
            let byte_start = char_to_byte[pos];
            let byte_end = char_to_byte[pos + 1];
            let s = from_utf8(&normalized_bytes[byte_start..byte_end])?;

            if let Some(&id) = self.piece_to_id.get(s) {
                tokens.push(BpeToken {
                    start: pos,
                    end: pos + 1,
                    id,
                    score: self.pieces[id as usize].score,
                });
            } else if self.byte_fallback {
                // Byte fallback
                let byte_val = normalized_bytes[byte_start];
                let byte_piece = format!("<0x{:02X}>", byte_val);
                if let Some(&id) = self.piece_to_id.get(&byte_piece) {
                    tokens.push(BpeToken {
                        start: pos,
                        end: pos + 1,
                        id,
                        score: self.pieces[id as usize].score,
                    });
                } else {
                    tokens.push(BpeToken {
                        start: pos,
                        end: pos + 1,
                        id: self.unk_id,
                        score: self.min_score - 10.0,
                    });
                }
            } else {
                tokens.push(BpeToken {
                    start: pos,
                    end: pos + 1,
                    id: self.unk_id,
                    score: self.min_score - 10.0,
                });
            }
        }

        // Repeatedly merge the highest-priority adjacent pair
        loop {
            if tokens.len() <= 1 {
                break;
            }

            // Find the best merge pair
            let mut best_merge_idx = None;
            let mut best_merge_score = f32::NEG_INFINITY;

            for i in 0..tokens.len() - 1 {
                let left = &tokens[i];
                let right = &tokens[i + 1];

                // Build the merged piece string
                let merged_start = left.start;
                let merged_end = right.end;
                let byte_start = char_to_byte[merged_start];
                let byte_end = char_to_byte[merged_end];
                let merged_str = from_utf8(&normalized_bytes[byte_start..byte_end])?;

                if let Some(&id) = self.piece_to_id.get(merged_str) {
                    let score = self.pieces[id as usize].score;
                    if score > best_merge_score {
                        best_merge_score = score;
                        best_merge_idx = Some(i);
                    }
                }
            }

            match best_merge_idx {
                Some(idx) => {
                    // Merge tokens[idx] and tokens[idx+1]
                    let left = &tokens[idx];
                    let right = &tokens[idx + 1];
                    let merged_start = left.start;
                    let merged_end = right.end;
                    let byte_start = char_to_byte[merged_start];
                    let byte_end = char_to_byte[merged_end];
                    let merged_str = from_utf8(&normalized_bytes[byte_start..byte_end])?;
                    let merged_id = self
                        .piece_to_id
                        .get(merged_str)
                        .copied()
                        .unwrap_or(self.unk_id);

                    let merged_token = BpeToken {
                        start: merged_start,
                        end: merged_end,
                        id: merged_id,
                        score: best_merge_score,
                    };

                    // Replace the two tokens with the merged one
                    tokens[idx] = merged_token;
                    tokens.remove(idx + 1);
                }
                None => {
                    // No more merges possible
                    break;
                }
            }
        }

        // Convert to output format
        let result: Vec<Token> = tokens
            .iter()
            .map(|t| {
                let piece_str: String = chars[t.start..t.end].iter().collect();
                Token {
                    id: t.id,
                    piece: piece_str,
                }
            })
            .collect();

        Ok(result)
    }

    /// Unigram Viterbi encoding.
    fn encode_unigram(&self, normalized: &str) -> Result<Vec<Token>, TtsError> {
        let chars: Vec<char> = normalized.chars().collect();
        let n = chars.len();

        if n == 0 {
            return Ok(Vec::new());
        }

        // Build character to byte offset mapping
        let mut char_to_byte = Vec::with_capacity(n + 1);
        let mut byte_offset = 0;
        for c in &chars {
            char_to_byte.push(byte_offset);
            byte_offset += c.len_utf8();
        }
        char_to_byte.push(byte_offset);

        let normalized_bytes = normalized.as_bytes();

        // Optimized Viterbi: best_path[i] = best score ending at position i
        #[derive(Clone)]
        struct BestPathNode {
            score: f32,
            starts_at: usize,
            id: i32,
        }

        let mut best_path: Vec<BestPathNode> = vec![
            BestPathNode {
                score: f32::NEG_INFINITY,
                starts_at: 0,
                id: -1,
            };
            n + 1
        ];
        best_path[0].score = 0.0;

        for pos in 0..n {
            let byte_start = char_to_byte[pos];
            let mut found_any = false;

            for end in (pos + 1)..=n.min(pos + 256) {
                let byte_end = char_to_byte[end];
                let substr = &normalized_bytes[byte_start..byte_end];

                if let Ok(s) = from_utf8(substr)
                    && let Some(&id) = self.piece_to_id.get(s)
                {
                    let piece = &self.pieces[id as usize];
                    if piece.piece_type != PieceType::Unused {
                        let score = best_path[pos].score + piece.score;
                        if score > best_path[end].score {
                            best_path[end] = BestPathNode {
                                score,
                                starts_at: pos,
                                id,
                            };
                        }
                        found_any = true;
                    }
                }
            }

            if !found_any && self.byte_fallback {
                let byte_val = normalized_bytes[byte_start];
                let byte_piece = format!("<0x{:02X}>", byte_val);
                if let Some(&id) = self.piece_to_id.get(&byte_piece) {
                    let score = best_path[pos].score + self.pieces[id as usize].score;
                    if score > best_path[pos + 1].score {
                        best_path[pos + 1] = BestPathNode {
                            score,
                            starts_at: pos,
                            id,
                        };
                    }
                }
            }

            if best_path[pos + 1].id == -1 && best_path[pos + 1].starts_at == 0 {
                let unk_score = best_path[pos].score + self.min_score - 10.0;
                best_path[pos + 1] = BestPathNode {
                    score: unk_score,
                    starts_at: pos,
                    id: self.unk_id,
                };
            }
        }

        // Backtrace
        let mut tokens = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let node = &best_path[pos];
            let piece_str: String = chars[node.starts_at..pos].iter().collect();
            tokens.push(Token {
                id: node.id,
                piece: piece_str,
            });
            pos = node.starts_at;
        }

        tokens.reverse();
        Ok(tokens)
    }
}

struct TrainerSpec {
    unk_id: i32,
    byte_fallback: bool,
    model_type: i32, // 1=UNIGRAM, 2=BPE, 3=WORD, 4=CHAR
}

struct NormalizerSpec {
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
}

// Protobuf decoding helpers

fn decode_varint(data: &[u8], pos: usize) -> Result<(usize, usize), TtsError> {
    let mut result = 0usize;
    let mut shift = 0;
    let mut p = pos;
    loop {
        if p >= data.len() {
            return Err(TtsError::EndOfVarint(format!(
                "unexpected end of varint at position {}",
                pos
            )));
        }
        let byte = data[p];
        result |= ((byte & 0x7F) as usize) << shift;
        p += 1;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(TtsError::VarintTooLong(format!(
                "varint at position {} is too long",
                pos
            )));
        }
    }
    Ok((result, p))
}

fn skip_field(data: &[u8], pos: usize, wire_type: usize) -> Result<usize, TtsError> {
    match wire_type {
        0 => {
            // Varint
            let (_, new_pos) = decode_varint(data, pos)?;
            Ok(new_pos)
        }
        1 => {
            // 64-bit
            Ok(pos + 8)
        }
        2 => {
            // Length-delimited
            let (len, new_pos) = decode_varint(data, pos)?;
            Ok(new_pos + len)
        }
        5 => {
            // 32-bit
            Ok(pos + 4)
        }
        _ => Err(TtsError::InvalidWireType(format!(
            "invalid wire type {} at position {}",
            wire_type, pos
        ))),
    }
}
