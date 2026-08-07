use std::collections::HashMap;
use std::path::Path;

use ort::memory::Allocator;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer as HfTokenizer;

use crate::errors::TranslatorError;
use crate::traits::{InferenceEngine, Tokenizer};

// ──────────────────────────────────────────────
//  HuggingFaceTokenizer
// ──────────────────────────────────────────────

pub struct HuggingFaceTokenizer {
    inner: HfTokenizer,
}

impl HuggingFaceTokenizer {
    pub fn from_file(path: &str) -> Result<Self, TranslatorError> {
        let inner =
            HfTokenizer::from_file(Path::new(path)).map_err(TranslatorError::TokenizerError)?;
        Ok(Self { inner })
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i64>, TranslatorError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(TranslatorError::TokenizerError)?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }

    fn decode(&self, ids: &[i64]) -> Result<String, TranslatorError> {
        let token_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner
            .decode(&token_ids, true)
            .map_err(TranslatorError::TokenizerError)
    }
}

// ──────────────────────────────────────────────
//  OnnxInferenceEngine
// ──────────────────────────────────────────────

const MAX_OUTPUT_LEN: usize = 256;
const EOS_TOKEN_ID: u32 = 43741;
const PAD_TOKEN_ID: u32 = 57059;
const DECODER_START_TOKEN_ID: i64 = 57059;
const NUM_LAYERS: usize = 6;
const NUM_HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const D_MODEL: usize = 1024;

pub struct OnnxInferenceEngine {
    encoder: Session,
    decoder: Session,
    max_source_len: usize,
}

impl OnnxInferenceEngine {
    pub fn load(encoder_path: &str, decoder_path: &str) -> Result<Self, TranslatorError> {
        let encoder = Session::builder()
            .map_err(TranslatorError::ModelLoadFailed)?
            .commit_from_file(encoder_path)
            .map_err(TranslatorError::ModelLoadFailed)?;
        let decoder = Session::builder()
            .map_err(TranslatorError::ModelLoadFailed)?
            .commit_from_file(decoder_path)
            .map_err(TranslatorError::ModelLoadFailed)?;
        Ok(Self {
            encoder,
            decoder,
            max_source_len: MAX_MODEL_TOKENS,
        })
    }

    pub fn with_max_source_len(mut self, max: usize) -> Self {
        self.max_source_len = max;
        self
    }

    /// Runs the encoder over a batch of padded sequences and returns the
    /// `last_hidden_state` tensor (`[batch, seq_len, D_MODEL]`) together with
    /// the attention-mask tensor (`[batch, seq_len]`).
    fn encode_input(
        &mut self,
        input_ids: &[Vec<i64>],
        attention_mask: &[Vec<i64>],
    ) -> Result<(Tensor<f32>, Tensor<i64>), TranslatorError> {
        let batch_size = input_ids.len();
        let source_len = input_ids[0].len();

        let mut ids_flat = Vec::with_capacity(batch_size * source_len);
        for row in input_ids {
            ids_flat.extend_from_slice(row);
        }
        let mut mask_flat = Vec::with_capacity(batch_size * source_len);
        for row in attention_mask {
            mask_flat.extend_from_slice(row);
        }

        let input_ids_tensor = Tensor::from_array(([batch_size, source_len], ids_flat))
            .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;
        let attention_mask_tensor = Tensor::from_array(([batch_size, source_len], mask_flat))
            .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;

        let encoder_inputs: Vec<(String, ort::value::DynValue)> = vec![
            ("input_ids".into(), input_ids_tensor.into_dyn()),
            (
                "attention_mask".into(),
                attention_mask_tensor.clone().into_dyn(),
            ),
        ];
        let encoder_outputs = self
            .encoder
            .run(encoder_inputs)
            .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;

        let hidden_states_value = &encoder_outputs["last_hidden_state"];
        let (_, hidden_states_data) = hidden_states_value
            .try_extract_tensor::<f32>()
            .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;
        let hidden_states_vec: Vec<f32> = hidden_states_data.to_vec();

        let encoder_hidden_states_tensor =
            Tensor::from_array(([batch_size, source_len, D_MODEL], hidden_states_vec))
                .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;

        Ok((encoder_hidden_states_tensor, attention_mask_tensor))
    }

    /// Autoregressively decodes a batch in lock-step, returning one generated
    /// token sequence per batch item (stripped of start/pad tokens).
    fn decode(
        &mut self,
        encoder_hidden_states: &Tensor<f32>,
        enc_attention_mask: &Tensor<i64>,
        batch_size: usize,
    ) -> Result<Vec<Vec<i64>>, TranslatorError> {
        let mut generated: Vec<Vec<i64>> = vec![Vec::new(); batch_size];
        let mut finished: Vec<bool> = vec![false; batch_size];
        let mut remaining = batch_size;
        let mut past_key_values: HashMap<String, Vec<f32>> = HashMap::new();
        let mut next_input: Vec<i64> = vec![DECODER_START_TOKEN_ID; batch_size];

        for _step in 0..MAX_OUTPUT_LEN {
            if remaining == 0 {
                break;
            }

            // Only the newest token per item is fed each step (dec_input_len = 1).
            let dec_input_ids_tensor = Tensor::from_array(([batch_size, 1], next_input.clone()))
                .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;

            let mut decoder_inputs: Vec<(String, ort::value::DynValue)> = vec![
                ("input_ids".into(), dec_input_ids_tensor.into_dyn()),
                (
                    "encoder_attention_mask".into(),
                    enc_attention_mask.clone().into_dyn(),
                ),
                (
                    "encoder_hidden_states".into(),
                    encoder_hidden_states.clone().into_dyn(),
                ),
            ];

            for layer in 0..NUM_LAYERS {
                for suffix in &["key", "value"] {
                    let name = format!("past_key_values.{}.decoder.{}", layer, suffix);
                    let tensor = if let Some(data) = past_key_values.get(&name) {
                        let seq_len = data.len() / (batch_size * NUM_HEADS * HEAD_DIM);
                        Tensor::from_array((
                            [batch_size, NUM_HEADS, seq_len, HEAD_DIM],
                            data.clone(),
                        ))
                        .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?
                    } else {
                        let shape = [batch_size as i64, NUM_HEADS as i64, 0, HEAD_DIM as i64];
                        Tensor::<f32>::new(&Allocator::default(), shape)
                            .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?
                    };
                    decoder_inputs.push((name, tensor.into_dyn()));
                }
            }

            let decoder_outputs = self
                .decoder
                .run(decoder_inputs)
                .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;

            let logits_value = &decoder_outputs["logits"];
            let (_, logits_data) = logits_value
                .try_extract_tensor::<f32>()
                .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;

            // logits shape is [batch, 1, vocab], laid out row-major.
            let vocab_size = logits_data.len() / batch_size;
            for b in 0..batch_size {
                if finished[b] {
                    continue;
                }
                let start = b * vocab_size;
                let last_token_logits = &logits_data[start..start + vocab_size];

                let mut best_token = 0u32;
                let mut best_score = f32::NEG_INFINITY;
                for (i, &score) in last_token_logits.iter().enumerate() {
                    let token = i as u32;
                    if score > best_score && token > 0 {
                        best_score = score;
                        best_token = token;
                    }
                }

                if best_token == EOS_TOKEN_ID || best_token == PAD_TOKEN_ID {
                    finished[b] = true;
                    remaining -= 1;
                    next_input[b] = EOS_TOKEN_ID as i64;
                } else {
                    generated[b].push(best_token as i64);
                    next_input[b] = best_token as i64;
                }
            }

            for layer in 0..NUM_LAYERS {
                for suffix in &["key", "value"] {
                    let output_name = format!("present.{}.decoder.{}", layer, suffix);
                    let cache_name = format!("past_key_values.{}.decoder.{}", layer, suffix);

                    if let Some(value) = decoder_outputs.get(&output_name) {
                        let (_, data) = value
                            .try_extract_tensor::<f32>()
                            .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;
                        past_key_values.insert(cache_name, data.to_vec());
                    }
                }
            }
        }

        for tokens in &mut generated {
            if tokens.first() == Some(&DECODER_START_TOKEN_ID) {
                tokens.remove(0);
            }
            tokens.retain(|&t| t != PAD_TOKEN_ID as i64);
        }

        Ok(generated)
    }
}

impl InferenceEngine for OnnxInferenceEngine {
    fn max_source_len(&self) -> usize {
        self.max_source_len
    }

    fn run_inference(&mut self, input_ids: &[i64]) -> Result<Vec<i64>, TranslatorError> {
        let batch = vec![input_ids.to_vec()];
        let mask = vec![vec![1i64; input_ids.len()]];
        let (encoder_hidden_states, enc_attention_mask) = self.encode_input(&batch, &mask)?;
        let mut outputs = self.decode(&encoder_hidden_states, &enc_attention_mask, 1)?;
        Ok(outputs.remove(0))
    }

    fn run_inference_batch(
        &mut self,
        input_ids: &[Vec<i64>],
        attention_mask: &[Vec<i64>],
    ) -> Result<Vec<Vec<i64>>, TranslatorError> {
        let batch_size = input_ids.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        if batch_size != attention_mask.len() {
            return Err(TranslatorError::InvalidBatchLength {
                expected: batch_size,
                actual: attention_mask.len(),
            });
        }

        let source_len = input_ids[0].len();
        for (row, mask_row) in input_ids.iter().zip(attention_mask.iter()) {
            if row.len() != source_len {
                return Err(TranslatorError::InvalidBatchLength {
                    expected: source_len,
                    actual: row.len(),
                });
            }
            if mask_row.len() != source_len {
                return Err(TranslatorError::InvalidBatchLength {
                    expected: source_len,
                    actual: mask_row.len(),
                });
            }
        }

        let (encoder_hidden_states, enc_attention_mask) =
            self.encode_input(input_ids, attention_mask)?;
        self.decode(&encoder_hidden_states, &enc_attention_mask, batch_size)
    }
}

// ──────────────────────────────────────────────
//  Smart chunking with safety buffer
// ──────────────────────────────────────────────

/// Hard limit on the number of source tokens the model can accept.
pub const MAX_MODEL_TOKENS: usize = 512;
/// Token buffer reserved for special tokens (`<s>`, `</s>`, `<pad>`) and
/// future context overlap.
pub const SAFETY_MARGIN: usize = 32;
/// Effective per-chunk limit after reserving the safety margin.
pub const TARGET_CHUNK_TOKENS: usize = MAX_MODEL_TOKENS - SAFETY_MARGIN;

/// Sentence- and paragraph-ending punctuation, in descending priority.
const SENTENCE_ENDERS: &[char] = &['.', '!', '?', ';'];

/// Boundary levels used during chunking, ordered by decreasing preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Boundary {
    Paragraph,
    Line,
    Sentence,
    Word,
}

impl Boundary {
    /// The next, lower-priority boundary to fall back to.
    fn next(self) -> Option<Self> {
        match self {
            Boundary::Paragraph => Some(Boundary::Line),
            Boundary::Line => Some(Boundary::Sentence),
            Boundary::Sentence => Some(Boundary::Word),
            Boundary::Word => None,
        }
    }
}

/// Configurable, boundary-aware text chunker.
///
/// Splits text into chunks that never exceed `max_model_tokens - safety_margin`
/// tokens. Breaks are preferred at paragraph (`\n\n`), newline (`\n`) and
/// sentence (`.!?;`) boundaries; continuous un-punctuated text falls back to
/// word-level (and finally character-level) splits so the target limit is
/// always honoured.
#[derive(Debug, Clone, Copy)]
pub struct Chunker {
    max_model_tokens: usize,
    safety_margin: usize,
}

impl Default for Chunker {
    fn default() -> Self {
        Self {
            max_model_tokens: MAX_MODEL_TOKENS,
            safety_margin: SAFETY_MARGIN,
        }
    }
}

impl Chunker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a chunker with explicit model limit and safety margin.
    pub fn with_limits(max_model_tokens: usize, safety_margin: usize) -> Self {
        assert!(
            safety_margin < max_model_tokens,
            "safety_margin ({safety_margin}) must be smaller than max_model_tokens ({max_model_tokens})"
        );
        Self {
            max_model_tokens,
            safety_margin,
        }
    }

    pub fn max_model_tokens(&self) -> usize {
        self.max_model_tokens
    }

    pub fn safety_margin(&self) -> usize {
        self.safety_margin
    }

    pub fn target_chunk_tokens(&self) -> usize {
        self.max_model_tokens - self.safety_margin
    }

    /// Splits `text` into chunks bounded by the effective target limit.
    ///
    /// `count_tokens` reports the token count of a piece of text, letting
    /// callers plug in a real tokenizer (or a cheap proxy in tests).
    pub fn split<E>(
        &self,
        text: &str,
        count_tokens: impl Fn(&str) -> Result<usize, E>,
    ) -> Result<Vec<String>, E> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let target = self.target_chunk_tokens();
        if count_tokens(text)? <= target {
            return Ok(vec![text.to_string()]);
        }

        let mut chunks: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut current_tokens = 0usize;

        self.pack_at_level(
            &split_keeping_separator(text, "\n\n"),
            Boundary::Paragraph,
            &mut chunks,
            &mut current,
            &mut current_tokens,
            &count_tokens,
        )?;

        if current_tokens > 0 {
            chunks.push(current.trim().to_string());
        }

        if chunks.is_empty() {
            chunks.push(text.to_string());
        }

        Ok(chunks)
    }

    /// Greedily packs `units` into chunks, descending to lower-priority
    /// boundaries whenever a single unit overflows the target.
    fn pack_at_level<E>(
        &self,
        units: &[String],
        level: Boundary,
        chunks: &mut Vec<String>,
        current: &mut String,
        current_tokens: &mut usize,
        count_tokens: &dyn Fn(&str) -> Result<usize, E>,
    ) -> Result<(), E> {
        let target = self.target_chunk_tokens();

        for unit in units {
            if unit.trim().is_empty() {
                continue;
            }
            let unit_tokens = count_tokens(unit)?;

            if unit_tokens > target {
                flush_chunk(chunks, current, current_tokens);
                match level.next() {
                    Some(next) if next != Boundary::Word => {
                        let sub = match next {
                            Boundary::Paragraph | Boundary::Line => {
                                split_keeping_separator(unit, "\n")
                            }
                            Boundary::Sentence => {
                                split_keeping_separator_chars(unit, SENTENCE_ENDERS)
                            }
                            Boundary::Word => unreachable!(),
                        };
                        self.pack_at_level(
                            &sub,
                            next,
                            chunks,
                            current,
                            current_tokens,
                            count_tokens,
                        )?;
                    }
                    _ => {
                        for piece in split_at_word_boundaries(unit, target, count_tokens)? {
                            chunks.push(piece);
                        }
                    }
                }
            } else if *current_tokens == 0 {
                *current = unit.clone();
                *current_tokens = unit_tokens;
            } else if *current_tokens + unit_tokens <= target {
                current.push_str(unit);
                *current_tokens += unit_tokens;
            } else {
                flush_chunk(chunks, current, current_tokens);
                *current = unit.clone();
                *current_tokens = unit_tokens;
            }
        }

        Ok(())
    }
}

fn flush_chunk(chunks: &mut Vec<String>, current: &mut String, current_tokens: &mut usize) {
    if *current_tokens > 0 {
        chunks.push(current.trim().to_string());
        current.clear();
        *current_tokens = 0;
    }
}

/// Splits `text` at every occurrence of `sep`, keeping the separator attached
/// to the piece that precedes it.
fn split_keeping_separator(text: &str, sep: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut rest = text;
    while let Some(idx) = rest.find(sep) {
        let end = idx + sep.len();
        pieces.push(rest[..end].to_string());
        rest = &rest[end..];
    }
    if !rest.is_empty() {
        pieces.push(rest.to_string());
    }
    pieces
}

/// Splits `text` after every occurrence of any char in `seps`, keeping the
/// punctuation attached to the piece that precedes it.
fn split_keeping_separator_chars(text: &str, seps: &[char]) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut start = 0;
    for (idx, ch) in text.char_indices() {
        if seps.contains(&ch) {
            let end = idx + ch.len_utf8();
            pieces.push(text[start..end].to_string());
            start = end;
        }
    }
    if start < text.len() {
        pieces.push(text[start..].to_string());
    }
    pieces
}

/// Fallback split that strictly honours `target` by breaking only at
/// whitespace/word boundaries.
fn split_at_word_boundaries<E>(
    text: &str,
    target: usize,
    count_tokens: &dyn Fn(&str) -> Result<usize, E>,
) -> Result<Vec<String>, E> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut pieces: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_tokens = 0usize;

    for word in words {
        let word_tokens = count_tokens(word)?;
        if word_tokens > target {
            if current_tokens > 0 {
                pieces.push(current.trim().to_string());
                current.clear();
                current_tokens = 0;
            }
            for piece in split_at_char_boundaries(word, target, count_tokens)? {
                pieces.push(piece);
            }
            continue;
        }
        if current_tokens > 0 && current_tokens + word_tokens > target {
            pieces.push(current.trim().to_string());
            current.clear();
            current_tokens = 0;
        }
        if current_tokens > 0 {
            current.push(' ');
        }
        current.push_str(word);
        current_tokens += word_tokens;
    }

    if current_tokens > 0 {
        pieces.push(current.trim().to_string());
    }

    Ok(pieces)
}

/// Last-resort split used when a single word alone exceeds the target.
fn split_at_char_boundaries<E>(
    text: &str,
    target: usize,
    count_tokens: &dyn Fn(&str) -> Result<usize, E>,
) -> Result<Vec<String>, E> {
    let mut pieces: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if !current.is_empty() && count_tokens(&current)? >= target {
            pieces.push(current.clone());
            current.clear();
        }
        current.push(ch);
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    Ok(pieces)
}

// ──────────────────────────────────────────────
//  Context overlap & conservative trimming
// ──────────────────────────────────────────────

/// Minimum size of the forward context-overlap window, in words.
pub const CONTEXT_OVERLAP_MIN_WORDS: usize = 16;
/// Maximum size of the forward context-overlap window, in words.
pub const CONTEXT_OVERLAP_MAX_WORDS: usize = 24;

/// Prepends a forward context-overlap window to every chunk `i > 0`.
///
/// For each chunk `i > 0`, the last
/// [`CONTEXT_OVERLAP_MIN_WORDS`..`CONTEXT_OVERLAP_MAX_WORDS`] words of the
/// previous chunk's *final* body are prepended so the model can see the
/// boundary context it otherwise lost. The combined `overlap + body` never
/// exceeds `target` tokens:
///
/// 1. the overlap is sized as large as possible within
///    `[CONTEXT_OVERLAP_MIN_WORDS, CONTEXT_OVERLAP_MAX_WORDS]` while still
///    leaving room for the full chunk body;
/// 2. only when even the minimum overlap overflows is the chunk body trimmed
///    from the front (the tail is kept so the next chunk's overlap stays
///    intact). This is the conservative last resort — content is never trimmed
///    when the overlap alone can absorb the overflow.
pub fn apply_context_overlap<E>(
    chunks: &[String],
    count_tokens: &dyn Fn(&str) -> Result<usize, E>,
    target: usize,
) -> Result<Vec<String>, E> {
    let mut bodies: Vec<String> = Vec::with_capacity(chunks.len());
    let mut augmented: Vec<String> = Vec::with_capacity(chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        if i == 0 {
            let body = chunk.trim().to_string();
            bodies.push(body.clone());
            augmented.push(body);
            continue;
        }

        let body_tokens = count_tokens(chunk)?;
        let prev_words: Vec<&str> = bodies[i - 1].split_whitespace().collect();
        let max_overlap = CONTEXT_OVERLAP_MAX_WORDS.min(prev_words.len());

        // Prefer the largest overlap that leaves room for the full body.
        let mut chosen_overlap: Option<String> = None;
        let mut overlap_size = max_overlap;
        while overlap_size >= CONTEXT_OVERLAP_MIN_WORDS {
            let overlap = prev_words[prev_words.len() - overlap_size..].join(" ");
            let overlap_tokens = count_tokens(&overlap)?;
            if body_tokens + overlap_tokens <= target {
                chosen_overlap = Some(overlap);
                break;
            }
            if overlap_size == CONTEXT_OVERLAP_MIN_WORDS {
                break;
            }
            overlap_size -= 1;
        }

        let (overlap, body) = match chosen_overlap {
            Some(overlap) => (overlap, chunk.trim().to_string()),
            None => {
                let min_overlap = CONTEXT_OVERLAP_MIN_WORDS.min(prev_words.len());
                let overlap = prev_words[prev_words.len() - min_overlap..].join(" ");
                let overlap_tokens = count_tokens(&overlap)?;
                let budget = target.saturating_sub(overlap_tokens);
                let body = trim_front_to_budget(chunk, budget, count_tokens)?;
                (overlap, body)
            }
        };

        bodies.push(body.clone());
        let mut combined = overlap;
        if !combined.is_empty() && !body.is_empty() {
            combined.push(' ');
        }
        combined.push_str(&body);
        augmented.push(combined);
    }

    Ok(augmented)
}

/// Removes words from the front of `text` until its token count fits within
/// `budget`.
///
/// The token count of a word-suffix is (non-strictly) monotonic in the trim
/// offset, so a binary search finds the smallest feasible offset in `O(log n)`
/// token counts instead of a linear scan.
fn trim_front_to_budget<E>(
    text: &str,
    budget: usize,
    count_tokens: &dyn Fn(&str) -> Result<usize, E>,
) -> Result<String, E> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() || count_tokens(text)? <= budget {
        return Ok(text.to_string());
    }

    // `lo` is infeasible (the full text), `hi` (the empty suffix) is feasible.
    let mut lo = 0usize;
    let mut hi = words.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if count_tokens(&words[mid..].join(" "))? <= budget {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Ok(words[lo..].join(" "))
}

/// Common English stop-words. Used to reject weak, stop-word-only overlap
/// matches that would otherwise mis-trim meaningful translation content.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet", "if", "then", "else", "of",
    "to", "in", "on", "at", "by", "with", "from", "into", "upon", "over", "under", "as", "is",
    "are", "was", "were", "be", "been", "being", "am", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "their", "our", "do", "does", "did", "have", "has", "had", "not", "no", "there", "here",
    "all", "each", "both", "some", "any", "more", "most", "such", "than", "up", "down", "out",
    "off", "about", "after", "before", "between", "against", "during", "through", "above", "below",
    "again", "further", "once", "only", "own", "same", "too", "very", "just", "while", "when",
    "where", "why", "how", "what", "which", "who", "whom", "whose",
];

fn is_stop_word(word: &str) -> bool {
    STOP_WORDS.contains(&word)
}

/// Normalizes a single word for boundary comparison: strips surrounding
/// punctuation and lowercases. Returns `None` for words that normalize to
/// nothing.
fn normalize_word(word: &str) -> Option<String> {
    let normalized = word
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

/// Removes the first `count` words of `words`, capped at `max_trim`.
fn trim_first_words(words: &[&str], count: usize, max_trim: usize) -> String {
    let trimmed = count.min(max_trim).min(words.len());
    words[trimmed..].join(" ")
}

/// Classic dynamic-programming Levenshtein edit distance between two strings.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();

    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];

    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let substitution = prev[j] + usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(substitution);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b.len()]
}

/// Normalized similarity of two words in `[0, 1]`, scaled by edit distance.
fn word_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let max_len = a.chars().count().max(b.chars().count());
    1.0 - (levenshtein(a, b) as f64 / max_len as f64)
}

/// Mean per-word similarity of two equally-sized word n-grams.
fn ngram_similarity(a: &[String], b: &[String]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 1.0;
    }
    let sum: f64 = a.iter().zip(b).map(|(x, y)| word_similarity(x, y)).sum();
    sum / a.len() as f64
}

/// Conservatively removes duplicated overlap text from a chunk translation.
///
/// After context overlap is prepended to chunk `i`, its translation begins by
/// re-translating source that chunk `i - 1` already covered. The trimmer
/// detects that duplicated prefix by comparing the *tail* of chunk `i - 1`'s
/// output with the *head* of chunk `i`'s output and strips the match.
///
/// The decision pipeline is strictly sequential and conservative:
///
/// 1. **Normalized boundary exact match** — case/whitespace/punctuation are
///    normalized and the longest contiguous prefix/suffix match is trimmed.
/// 2. **Strict multi-word fuzzy match** — if no exact match, `2..=6` word
///    n-grams are compared with normalized edit-distance similarity; trimming
///    requires the similarity to clearly exceed the high threshold, a match
///    anchored at both boundaries, and at least two meaningful (non-stop-word)
///    words.
/// 3. **Conservative fallback** — single-word matches, stop-word-only matches,
///    near-threshold (ambiguous) matches, and everything else is left untrimmed.
///
/// All comparisons are restricted to the last `search_window_words` of the
/// previous output and the first `search_window_words` of the current output,
/// and no more than `max_trim_words` words are ever removed.
#[derive(Debug, Clone)]
pub struct OverlapTrimmer {
    search_window_words: usize,
    max_trim_words: usize,
    ngram_min: usize,
    ngram_max: usize,
    similarity_threshold: f64,
    confidence_margin: f64,
}

impl Default for OverlapTrimmer {
    fn default() -> Self {
        Self::new()
    }
}

impl OverlapTrimmer {
    pub fn new() -> Self {
        Self {
            search_window_words: 20,
            max_trim_words: 20,
            ngram_min: 2,
            ngram_max: 6,
            similarity_threshold: 0.85,
            confidence_margin: 0.05,
        }
    }

    /// Sets the size of the boundary search window (in words) on each side.
    pub fn with_search_window(mut self, words: usize) -> Self {
        self.search_window_words = words.max(1);
        self
    }

    /// Sets the hard upper bound on how many words may be trimmed.
    pub fn with_max_trim_words(mut self, words: usize) -> Self {
        self.max_trim_words = words.max(1);
        self
    }

    /// Trims duplicated overlap text from the head of `cur_output`, using
    /// `prev_output` as the reference for what was already translated.
    pub fn trim(&self, prev_output: &str, cur_output: &str) -> String {
        let prev_words: Vec<&str> = prev_output.split_whitespace().collect();
        let cur_words: Vec<&str> = cur_output.split_whitespace().collect();
        if prev_words.is_empty() || cur_words.len() <= 1 {
            return cur_output.to_string();
        }

        let prev_norm: Vec<String> = prev_words
            .iter()
            .filter_map(|w| normalize_word(w))
            .collect();
        let cur_norm: Vec<String> = cur_words.iter().filter_map(|w| normalize_word(w)).collect();
        if prev_norm.is_empty() || cur_norm.is_empty() {
            return cur_output.to_string();
        }

        let window = self.search_window_words;
        let pw = &prev_norm[prev_norm.len().saturating_sub(window)..];
        let cw = &cur_norm[..cur_norm.len().min(window)];

        // 1. Normalized boundary exact match: longest common suffix/prefix.
        let k_max = pw.len().min(cw.len()).min(self.max_trim_words);
        let mut exact_k = 0usize;
        for k in 1..=k_max {
            if pw[pw.len() - k..] == cw[..k] {
                exact_k = k;
            }
        }
        if exact_k >= 2 {
            let matched = &cw[..exact_k];
            if matched.iter().any(|word| !is_stop_word(word)) {
                return trim_first_words(&cur_words, exact_k, self.max_trim_words);
            }
        }

        // 2. Strict multi-word fuzzy match on 2..=6 word n-grams. Iterate from
        //    the largest n-gram down so the longest anchored, high-confidence
        //    span wins. The confidence margin keeps near-border matches out.
        for n in (self.ngram_min..=self.ngram_max).rev() {
            if n > pw.len() || n > cw.len() {
                continue;
            }
            let prev_gram = &pw[pw.len() - n..];
            let cur_gram = &cw[..n];
            let similarity = ngram_similarity(prev_gram, cur_gram);
            if similarity < self.similarity_threshold + self.confidence_margin {
                continue;
            }
            let meaningful = cur_gram.iter().filter(|word| !is_stop_word(word)).count();
            if meaningful < 2 {
                continue;
            }
            return trim_first_words(&cur_words, n, self.max_trim_words);
        }

        // 3. Conservative fallback: keep the output untouched.
        cur_output.to_string()
    }
}

// ──────────────────────────────────────────────
//  Length bucketing & batching
// ──────────────────────────────────────────────

/// A padded batch ready for inference.
#[derive(Debug, Clone)]
pub struct ChunkBatch {
    /// Original chunk indices, in the order the rows appear in [`ChunkBatch::input_ids`].
    pub order: Vec<usize>,
    /// Padded token ids, shape `[batch_size, max_len]`.
    pub input_ids: Vec<Vec<i64>>,
    /// Attention mask: `1` for real tokens, `0` for padding.
    pub attention_mask: Vec<Vec<i64>>,
    /// Padded sequence length shared by every row.
    pub max_len: usize,
}

/// Groups chunks into length-bucketed batches of at most `batch_size`.
///
/// Chunks are sorted by token length first so that similar-length sequences
/// share a batch and zero-padding is minimised. Each batch is padded with
/// `pad_id` up to its longest sequence and given a matching attention mask.
/// [`ChunkBatch::order`] records each chunk's original index so decoded
/// outputs can be stitched back in source order.
pub fn build_batches(
    chunks: &[(usize, Vec<i64>)],
    batch_size: usize,
    pad_id: i64,
) -> Vec<ChunkBatch> {
    if chunks.is_empty() {
        return Vec::new();
    }
    let batch_size = batch_size.max(1);

    let mut sorted: Vec<(usize, Vec<i64>)> = chunks.to_vec();
    sorted.sort_by_key(|(index, ids)| (ids.len(), *index));

    let mut batches: Vec<ChunkBatch> = Vec::new();
    for group in sorted.chunks(batch_size) {
        let max_len = group.iter().map(|(_, ids)| ids.len()).max().unwrap_or(0);
        let mut order = Vec::with_capacity(group.len());
        let mut input_ids = Vec::with_capacity(group.len());
        let mut attention_mask = Vec::with_capacity(group.len());

        for (index, ids) in group {
            order.push(*index);
            let mut padded = ids.clone();
            padded.resize(max_len, pad_id);
            input_ids.push(padded);
            attention_mask.push(mask_for(ids.len(), max_len));
        }

        batches.push(ChunkBatch {
            order,
            input_ids,
            attention_mask,
            max_len,
        });
    }

    batches
}

/// Builds the attention mask for a row: `1` for the first `real_len`
/// positions, `0` for the remaining `max_len - real_len` padding.
fn mask_for(real_len: usize, max_len: usize) -> Vec<i64> {
    let mut mask = vec![1i64; real_len];
    mask.resize(max_len, 0i64);
    mask
}

// ──────────────────────────────────────────────
//  Auto-probing & dynamic batch size detection
// ──────────────────────────────────────────────

/// Default ceiling for automatic batch-size probing.
pub const DEFAULT_MAX_BATCH_SIZE: usize = 8;

/// Batch sizes to probe: `1, 2, 4, 8, ...` up to and including `max`.
fn probe_sizes(max: usize) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut size = 1usize;
    while size <= max {
        sizes.push(size);
        match size.checked_mul(2) {
            Some(next) => size = next,
            None => break,
        }
    }
    sizes
}

/// Runs a single probe inference at `batch_size`, returning `Ok` when the
/// engine accepts the batch shape.
fn run_probe<E: InferenceEngine>(engine: &mut E, batch_size: usize) -> Result<(), TranslatorError> {
    let input: Vec<Vec<i64>> = vec![vec![1i64, 2, 3, 4]; batch_size];
    let mask: Vec<Vec<i64>> = vec![vec![1i64; 4]; batch_size];
    engine.run_inference_batch(&input, &mask).map(|_| ())
}

/// Probes batch sizes `1 -> 2 -> 4 -> ...` (up to `max_batch_size`) and returns
/// the largest working batch size.
///
/// Recoverable execution errors (out of memory, unsupported tensor shapes,
/// execution-provider dynamic-shape failures) stop probing and fall back to the
/// last known working size. All other errors are considered unrecoverable and
/// are propagated immediately so callers fail fast.
pub fn probe_batch_size<E: InferenceEngine>(
    engine: &mut E,
    max_batch_size: usize,
) -> Result<usize, TranslatorError> {
    let mut working = 1usize;
    for size in probe_sizes(max_batch_size).into_iter().skip(1) {
        match run_probe(engine, size) {
            Ok(()) => {
                log::info!("Probe: batch size {size} OK");
                working = size;
            }
            Err(e) if e.is_recoverable_batch_error() => {
                log::warn!("Probe: batch size {size} failed ({e}); falling back to {working}");
                break;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(working)
}

// ──────────────────────────────────────────────
//  Translator (generic)
// ──────────────────────────────────────────────

pub struct Translator<T: Tokenizer, E: InferenceEngine> {
    tokenizer: T,
    engine: E,
    max_batch_size: Option<usize>,
    probed_batch_size: Option<usize>,
}

impl<T: Tokenizer, E: InferenceEngine> Translator<T, E> {
    pub fn new(tokenizer: T, engine: E) -> Self {
        Self {
            tokenizer,
            engine,
            max_batch_size: None,
            probed_batch_size: None,
        }
    }

    /// Sets an optional ceiling for auto-probed batch sizes.
    pub fn with_max_batch_size(mut self, max: usize) -> Self {
        self.max_batch_size = Some(max.max(1));
        self
    }

    /// The batch size determined by probing, if probing has run.
    pub fn probed_batch_size(&self) -> Option<usize> {
        self.probed_batch_size
    }

    /// Probes the engine for a working batch size and caches the result.
    pub fn probe_batch_size(&mut self) -> Result<usize, TranslatorError> {
        if let Some(size) = self.probed_batch_size {
            return Ok(size);
        }
        let max = self.max_batch_size.unwrap_or(DEFAULT_MAX_BATCH_SIZE);
        let size = probe_batch_size(&mut self.engine, max)?;
        self.probed_batch_size = Some(size);
        Ok(size)
    }

    pub fn translate(&mut self, text: &str) -> Result<String, TranslatorError> {
        if text.trim().is_empty() {
            return Err(TranslatorError::EmptyInput);
        }

        let chunker = Chunker::with_limits(self.engine.max_source_len(), SAFETY_MARGIN);
        let chunks = chunker.split(text, |piece| self.tokenizer.count_tokens(piece))?;

        log::info!(
            "Split input ({} chars) into {} chunk(s)",
            text.len(),
            chunks.len()
        );

        if chunks.len() == 1 {
            return self.translate_chunk(&chunks[0]);
        }

        // Prepend a forward context-overlap window to every chunk i > 0 so the
        // model sees the boundary context, then strip the duplicated re-translation
        // from each subsequent chunk's output with the conservative OverlapTrimmer.
        let target = chunker.target_chunk_tokens();
        let augmented =
            apply_context_overlap(&chunks, &|piece| self.tokenizer.count_tokens(piece), target)?;
        let trimmer = OverlapTrimmer::default();
        log::info!(
            "Applied context overlap to {} chunk(s); target {target} tokens",
            augmented.len().saturating_sub(1)
        );

        let max_source = self.engine.max_source_len();
        let mut encoded: Vec<(usize, Vec<i64>)> = Vec::with_capacity(augmented.len());
        for (index, chunk) in augmented.iter().enumerate() {
            let ids = self.tokenizer.encode(chunk)?;
            if ids.len() > max_source {
                return Err(TranslatorError::ContextWindowExceeded {
                    max: max_source,
                    actual: ids.len(),
                });
            }
            encoded.push((index, ids));
        }

        let batch_size = self.probe_batch_size()?;
        log::info!("Probed batch size: {batch_size}");

        let batches = build_batches(&encoded, batch_size, PAD_TOKEN_ID as i64);
        let mut decoded: Vec<(usize, String)> = Vec::with_capacity(augmented.len());

        for (batch_index, batch) in batches.iter().enumerate() {
            log::info!(
                "Translating batch {}/{} ({} sequences, max len {})",
                batch_index + 1,
                batches.len(),
                batch.input_ids.len(),
                batch.max_len
            );
            let outputs = self
                .engine
                .run_inference_batch(&batch.input_ids, &batch.attention_mask)?;
            debug_assert_eq!(outputs.len(), batch.order.len());
            for (output, &original) in outputs.iter().zip(batch.order.iter()) {
                let translated = self.tokenizer.decode(output)?;
                decoded.push((original, translated));
            }
        }

        decoded.sort_by_key(|(original, _)| *original);
        let raw_outputs: Vec<String> = decoded.into_iter().map(|(_, text)| text).collect();

        let mut results: Vec<String> = Vec::with_capacity(raw_outputs.len());
        for (i, output) in raw_outputs.iter().enumerate() {
            if i == 0 {
                results.push(output.clone());
            } else {
                results.push(trimmer.trim(&raw_outputs[i - 1], output));
            }
        }

        Ok(results.join(" "))
    }

    fn translate_chunk(&mut self, text: &str) -> Result<String, TranslatorError> {
        let input_ids = self.tokenizer.encode(text)?;

        let max_len = self.engine.max_source_len();
        if input_ids.len() > max_len {
            return Err(TranslatorError::ContextWindowExceeded {
                max: max_len,
                actual: input_ids.len(),
            });
        }

        let output_ids = self.engine.run_inference(&input_ids)?;
        self.tokenizer.decode(&output_ids)
    }

    /// Splits text into chunks bounded by [`TARGET_CHUNK_TOKENS`], preferring
    /// sentence/paragraph boundaries.
    ///
    /// Uses a whitespace word count as a token proxy so it works without a
    /// tokenizer; [`Translator::translate`] performs tokenizer-accurate
    /// chunking through [`Chunker`].
    pub fn split_into_chunks(text: &str) -> Vec<String> {
        let estimator: fn(&str) -> Result<usize, std::convert::Infallible> =
            |piece| Ok(piece.split_whitespace().count());
        Chunker::default()
            .split(text, estimator)
            .unwrap_or_else(|_| vec![text.trim().to_string()])
    }
}

impl Translator<HuggingFaceTokenizer, OnnxInferenceEngine> {
    pub fn load_model(
        encoder_path: &str,
        decoder_path: &str,
        tokenizer_path: &str,
    ) -> Result<Self, TranslatorError> {
        let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path)?;
        let engine = OnnxInferenceEngine::load(encoder_path, decoder_path)?;
        Ok(Self::new(tokenizer, engine))
    }
}

pub type DefaultTranslator = Translator<HuggingFaceTokenizer, OnnxInferenceEngine>;

// ──────────────────────────────────────────────
//  Mocks (testing only)
// ──────────────────────────────────────────────

pub mod mocks {
    use crate::errors::TranslatorError;
    use crate::traits::{InferenceEngine, Tokenizer};

    pub struct MockTokenizer {
        pub encode_ids: Vec<i64>,
        pub decode_text: String,
        pub fail_encode: bool,
        pub fail_decode: bool,
        /// Encode returns `[word_count]` and token counting uses the real word
        /// count (used to simulate varying sequence lengths).
        pub echo_len: bool,
        /// Decode renders each token as `t{id}` instead of `decode_text`.
        pub dynamic_decode: bool,
    }

    impl Default for MockTokenizer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MockTokenizer {
        pub fn new() -> Self {
            Self {
                encode_ids: vec![1, 2, 3],
                decode_text: "mock translation".into(),
                fail_encode: false,
                fail_decode: false,
                echo_len: false,
                dynamic_decode: false,
            }
        }
    }

    impl Tokenizer for MockTokenizer {
        fn count_tokens(&self, text: &str) -> Result<usize, TranslatorError> {
            if self.echo_len {
                Ok(text.split_whitespace().count())
            } else {
                Ok(self.encode_ids.len())
            }
        }

        fn encode(&self, text: &str) -> Result<Vec<i64>, TranslatorError> {
            if self.fail_encode {
                Err(TranslatorError::EmptyInput)
            } else if self.echo_len {
                Ok(vec![text.split_whitespace().count() as i64])
            } else {
                Ok(self.encode_ids.clone())
            }
        }

        fn decode(&self, ids: &[i64]) -> Result<String, TranslatorError> {
            if self.fail_decode {
                Err(TranslatorError::EmptyInput)
            } else if self.dynamic_decode {
                Ok(ids
                    .iter()
                    .map(|id| format!("t{id}"))
                    .collect::<Vec<_>>()
                    .join(" "))
            } else {
                Ok(self.decode_text.clone())
            }
        }
    }

    pub struct MockInferenceEngine {
        pub output_ids: Vec<i64>,
        pub fail: bool,
        pub max_source: usize,
        /// Batch inference echoes the first token of each padded row.
        pub echo_input: bool,
        /// Batch sizes above this fail with a recoverable (OOM-shaped) error.
        pub max_working_batch: usize,
        /// When a batch is too large, fail with an unrecoverable error instead.
        pub unrecoverable_batch_failure: bool,
        /// Records every batch size passed to [`MockInferenceEngine::run_inference_batch`].
        pub batch_sizes_seen: Vec<usize>,
    }

    impl Default for MockInferenceEngine {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MockInferenceEngine {
        pub fn new() -> Self {
            Self {
                output_ids: vec![4, 5, 6],
                fail: false,
                max_source: 512,
                echo_input: false,
                max_working_batch: 8,
                unrecoverable_batch_failure: false,
                batch_sizes_seen: Vec::new(),
            }
        }
    }

    impl InferenceEngine for MockInferenceEngine {
        fn max_source_len(&self) -> usize {
            self.max_source
        }

        fn run_inference(&mut self, _input_ids: &[i64]) -> Result<Vec<i64>, TranslatorError> {
            if self.fail {
                Err(TranslatorError::EmptyInput)
            } else {
                Ok(self.output_ids.clone())
            }
        }

        fn run_inference_batch(
            &mut self,
            input_ids: &[Vec<i64>],
            _attention_mask: &[Vec<i64>],
        ) -> Result<Vec<Vec<i64>>, TranslatorError> {
            if self.fail {
                return Err(TranslatorError::EmptyInput);
            }
            let batch_size = input_ids.len();
            self.batch_sizes_seen.push(batch_size);

            if batch_size > self.max_working_batch {
                if self.unrecoverable_batch_failure {
                    return Err(TranslatorError::EngineExecutionFailed {
                        source: ort::Error::new("Failed to load model: missing model file"),
                    });
                }
                return Err(TranslatorError::EngineExecutionFailed {
                    source: ort::Error::new("Out of memory allocating tensor shape [8, 512]"),
                });
            }

            if self.echo_input {
                Ok(input_ids
                    .iter()
                    .map(|row| row.first().map_or_else(Vec::new, |&token| vec![token]))
                    .collect())
            } else {
                Ok(vec![self.output_ids.clone(); batch_size])
            }
        }
    }

    #[test]
    fn test_mock_translator_basic() {
        let tokenizer = MockTokenizer::new();
        let engine = MockInferenceEngine::new();
        let mut translator = super::Translator::new(tokenizer, engine);
        let result = translator.translate("hello");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "mock translation");
    }

    #[test]
    fn test_mock_translator_empty_input() {
        let tokenizer = MockTokenizer::new();
        let engine = MockInferenceEngine::new();
        let mut translator = super::Translator::new(tokenizer, engine);
        let result = translator.translate("");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TranslatorError::EmptyInput));
    }

    #[test]
    fn test_mock_translator_encode_failure() {
        let tokenizer = MockTokenizer {
            fail_encode: true,
            ..MockTokenizer::new()
        };
        let engine = MockInferenceEngine::new();
        let mut translator = super::Translator::new(tokenizer, engine);
        let result = translator.translate("hello");
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_translator_decode_failure() {
        let tokenizer = MockTokenizer {
            fail_decode: true,
            ..MockTokenizer::new()
        };
        let engine = MockInferenceEngine::new();
        let mut translator = super::Translator::new(tokenizer, engine);
        let result = translator.translate("hello");
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_translator_inference_failure() {
        let tokenizer = MockTokenizer::new();
        let engine = MockInferenceEngine {
            fail: true,
            ..MockInferenceEngine::new()
        };
        let mut translator = super::Translator::new(tokenizer, engine);
        let result = translator.translate("hello");
        assert!(result.is_err());
    }

    #[test]
    fn test_split_into_chunks() {
        let chunks = super::Translator::<MockTokenizer, MockInferenceEngine>::split_into_chunks(
            "Hello. How are you? I am fine.",
        );
        assert!(!chunks.is_empty());
    }
}

// ──────────────────────────────────────────────
//  Smart chunking tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod chunking_tests {
    use super::{Chunker, MAX_MODEL_TOKENS, SAFETY_MARGIN, SENTENCE_ENDERS, TARGET_CHUNK_TOKENS};

    /// Token proxy: count whitespace-separated words.
    fn count_words(text: &str) -> Result<usize, std::convert::Infallible> {
        Ok(text.split_whitespace().count())
    }

    /// Token proxy: count characters.
    fn count_chars(text: &str) -> Result<usize, std::convert::Infallible> {
        Ok(text.chars().count())
    }

    fn count_with(
        chunker: &Chunker,
        text: &str,
        count: fn(&str) -> Result<usize, std::convert::Infallible>,
    ) -> Vec<String> {
        chunker.split(text, count).unwrap()
    }

    /// Builds a paragraph of `n` copies of `word` separated by spaces.
    fn paragraph(word: &str, n: usize) -> String {
        std::iter::repeat_n(word, n).collect::<Vec<_>>().join(" ")
    }

    /// Builds a sentence of `n` words terminated by sentence punctuation.
    fn sentence(word: &str, n: usize) -> String {
        format!("{}.", paragraph(word, n))
    }

    #[test]
    fn test_constants_reserve_safety_margin() {
        assert_eq!(MAX_MODEL_TOKENS, 512);
        assert_eq!(SAFETY_MARGIN, 32);
        assert_eq!(TARGET_CHUNK_TOKENS, 480);
    }

    #[test]
    fn test_short_text_is_single_chunk() {
        let chunker = Chunker::default();
        let chunks = count_with(&chunker, "Hello world. This is a short text!", count_words);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world. This is a short text!");
    }

    #[test]
    fn test_empty_text_yields_no_chunks() {
        let chunker = Chunker::default();
        assert!(count_with(&chunker, "   \n  ", count_words).is_empty());
    }

    #[test]
    fn test_paragraph_boundaries_respected() {
        let chunker = Chunker::default();
        let text = format!(
            "{}\n\n{}\n\n{}",
            paragraph("alpha", 200),
            paragraph("beta", 200),
            paragraph("gamma", 200),
        );
        let chunks = count_with(&chunker, &text, count_words);

        assert_eq!(
            chunks.len(),
            2,
            "expected two paragraph-aligned chunks: {chunks:?}"
        );
        assert_eq!(
            chunks[0].split("\n\n").count(),
            2,
            "chunk 0 must contain whole paragraphs only"
        );
        assert_eq!(
            chunks[1].split("\n\n").count(),
            1,
            "chunk 1 must contain whole paragraphs only"
        );
        for chunk in &chunks {
            assert!(chunk.split_whitespace().count() <= chunker.target_chunk_tokens());
        }
    }

    #[test]
    fn test_sentence_boundaries_respected() {
        let chunker = Chunker::default();
        let text = (0..20)
            .map(|i| sentence(&format!("word{i}"), 40))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = count_with(&chunker, &text, count_words);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.split_whitespace().count() <= chunker.target_chunk_tokens());
            let trimmed = chunk.trim_end();
            assert!(
                trimmed.ends_with(SENTENCE_ENDERS),
                "chunk must break at a sentence boundary: {chunk:?}"
            );
        }
    }

    #[test]
    fn test_giant_unpunctuated_sentence_falls_back_to_word_splits() {
        let chunker = Chunker::default();
        let text = (0..1000)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(text.split_whitespace().count(), 1000);

        let chunks = count_with(&chunker, &text, count_words);

        assert_eq!(
            chunks.len(),
            3,
            "1000 words should split into 3 word-boundary chunks: {chunks:?}"
        );
        for chunk in &chunks {
            assert!(
                chunk.split_whitespace().count() <= chunker.target_chunk_tokens(),
                "chunk exceeds target limit: {chunk:?}"
            );
        }
        let reconstructed = chunks.join(" ");
        assert_eq!(
            reconstructed, text,
            "word-boundary fallback must not lose content"
        );
    }

    #[test]
    fn test_no_chunk_exceeds_target_for_mixed_text() {
        let chunker = Chunker::with_limits(100, 20);
        let text = format!(
            "{}\n\n{}\n{} {}\n\n{}",
            sentence("first", 40),
            sentence("second", 40),
            sentence("third", 40),
            sentence("fourth", 40),
            sentence("fifth", 40),
        );
        let chunks = count_with(&chunker, &text, count_words);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(
                chunk.split_whitespace().count() <= chunker.target_chunk_tokens(),
                "chunk exceeds target limit: {chunk:?}"
            );
        }
    }

    #[test]
    fn test_single_word_exceeding_target_is_hard_split() {
        let chunker = Chunker::with_limits(100, 20);
        let word = "x".repeat(600);
        let chunks = count_with(&chunker, &word, count_chars);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(
                chunk.chars().count() <= chunker.target_chunk_tokens(),
                "chunk exceeds target limit: {chunk:?}"
            );
        }
        let reconstructed: String = chunks.concat();
        assert_eq!(reconstructed, word);
    }
}

// ──────────────────────────────────────────────
//  Length bucketing, probing & batch pipeline tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod batching_tests {
    use super::mocks::{MockInferenceEngine, MockTokenizer};
    use super::{build_batches, probe_batch_size, probe_sizes, Translator};
    use crate::errors::TranslatorError;

    const PAD: i64 = 57059;

    #[test]
    fn test_probe_sizes_doubling_sequence() {
        assert_eq!(probe_sizes(1), vec![1]);
        assert_eq!(probe_sizes(3), vec![1, 2]);
        assert_eq!(probe_sizes(8), vec![1, 2, 4, 8]);
        assert_eq!(probe_sizes(100), vec![1, 2, 4, 8, 16, 32, 64]);
    }

    #[test]
    fn test_build_batches_sorts_and_pads() {
        let chunks = vec![
            (0, vec![1, 2, 3, 4, 5]),
            (1, vec![1, 2]),
            (2, vec![1, 2, 3]),
            (3, vec![1, 2, 3, 4]),
        ];
        let batches = build_batches(&chunks, 2, PAD);

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].order, vec![1, 2], "shortest chunks first");
        assert_eq!(batches[0].max_len, 3);
        assert_eq!(batches[0].input_ids[0], vec![1, 2, PAD]);
        assert_eq!(batches[0].attention_mask[0], vec![1, 1, 0]);
        assert_eq!(batches[0].input_ids[1], vec![1, 2, 3]);
        assert_eq!(batches[0].attention_mask[1], vec![1, 1, 1]);

        assert_eq!(batches[1].order, vec![3, 0]);
        assert_eq!(batches[1].max_len, 5);
        assert_eq!(batches[1].input_ids[0], vec![1, 2, 3, 4, PAD]);
        assert_eq!(batches[1].attention_mask[0], vec![1, 1, 1, 1, 0]);
        assert_eq!(batches[1].input_ids[1], vec![1, 2, 3, 4, 5]);
        assert_eq!(batches[1].attention_mask[1], vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_build_batches_covers_every_chunk_exactly_once() {
        let chunks: Vec<(usize, Vec<i64>)> = (0..7).map(|i| (i, vec![1i64; i + 1])).collect();
        let batches = build_batches(&chunks, 3, PAD);

        let mut seen = Vec::new();
        for batch in &batches {
            assert!(batch.input_ids.len() <= 3);
            assert_eq!(batch.input_ids.len(), batch.order.len());
            assert_eq!(batch.attention_mask.len(), batch.order.len());
            for (row, mask) in batch.input_ids.iter().zip(batch.attention_mask.iter()) {
                assert_eq!(row.len(), batch.max_len);
                assert_eq!(mask.len(), batch.max_len);
            }
            seen.extend(batch.order.iter().copied());
        }
        seen.sort();
        assert_eq!(seen, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn test_build_batches_empty_input() {
        assert!(build_batches(&[], 4, PAD).is_empty());
    }

    #[test]
    fn test_probe_reaches_max_when_all_succeed() {
        let mut engine = MockInferenceEngine::new();
        let size = probe_batch_size(&mut engine, 8).unwrap();
        assert_eq!(size, 8);
        assert_eq!(engine.batch_sizes_seen, vec![2, 4, 8]);
    }

    #[test]
    fn test_probe_falls_back_on_recoverable_error() {
        let mut engine = MockInferenceEngine {
            max_working_batch: 2,
            ..MockInferenceEngine::new()
        };
        let size = probe_batch_size(&mut engine, 8).unwrap();
        assert_eq!(size, 2, "must fall back to the last working batch size");
        assert_eq!(engine.batch_sizes_seen, vec![2, 4]);
    }

    #[test]
    fn test_probe_propagates_unrecoverable_error() {
        let mut engine = MockInferenceEngine {
            max_working_batch: 2,
            unrecoverable_batch_failure: true,
            ..MockInferenceEngine::new()
        };
        let err = probe_batch_size(&mut engine, 8).unwrap_err();
        assert!(matches!(err, TranslatorError::EngineExecutionFailed { .. }));
        assert!(!err.is_recoverable_batch_error());
    }

    #[test]
    fn test_translator_batches_and_preserves_order() {
        let tokenizer = MockTokenizer {
            echo_len: true,
            dynamic_decode: true,
            ..MockTokenizer::new()
        };
        let engine = MockInferenceEngine {
            echo_input: true,
            ..MockInferenceEngine::new()
        };
        let mut translator = Translator::new(tokenizer, engine);

        // Three sentences with distinct word counts. Greedy packing produces
        // two chunks: "S1 S2" (400 words) then "S3" (200 words).
        let text = format!(
            "{} {} {}",
            sentence_of("alpha", 200),
            sentence_of("beta", 200),
            sentence_of("gamma", 200),
        );

        let result = translator.translate(&text).unwrap();

        // Chunk 0 (400 words) encodes to [400] and decodes to "t400". Chunk 1
        // gets a 24-word context overlap prepended: 24 + 200 = 224 words, so it
        // decodes to "t224". Original order must be preserved despite length
        // sorting, and the OverlapTrimmer must not touch the mock outputs.
        assert_eq!(result, "t400 t224");
        assert_eq!(translator.probed_batch_size(), Some(8));
    }

    #[test]
    fn test_translator_falls_back_after_probe_failure() {
        let tokenizer = MockTokenizer {
            echo_len: true,
            dynamic_decode: true,
            ..MockTokenizer::new()
        };
        let engine = MockInferenceEngine {
            echo_input: true,
            max_working_batch: 2,
            ..MockInferenceEngine::new()
        };
        let mut translator = Translator::new(tokenizer, engine);

        let text = format!(
            "{} {} {}",
            sentence_of("alpha", 200),
            sentence_of("beta", 200),
            sentence_of("gamma", 200),
        );

        let result = translator.translate(&text).unwrap();
        assert_eq!(result, "t400 t224");
        assert_eq!(translator.probed_batch_size(), Some(2));
    }

    #[test]
    fn test_translator_propagates_unrecoverable_probe_error() {
        let tokenizer = MockTokenizer {
            echo_len: true,
            ..MockTokenizer::new()
        };
        let engine = MockInferenceEngine {
            max_working_batch: 2,
            unrecoverable_batch_failure: true,
            ..MockInferenceEngine::new()
        };
        let mut translator = Translator::new(tokenizer, engine);

        let text = format!(
            "{} {} {}",
            sentence_of("alpha", 200),
            sentence_of("beta", 200),
            sentence_of("gamma", 200),
        );

        let err = translator.translate(&text).unwrap_err();
        assert!(matches!(err, TranslatorError::EngineExecutionFailed { .. }));
        assert!(translator.probed_batch_size().is_none());
    }

    #[test]
    fn test_translator_with_max_batch_size_uses_ceiling() {
        let tokenizer = MockTokenizer {
            echo_len: true,
            dynamic_decode: true,
            ..MockTokenizer::new()
        };
        let engine = MockInferenceEngine {
            echo_input: true,
            ..MockInferenceEngine::new()
        };
        let mut translator = Translator::new(tokenizer, engine).with_max_batch_size(2);

        let text = format!(
            "{} {} {}",
            sentence_of("alpha", 200),
            sentence_of("beta", 200),
            sentence_of("gamma", 200),
        );

        let result = translator.translate(&text).unwrap();
        assert_eq!(result, "t400 t224");
        assert_eq!(translator.probed_batch_size(), Some(2));
    }

    /// Builds a sentence of `n` copies of `word`, terminated with a period.
    fn sentence_of(word: &str, n: usize) -> String {
        format!(
            "{}.",
            std::iter::repeat_n(word, n).collect::<Vec<_>>().join(" ")
        )
    }
}

// ──────────────────────────────────────────────
//  Context overlap & OverlapTrimmer tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod overlap_tests {
    use super::{
        apply_context_overlap, OverlapTrimmer, CONTEXT_OVERLAP_MAX_WORDS, CONTEXT_OVERLAP_MIN_WORDS,
    };

    /// Token proxy: count whitespace-separated words.
    fn count_words(text: &str) -> Result<usize, std::convert::Infallible> {
        Ok(text.split_whitespace().count())
    }

    /// Builds a string of `n` words `prefix0 prefix1 ...`.
    fn words(n: usize, prefix: &str) -> String {
        (0..n)
            .map(|i| format!("{prefix}{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn tail_words(text: &str, n: usize) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        words[words.len().saturating_sub(n)..].join(" ")
    }

    #[test]
    fn test_overlap_single_chunk_unchanged() {
        let chunks = vec![words(100, "a")];
        let augmented = apply_context_overlap(&chunks, &count_words, 480).unwrap();
        assert_eq!(augmented, chunks);
    }

    #[test]
    fn test_overlap_prepends_previous_tail() {
        let chunks = vec![words(100, "a"), words(50, "b")];
        let augmented = apply_context_overlap(&chunks, &count_words, 480).unwrap();

        assert_eq!(augmented.len(), 2);
        assert_eq!(augmented[0], chunks[0]);
        assert!(
            augmented[1].starts_with(&tail_words(&chunks[0], CONTEXT_OVERLAP_MAX_WORDS)),
            "chunk 1 must be prefixed with the tail of chunk 0"
        );
        assert!(augmented[1].ends_with(&chunks[1]));
        assert!(
            augmented[1].split_whitespace().count() <= 480,
            "overlap + body must honour the target budget"
        );
    }

    #[test]
    fn test_overlap_shrinks_window_before_trimming_body() {
        // Body of 460 words cannot take the full 24-word overlap (484 > 480)
        // but fits a 20-word overlap exactly, so the overlap shrinks and the
        // body is preserved whole.
        let chunks = vec![words(480, "a"), words(460, "b")];
        let augmented = apply_context_overlap(&chunks, &count_words, 480).unwrap();

        let count = augmented[1].split_whitespace().count();
        assert_eq!(count, 480);
        assert!(
            augmented[1].starts_with(&tail_words(&chunks[0], 20)),
            "overlap must shrink to 20 words: {}",
            augmented[1]
        );
        assert!(
            augmented[1].ends_with("b459"),
            "body must be preserved: {}",
            augmented[1]
        );
    }

    #[test]
    fn test_overlap_budget_strictly_honored_when_body_at_limit() {
        // Both chunks sit exactly at the target; even the 16-word minimum
        // overlap overflows, so the body's front is trimmed to make room.
        let chunks = vec![words(480, "a"), words(480, "b")];
        let augmented = apply_context_overlap(&chunks, &count_words, 480).unwrap();

        let count = augmented[1].split_whitespace().count();
        assert_eq!(
            count, 480,
            "overlap + trimmed body must stay at exactly 480 tokens"
        );
        assert!(
            augmented[1].starts_with(&tail_words(&chunks[0], CONTEXT_OVERLAP_MIN_WORDS)),
            "minimum overlap must be preserved"
        );
        assert!(
            augmented[1].ends_with("b479"),
            "the body tail must be kept intact"
        );
    }

    #[test]
    fn test_exact_boundary_trim_normalized_case_and_punctuation() {
        let prev = "The cat sat toward the table";
        let cur = "Toward the table, the cat purred";
        let trimmed = OverlapTrimmer::default().trim(prev, cur);

        assert_eq!(trimmed, "the cat purred");
    }

    #[test]
    fn test_fuzzy_ngram_trim_toward_vs_towards() {
        // Exact match fails ("toward" vs "towards"), but the 3-gram
        // "toward the table" / "towards the table" is a high-confidence,
        // anchored fuzzy match with two meaningful words.
        let prev = "The dog walked toward the table";
        let cur = "Towards the table, it sniffed";
        let trimmed = OverlapTrimmer::default().trim(prev, cur);

        assert_eq!(trimmed, "it sniffed");
    }

    #[test]
    fn test_no_trim_for_single_word_match() {
        let prev = "She went home";
        let cur = "Home is where the heart is";
        assert_eq!(OverlapTrimmer::default().trim(prev, cur), cur);
    }

    #[test]
    fn test_no_trim_for_stop_word_only_match() {
        let prev = "Go to the and the";
        let cur = "the and the way home";
        assert_eq!(OverlapTrimmer::default().trim(prev, cur), cur);
    }

    #[test]
    fn test_no_trim_for_semantically_divergent_text() {
        // "walked toward the table" vs "walked toward the door": the n-gram
        // similarity falls below the threshold, so nothing is trimmed.
        let prev = "He walked toward the table";
        let cur = "Walked toward the door, she said";
        assert_eq!(OverlapTrimmer::default().trim(prev, cur), cur);
    }

    #[test]
    fn test_upper_bound_limits_maximum_trimmed_words() {
        let dup = std::iter::repeat_n("dup", 30).collect::<Vec<_>>().join(" ");
        let prev = format!("x y {dup}");
        let cur = format!("{dup} z w");

        let trimmed = OverlapTrimmer::default().trim(&prev, &cur);
        let trimmed_words: Vec<&str> = trimmed.split_whitespace().collect();

        assert_eq!(trimmed_words.len(), 32 - 20);
        assert!(
            trimmed_words
                .iter()
                .all(|w| *w == "dup" || *w == "z" || *w == "w"),
            "unexpected words remain after trimming: {trimmed_words:?}"
        );
        assert_eq!(
            trimmed_words[0], "dup",
            "duplicated words beyond the upper bound must be left untouched"
        );
    }
}

// ──────────────────────────────────────────────
//  Integration tests (real ONNX model)
// ──────────────────────────────────────────────

#[cfg(test)]
mod integration_tests {
    use std::path::Path;

    use super::{probe_batch_size, OnnxInferenceEngine};
    use crate::traits::InferenceEngine;

    fn model_paths() -> Option<(&'static str, &'static str)> {
        let candidates = [
            (
                "models/encoder_model_int8.onnx",
                "models/decoder_model_int8.onnx",
            ),
            ("models/encoder_model.onnx", "models/decoder_model.onnx"),
        ];
        candidates
            .iter()
            .copied()
            .find(|(enc, dec)| Path::new(enc).exists() && Path::new(dec).exists())
    }

    #[test]
    #[ignore = "requires the ONNX models under models/"]
    fn test_real_engine_batch_inference() {
        let Some((enc, dec)) = model_paths() else {
            return;
        };
        let mut engine = OnnxInferenceEngine::load(enc, dec).unwrap();

        let input = vec![
            vec![1i64, 2, 3, 4, 5, 6],
            vec![1, 2, 3, 57059, 57059, 57059],
        ];
        let mask = vec![vec![1i64, 1, 1, 1, 1, 1], vec![1, 1, 1, 0, 0, 0]];
        let outputs = engine.run_inference_batch(&input, &mask).unwrap();
        assert_eq!(outputs.len(), 2, "one output sequence per batch item");
        assert!(!outputs[0].is_empty(), "batch item 0 must generate tokens");
        assert!(!outputs[1].is_empty(), "batch item 1 must generate tokens");

        let probed = probe_batch_size(&mut engine, 4).unwrap();
        assert!((1..=4).contains(&probed));
    }

    #[test]
    #[ignore = "requires the ONNX models under models/"]
    fn test_real_engine_rejects_mismatched_batch() {
        let Some((enc, dec)) = model_paths() else {
            return;
        };
        let mut engine = OnnxInferenceEngine::load(enc, dec).unwrap();

        let input = vec![vec![1i64, 2, 3], vec![4, 5]];
        let mask = vec![vec![1i64; 3], vec![1, 1, 1]];
        let err = engine.run_inference_batch(&input, &mask).unwrap_err();
        assert!(matches!(
            err,
            super::TranslatorError::InvalidBatchLength { .. }
        ));
    }
}
