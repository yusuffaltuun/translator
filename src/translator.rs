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
            max_source_len: 512,
        })
    }

    pub fn with_max_source_len(mut self, max: usize) -> Self {
        self.max_source_len = max;
        self
    }
}

impl InferenceEngine for OnnxInferenceEngine {
    fn max_source_len(&self) -> usize {
        self.max_source_len
    }

    fn run_inference(&mut self, input_ids: &[i64]) -> Result<Vec<i64>, TranslatorError> {
        let source_len = input_ids.len();
        let attention_mask: Vec<i64> = vec![1i64; source_len];

        // ── Encoder forward pass ──
        let input_ids_tensor = Tensor::from_array(([1, source_len], input_ids.to_vec()))
            .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;
        let attention_mask_tensor =
            Tensor::from_array(([1, source_len], attention_mask.clone()))
                .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;

        let encoder_inputs: Vec<(String, ort::value::DynValue)> = vec![
            ("input_ids".into(), input_ids_tensor.into_dyn()),
            ("attention_mask".into(), attention_mask_tensor.into_dyn()),
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
            Tensor::from_array(([1, source_len, D_MODEL], hidden_states_vec))
                .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;
        let enc_attention_mask_tensor = Tensor::from_array(([1, source_len], attention_mask))
            .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;

        // ── Autoregressive decoder loop ──
        let mut generated_tokens: Vec<i64> = Vec::new();
        let mut past_key_values: HashMap<String, Vec<f32>> = HashMap::new();
        let mut next_input: Vec<i64> = vec![DECODER_START_TOKEN_ID];

        for _step in 0..MAX_OUTPUT_LEN {
            let dec_input_len = next_input.len();
            let dec_input_ids_tensor = Tensor::from_array(([1, dec_input_len], next_input.clone()))
                .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;

            let mut decoder_inputs: Vec<(String, ort::value::DynValue)> = vec![
                ("input_ids".into(), dec_input_ids_tensor.into_dyn()),
                (
                    "encoder_attention_mask".into(),
                    enc_attention_mask_tensor.clone().into_dyn(),
                ),
                (
                    "encoder_hidden_states".into(),
                    encoder_hidden_states_tensor.clone().into_dyn(),
                ),
            ];

            for layer in 0..NUM_LAYERS {
                for suffix in &["key", "value"] {
                    let name = format!("past_key_values.{}.decoder.{}", layer, suffix);
                    let cached = past_key_values.get(&name);

                    let tensor = if let Some(data) = cached {
                        let seq_len = data.len() / (NUM_HEADS * HEAD_DIM);
                        Tensor::from_array(([1, NUM_HEADS, seq_len, HEAD_DIM], data.clone()))
                            .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?
                    } else {
                        let shape = [1, NUM_HEADS as i64, 0, HEAD_DIM as i64];
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

            let vocab_size = logits_data.len() / dec_input_len;
            let last_token_logits =
                &logits_data[(dec_input_len - 1) * vocab_size..dec_input_len * vocab_size];

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
                break;
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

            generated_tokens.push(best_token as i64);
            next_input = vec![best_token as i64];
        }

        if generated_tokens.first() == Some(&DECODER_START_TOKEN_ID) {
            generated_tokens.remove(0);
        }
        generated_tokens.retain(|&t| t != PAD_TOKEN_ID as i64);

        Ok(generated_tokens)
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
//  Translator (generic)
// ──────────────────────────────────────────────

pub struct Translator<T: Tokenizer, E: InferenceEngine> {
    tokenizer: T,
    engine: E,
}

impl<T: Tokenizer, E: InferenceEngine> Translator<T, E> {
    pub fn new(tokenizer: T, engine: E) -> Self {
        Self { tokenizer, engine }
    }

    pub fn translate(&mut self, text: &str) -> Result<String, TranslatorError> {
        if text.trim().is_empty() {
            return Err(TranslatorError::EmptyInput);
        }

        let chunker = Chunker::with_limits(self.engine.max_source_len(), SAFETY_MARGIN);
        let chunks = chunker.split(text, |piece| -> Result<usize, TranslatorError> {
            Ok(self.tokenizer.encode(piece)?.len())
        })?;

        log::info!(
            "Split input ({} chars) into {} chunk(s)",
            text.len(),
            chunks.len()
        );

        if chunks.len() == 1 {
            return self.translate_chunk(&chunks[0]);
        }

        let mut results = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            log::info!("Translating chunk {}/{}", i + 1, chunks.len());
            let translated = self.translate_chunk(chunk)?;
            results.push(translated);
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
            }
        }
    }

    impl Tokenizer for MockTokenizer {
        fn encode(&self, _text: &str) -> Result<Vec<i64>, TranslatorError> {
            if self.fail_encode {
                Err(TranslatorError::EmptyInput)
            } else {
                Ok(self.encode_ids.clone())
            }
        }

        fn decode(&self, _ids: &[i64]) -> Result<String, TranslatorError> {
            if self.fail_decode {
                Err(TranslatorError::EmptyInput)
            } else {
                Ok(self.decode_text.clone())
            }
        }
    }

    pub struct MockInferenceEngine {
        pub output_ids: Vec<i64>,
        pub fail: bool,
        pub max_source: usize,
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
