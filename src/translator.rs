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

        let chunks = Self::split_into_chunks(text);

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

    fn split_into_chunks(text: &str) -> Vec<String> {
        let mut chunks: Vec<String> = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);

            let dominated_by_sentence_end = current.ends_with(". ")
                || current.ends_with("! ")
                || current.ends_with("? ")
                || current.ends_with(".\n")
                || current.ends_with("!\n")
                || current.ends_with("?\n");
            let dominated_by_newline = current.ends_with("\n\n");

            if dominated_by_sentence_end || dominated_by_newline {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    chunks.push(trimmed);
                }
                current.clear();
            }
        }

        let remainder = current.trim().to_string();
        if !remainder.is_empty() {
            chunks.push(remainder);
        }

        if chunks.is_empty() {
            chunks.push(text.trim().to_string());
        }

        let mut merged: Vec<String> = Vec::new();
        for chunk in chunks {
            let word_count = chunk.split_whitespace().count();
            if word_count < 4 && !merged.is_empty() {
                let prev = merged.pop().unwrap();
                merged.push(format!("{} {}", prev, chunk));
            } else if !chunk.is_empty() {
                merged.push(chunk);
            }
        }

        if merged.is_empty() {
            merged.push(text.trim().to_string());
        }

        merged
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

#[cfg(test)]
pub mod mocks {
    use crate::errors::TranslatorError;
    use crate::traits::{InferenceEngine, Tokenizer};

    pub struct MockTokenizer {
        pub encode_ids: Vec<i64>,
        pub decode_text: String,
        pub fail_encode: bool,
        pub fail_decode: bool,
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
