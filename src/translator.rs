use std::collections::HashMap;
use std::path::Path;

use ort::memory::Allocator;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::errors::TranslatorError;

const MAX_SOURCE_LEN: usize = 512;
const MAX_OUTPUT_LEN: usize = 256;
const EOS_TOKEN_ID: u32 = 43741;
const PAD_TOKEN_ID: u32 = 57059;
const DECODER_START_TOKEN_ID: i64 = 57059;
const NUM_LAYERS: usize = 6;
const NUM_HEADS: usize = 16;
const HEAD_DIM: usize = 64;

pub struct Translator {
    encoder: Session,
    decoder: Session,
    tokenizer: Tokenizer,
}

impl Translator {
    pub fn load_model(
        encoder_path: &str,
        decoder_path: &str,
        tokenizer_path: &str,
    ) -> Result<Self, TranslatorError> {
        let encoder = Session::builder()
            .map_err(TranslatorError::ModelLoadFailed)?
            .commit_from_file(encoder_path)
            .map_err(TranslatorError::ModelLoadFailed)?;

        let decoder = Session::builder()
            .map_err(TranslatorError::ModelLoadFailed)?
            .commit_from_file(decoder_path)
            .map_err(TranslatorError::ModelLoadFailed)?;

        let tokenizer = Tokenizer::from_file(Path::new(tokenizer_path))
            .map_err(TranslatorError::TokenizerError)?;

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
        })
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

    fn translate_chunk(&mut self, text: &str) -> Result<String, TranslatorError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(TranslatorError::TokenizerError)?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|m| *m as i64)
            .collect();

        if input_ids.len() > MAX_SOURCE_LEN {
            return Err(TranslatorError::ContextWindowExceeded {
                max: MAX_SOURCE_LEN,
                actual: input_ids.len(),
            });
        }

        let seq_len = input_ids.len();

        let input_ids_tensor = Tensor::from_array(([1, seq_len], input_ids.clone()))
            .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;
        let attention_mask_tensor = Tensor::from_array(([1, seq_len], attention_mask.clone()))
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
            Tensor::from_array(([1, seq_len, 1024], hidden_states_vec))
                .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;
        let enc_attention_mask_tensor = Tensor::from_array(([1, seq_len], attention_mask.clone()))
            .map_err(|e| TranslatorError::EngineExecutionFailed { source: e })?;

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

        if generated_tokens.first() == Some(&(DECODER_START_TOKEN_ID)) {
            generated_tokens.remove(0);
        }
        generated_tokens.retain(|&t| t != PAD_TOKEN_ID as i64);
        let token_ids: Vec<u32> = generated_tokens.iter().map(|&t| t as u32).collect();

        self.tokenizer
            .decode(&token_ids, true)
            .map_err(TranslatorError::TokenizerError)
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
}
