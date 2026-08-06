use crate::errors::TranslatorError;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i64>, TranslatorError>;
    fn decode(&self, ids: &[i64]) -> Result<String, TranslatorError>;

    /// Number of tokens `text` produces. Defaults to the encoded length, which
    /// is the correct behaviour for real tokenizers; test doubles can override
    /// it with a cheap proxy.
    fn count_tokens(&self, text: &str) -> Result<usize, TranslatorError> {
        Ok(self.encode(text)?.len())
    }
}

pub trait InferenceEngine {
    fn run_inference(&mut self, input_ids: &[i64]) -> Result<Vec<i64>, TranslatorError>;

    /// Runs batched inference over padded sequences.
    ///
    /// `input_ids` is `[batch_size, sequence_length]` (all rows share the same
    /// padded length) and `attention_mask` marks real tokens with `1` and
    /// padding with `0`. The default implementation falls back to running each
    /// sequence individually via [`InferenceEngine::run_inference`]; engines
    /// that support true batching should override this.
    fn run_inference_batch(
        &mut self,
        input_ids: &[Vec<i64>],
        attention_mask: &[Vec<i64>],
    ) -> Result<Vec<Vec<i64>>, TranslatorError> {
        debug_assert_eq!(input_ids.len(), attention_mask.len());
        let mut outputs = Vec::with_capacity(input_ids.len());
        for ids in input_ids {
            outputs.push(self.run_inference(ids)?);
        }
        Ok(outputs)
    }

    fn max_source_len(&self) -> usize {
        512
    }
}
