use crate::errors::TranslatorError;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i64>, TranslatorError>;
    fn decode(&self, ids: &[i64]) -> Result<String, TranslatorError>;
}

pub trait InferenceEngine {
    fn run_inference(&mut self, input_ids: &[i64]) -> Result<Vec<i64>, TranslatorError>;

    fn max_source_len(&self) -> usize {
        512
    }
}
