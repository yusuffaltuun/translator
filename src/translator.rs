use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

pub struct Translator {
    session: Session,
    tokenizer: Tokenizer,
}

impl Translator {
    pub fn load_model(model_path: &str, tokenizer_path: &str) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("session builder error: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("failed to load model: {}", e))?;

        let tokenizer = Tokenizer::from_file(Path::new(tokenizer_path))
            .map_err(|e| format!("failed to load tokenizer: {}", e))?;

        Ok(Self { session, tokenizer })
    }

    pub fn translate(&mut self, text: &str) -> Result<String, String> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenization error: {}", e))?;

        let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let attention_mask: Vec<i32> = encoding.get_attention_mask().iter().map(|&m| m as i32).collect();
        let seq_len = input_ids.len();

        let input_ids_tensor = Tensor::from_array(([1, seq_len], input_ids))
            .map_err(|e| format!("input_ids tensor error: {}", e))?;
        let attention_mask_tensor = Tensor::from_array(([1, seq_len], attention_mask))
            .map_err(|e| format!("attention_mask tensor error: {}", e))?;
        let num_beams_tensor = Tensor::from_array(([1], vec![1i32]))
            .map_err(|e| format!("num_beams tensor error: {}", e))?;
        let min_length_tensor = Tensor::from_array(([1], vec![1i32]))
            .map_err(|e| format!("min_length tensor error: {}", e))?;
        let max_length_tensor = Tensor::from_array(([1], vec![128i32]))
            .map_err(|e| format!("max_length tensor error: {}", e))?;
        let length_penalty_tensor = Tensor::from_array(([1], vec![1.0f32]))
            .map_err(|e| format!("length_penalty tensor error: {}", e))?;
        let repetition_penalty_tensor = Tensor::from_array(([1], vec![3.0f32]))
            .map_err(|e| format!("repetition_penalty tensor error: {}", e))?;

        let input_pairs: Vec<(String, ort::value::DynValue)> = vec![
            ("input_ids".into(), input_ids_tensor.into_dyn()),
            ("attention_mask".into(), attention_mask_tensor.into_dyn()),
            ("num_beams".into(), num_beams_tensor.into_dyn()),
            ("min_length".into(), min_length_tensor.into_dyn()),
            ("max_length".into(), max_length_tensor.into_dyn()),
            ("length_penalty".into(), length_penalty_tensor.into_dyn()),
            ("repetition_penalty".into(), repetition_penalty_tensor.into_dyn()),
        ];

        let outputs = self
            .session
            .run(input_pairs)
            .map_err(|e| format!("inference error: {}", e))?;

        let sequences_value = &outputs["sequences"];
        let (_, sequences_data) = sequences_value
            .try_extract_tensor::<i32>()
            .map_err(|e| format!("output extraction error: {}", e))?;

        let token_ids: Vec<u32> = sequences_data
            .iter()
            .filter(|&&t| t > 0)
            .map(|&t| t as u32)
            .collect();

        self.tokenizer
            .decode(&token_ids, true)
            .map_err(|e| format!("decode error: {}", e))
    }
}
