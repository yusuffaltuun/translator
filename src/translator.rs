use ort::session::Session;

pub struct Translator {
    session: Session,
}

impl Translator {
    pub fn load_model(path: &str) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("session builder error: {}", e))?
            .commit_from_file(path)
            .map_err(|e| format!("failed to load model: {}", e))?;

        Ok(Self { session })
    }

    pub fn translate(&self, text: &str) -> Result<String, String> {
        let _ = text;
        todo!("inference logic will be implemented in next micro-step")
    }
}
