use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Fatal,
    Recoverable,
    Transient,
}

#[derive(Error, Debug)]
pub enum TranslatorError {
    #[error("failed to load model: {0}")]
    ModelLoadFailed(#[from] ort::Error),

    #[error("engine execution failed: {source}")]
    EngineExecutionFailed {
        #[source]
        source: ort::Error,
    },

    #[error("tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),

    #[error("input text is empty")]
    EmptyInput,

    #[error("context window exceeded: max={max}, actual={actual}")]
    ContextWindowExceeded { max: usize, actual: usize },

    #[error("unsupported language pair: {from} -> {to}")]
    UnsupportedLanguagePair { from: String, to: String },
}

impl TranslatorError {
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            TranslatorError::ModelLoadFailed(_) => ErrorSeverity::Fatal,
            TranslatorError::EngineExecutionFailed { .. } => ErrorSeverity::Transient,
            TranslatorError::TokenizerError(_) => ErrorSeverity::Recoverable,
            TranslatorError::EmptyInput => ErrorSeverity::Recoverable,
            TranslatorError::ContextWindowExceeded { .. } => ErrorSeverity::Recoverable,
            TranslatorError::UnsupportedLanguagePair { .. } => ErrorSeverity::Recoverable,
        }
    }
}
