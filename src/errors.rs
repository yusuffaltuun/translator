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

    #[error("batch sequence length mismatch: expected {expected}, got {actual}")]
    InvalidBatchLength { expected: usize, actual: usize },

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
            TranslatorError::InvalidBatchLength { .. } => ErrorSeverity::Recoverable,
            TranslatorError::UnsupportedLanguagePair { .. } => ErrorSeverity::Recoverable,
        }
    }

    /// Whether this error is a *recoverable* batch-related failure.
    ///
    /// During auto-probing, recoverable errors (out of memory, unsupported
    /// tensor shape, execution-provider dynamic-shape failures) cause the
    /// prober to fall back to the last known working batch size. Everything
    /// else — missing/corrupt model files, tokenizer failures, and so on — is
    /// treated as unrecoverable and must fail fast.
    pub fn is_recoverable_batch_error(&self) -> bool {
        match self {
            TranslatorError::EngineExecutionFailed { source } => {
                let message = source.message().to_ascii_lowercase();
                let code = source.code();
                code == ort::ErrorCode::ExecutionProviderFailure
                    || code == ort::ErrorCode::InvalidArgument
                    || [
                        "out of memory",
                        "oom",
                        "not enough memory",
                        "failed to allocate",
                        "shape",
                        "unsupported",
                        "dynamic",
                        "not implemented",
                    ]
                    .iter()
                    .any(|keyword| message.contains(keyword))
            }
            _ => false,
        }
    }
}
