use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AnthropicError {
    #[error("network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("malformed request: {0}")]
    BadRequest(String),

    #[error("api error: {0}")]
    ApiError(String),

    #[error("unauthorized; check your API key")]
    Unauthorized,

    #[error("failed to deserialize response: {0}")]
    DeserializationError(#[from] serde_json::Error),

    #[error("unknown error: {0}")]
    Unknown(String),

    #[error("unexpected error occurred")]
    UnexpectedError,

    #[error("stream failed: {0}")]
    StreamError(StreamError),
}

impl From<backoff::Error<AnthropicError>> for AnthropicError {
    fn from(err: backoff::Error<AnthropicError>) -> Self {
        match err {
            backoff::Error::Permanent(err) => err,
            backoff::Error::Transient { .. } => AnthropicError::UnexpectedError,
        }
    }
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq, Serialize)]
pub struct StreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: Option<String>,
    pub error: Option<serde_json::Value>,
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Error ({}): {}",
            self.error_type,
            self.message.clone().unwrap_or_else(|| self
                .error
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_default())
        ))
    }
}

pub(crate) fn map_deserialization_error(e: serde_json::Error, _bytes: &[u8]) -> AnthropicError {
    AnthropicError::DeserializationError(e)
}
