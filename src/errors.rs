use thiserror::Error;

#[derive(Debug, Error)]
pub enum CreateMessagesError {
    #[error(transparent)]
    AnthropicError(#[from] AnthropicError),
}

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

    #[error("unknown error: {0}")]
    Unknown(String),

    #[error("unexpected error occurred")]
    UnexpectedError,
}

impl From<backoff::Error<AnthropicError>> for AnthropicError {
    fn from(err: backoff::Error<AnthropicError>) -> Self {
        match err {
            backoff::Error::Permanent(err) => err,
            backoff::Error::Transient { .. } => AnthropicError::UnexpectedError,
        }
    }
}
