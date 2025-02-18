use backoff::{Error as BackoffError, ExponentialBackoff, ExponentialBackoffBuilder};
use derive_builder::Builder;
use reqwest::StatusCode;
use secrecy::ExposeSecret;
use serde::{de::DeserializeOwned, Serialize};
use std::time::Duration;

use crate::{errors::AnthropicError, messages::Messages};

const BASE_URL: &str = "https://api.anthropic.com";

/// Main entry point for the Anthropic API
///
/// By default will use the `ANTHROPIC_API_KEY` environment variable
///
/// # Example
///
/// ```no_run
/// # use async_anthropic::types::*;
/// # async fn run() {
/// let client = async_anthropic::Client::default();
///
/// let request = CreateMessagesRequestBuilder::default()
///    .model("claude-3.5-sonnet")
///    .messages(vec![MessageBuilder::default()
///        .role(MessageRole::User)
///        .content("Hello world!")
///        .build()
///        .unwrap()])
///    .build()
///    .unwrap();
///
/// client.messages().create(request).await.unwrap();
/// # }
/// ```
#[derive(Clone, Debug, Builder)]
#[builder(setter(into, strip_option))]
pub struct Client {
    #[builder(default)]
    http_client: reqwest::Client,
    #[builder(default)]
    base_url: String,
    #[builder(default = default_api_key())]
    api_key: secrecy::SecretString,
    #[builder(default)]
    version: String,
    #[builder(default)]
    beta: Option<String>,
    #[builder(default)]
    backoff: ExponentialBackoff,
}

impl Default for Client {
    fn default() -> Self {
        // Load backoff settings from configuration
        let backoff = ExponentialBackoffBuilder::default()
            .with_initial_interval(Duration::from_secs(15))
            .with_multiplier(2.0)
            .with_randomization_factor(0.05)
            .with_max_elapsed_time(Some(Duration::from_secs(120)))
            .build();

        Self {
            http_client: reqwest::Client::new(),
            api_key: default_api_key(), // Default env?
            version: "2023-06-01".to_string(),
            beta: None,
            base_url: BASE_URL.to_string(),
            backoff,
        }
    }
}

fn default_api_key() -> secrecy::SecretString {
    if cfg!(test) {
        return "test".into();
    }
    std::env::var("ANTHROPIC_API_KEY")
        .unwrap_or_else(|_| {
            tracing::warn!("Default Anthropic client initialized without api key");
            String::new()
        })
        .into()
}

impl Client {
    /// Build a new client from an API key
    pub fn from_api_key(api_key: impl Into<secrecy::SecretString>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Create a new client builder
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// Set a custom backoff strategy
    pub fn with_backoff(mut self, backoff: ExponentialBackoff) -> Self {
        self.backoff = backoff;
        self
    }

    /// Call the messages api
    pub fn messages(&self) -> Messages {
        Messages::new(self)
    }

    /// Make post request to the API
    ///
    /// This includes all headers and error handling
    pub async fn post<I, O>(&self, path: &str, request: I) -> Result<O, AnthropicError>
    where
        I: Serialize,
        O: DeserializeOwned,
    {
        backoff::future::retry(self.backoff.clone(), || async {
            let mut request = self
                .http_client
                .post(format!(
                    "{}/{}",
                    &self.base_url.trim_end_matches('/'),
                    &path.trim_start_matches('/')
                ))
                .header("x-api-key", self.api_key.expose_secret())
                .header("anthropic-version", &self.version)
                .header("content-type", "application/json")
                .json(&request);

            if let Some(beta_value) = &self.beta {
                request = request.header("anthropic-beta", beta_value);
            }

            let response = request
                .send()
                .await
                .map_err(AnthropicError::NetworkError)
                .map_err(backoff::Error::Permanent)?;
            let status = response.status();

            // 529 is the status code for overloaded requests
            let overloaded_status = StatusCode::from_u16(529).expect("529 is a valid status code");

            match status {
                StatusCode::OK => {
                    let response = response
                        .json::<O>()
                        .await
                        .map_err(AnthropicError::NetworkError)
                        .map_err(backoff::Error::Permanent)?;
                    Ok(response)
                }
                StatusCode::BAD_REQUEST => {
                    let text = response
                        .text()
                        .await
                        .map_err(AnthropicError::NetworkError)
                        .map_err(backoff::Error::Permanent)?;
                    Err(BackoffError::Permanent(AnthropicError::BadRequest(text)))
                }
                StatusCode::UNAUTHORIZED => {
                    Err(BackoffError::Permanent(AnthropicError::Unauthorized))
                }

                _ if status == StatusCode::TOO_MANY_REQUESTS || status == overloaded_status => {
                    let text = response
                        .text()
                        .await
                        .map_err(AnthropicError::NetworkError)
                        .map_err(backoff::Error::Permanent)?;

                    // Rate limited retry...
                    tracing::warn!("Rate limited: {}", text);
                    Err(backoff::Error::Transient {
                        err: AnthropicError::ApiError(text),
                        retry_after: None,
                    })
                }
                _ => {
                    let text = response
                        .text()
                        .await
                        .map_err(AnthropicError::NetworkError)
                        .map_err(backoff::Error::Permanent)?;
                    Err(BackoffError::Permanent(AnthropicError::Unknown(text)))
                }
            }
        })
        .await
    }
}
