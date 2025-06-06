use backon::{ExponentialBuilder, Retryable as _};
use derive_builder::Builder;
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt as _};
use secrecy::ExposeSecret;
use serde::{de::DeserializeOwned, Serialize};
use std::{pin::Pin, time::Duration};
use tokio_stream::{Stream, StreamExt as _};

use crate::{
    errors::{map_deserialization_error, AnthropicError, StreamError},
    messages::Messages,
    models::Models,
};

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
    backoff: ExponentialBuilder,
}

impl Default for Client {
    fn default() -> Self {
        // Load backoff settings from configuration
        let backoff = ExponentialBuilder::default()
            .with_min_delay(Duration::from_secs(15))
            .with_factor(2.0)
            .with_jitter()
            .with_max_delay(Duration::from_secs(120));

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
    pub fn with_backoff(mut self, backoff: ExponentialBuilder) -> Self {
        self.backoff = backoff;
        self
    }

    /// Call the messages api
    pub fn messages(&self) -> Messages {
        Messages::new(self)
    }

    pub fn models(&self) -> Models {
        Models::new(self)
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-api-key", self.api_key.expose_secret().parse().unwrap());
        headers.insert("anthropic-version", self.version.parse().unwrap());
        if let Some(beta_value) = &self.beta {
            headers.insert("anthropic-beta", beta_value.parse().unwrap());
        }
        headers
    }

    fn format_url(&self, path: &str) -> String {
        format!(
            "{}/{}",
            &self.base_url.trim_end_matches('/'),
            &path.trim_start_matches('/')
        )
    }

    pub async fn get<O>(&self, path: &str) -> Result<O, AnthropicError>
    where
        O: DeserializeOwned,
    {
        let request = || async {
            let response = self
                .http_client
                .get(self.format_url(path))
                .headers(self.headers())
                .send()
                .await
                .map_err(AnthropicError::NetworkError)?;

            handle_response(response).await
        };

        request
            .retry(self.backoff)
            .sleep(tokio::time::sleep)
            .when(|e| matches!(e, AnthropicError::RateLimit))
            .await
    }

    /// Make post request to the API
    ///
    /// This includes all headers and error handling
    pub async fn post<I, O>(&self, path: &str, request: I) -> Result<O, AnthropicError>
    where
        I: Serialize,
        O: DeserializeOwned,
    {
        let request = || async {
            let mut request = self
                .http_client
                .post(self.format_url(path))
                .headers(self.headers())
                .json(&request);

            if let Some(beta_value) = &self.beta {
                request = request.header("anthropic-beta", beta_value);
            }

            let response = request.send().await.map_err(AnthropicError::NetworkError)?;

            handle_response(response).await
        };

        request
            .retry(self.backoff)
            .sleep(tokio::time::sleep)
            .when(|e| matches!(e, AnthropicError::RateLimit))
            .await
    }

    pub(crate) async fn post_stream<I, O, const N: usize>(
        &self,
        path: &str,
        request: I,
        event_types: [&'static str; N],
    ) -> Pin<Box<dyn Stream<Item = Result<O, AnthropicError>> + Send>>
    where
        I: Serialize,
        O: DeserializeOwned + Send + 'static,
    {
        let event_source = self
            .http_client
            .post(self.format_url(path))
            .headers(self.headers())
            .json(&request)
            .eventsource()
            .unwrap();

        stream(event_source, event_types).await
    }
}

async fn handle_response<O>(response: reqwest::Response) -> Result<O, AnthropicError>
where
    O: DeserializeOwned,
{
    let status = response.status();

    // 529 is the status code for overloaded requests
    let overloaded_status = StatusCode::from_u16(529).expect("529 is a valid status code");

    match status {
        StatusCode::OK => response
            .json::<O>()
            .await
            .map_err(AnthropicError::NetworkError),
        StatusCode::BAD_REQUEST => {
            let text = response
                .text()
                .await
                .map_err(AnthropicError::NetworkError)?;

            Err(AnthropicError::BadRequest(text))
        }
        StatusCode::UNAUTHORIZED => Err(AnthropicError::Unauthorized),
        _ if status == StatusCode::TOO_MANY_REQUESTS || status == overloaded_status => {
            let text = response
                .text()
                .await
                .map_err(AnthropicError::NetworkError)?;

            tracing::warn!("Rate limited: {}", text);
            Err(AnthropicError::RateLimit)
        }
        _ => {
            let text = response
                .text()
                .await
                .map_err(AnthropicError::NetworkError)?;

            Err(AnthropicError::Unknown(text))
        }
    }
}

async fn stream<O, const N: usize>(
    mut event_source: EventSource,
    event_types: [&'static str; N],
) -> Pin<Box<dyn Stream<Item = Result<O, AnthropicError>> + Send>>
where
    O: DeserializeOwned + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    tokio::spawn(async move {
        while let Some(ev) = event_source.next().await {
            tracing::trace!("Streaming event: {ev:?}");
            match ev {
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        let event = message.event.as_str();
                        if event == "ping" {
                            continue;
                        }

                        let response = if event == "error" {
                            match serde_json::from_str::<StreamError>(&message.data) {
                                Ok(e) => Err(AnthropicError::StreamError(e)),
                                Err(e) => {
                                    Err(map_deserialization_error(e, message.data.as_bytes()))
                                }
                            }
                        } else if event_types.contains(&event) {
                            match serde_json::from_str::<O>(&message.data) {
                                Ok(output) => Ok(output),
                                Err(e) => {
                                    Err(map_deserialization_error(e, message.data.as_bytes()))
                                }
                            }
                        } else {
                            Err(AnthropicError::StreamError(StreamError {
                                error_type: "unknown_event_type".to_string(),
                                message: Some(format!("Unknown event type: {event}")),
                                error: None,
                            }))
                        };
                        let cancel = response.is_err();
                        if tx.send(response).is_err() || cancel {
                            // rx dropped or other error
                            break;
                        }
                    }
                },
                Err(e) => {
                    if let reqwest_eventsource::Error::StreamEnded = e {
                        break;
                    }
                    if tx
                        .send(Err(AnthropicError::StreamError(StreamError {
                            error_type: "sse_error".to_string(),
                            message: Some(e.to_string()),
                            error: match e {
                                reqwest_eventsource::Error::InvalidContentType(_, response)
                                | reqwest_eventsource::Error::InvalidStatusCode(_, response) => {
                                    Some(response.text().await.unwrap_or_default().into())
                                }
                                _ => None,
                            },
                        })))
                        .is_err()
                    {
                        // rx dropped
                        break;
                    }
                }
            }
        }

        event_source.close();
    });

    Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
}
