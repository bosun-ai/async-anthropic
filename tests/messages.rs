use async_anthropic::{
    errors::{AnthropicError, CreateMessagesError},
    types::{CreateMessagesRequestBuilder, MessageBuilder, MessageContent, MessageRole},
    Client,
};
use async_trait::async_trait;
use backoff::ExponentialBackoffBuilder;
use serde_json::json;
use std::{sync::Arc, sync::Mutex, time::Duration};
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

// Helper trait for setting up and tearing down mock server
#[async_trait]
pub trait MockApp {
    async fn setup() -> MockServer;
}

struct TestSetup;

#[async_trait]
impl MockApp for TestSetup {
    async fn setup() -> MockServer {
        MockServer::start().await
    }
}

#[tokio::test]
async fn test_client_build_request() {
    let secret_key = "test_secret";

    let request = Client::builder().api_key(secret_key).build();

    assert!(request.is_ok());
}

#[test_log::test(tokio::test)]
async fn test_successful_request_execution() {
    let server = TestSetup::setup().await;
    let secret_key = "test_secret";

    // Mock successful response
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "mocked response"}]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::builder()
        .api_key(secret_key)
        .base_url(server.uri())
        .build()
        .unwrap();

    let request = CreateMessagesRequestBuilder::default()
        .model("test-model".to_string())
        .stream(true)
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("Hello world!")
            .build()
            .unwrap()])
        .build()
        .unwrap();

    let result = client.messages().create(request).await.unwrap();

    if let Some(content) = result.content {
        if let MessageContent::Text(text) = &content[0] {
            assert_eq!(text.text, "mocked response");
        }
    }
}

#[tokio::test]
async fn test_with_backoff_basic() {
    let server = TestSetup::setup().await;
    let secret_key = "test_secret";

    // Mock 429 Too Many Requests, expecting retries
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(429)
                .set_body_string("Too Many Requests")
                .set_delay(Duration::from_millis(10)),
        )
        .up_to_n_times(20)
        .expect(1..)
        .mount(&server)
        .await;

    let backoff = ExponentialBackoffBuilder::default()
        .with_initial_interval(Duration::from_millis(10))
        .with_multiplier(2.0)
        .with_randomization_factor(0.0)
        .with_max_elapsed_time(Some(Duration::from_millis(100)))
        .build();

    let client = Client::builder()
        .base_url(server.uri())
        .api_key(secret_key)
        .build()
        .unwrap()
        .with_backoff(backoff);

    let request = CreateMessagesRequestBuilder::default()
        .model("test-model".to_string())
        .stream(true)
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("Hello world!")
            .build()
            .unwrap()])
        .build()
        .unwrap();

    let result = client.messages().create(request).await;

    assert!(result.is_err());
    assert!(
        matches!(
            result.as_ref().unwrap_err(),
            CreateMessagesError::AnthropicError(AnthropicError::ApiError(_))
        ),
        "actual: {:?}",
        &result
    )
}

pub struct RetryResponder {
    num_calls_before_success: Arc<Mutex<u64>>,
    calls_made: Arc<Mutex<u64>>,
}

impl RetryResponder {
    pub fn new(num_calls_before_success: u64) -> Self {
        Self {
            num_calls_before_success: Arc::new(Mutex::new(num_calls_before_success)),
            calls_made: Arc::new(Mutex::new(0)),
        }
    }
}

impl wiremock::Respond for RetryResponder {
    fn respond(&self, _: &wiremock::Request) -> ResponseTemplate {
        let i = *self.calls_made.lock().unwrap();
        let succ_calls = *self.num_calls_before_success.lock().unwrap();

        if i < succ_calls {
            *self.calls_made.lock().unwrap() += 1;
            ResponseTemplate::new(429)
                .set_body_string("Too Many Requests")
                .set_delay(Duration::from_millis(10))
        } else {
            ResponseTemplate::new(200).set_body_json(json!({
                "content": [{"type": "text", "text": "retried response"}]
            }))
        }
    }
}

#[tokio::test]
async fn test_default_backoff_retries() {
    let server = TestSetup::setup().await;
    let secret_key = "test_secret";

    let bad_calls = 3;
    let expected_calls = bad_calls + 1; // + 1 good call at the end

    let rr = RetryResponder::new(bad_calls);

    // Mock 429 Too Many Requests, expecting retries
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(rr)
        .expect(expected_calls)
        .mount(&server)
        .await;

    let backoff = ExponentialBackoffBuilder::default()
        .with_initial_interval(Duration::from_millis(10))
        .with_multiplier(2.0)
        .with_randomization_factor(0.0)
        .with_max_elapsed_time(Some(Duration::from_millis(200)))
        .build();

    let client = Client::builder()
        .base_url(server.uri())
        .api_key(secret_key)
        .build()
        .unwrap()
        .with_backoff(backoff);

    let request = CreateMessagesRequestBuilder::default()
        .model("test-model".to_string())
        .stream(true)
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("Hello world!")
            .build()
            .unwrap()])
        .build()
        .unwrap();

    let result = client.messages().create(request).await;
    assert!(result.is_ok());
    let result = result.unwrap();

    if let Some(content) = result.content {
        if let MessageContent::Text(text) = &content[0] {
            assert_eq!(text.text, "retried response");
        }
    }
}

#[tokio::test]
#[ignore = "streaming not implemented"]
async fn test_streaming_response() {
    let server = TestSetup::setup().await;
    let secret_key = "test_secret";

    // Mock streaming response
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_string("event: content_block_start\ndata: {\"type\": \"text\", \"text\": \"streamed chunk\"}\n\n"))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::builder()
        .base_url(server.uri())
        .api_key(secret_key)
        .build()
        .unwrap();

    let request = CreateMessagesRequestBuilder::default()
        .model("test-model".to_string())
        .stream(true)
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("Hello world!")
            .build()
            .unwrap()])
        .build()
        .unwrap();

    let result = client.messages().create(request).await.unwrap();

    if let Some(content) = result.content {
        if let MessageContent::Text(text) = &content[0] {
            assert_eq!(text.text, "streamed chunk");
        }
    }
}

#[tokio::test]
async fn test_error_handling_bad_request() {
    let server = TestSetup::setup().await;
    let secret_key = "test_secret";

    // Mock 400 Bad Request response
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(400).set_body_string("Bad request"))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::builder()
        .base_url(server.uri())
        .api_key(secret_key)
        .build()
        .unwrap();

    let request = CreateMessagesRequestBuilder::default()
        .model("test-model".to_string())
        .stream(true)
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("Hello world!")
            .build()
            .unwrap()])
        .build()
        .unwrap();

    let result = client.messages().create(request).await;

    assert!(result.is_err());
    assert!(
        matches!(
            result.as_ref().unwrap_err(),
            CreateMessagesError::AnthropicError(AnthropicError::BadRequest(_))
        ),
        "actual: {:?}",
        &result
    )
}

#[tokio::test]
async fn test_error_handling_unauthorized() {
    let server = TestSetup::setup().await;
    let secret_key = "test_secret";

    // Mock 401 Unauthorized response
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::builder()
        .base_url(server.uri())
        .api_key(secret_key)
        .build()
        .unwrap();

    let request = CreateMessagesRequestBuilder::default()
        .model("test-model".to_string())
        .stream(true)
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("Hello world!")
            .build()
            .unwrap()])
        .build()
        .unwrap();

    let result = client.messages().create(request).await;

    assert!(result.is_err());
    assert!(
        matches!(
            result.as_ref().unwrap_err(),
            CreateMessagesError::AnthropicError(AnthropicError::Unauthorized)
        ),
        "actual: {:?}",
        &result
    )
}
