use crate::{
    errors::AnthropicError,
    types::{CreateMessagesRequest, CreateMessagesResponse, CreateMessagesResponseStream},
    Client,
};

pub const DEFAULT_MAX_TOKENS: i32 = 2048;

#[derive(Debug, Clone)]
pub struct Messages<'c> {
    client: &'c Client,
}

impl Messages<'_> {
    pub fn new(client: &Client) -> Messages<'_> {
        Messages { client }
    }

    #[tracing::instrument(skip_all)]
    pub async fn create(
        &self,
        request: impl Into<CreateMessagesRequest>,
    ) -> Result<CreateMessagesResponse, AnthropicError> {
        let mut request = request.into();
        request.stream = false;

        self.client.post("/v1/messages", request).await
    }

    #[tracing::instrument(skip_all)]
    pub async fn create_stream(
        &self,
        request: impl Into<CreateMessagesRequest>,
    ) -> CreateMessagesResponseStream {
        let mut request = request.into();
        request.stream = true;

        self.client
            .post_stream(
                "/v1/messages",
                request,
                [
                    "message_start",
                    "message_delta",
                    "message_stop",
                    "content_block_start",
                    "content_block_delta",
                    "content_block_stop",
                ],
            )
            .await
    }
}
