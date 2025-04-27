use crate::{
    errors::AnthropicError,
    types::{GetModelResponse, ListModelsResponse},
    Client,
};

pub const DEFAULT_MAX_TOKENS: i32 = 2048;

#[derive(Debug, Clone)]
pub struct Models<'c> {
    client: &'c Client,
}

impl Models<'_> {
    pub fn new(client: &Client) -> Models {
        Models { client }
    }

    #[tracing::instrument(skip_all)]
    pub async fn list(&self) -> Result<ListModelsResponse, AnthropicError> {
        self.client.get("/v1/models").await
    }

    #[tracing::instrument(skip_all)]
    pub async fn get(&self, model_id: impl AsRef<str>) -> Result<GetModelResponse, AnthropicError> {
        self.client
            .get(&format!("/v1/models/{}", model_id.as_ref()))
            .await
    }
}
