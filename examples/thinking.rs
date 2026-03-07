use dotenv::dotenv;
use async_anthropic::{
    types::{CreateMessagesRequestBuilder, MessageBuilder, MessageRole, Thinking},
    Client,
};
use tokio_stream::StreamExt as _;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let client = Client::default();

    let request = CreateMessagesRequestBuilder::default()
        .model("claude-sonnet-4-6")
        .max_tokens(4096)
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("What is 15 * 37? Think step by step.")
            .build()
            .unwrap()])
        .thinking(Thinking::Enabled { budget_tokens: 2048 })
        .build()
        .unwrap();

    let mut stream = client.messages().create_stream(request).await;

    while let Some(response) = stream.next().await {
        match response {
            Ok(msg) => println!("{msg:?}"),
            Err(e) => eprintln!("Error: {e:?}"),
        }
    }

    Ok(())
}
