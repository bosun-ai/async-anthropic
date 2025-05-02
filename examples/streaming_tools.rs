// examples/basic_usage.rs

use async_anthropic::{
    types::{CreateMessagesRequestBuilder, MessageBuilder, MessageRole},
    Client,
};
use serde_json::json;
use tokio_stream::StreamExt as _;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::default();

    let request = CreateMessagesRequestBuilder::default()
        .model("claude-3-5-sonnet-20241022")
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("What is the weather like in San Francisco?")
            .build()
            .unwrap()])
        .tools([json!({
          "name": "get_weather",
          "description": "Get the current weather in a given location",
          "input_schema": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              }
            },
            "required": ["location"]
          }
        })
        .as_object()
        .unwrap()
        .to_owned()])
        .build()
        .unwrap();

    let mut stream = client.messages().create_stream(request).await;

    while let Some(response) = stream.next().await {
        match response {
            Ok(msg) => println!("{:?}", msg),
            Err(e) => eprintln!("Error: {:?}", e),
        }
    }

    Ok(())
}
