use async_anthropic::{
    types::{
        CreateMessagesRequestBuilder, MessageBuilder, MessageRole, ToolChoice, ToolResultBuilder,
    },
    Client,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::default();

    let mut messages = vec![];

    messages.push(
        MessageBuilder::default()
            .role(MessageRole::User)
            .content("What is the weather like in San Francisco?")
            .build()
            .unwrap(),
    );

    let request = CreateMessagesRequestBuilder::default()
        .model("claude-3-5-sonnet-20241022")
        .messages(messages.as_slice())
        // TODO: Type the tool spec so we can skip the shenanigans
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
        .tool_choice(ToolChoice::auto())
        .build()
        .unwrap();

    let response = client.messages().create(request).await?;

    println!("1. ---");
    println!("{response:?}");

    for message in response.messages() {
        messages.push(message.clone());

        for tool_use in message.tool_uses() {
            println!("Tool use: {tool_use:?}");
            let location: String =
                serde_json::from_value(tool_use.input["location"].clone()).unwrap();

            messages.push(
                MessageBuilder::default()
                    .role(MessageRole::User)
                    .content(
                        ToolResultBuilder::default()
                            .tool_use_id(&tool_use.id)
                            .content(format!("Pretty warm in {location}"))
                            .build()
                            .unwrap(),
                    )
                    .build()
                    .unwrap(),
            );
        }
    }

    let request = CreateMessagesRequestBuilder::default()
        .model("claude-3-5-sonnet-20241022")
        .messages(messages.as_slice())
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

    let response = client.messages().create(request).await?;

    println!("2. ---");
    println!("{response:?}");

    // 2. ---
    // CreateMessagesResponse { id: Some("msg_019EVre2rdkCFwusZpGPgzDp"), content: Some([Text(Text { text: "According to the weather report, it's pretty warm in San Francisco right now." })]), model: Some("claude-3-5-sonnet-20241022"), stop_reason: Some("end_turn"), stop_sequence: None, usage: Some(Usage { input_tokens: Some(516), output_tokens: Some(20) }) }
    Ok(())
}
