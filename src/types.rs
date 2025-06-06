use std::{
    ops::{Deref, DerefMut},
    pin::Pin,
};

use derive_builder::Builder;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;
use tokio_stream::Stream;

use crate::{errors::AnthropicError, messages};

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub enum ToolChoice {
    None,
    Auto {
        disable_parallel_tool_use: bool,
    },
    Any {
        disable_parallel_tool_use: bool,
    },
    Tool {
        name: String,
        disable_parallel_tool_use: bool,
    },
}

impl ToolChoice {
    /// Instruct the model to not use any tools.
    #[must_use]
    pub fn none() -> Self {
        ToolChoice::None
    }

    /// Instruct the model to use zero, one, or more tools.
    #[must_use]
    pub fn auto() -> Self {
        ToolChoice::Auto {
            disable_parallel_tool_use: false,
        }
    }

    /// Instruct the model to use one, or more tools.
    #[must_use]
    pub fn any() -> Self {
        ToolChoice::Any {
            disable_parallel_tool_use: false,
        }
    }

    /// Instruct the model to use the specified tool.
    #[must_use]
    pub fn tool(name: String) -> Self {
        ToolChoice::Tool {
            name,
            disable_parallel_tool_use: false,
        }
    }

    /// Enable or disable parallel tool use for this tool choice.
    #[must_use]
    pub fn with_disable_parallel_tool_use(self, disable_parallel_tool_use: bool) -> Self {
        match self {
            ToolChoice::None => ToolChoice::None,
            ToolChoice::Auto { .. } => ToolChoice::Auto {
                disable_parallel_tool_use,
            },
            ToolChoice::Any { .. } => ToolChoice::Any {
                disable_parallel_tool_use,
            },
            ToolChoice::Tool { name, .. } => ToolChoice::Tool {
                name,
                disable_parallel_tool_use,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder, PartialEq)]
#[builder(setter(into))]
pub struct ExtendedThinking {
    #[serde(rename = "type")]
    pub kind: String,
    pub budget_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder, PartialEq, Default)]
#[builder(setter(into, strip_option), default)]
pub struct Message {
    pub role: MessageRole,
    pub content: MessageContentList,
}

impl Message {
    /// Returns all the tool uses in the message
    pub fn tool_uses(&self) -> Vec<ToolUse> {
        self.content
            .0
            .iter()
            .filter_map(|c| match c {
                MessageContent::ToolUse(tool_use) => Some(tool_use.clone()),
                _ => None,
            })
            .collect()
    }

    /// Returns the first text content in the message
    pub fn text(&self) -> Option<String> {
        self.content
            .0
            .iter()
            .filter_map(|c| match c {
                MessageContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .next()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MessageContentList(pub Vec<MessageContent>);

impl Deref for MessageContentList {
    type Target = Vec<MessageContent>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MessageContentList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    #[default]
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(setter(into, strip_option))]
pub struct CreateMessagesRequest {
    pub messages: Vec<Message>,
    pub model: String,
    #[builder(default = messages::DEFAULT_MAX_TOKENS)]
    pub max_tokens: i32,
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ExtendedThinking>,
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[builder(default = "false")]
    pub stream: bool, // Optional default false
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub temperature: Option<f32>, // 0 < x < 1
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub tool_choice: Option<ToolChoice>,
    // TODO: Type this
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub tools: Option<Vec<serde_json::Map<String, Value>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub top_k: Option<u32>, // > 0
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub top_p: Option<f32>, // 0 < x < 1
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub system: Option<String>, // 0 < x < 1
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(setter(into, strip_option))]
pub struct CreateMessagesResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub content: Option<Vec<MessageContent>>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl CreateMessagesResponse {
    /// Returns the content as Messages so they are more easily reusable
    pub fn messages(&self) -> Vec<Message> {
        let Some(content) = &self.content else {
            return vec![];
        };
        content
            .iter()
            .map(|c| Message {
                role: MessageRole::Assistant,
                content: c.clone().into(),
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    ToolUse(ToolUse),
    ToolResult(ToolResult),
    Text(Text),
    Thinking(Thinking),

    /// See Anthropic's docs for more information:
    ///
    /// > Occasionally Claude’s internal reasoning will be flagged by our safety
    /// > systems. When this occurs, we encrypt some or all of the thinking
    /// > block and return it to you as a redacted_thinking block.
    /// > redacted_thinking blocks are decrypted when passed back to the API,
    /// > allowing Claude to continue its response without losing context.
    ///
    /// See:
    /// <https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#thinking-redaction>
    RedactedThinking {
        data: String,
    },
}

impl MessageContent {
    pub fn as_tool_use(&self) -> Option<&ToolUse> {
        if let MessageContent::ToolUse(tool_use) = self {
            Some(tool_use)
        } else {
            None
        }
    }

    pub fn as_tool_result(&self) -> Option<&ToolResult> {
        if let MessageContent::ToolResult(tool_result) = self {
            Some(tool_result)
        } else {
            None
        }
    }

    pub fn as_text(&self) -> Option<&Text> {
        if let MessageContent::Text(text) = self {
            Some(text)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default, Builder)]
#[builder(setter(into, strip_option), default)]
pub struct ToolUse {
    pub id: String,
    pub input: Value,
    pub name: String,
}

impl From<ToolUse> for MessageContent {
    fn from(tool_use: ToolUse) -> Self {
        MessageContent::ToolUse(tool_use)
    }
}

impl From<ToolUse> for MessageContentList {
    fn from(tool_use: ToolUse) -> Self {
        MessageContentList(vec![tool_use.into()])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default, Builder)]
#[builder(setter(into, strip_option), default)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub content: Option<String>,
    pub is_error: bool,
}

impl From<ToolResult> for MessageContent {
    fn from(tool_result: ToolResult) -> Self {
        MessageContent::ToolResult(tool_result)
    }
}

impl From<ToolResult> for MessageContentList {
    fn from(tool_result: ToolResult) -> Self {
        MessageContentList(vec![tool_result.into()])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default, Builder)]
#[builder(setter(into, strip_option), default)]
pub struct Text {
    pub text: String,
}

impl<S: AsRef<str>> From<S> for Text {
    fn from(s: S) -> Self {
        Text {
            text: s.as_ref().to_string(),
        }
    }
}

impl From<Text> for MessageContent {
    fn from(text: Text) -> Self {
        MessageContent::Text(text)
    }
}

impl From<Text> for MessageContentList {
    fn from(text: Text) -> Self {
        MessageContentList(vec![text.into()])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default, Builder)]
#[builder(setter(into, strip_option), default)]
pub struct Thinking {
    pub thinking: String,
    pub signature: Option<String>,
}

impl Thinking {
    pub fn with_signature(mut self, signature: String) -> Self {
        self.signature = Some(signature);
        self
    }
}

impl<S: AsRef<str>> From<S> for Thinking {
    fn from(s: S) -> Self {
        Thinking {
            thinking: s.as_ref().to_string(),
            signature: None,
        }
    }
}

impl From<Thinking> for MessageContent {
    fn from(thinking: Thinking) -> Self {
        MessageContent::Thinking(thinking)
    }
}

impl From<Thinking> for MessageContentList {
    fn from(thinking: Thinking) -> Self {
        MessageContentList(vec![thinking.into()])
    }
}

impl<S: AsRef<str>> From<S> for MessageContent {
    fn from(s: S) -> Self {
        MessageContent::Text(Text {
            text: s.as_ref().to_string(),
        })
    }
}

impl<S: AsRef<str>> From<S> for Message {
    fn from(s: S) -> Self {
        MessageBuilder::default()
            .role(MessageRole::User)
            .content(s.as_ref().to_string())
            .build()
            .expect("infallible")
    }
}

// Any single AsRef<str> can be converted to a MessageContent, in a list as a single item
impl<S: AsRef<str>> From<S> for MessageContentList {
    fn from(s: S) -> Self {
        MessageContentList(vec![s.as_ref().into()])
    }
}

impl From<MessageContent> for MessageContentList {
    fn from(content: MessageContent) -> Self {
        MessageContentList(vec![content])
    }
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            ToolChoice::None => serde::Serialize::serialize(
                &serde_json::json!({
                  "type": "none",
                }),
                serializer,
            ),
            ToolChoice::Auto {
                disable_parallel_tool_use,
            } => serde::Serialize::serialize(
                &serde_json::json!({
                  "type": "auto",
                  "disable_parallel_tool_use": disable_parallel_tool_use,
                }),
                serializer,
            ),
            ToolChoice::Any {
                disable_parallel_tool_use,
            } => serde::Serialize::serialize(
                &serde_json::json!({
                  "type": "any",
                  "disable_parallel_tool_use": disable_parallel_tool_use,
                }),
                serializer,
            ),
            ToolChoice::Tool {
                name,
                disable_parallel_tool_use,
            } => serde::Serialize::serialize(
                &serde_json::json!({
                    "type": "tool",
                    "name": name,
                    "disable_parallel_tool_use": disable_parallel_tool_use
                }),
                serializer,
            ),
        }
    }
}
#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ContentBlockDelta {
    TextDelta { text: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum MessagesStreamEvent {
    MessageStart {
        message: MessageStart,
        usage: Option<Usage>,
    },
    ContentBlockStart {
        index: usize,
        content_block: MessageContent,
    },
    ContentBlockDelta {
        index: usize,
        delta: ContentBlockDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
        #[serde(default)]
        usage: Option<Usage>,
    },
    MessageStop,
}
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct MessageStart {
    pub id: String,
    pub model: String,
    pub role: String,
    pub content: Vec<MessageContent>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

pub type CreateMessagesResponseStream =
    Pin<Box<dyn Stream<Item = Result<MessagesStreamEvent, AnthropicError>> + Send>>;

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ListModelsResponse {
    #[serde(default)]
    pub data: Vec<Model>,

    #[serde(default)]
    pub first_id: Option<String>,
    pub has_more: bool,
    #[serde(default)]
    pub last_id: Option<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct Model {
    pub created_at: String,
    pub display_name: String,
    pub id: String,
    #[serde(rename = "type")]
    pub model_type: String,
}

pub type GetModelResponse = Model;

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test_log::test(tokio::test)]
    async fn test_deserialize_response() {
        let response = json!({
          "id": "msg_01KkaCASJuaAgTWD2wqdbwC8",
          "type": "message",
          "role": "assistant",
          "model": "claude-3-5-sonnet-20241022",
          "content": [
            {
              "type": "text",
              "text": "Hi! How can I help you today?"
            }
          ],
          "stop_reason": "end_turn",
          "stop_sequence": null,
          "usage": {
            "input_tokens": 10,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 12
          }
        })
        .to_string();

        let response = serde_json::from_str::<CreateMessagesResponse>(&response).unwrap();

        let usage = response.usage.as_ref().unwrap();

        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(12));
        assert_eq!(
            response.id,
            Some("msg_01KkaCASJuaAgTWD2wqdbwC8".to_string())
        );
        assert_eq!(
            response.model,
            Some("claude-3-5-sonnet-20241022".to_string())
        );
        assert_eq!(response.stop_reason, Some("end_turn".to_string()));
        assert_eq!(response.stop_sequence, None);
        assert_eq!(
            response
                .messages()
                .first()
                .unwrap()
                .content
                .first()
                .unwrap()
                .as_text(),
            Some(&Text {
                text: "Hi! How can I help you today?".to_string(),
            })
        );
    }

    #[test_log::test(tokio::test)]
    async fn test_from_str() {
        let message: Message = "Hello world!".into();

        assert_eq!(
            message,
            Message {
                role: MessageRole::User,
                content: MessageContentList(vec![MessageContent::Text(Text {
                    text: "Hello world!".to_string()
                })]),
            }
        );

        assert_eq!(message.text(), Some("Hello world!".to_string()));
    }
}
