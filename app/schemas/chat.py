# Agentic RAG - chat request/response schemas.

from pydantic import BaseModel, Field, field_validator

# Hard bounds on untrusted chat input (OWASP Input Validation [13]): a single
# message is capped in size and the history list is bounded, so one request
# can't force unbounded embedding/LLM work or memory use.
MAX_MESSAGE_CHARS = 32000
MAX_MESSAGES = 100


class ChatRequest(BaseModel):
    model: str | None = Field(default=None, max_length=128)
    messages: list[dict] = Field(default_factory=list, max_length=MAX_MESSAGES)
    stream: bool = False
    conversation_id: str | None = Field(default=None, max_length=64)
    collection: str | None = Field(default=None, max_length=128)
    # Optional metadata filter: {user_id, date_from, date_to, tags, tags_mode}.
    filters: dict | None = None

    @field_validator("messages")
    @classmethod
    def _cap_message_length(cls, v):
        for m in v:
            content = m.get("content")
            if content is not None and len(str(content)) > MAX_MESSAGE_CHARS:
                raise ValueError("message content too long")
        return v
