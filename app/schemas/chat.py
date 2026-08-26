# Agentic RAG - chat request/response schemas.

from pydantic import BaseModel


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    stream: bool = False
    conversation_id: str | None = None
    collection: str | None = None
    # Optional metadata filter: {user_id, date_from, date_to, tags, tags_mode}.
    filters: dict | None = None
