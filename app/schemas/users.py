# Agentic RAG - user / auth request schemas.

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(
        min_length=3, max_length=32,
        pattern=r"^[a-zA-Z0-9_.-]+$",
        description="3-32 chars: letters, digits, _ . -",
    )
    password: str
    display_name: str | None = Field(default=None, max_length=64)


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
