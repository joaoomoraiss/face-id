from pydantic import BaseModel


class RegisterRequest(BaseModel):
    name: str
    email: str
    cargo: str | None = None


class LoginRequest(BaseModel):
    email: str
