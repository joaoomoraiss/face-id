from pydantic import BaseModel


class RegisterRequest(BaseModel):
    name: str
    email: str


class LoginRequest(BaseModel):
    email: str
