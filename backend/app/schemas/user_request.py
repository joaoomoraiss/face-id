from pydantic import BaseModel, Field


class CreateUserRequest(BaseModel):
    name: str
    email: str
    cargo: str | None = None


class UpdateUserRequest(BaseModel):
    name: str | None = None
    email: str | None = None
    cargo: str | None = None


class ListUsersRequest(BaseModel):
    skip: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=100)
