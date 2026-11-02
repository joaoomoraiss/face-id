from datetime import datetime

from pydantic import BaseModel

from app.models.user import User, role_value


class RegisterResponse(BaseModel):
    id: str
    name: str
    email: str
    role: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_user(cls, user: User) -> "RegisterResponse":
        return cls(
            id=str(user.id),
            name=user.name,
            email=user.email,
            role=role_value(user.role),
            created_at=user.created_at,
            updated_at=user.updated_at,
        )


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
