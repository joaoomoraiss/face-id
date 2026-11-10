from datetime import datetime

from pydantic import BaseModel

from app.models.user import User, role_value


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: str
    cargo: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_user(cls, user: User) -> "UserResponse":
        # embedding fica de fora da resposta
        return cls(
            id=str(user.id),
            name=user.name,
            email=user.email,
            role=role_value(user.role),
            cargo=user.cargo,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )


class UserListResponse(BaseModel):
    items: list[UserResponse]
    total: int
