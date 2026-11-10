import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, String, func
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Role(enum.Enum):
    GERAL = "GERAL PERMISSIONS"
    DIRECTOR = "DIRECTOR PERMISSIONS"
    MINISTRY = "MINISTRY PERMISSIONS"


# nivel de acesso de cada role. quanto maior, mais pode
ROLE_LEVEL = {
    Role.GERAL.value: 1,
    Role.DIRECTOR.value: 2,
    Role.MINISTRY.value: 3,
}

CARGO_MAX_LENGTH = 120


# grava o valor ("GERAL PERMISSIONS"...) no banco, n o nome GERAL/DIRECTOR/MINISTRY
def _role_values(roles: list[Role]) -> list[str]:
    return [role.value for role in roles]


# as vezes chega enum, as vezes string. token e json usam a string
def role_value(role: Role | str) -> str:
    if isinstance(role, Role):
        return role.value
    return role


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        index=True,
        default=uuid.uuid4,
    )
    # nome pode repetir, login é pelo email
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    # todo mundo nasce geral. director/ministry eu troco direto no banco
    role: Mapped[Role] = mapped_column(
        Enum(Role, values_callable=_role_values),
        default=Role.GERAL,
        nullable=False,
    )
    # cargo da pessoa (ex: professor). texto livre, n tem nada a ver com permissao
    cargo: Mapped[str | None] = mapped_column(String(CARGO_MAX_LENGTH), nullable=True)
    # vetor do rosto (facenet), double precision[] no postgres
    embedding: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
