import asyncio
from collections.abc import Callable
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.exceptions.user import InvalidFieldException, UserAlreadyExistsException, UserNotFoundException
from app.models.user import User
from app.services.auth import clean_cargo, commit_user, parse_user_id
from app.services.face import Embedder


class UserService:
    def __init__(
        self,
        face_service: Embedder,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.face_service = face_service
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def update_user(
        self,
        db: Session,
        user_id: str,
        *,
        name: str | None = None,
        email: str | None = None,
        image: bytes | None = None,
        content_type: str | None = None,
        cargo: str | None = None,
    ) -> User:
        # patch vazio n faz sentido
        if name is None and email is None and image is None and cargo is None:
            raise InvalidFieldException("Informe ao menos um campo")
        new_cargo = clean_cargo(cargo)
        current = await self.get_user(db, user_id)
        new_name = current.name if name is None else name
        new_email = current.email if email is None else email
        # so checa email se mudou, e ignora o proprio user
        if new_email != current.email:
            owner = db.scalar(select(User).where(User.email == new_email))
            if owner is not None and owner.id != current.id:
                raise UserAlreadyExistsException()
        embedding = current.embedding
        if image is not None:
            embedding = await asyncio.to_thread(self.face_service.extract_embedding, image, content_type)
        # role fica como ta, update n promove ninguem
        current.name = new_name
        current.email = new_email
        current.embedding = list(embedding)
        # None = n mandou, fica como ta
        if cargo is not None:
            current.cargo = new_cargo
        current.updated_at = self._clock()
        return commit_user(db, current)

    async def get_user(self, db: Session, user_id: str) -> User:
        parsed = parse_user_id(user_id)
        user = db.get(User, parsed) if parsed is not None else None
        if user is None:
            raise UserNotFoundException()
        return user

    async def delete_user(self, db: Session, user_id: str) -> None:
        parsed = parse_user_id(user_id)
        user = db.get(User, parsed) if parsed is not None else None
        if user is None:
            raise UserNotFoundException()
        db.delete(user)
        db.commit()

    # sem paginacao, devolve todo mundo
    async def list_all_users(self, db: Session) -> list[User]:
        return list(db.scalars(select(User).order_by(User.created_at.desc())))

    async def list_users(self, db: Session, *, skip: int, limit: int) -> tuple[list[User], int]:
        total = db.scalar(select(func.count()).select_from(User)) or 0
        # mais novo primeiro
        items = list(
            db.scalars(
                select(User).order_by(User.created_at.desc()).offset(skip).limit(limit)
            )
        )
        return items, total
