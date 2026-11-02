import asyncio
import math
import uuid
from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.exceptions.auth import InvalidCredentialsException
from app.exceptions.face import FaceProcessingException
from app.exceptions.user import UserAlreadyExistsException
from app.models.user import Role, User
from app.services.face import Embedder


# quanto menor, mais parecido. corte do facenet é 0.30
def cosine_distance(left: list[float], right: list[float]) -> float:
    # tamanho diferente ou vazio n da pra comparar
    if len(left) == 0 or len(left) != len(right):
        raise FaceProcessingException("Não foi possível processar o rosto")
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    # nan/inf estraga a conta e ainda passa batido no if
    for value in first:
        if not math.isfinite(value):
            raise FaceProcessingException("Não foi possível processar o rosto")
    for value in second:
        if not math.isfinite(value):
            raise FaceProcessingException("Não foi possível processar o rosto")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        raise FaceProcessingException("Não foi possível processar o rosto")
    distance = float(1.0 - (float(np.dot(first, second)) / denominator))
    if not math.isfinite(distance):
        raise FaceProcessingException("Não foi possível processar o rosto")
    return distance


# id na url vem string. se n for uuid, trato como n achou
def parse_user_id(user_id: uuid.UUID | str) -> uuid.UUID | None:
    if isinstance(user_id, uuid.UUID):
        return user_id
    try:
        return uuid.UUID(str(user_id))
    except (ValueError, TypeError, AttributeError):
        return None


# unique do email tbm estoura aqui se duas request passarem juntas
def commit_user(db: Session, user: User) -> User:
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise UserAlreadyExistsException() from None
    db.refresh(user)
    return user


class AuthService:
    def __init__(
        self,
        face_service: Embedder,
        *,
        distance_threshold: float,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.face_service = face_service
        self.distance_threshold = distance_threshold
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def register(
        self,
        db: Session,
        *,
        name: str,
        email: str,
        image: bytes,
        content_type: str | None,
    ) -> User:
        existing = db.scalar(select(User).where(User.email == email))
        if existing is not None:
            raise UserAlreadyExistsException()
        # deepface é pesado, joga pra outra thread pra n travar o event loop
        embedding = await asyncio.to_thread(self.face_service.extract_embedding, image, content_type)
        now = self._clock()
        user = User(
            name=name,
            email=email,
            role=Role.GERAL,  # cadastro n escolhe role
            embedding=list(embedding),
            created_at=now,
            updated_at=now,
        )
        db.add(user)
        return commit_user(db, user)

    async def authenticate(
        self,
        db: Session,
        *,
        email: str,
        image: bytes,
        content_type: str | None,
    ) -> User:
        user = db.scalar(select(User).where(User.email == email))
        # email q n existe e rosto diferente devolvem a mesma msg
        if user is None:
            raise InvalidCredentialsException()
        probe = await asyncio.to_thread(self.face_service.extract_embedding, image, content_type)
        distance = cosine_distance(list(user.embedding), probe)
        # 1e-9 é folga de float, senao o limite exato as vezes reprova
        if distance > self.distance_threshold + 1e-9:
            raise InvalidCredentialsException()
        return user
