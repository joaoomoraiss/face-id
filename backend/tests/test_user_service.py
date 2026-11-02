import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.user import User
from tests.db import open_test_session


class ScriptedFace:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = list(embeddings)
        self.calls = 0

    def extract_embedding(self, content: bytes, content_type: str | None) -> list[float]:
        self.calls += 1
        return self.embeddings.pop(0)


class Clock:
    def __init__(self) -> None:
        self.current = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        value = self.current
        self.current += timedelta(seconds=1)
        return value


def _services(embeddings: list[list[float]], threshold: float = 0.30):
    from app.services.auth import AuthService
    from app.services.user import UserService

    face = ScriptedFace(embeddings)
    clock = Clock()
    db = open_test_session()
    auth = AuthService(
        face,
        distance_threshold=threshold,
        clock=clock,
    )
    user = UserService(face, clock=clock)
    return auth, user, db, face


async def test_register_stores_user_role_and_email():
    auth, _user, db, _face = _services([[1.0, 0.0]])
    user = await auth.register(db, 
        name="Ana",
        email="ana@example.com",
        image=b"img",
        content_type="image/png",
    )
    from app.models.user import Role

    assert user.role == Role.GERAL
    assert user.email == "ana@example.com"
    assert user.name == "Ana"
    assert user.embedding == [1.0, 0.0]
    assert user.created_at == user.updated_at
    stored = db.scalar(select(User).where(User.email == "ana@example.com"))
    assert stored is not None
    assert stored.id == user.id


async def test_register_rejects_duplicate_email():
    from app.exceptions.user import UserAlreadyExistsException

    auth, _user, db, _face = _services([[1.0, 0.0], [1.0, 0.0]])
    await auth.register(db, name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    with pytest.raises(UserAlreadyExistsException):
        await auth.register(db, name="Ana 2", email="ana@example.com", image=b"img", content_type="image/png")


async def test_login_accepts_distance_equal_to_threshold():
    auth, _user, db, _face = _services([[1.0, 0.0], [0.7, 0.714142842854285]])
    await auth.register(db, name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    user = await auth.authenticate(db, email="ana@example.com", image=b"img", content_type="image/png")
    assert user.email == "ana@example.com"


async def test_login_rejects_unknown_email_without_calling_face():
    from app.exceptions.auth import InvalidCredentialsException

    auth, _user, db, face = _services([[1.0, 0.0]])
    with pytest.raises(InvalidCredentialsException):
        await auth.authenticate(db, email="missing@example.com", image=b"img", content_type="image/png")
    assert face.calls == 0


async def test_login_rejects_distant_face():
    from app.exceptions.auth import InvalidCredentialsException

    auth, _user, db, _face = _services([[1.0, 0.0], [0.0, 1.0]])
    await auth.register(db, name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    with pytest.raises(InvalidCredentialsException):
        await auth.authenticate(db, email="ana@example.com", image=b"img", content_type="image/png")


async def test_login_rejects_non_finite_probe_embedding():
    from app.exceptions.face import FaceProcessingException

    auth, _user, db, _face = _services([[1.0, 0.0], [float("nan"), 0.0]])
    await auth.register(db, name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    with pytest.raises(FaceProcessingException, match="Não foi possível processar o rosto"):
        await auth.authenticate(db, email="ana@example.com", image=b"img", content_type="image/png")


async def test_update_replaces_embedding_and_keeps_same_email():
    auth, user, db, _face = _services([[1.0, 0.0], [0.0, 1.0]])
    created = await auth.register(db, name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    updated = await user.update_user(db, 
        created.id,
        name="Ana Souza",
        email="ana@example.com",
        image=b"img",
        content_type="image/png",
    )
    assert updated.name == "Ana Souza"
    assert updated.email == "ana@example.com"
    assert updated.embedding == [0.0, 1.0]
    assert updated.updated_at > updated.created_at


async def test_update_rejects_email_owned_by_someone_else():
    from app.exceptions.user import UserAlreadyExistsException

    auth, user, db, _face = _services([[1.0, 0.0], [0.0, 1.0]])
    await auth.register(db, name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    other = await auth.register(db, name="Bia", email="bia@example.com", image=b"img", content_type="image/png")
    with pytest.raises(UserAlreadyExistsException):
        await user.update_user(db, other.id, email="ana@example.com")


async def test_update_requires_a_field():
    from app.exceptions.user import InvalidFieldException

    auth, user, db, _face = _services([[1.0, 0.0]])
    created = await auth.register(db, name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    with pytest.raises(InvalidFieldException, match="Informe ao menos um campo"):
        await user.update_user(db, created.id)


async def test_list_is_newest_first_and_delete_removes_user():
    from app.exceptions.user import UserNotFoundException

    auth, user, db, _face = _services([[1.0, 0.0], [0.0, 1.0]])
    first = await auth.register(db, name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    second = await auth.register(db, name="Bia", email="bia@example.com", image=b"img", content_type="image/png")
    page, total = await user.list_users(db, skip=0, limit=1)
    assert total == 2
    assert [user_item.id for user_item in page] == [second.id]
    await user.delete_user(db, first.id)
    with pytest.raises(UserNotFoundException):
        await user.get_user(db, first.id)


async def test_face_extraction_runs_off_the_event_loop():
    from app.services.auth import AuthService
    from app.services.user import UserService

    threads: list[int] = []

    class ThreadFace:
        def extract_embedding(self, content: bytes, content_type: str | None) -> list[float]:
            threads.append(threading.get_ident())
            return [1.0, 0.0]

    db = open_test_session()
    face = ThreadFace()
    clock = lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    auth = AuthService(face, distance_threshold=0.30, clock=clock)
    user_service = UserService(face, clock=clock)
    await auth.register(db, name="Ana", email="ana@example.com", image=b"x", content_type="image/png")
    await auth.authenticate(db, email="ana@example.com", image=b"x", content_type="image/png")
    user = db.scalar(select(User).where(User.email == "ana@example.com"))
    await user_service.update_user(db, user.id, image=b"x", content_type="image/png")
    assert len(threads) == 3
    assert threading.get_ident() not in threads
