import os

# testes sempre no banco facecheck_test, nunca no DATABASE_URL do .env (apagam as tabelas)
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://facecheck:facecheck@localhost:5432/facecheck_test",
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-facecheck-0123456789")
os.environ.setdefault("FACE_DISTANCE_METRIC", "cosine")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import Settings
from app.database import get_db
from app.main import app
from app.services.auth import AuthService
from app.services.face import FaceService
from app.services.user import UserService
from tests.db import open_test_session

PNG_HEADER = "image/png"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url=TEST_DATABASE_URL,
        jwt_secret_key="test-secret-key-for-facecheck-0123456789",
        face_distance_threshold=0.30,
    )


@pytest.fixture
def db() -> Session:
    session = open_test_session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(settings: Settings, db: Session):
    pending: list[list[float] | Exception] = []

    def represent(image, *, model_name, detector_backend):
        item = pending.pop(0)
        if isinstance(item, Exception):
            raise item
        return [{"embedding": item}]

    face = FaceService(model_name="Facenet512", detector_backend="opencv", represent=represent)
    auth = AuthService(face, distance_threshold=settings.face_distance_threshold)
    users = UserService(face)
    previous = (app.state.settings, app.state.auth_service, app.state.user_service)
    app.state.settings = settings
    app.state.auth_service = auth
    app.state.user_service = users

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.state.face_script = pending
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app.state.settings, app.state.auth_service, app.state.user_service = previous
