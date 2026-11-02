from pathlib import Path

import pytest
from pydantic import ValidationError

PG_URL = "postgresql+psycopg://facecheck:facecheck@localhost:5432/facecheck"


def test_settings_use_facecheck_defaults():
    from app.config import Settings

    settings = Settings(
        _env_file=None,
        database_url=PG_URL,
        jwt_secret_key="test-secret-key-for-facecheck-0123456789",
    )
    assert settings.database_url == PG_URL
    assert settings.jwt_algorithm == "HS256"
    assert settings.access_token_expire_minutes == 60
    assert settings.face_model_name == "Facenet512"
    assert settings.face_detector_backend == "opencv"
    assert settings.face_distance_metric == "cosine"
    assert settings.face_distance_threshold == 0.30


@pytest.mark.parametrize("url", ["  ", "mysql://facecheck:facecheck@localhost:3306/facecheck"])
def test_settings_reject_non_postgres_url(url):
    from app.config import Settings

    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            database_url=url,
            jwt_secret_key="test-secret-key-for-facecheck-0123456789",
        )
    assert "DATABASE_URL" in str(exc.value)


def test_settings_reject_blank_jwt_secret():
    from app.config import Settings

    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            database_url=PG_URL,
            jwt_secret_key="",
        )
    assert "JWT_SECRET_KEY" in str(exc.value)


def test_settings_reject_non_cosine_metric():
    from app.config import Settings

    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            database_url=PG_URL,
            jwt_secret_key="test-secret-key-for-facecheck-0123456789",
            face_distance_metric="euclidean",
        )
    assert "FACE_DISTANCE_METRIC" in str(exc.value)


def test_load_settings_names_missing_variables(monkeypatch, tmp_path: Path):
    from app.config import load_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        load_settings()
    message = str(exc.value)
    assert "DATABASE_URL" in message
    assert "JWT_SECRET_KEY" in message


def test_load_settings_names_non_postgres_url(monkeypatch, tmp_path: Path):
    from app.config import load_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "mysql://facecheck:facecheck@localhost:3306/facecheck")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-for-facecheck-0123456789")
    monkeypatch.setenv("FACE_DISTANCE_METRIC", "cosine")
    with pytest.raises(SystemExit) as exc:
        load_settings()
    assert "DATABASE_URL" in str(exc.value)


def test_load_settings_names_blank_jwt_secret(monkeypatch, tmp_path: Path):
    from app.config import load_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", PG_URL)
    monkeypatch.setenv("JWT_SECRET_KEY", "")
    monkeypatch.setenv("FACE_DISTANCE_METRIC", "cosine")
    with pytest.raises(SystemExit) as exc:
        load_settings()
    assert "JWT_SECRET_KEY" in str(exc.value)


def test_load_settings_names_non_cosine_metric(monkeypatch, tmp_path: Path):
    from app.config import load_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", PG_URL)
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-for-facecheck-0123456789")
    monkeypatch.setenv("FACE_DISTANCE_METRIC", "euclidean")
    with pytest.raises(SystemExit) as exc:
        load_settings()
    assert "FACE_DISTANCE_METRIC" in str(exc.value)


def test_database_engine_uses_postgres_test_database():
    import app.database as database

    url = database.engine.url
    assert url.get_backend_name() == "postgresql"
    assert url.database == "facecheck_test"
