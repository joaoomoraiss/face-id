from pydantic import ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_NAMES = {
    "database_url": "DATABASE_URL",
    "jwt_secret_key": "JWT_SECRET_KEY",
    "face_distance_metric": "FACE_DISTANCE_METRIC",
}

_MSG_ENV_LABELS = ("DATABASE_URL", "JWT_SECRET_KEY", "FACE_DISTANCE_METRIC")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # postgres. ex: postgresql+psycopg://facecheck:facecheck@localhost:5432/facecheck
    database_url: str
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    face_model_name: str = "Facenet512"
    face_detector_backend: str = "opencv"
    face_distance_metric: str = "cosine"
    face_distance_threshold: float = 0.30

    @model_validator(mode="after")
    def reject_invalid_values(self) -> "Settings":
        if not self.database_url.strip().startswith("postgresql"):
            raise ValueError("DATABASE_URL must be a postgresql URL")
        if not self.jwt_secret_key.strip():
            raise ValueError("JWT_SECRET_KEY must not be empty")
        if self.face_distance_metric != "cosine":
            raise ValueError("FACE_DISTANCE_METRIC must be cosine")
        return self


def load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        labels: list[str] = []
        for error in exc.errors():
            loc = error.get("loc", ())
            if loc:
                field = str(loc[0])
                labels.append(_ENV_NAMES.get(field, field))
                continue
            msg = error.get("msg", "")
            label = next((name for name in _MSG_ENV_LABELS if name in msg), None)
            labels.append(label if label is not None else "config")
        unique = list(dict.fromkeys(labels))
        raise SystemExit("Configuração inválida: " + ", ".join(unique)) from exc
