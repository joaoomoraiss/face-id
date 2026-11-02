# FaceCheck Facial Auth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI app that registers and logs users in with a face photo, stores the DeepFace embedding in MongoDB, and protects user routes with a JWT and roles.

**Architecture:** DeepFace runs only inside `FaceService`. `UserService` applies the account rules and compares one stored embedding with cosine distance. HTTP routes translate domain errors. Tests inject a fake representer and an in-memory repository, so they never download model weights or open MongoDB. The real Beanie repository is wired in the app lifespan.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, Motor, Beanie, DeepFace, PyJWT, Pydantic Settings, python-multipart, Pillow, NumPy, pytest, pytest-asyncio.

## Global Constraints

- Login bem-sucedido devolve um JWT. Rotas de usuário exigem `Authorization: Bearer`.
- Cadastro público e `POST /users` recebem nome, e-mail e uma foto. Os dois chamam o mesmo serviço e gravam `role=user`.
- A API não altera `role`. Promoção a admin é manual no MongoDB.
- `user` só usa `GET /users/me` e `PATCH /users/me`. Qualquer rota de admin responde `403` com `Permissão insuficiente`, inclusive quando o id é o próprio.
- `admin` lista, lê, cria, atualiza e apaga qualquer usuário, inclusive a si mesmo.
- Login é 1:1: e-mail mais foto. E-mail ausente não chama DeepFace. E-mail ausente e rosto diferente respondem `401` com `Credenciais inválidas`.
- Permissão lê `role` do documento recarregado pelo `sub`. Token de id apagado responde `401` com `Não autenticado`.
- `MONGODB_URI` ou `JWT_SECRET_KEY` vazio impede a subida. A mensagem cita o nome da variável.
- `FACE_DISTANCE_METRIC` diferente de `cosine` impede a subida.
- Defaults: database `facecheck`, algoritmo `HS256`, expiração `60` minutos, modelo `Facenet512`, detector `opencv`, limiar `0.30`. Distância menor ou igual ao limiar autentica.
- Foto no campo `image`, multipart. JPEG, PNG ou WEBP. Máximo 5 MB. Mensagens: `Arquivo de imagem ausente`, `Tipo de imagem inválido`, `Imagem maior que 5 MB`, `Nenhum rosto encontrado`, `Mais de um rosto encontrado`, `Não foi possível processar o rosto`.
- E-mail duplicado: `409` `E-mail já cadastrado`. Id admin inexistente: `404` `Usuário não encontrado`.
- Resposta pública: `id`, `name`, `email`, `role`, `created_at`, `updated_at`. Nunca inclui `embedding`.
- E-mail guardado em minúsculas. Nome sem espaços nas pontas, de 1 a 120 caracteres.
- Lista: `skip` padrão 0, `limit` padrão 50 e máximo 100, ordem `created_at` decrescente, corpo `items` e `total`.
- `GET /health` fica na raiz. `200` `{"status": "ok"}` com ping. `503` quando o ping falha.
- Prefixo `/api/v1`. Testes não usam MongoDB real nem pesos do DeepFace.
- `FACE_MODEL_NAME` não pode mudar depois que existirem usuários.

---

## File structure

- `app/core/config.py` — lê o `.env` e recusa configuração inválida.
- `app/core/security.py` — cria e decodifica o JWT.
- `app/services/face.py` — valida a imagem e chama DeepFace.
- `app/models/user.py` — usuário de domínio, erros e protocolo do repositório. A tarefa 7 acrescenta o documento Beanie.
- `app/services/users.py` — cadastro, login, atualização, lista e exclusão.
- `app/schemas/auth.py` — resposta do token.
- `app/schemas/user.py` — usuário público e lista.
- `app/api/deps.py` — usuário atual, admin e tradução dos erros.
- `app/api/v1/endpoints/auth.py` — register e login.
- `app/api/v1/endpoints/users.py` — perfil e CRUD admin.
- `app/api/v1/router.py` — junta os routers.
- `app/db/mongodb.py` — cliente, repositório Mongo e abertura do Beanie.
- `app/main.py` — fábrica da aplicação, lifespan e health.
- `tests/fakes.py` — repositório em memória.
- `tests/images.py` — PNG de 1 pixel.
- `tests/conftest.py` — cliente HTTP de teste.
- `tests/test_config.py`, `tests/test_face.py`, `tests/test_security.py`, `tests/test_user_service.py`, `tests/test_auth.py`, `tests/test_users.py`, `tests/test_mongo.py` — cobertura pedida na spec.

Comandos rodam na raiz do projeto, no PowerShell. O interpretador dos testes é `.venv\Scripts\python`.

### Task 1: Settings and project skeleton

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/core/__init__.py`
- Create: `app/core/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: none.
- Produces: `Settings` with fields `mongodb_uri`, `mongodb_db_name`, `jwt_secret_key`, `jwt_algorithm`, `access_token_expire_minutes`, `face_model_name`, `face_detector_backend`, `face_distance_metric`, `face_distance_threshold`. Produces `load_settings() -> Settings`, which raises `SystemExit` whose text contains `MONGODB_URI`, `JWT_SECRET_KEY`, or `FACE_DISTANCE_METRIC`.

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` empty and `tests/test_config.py`:

```python
from pathlib import Path

import pytest
from pydantic import ValidationError


def test_settings_use_facecheck_defaults():
    from app.core.config import Settings

    settings = Settings(
        _env_file=None,
        mongodb_uri="mongodb://localhost:27017",
        jwt_secret_key="secret",
    )
    assert settings.mongodb_db_name == "facecheck"
    assert settings.jwt_algorithm == "HS256"
    assert settings.access_token_expire_minutes == 60
    assert settings.face_model_name == "Facenet512"
    assert settings.face_detector_backend == "opencv"
    assert settings.face_distance_metric == "cosine"
    assert settings.face_distance_threshold == 0.30


def test_settings_reject_blank_mongodb_uri():
    from app.core.config import Settings

    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, mongodb_uri="  ", jwt_secret_key="secret")
    assert "MONGODB_URI" in str(exc.value)


def test_settings_reject_blank_jwt_secret():
    from app.core.config import Settings

    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            mongodb_uri="mongodb://localhost:27017",
            jwt_secret_key="",
        )
    assert "JWT_SECRET_KEY" in str(exc.value)


def test_settings_reject_non_cosine_metric():
    from app.core.config import Settings

    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            mongodb_uri="mongodb://localhost:27017",
            jwt_secret_key="secret",
            face_distance_metric="euclidean",
        )
    assert "FACE_DISTANCE_METRIC" in str(exc.value)


def test_load_settings_names_missing_variables(monkeypatch, tmp_path: Path):
    from app.core.config import load_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        load_settings()
    message = str(exc.value)
    assert "MONGODB_URI" in message
    assert "JWT_SECRET_KEY" in message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_config.py -v`

Expected: FAIL with `ModuleNotFoundError` or `No module named 'app'`. If `.venv` does not exist yet, the same command without the prefix also fails; create the venv in step 3 before re-running.

- [ ] **Step 3: Write minimal implementation**

Create empty `app/__init__.py` and `app/core/__init__.py`.

Create `requirements.txt`:

```text
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
motor>=3.6.0
beanie>=1.27.0
pydantic-settings>=2.6.0
python-multipart>=0.0.17
PyJWT>=2.10.0
deepface>=0.0.93
numpy>=1.26.0
Pillow>=10.4.0
email-validator>=2.2.0
pytest>=8.3.0
pytest-asyncio>=0.24.0
httpx>=0.27.0
```

Create `pytest.ini`:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

Create `.gitignore`:

```text
.env
.venv/
__pycache__/
.pytest_cache/
*.pyc
.deepface/
```

Create `.env.example`:

```text
MONGODB_URI=
MONGODB_DB_NAME=facecheck
JWT_SECRET_KEY=
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
FACE_MODEL_NAME=Facenet512
FACE_DETECTOR_BACKEND=opencv
FACE_DISTANCE_METRIC=cosine
FACE_DISTANCE_THRESHOLD=0.30
```

Create `app/core/config.py`:

```python
from pydantic import ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_NAMES = {
    "mongodb_uri": "MONGODB_URI",
    "jwt_secret_key": "JWT_SECRET_KEY",
    "face_distance_metric": "FACE_DISTANCE_METRIC",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    mongodb_uri: str
    mongodb_db_name: str = "facecheck"
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    face_model_name: str = "Facenet512"
    face_detector_backend: str = "opencv"
    face_distance_metric: str = "cosine"
    face_distance_threshold: float = 0.30

    @model_validator(mode="after")
    def reject_invalid_values(self) -> "Settings":
        if not self.mongodb_uri.strip():
            raise ValueError("MONGODB_URI must not be empty")
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
            field = str(loc[0]) if loc else "config"
            labels.append(_ENV_NAMES.get(field, field))
        unique = list(dict.fromkeys(labels))
        raise SystemExit("Configuração inválida: " + ", ".join(unique)) from exc
```

Install dependencies. DeepFace puxa TensorFlow; a instalação pode levar vários minutos:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_config.py -v`

Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

If `git status` says this is not a repository, run `git init` first.

```powershell
git add requirements.txt pytest.ini .gitignore .env.example app/__init__.py app/core/__init__.py app/core/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: load settings from the environment"
```

### Task 2: Face embedding extraction

**Files:**
- Create: `app/services/__init__.py`
- Create: `app/services/face.py`
- Create: `tests/images.py`
- Create: `tests/test_face.py`

**Interfaces:**
- Consumes: nothing from task 1 at runtime.
- Produces: `FaceError(Exception)`. `Embedder.extract_embedding(content: bytes, content_type: str | None) -> list[float]`. `FaceService(model_name: str, detector_backend: str, represent: RepresentFn | None = None)`. `RepresentFn` is called as `represent(image, model_name=..., detector_backend=...)` and returns `list[dict]`. `deepface_represent(image, *, model_name: str, detector_backend: str) -> list[dict]` imports DeepFace only when called. `tests/images.py` produces `PNG_1X1: bytes`.

- [ ] **Step 1: Write the failing test**

Create `tests/images.py`:

```python
import base64

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
```

Create `tests/test_face.py`:

```python
import pytest

from tests.images import PNG_1X1


def _service(result):
    from app.services.face import FaceService

    def represent(image, *, model_name, detector_backend):
        if isinstance(result, Exception):
            raise result
        return result

    return FaceService(
        model_name="Facenet512",
        detector_backend="opencv",
        represent=represent,
    )


def test_extracts_one_embedding():
    service = _service([{"embedding": [1, 0]}])
    assert service.extract_embedding(PNG_1X1, "image/png") == [1.0, 0.0]


def test_rejects_empty_file():
    from app.services.face import FaceError

    with pytest.raises(FaceError, match="Arquivo de imagem ausente"):
        _service([]).extract_embedding(b"", "image/png")


def test_rejects_invalid_type():
    from app.services.face import FaceError

    with pytest.raises(FaceError, match="Tipo de imagem inválido"):
        _service([]).extract_embedding(PNG_1X1, "text/plain")


def test_rejects_file_over_5mb():
    from app.services.face import FaceError

    content = b"a" * (5 * 1024 * 1024 + 1)
    with pytest.raises(FaceError, match="Imagem maior que 5 MB"):
        _service([]).extract_embedding(content, "image/jpeg")


def test_rejects_missing_face():
    from app.services.face import FaceError

    service = _service(ValueError("Face could not be detected"))
    with pytest.raises(FaceError, match="Nenhum rosto encontrado"):
        service.extract_embedding(PNG_1X1, "image/jpeg")


def test_rejects_multiple_faces():
    from app.services.face import FaceError

    service = _service([{"embedding": [1.0]}, {"embedding": [0.0]}])
    with pytest.raises(FaceError, match="Mais de um rosto encontrado"):
        service.extract_embedding(PNG_1X1, "image/webp")


def test_rejects_undecodable_image():
    from app.services.face import FaceError

    with pytest.raises(FaceError, match="Não foi possível processar o rosto"):
        _service([]).extract_embedding(b"not-an-image", "image/png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_face.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.services'`.

- [ ] **Step 3: Write minimal implementation**

Create empty `app/services/__init__.py`.

Create `app/services/face.py`:

```python
from io import BytesIO
from typing import Callable, Protocol

import numpy as np
from PIL import Image

MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


class FaceError(Exception):
    pass


class Embedder(Protocol):
    def extract_embedding(self, content: bytes, content_type: str | None) -> list[float]:
        pass


RepresentFn = Callable[..., list[dict]]


def deepface_represent(image: np.ndarray, *, model_name: str, detector_backend: str) -> list[dict]:
    from deepface import DeepFace

    return DeepFace.represent(
        img_path=image,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=True,
        align=True,
    )


class FaceService:
    def __init__(
        self,
        *,
        model_name: str,
        detector_backend: str,
        represent: RepresentFn | None = None,
    ) -> None:
        self.model_name = model_name
        self.detector_backend = detector_backend
        self._represent = represent or deepface_represent

    def extract_embedding(self, content: bytes, content_type: str | None) -> list[float]:
        if not content:
            raise FaceError("Arquivo de imagem ausente")
        media_type = (content_type or "").split(";", 1)[0].strip().lower()
        if media_type not in ALLOWED_CONTENT_TYPES:
            raise FaceError("Tipo de imagem inválido")
        if len(content) > MAX_IMAGE_BYTES:
            raise FaceError("Imagem maior que 5 MB")
        image = _to_bgr(content)
        try:
            faces = self._represent(
                image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
            )
        except FaceError:
            raise
        except Exception as exc:
            text = str(exc).lower()
            if "face could not be detected" in text or "no face" in text:
                raise FaceError("Nenhum rosto encontrado") from exc
            raise FaceError("Não foi possível processar o rosto") from exc
        if not faces:
            raise FaceError("Nenhum rosto encontrado")
        if len(faces) > 1:
            raise FaceError("Mais de um rosto encontrado")
        embedding = faces[0].get("embedding")
        if not embedding:
            raise FaceError("Não foi possível processar o rosto")
        return [float(value) for value in embedding]


def _to_bgr(content: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(content)) as img:
            rgb = np.array(img.convert("RGB"))
    except Exception as exc:
        raise FaceError("Não foi possível processar o rosto") from exc
    return rgb[:, :, ::-1].copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_face.py -v`

Expected: PASS, 7 tests. DeepFace is not imported.

- [ ] **Step 5: Commit**

```powershell
git add app/services/__init__.py app/services/face.py tests/images.py tests/test_face.py
git commit -m "feat: extract a single face embedding"
```

### Task 3: JWT access tokens

**Files:**
- Create: `app/core/security.py`
- Create: `tests/test_security.py`

**Interfaces:**
- Consumes: none.
- Produces: `create_access_token(*, user_id: str, role: str, secret: str, algorithm: str, expires_minutes: int, now: datetime | None = None) -> str`. `decode_access_token(token: str, secret: str, algorithm: str) -> dict` with `sub`, `role`, and `exp`. Invalid or expired tokens raise `jwt.PyJWTError`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_security.py`:

```python
from datetime import datetime, timedelta, timezone

import jwt
import pytest


def test_token_round_trip_contains_sub_and_role():
    from app.core.security import create_access_token, decode_access_token

    now = datetime(2026, 10, 22, tzinfo=timezone.utc)
    token = create_access_token(
        user_id="abc123",
        role="user",
        secret="secret",
        algorithm="HS256",
        expires_minutes=60,
        now=now,
    )
    payload = decode_access_token(token, "secret", "HS256")
    assert payload["sub"] == "abc123"
    assert payload["role"] == "user"
    assert payload["exp"] == int((now + timedelta(minutes=60)).timestamp())


def test_rejects_expired_token():
    from app.core.security import create_access_token, decode_access_token

    token = create_access_token(
        user_id="abc123",
        role="admin",
        secret="secret",
        algorithm="HS256",
        expires_minutes=1,
        now=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(token, "secret", "HS256")


def test_rejects_wrong_secret():
    from app.core.security import create_access_token, decode_access_token

    token = create_access_token(
        user_id="abc123",
        role="user",
        secret="secret",
        algorithm="HS256",
        expires_minutes=60,
    )
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(token, "other", "HS256")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_security.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.security'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/core/security.py`:

```python
from datetime import datetime, timedelta, timezone

import jwt


def create_access_token(
    *,
    user_id: str,
    role: str,
    secret: str,
    algorithm: str,
    expires_minutes: int,
    now: datetime | None = None,
) -> str:
    issued = now or datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "role": role,
        "exp": issued + timedelta(minutes=expires_minutes),
    }
    token = jwt.encode(payload, secret, algorithm=algorithm)
    if isinstance(token, bytes):
        return token.decode("utf-8")
    return token


def decode_access_token(token: str, secret: str, algorithm: str) -> dict:
    return jwt.decode(token, secret, algorithms=[algorithm])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_security.py -v`

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```powershell
git add app/core/security.py tests/test_security.py
git commit -m "feat: issue and decode access tokens"
```

### Task 4: User service

**Files:**
- Create: `app/models/__init__.py`
- Create: `app/models/user.py`
- Create: `app/services/users.py`
- Create: `tests/fakes.py`
- Create: `tests/test_user_service.py`

**Interfaces:**
- Consumes: `Embedder.extract_embedding(content: bytes, content_type: str | None) -> list[float]` and `FaceError`.
- Produces:
  - `User` frozen dataclass: `id: str`, `name: str`, `email: str`, `role: str`, `embedding: list[float]`, `created_at: datetime`, `updated_at: datetime`.
  - `EmailAlreadyExists`, `InvalidCredentials`, `UserNotFound`.
  - `UserRepository` protocol: `get_by_email(email: str) -> User | None`, `get_by_id(user_id: str) -> User | None`, `add(*, name: str, email: str, role: str, embedding: list[float], created_at: datetime, updated_at: datetime) -> User`, `save(user: User) -> User`, `remove(user_id: str) -> bool`, `list_page(skip: int, limit: int) -> tuple[list[User], int]`.
  - `UserService(repository, face_service, *, distance_threshold: float, clock: Callable[[], datetime] | None = None)`.
  - `register(*, name: str, email: str, image: bytes, content_type: str | None) -> User`.
  - `authenticate(*, email: str, image: bytes, content_type: str | None) -> User`.
  - `update_user(user_id: str, *, name: str | None = None, email: str | None = None, image: bytes | None = None, content_type: str | None = None) -> User`.
  - `get_user(user_id: str) -> User`, `delete_user(user_id: str) -> None`, `list_users(*, skip: int, limit: int) -> tuple[list[User], int]`.
  - `cosine_distance(left: list[float], right: list[float]) -> float`.
  - `InMemoryUserRepository.seed(user: User) -> None` for tests. `add` assigns `str(uuid4())`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_user_service.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from tests.fakes import InMemoryUserRepository


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


def _service(embeddings: list[list[float]], threshold: float = 0.30):
    from app.services.users import UserService

    face = ScriptedFace(embeddings)
    clock = Clock()
    repo = InMemoryUserRepository()
    service = UserService(
        repo,
        face,
        distance_threshold=threshold,
        clock=clock,
    )
    return service, repo, face


async def test_register_stores_user_role_and_normalized_email():
    service, repo, _face = _service([[1.0, 0.0]])
    user = await service.register(
        name="  Ana  ",
        email="Ana@Example.com",
        image=b"img",
        content_type="image/png",
    )
    assert user.role == "user"
    assert user.email == "ana@example.com"
    assert user.name == "Ana"
    assert user.embedding == [1.0, 0.0]
    assert user.created_at == user.updated_at
    stored = await repo.get_by_email("ana@example.com")
    assert stored is not None
    assert stored.id == user.id


async def test_register_rejects_duplicate_email():
    from app.models.user import EmailAlreadyExists

    service, _repo, _face = _service([[1.0, 0.0], [1.0, 0.0]])
    await service.register(name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    with pytest.raises(EmailAlreadyExists):
        await service.register(name="Ana 2", email="ANA@example.com", image=b"img", content_type="image/png")


async def test_register_rejects_blank_name():
    service, _repo, _face = _service([[1.0, 0.0]])
    with pytest.raises(ValueError, match="Nome inválido"):
        await service.register(name="   ", email="ana@example.com", image=b"img", content_type="image/png")


async def test_login_accepts_distance_equal_to_threshold():
    service, _repo, _face = _service([[1.0, 0.0], [0.7, 0.714142842854285]])
    await service.register(name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    user = await service.authenticate(email="ana@example.com", image=b"img", content_type="image/png")
    assert user.email == "ana@example.com"


async def test_login_rejects_unknown_email_without_calling_face():
    from app.models.user import InvalidCredentials

    service, _repo, face = _service([[1.0, 0.0]])
    with pytest.raises(InvalidCredentials):
        await service.authenticate(email="missing@example.com", image=b"img", content_type="image/png")
    assert face.calls == 0


async def test_login_rejects_distant_face():
    from app.models.user import InvalidCredentials

    service, _repo, _face = _service([[1.0, 0.0], [0.0, 1.0]])
    await service.register(name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    with pytest.raises(InvalidCredentials):
        await service.authenticate(email="ana@example.com", image=b"img", content_type="image/png")


async def test_update_replaces_embedding_and_keeps_same_email():
    service, _repo, _face = _service([[1.0, 0.0], [0.0, 1.0]])
    created = await service.register(name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    updated = await service.update_user(
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
    from app.models.user import EmailAlreadyExists

    service, _repo, _face = _service([[1.0, 0.0], [0.0, 1.0]])
    await service.register(name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    other = await service.register(name="Bia", email="bia@example.com", image=b"img", content_type="image/png")
    with pytest.raises(EmailAlreadyExists):
        await service.update_user(other.id, email="ana@example.com")


async def test_update_requires_a_field():
    service, _repo, _face = _service([[1.0, 0.0]])
    created = await service.register(name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    with pytest.raises(ValueError, match="Informe ao menos um campo"):
        await service.update_user(created.id)


async def test_list_is_newest_first_and_delete_removes_user():
    from app.models.user import UserNotFound

    service, _repo, _face = _service([[1.0, 0.0], [0.0, 1.0]])
    first = await service.register(name="Ana", email="ana@example.com", image=b"img", content_type="image/png")
    second = await service.register(name="Bia", email="bia@example.com", image=b"img", content_type="image/png")
    page, total = await service.list_users(skip=0, limit=1)
    assert total == 2
    assert [user.id for user in page] == [second.id]
    await service.delete_user(first.id)
    with pytest.raises(UserNotFound):
        await service.get_user(first.id)
```

The vector `[0.7, 0.714142842854285]` is the unit vector whose cosine distance to `[1, 0]` is 0.30. Login must accept that boundary.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_user_service.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'tests.fakes'`.

- [ ] **Step 3: Write minimal implementation**

Create `tests/fakes.py`:

```python
from uuid import uuid4

from app.models.user import User


class InMemoryUserRepository:
    def __init__(self) -> None:
        self.users: dict[str, User] = {}

    def seed(self, user: User) -> None:
        self.users[user.id] = user

    async def get_by_email(self, email: str) -> User | None:
        for user in self.users.values():
            if user.email == email:
                return user
        return None

    async def get_by_id(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    async def add(
        self,
        *,
        name: str,
        email: str,
        role: str,
        embedding: list[float],
        created_at,
        updated_at,
    ) -> User:
        user = User(
            id=str(uuid4()),
            name=name,
            email=email,
            role=role,
            embedding=list(embedding),
            created_at=created_at,
            updated_at=updated_at,
        )
        self.users[user.id] = user
        return user

    async def save(self, user: User) -> User:
        self.users[user.id] = user
        return user

    async def remove(self, user_id: str) -> bool:
        return self.users.pop(user_id, None) is not None

    async def list_page(self, skip: int, limit: int) -> tuple[list[User], int]:
        ordered = sorted(self.users.values(), key=lambda item: item.created_at, reverse=True)
        return ordered[skip : skip + limit], len(ordered)
```

Create empty `app/models/__init__.py`.

Create `app/models/user.py`:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


class EmailAlreadyExists(Exception):
    pass


class InvalidCredentials(Exception):
    pass


class UserNotFound(Exception):
    pass


@dataclass(frozen=True)
class User:
    id: str
    name: str
    email: str
    role: str
    embedding: list[float]
    created_at: datetime
    updated_at: datetime


class UserRepository(Protocol):
    async def get_by_email(self, email: str) -> User | None:
        pass

    async def get_by_id(self, user_id: str) -> User | None:
        pass

    async def add(
        self,
        *,
        name: str,
        email: str,
        role: str,
        embedding: list[float],
        created_at: datetime,
        updated_at: datetime,
    ) -> User:
        pass

    async def save(self, user: User) -> User:
        pass

    async def remove(self, user_id: str) -> bool:
        pass

    async def list_page(self, skip: int, limit: int) -> tuple[list[User], int]:
        pass
```

Create `app/services/users.py`:

```python
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
from pydantic import EmailStr, TypeAdapter, ValidationError

from app.models.user import EmailAlreadyExists, InvalidCredentials, User, UserNotFound, UserRepository
from app.services.face import Embedder, FaceError

_EMAIL = TypeAdapter(EmailStr)


def cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) == 0 or len(left) != len(right):
        raise FaceError("Não foi possível processar o rosto")
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        raise FaceError("Não foi possível processar o rosto")
    return float(1.0 - (float(np.dot(first, second)) / denominator))


def normalize_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 120:
        raise ValueError("Nome inválido")
    return cleaned


def normalize_email(value: str) -> str:
    try:
        return str(_EMAIL.validate_python(value.strip().lower()))
    except ValidationError as exc:
        raise ValueError("E-mail inválido") from exc


class UserService:
    def __init__(
        self,
        repository: UserRepository,
        face_service: Embedder,
        *,
        distance_threshold: float,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.face_service = face_service
        self.distance_threshold = distance_threshold
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def register(self, *, name: str, email: str, image: bytes, content_type: str | None) -> User:
        cleaned_name = normalize_name(name)
        cleaned_email = normalize_email(email)
        if await self.repository.get_by_email(cleaned_email) is not None:
            raise EmailAlreadyExists()
        embedding = self.face_service.extract_embedding(image, content_type)
        now = self._clock()
        return await self.repository.add(
            name=cleaned_name,
            email=cleaned_email,
            role="user",
            embedding=embedding,
            created_at=now,
            updated_at=now,
        )

    async def authenticate(self, *, email: str, image: bytes, content_type: str | None) -> User:
        cleaned_email = normalize_email(email)
        user = await self.repository.get_by_email(cleaned_email)
        if user is None:
            raise InvalidCredentials()
        probe = self.face_service.extract_embedding(image, content_type)
        distance = cosine_distance(user.embedding, probe)
        if distance > self.distance_threshold + 1e-9:
            raise InvalidCredentials()
        return user

    async def update_user(
        self,
        user_id: str,
        *,
        name: str | None = None,
        email: str | None = None,
        image: bytes | None = None,
        content_type: str | None = None,
    ) -> User:
        if name is None and email is None and image is None:
            raise ValueError("Informe ao menos um campo")
        current = await self.get_user(user_id)
        new_name = current.name if name is None else normalize_name(name)
        new_email = current.email if email is None else normalize_email(email)
        if new_email != current.email:
            owner = await self.repository.get_by_email(new_email)
            if owner is not None and owner.id != current.id:
                raise EmailAlreadyExists()
        embedding = current.embedding
        if image is not None:
            embedding = self.face_service.extract_embedding(image, content_type)
        updated = replace(
            current,
            name=new_name,
            email=new_email,
            embedding=list(embedding),
            updated_at=self._clock(),
        )
        return await self.repository.save(updated)

    async def get_user(self, user_id: str) -> User:
        user = await self.repository.get_by_id(user_id)
        if user is None:
            raise UserNotFound()
        return user

    async def delete_user(self, user_id: str) -> None:
        if not await self.repository.remove(user_id):
            raise UserNotFound()

    async def list_users(self, *, skip: int, limit: int) -> tuple[list[User], int]:
        return await self.repository.list_page(skip, limit)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_user_service.py -v`

Expected: PASS, 10 tests. The boundary vector's raw cosine distance is `0.30000000000000004`. The `1e-9` tolerance treats that value as equal to `0.30`. A distance of `1.0` still fails.

- [ ] **Step 5: Commit**

```powershell
git add app/models/__init__.py app/models/user.py app/services/users.py tests/fakes.py tests/test_user_service.py
git commit -m "feat: register and match users by face embedding"
```

### Task 5: Register and login routes

**Files:**
- Create: `app/schemas/__init__.py`
- Create: `app/schemas/auth.py`
- Create: `app/schemas/user.py`
- Create: `app/api/__init__.py`
- Create: `app/api/deps.py`
- Create: `app/api/v1/__init__.py`
- Create: `app/api/v1/endpoints/__init__.py`
- Create: `app/api/v1/endpoints/auth.py`
- Create: `app/api/v1/router.py`
- Create: `app/main.py`
- Create: `tests/conftest.py`
- Create: `tests/test_auth.py`

**Interfaces:**
- Consumes: `load_settings`, `Settings`, `create_access_token`, `decode_access_token`, `UserService.register`, `UserService.authenticate`, `User`, `FaceError`, `EmailAlreadyExists`, `InvalidCredentials`, `UserNotFound`, `PNG_1X1`, `InMemoryUserRepository`.
- Produces: `UserPublic.from_user(user: User) -> UserPublic`. `TokenResponse(access_token: str, token_type: str)`. `raise_domain(exc: Exception) -> None`. `get_current_user` returns `User` or raises `401` `Não autenticado`. `require_admin` raises `403` `Permissão insuficiente`. `create_app(*, settings: Settings | None = None, user_service: UserService | None = None) -> FastAPI`. When `user_service` is passed, `app.state.skip_database` is `True` and the lifespan does not open MongoDB. Module-level `app` calls `create_app()`. Routes: `POST /api/v1/auth/register` returns `201`; `POST /api/v1/auth/login` returns `200`.

- [ ] **Step 1: Write the failing test**

Create `tests/conftest.py`:

```python
import os

os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("FACE_DISTANCE_METRIC", "cosine")

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.face import FaceService
from app.services.users import UserService
from tests.fakes import InMemoryUserRepository

PNG_HEADER = "image/png"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        mongodb_uri="mongodb://localhost:27017",
        jwt_secret_key="test-secret",
        face_distance_threshold=0.30,
    )


@pytest.fixture
def repository() -> InMemoryUserRepository:
    return InMemoryUserRepository()


@pytest.fixture
def client(settings: Settings, repository: InMemoryUserRepository):
    pending: list[list[float] | Exception] = []

    def represent(image, *, model_name, detector_backend):
        item = pending.pop(0)
        if isinstance(item, Exception):
            raise item
        return [{"embedding": item}]

    face = FaceService(model_name="Facenet512", detector_backend="opencv", represent=represent)
    service = UserService(repository, face, distance_threshold=settings.face_distance_threshold)
    app = create_app(settings=settings, user_service=service)
    app.state.face_script = pending
    with TestClient(app) as test_client:
        yield test_client
```

Create `tests/test_auth.py`:

```python
from tests.images import PNG_1X1


def _files():
    return {"image": ("face.png", PNG_1X1, "image/png")}


def test_register_hides_embedding_and_lowercases_email(client):
    client.app.state.face_script.append([1.0, 0.0])
    response = client.post(
        "/api/v1/auth/register",
        data={"name": "Ana", "email": "Ana@Example.com"},
        files=_files(),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "ana@example.com"
    assert body["role"] == "user"
    assert "embedding" not in body
    assert set(body) == {"id", "name", "email", "role", "created_at", "updated_at"}


def test_register_duplicate_email_returns_409(client):
    client.app.state.face_script.extend([[1.0, 0.0], [1.0, 0.0]])
    payload = {"name": "Ana", "email": "ana@example.com"}
    assert client.post("/api/v1/auth/register", data=payload, files=_files()).status_code == 201
    response = client.post("/api/v1/auth/register", data=payload, files=_files())
    assert response.status_code == 409
    assert response.json()["detail"] == "E-mail já cadastrado"


def test_register_without_one_face_returns_400(client):
    client.app.state.face_script.append(ValueError("Face could not be detected"))
    response = client.post(
        "/api/v1/auth/register",
        data={"name": "Ana", "email": "ana@example.com"},
        files=_files(),
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Nenhum rosto encontrado"


def test_register_missing_image_returns_400(client):
    response = client.post(
        "/api/v1/auth/register",
        data={"name": "Ana", "email": "ana@example.com"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Arquivo de imagem ausente"


def test_login_returns_bearer_token(client):
    client.app.state.face_script.extend([[1.0, 0.0], [1.0, 0.0]])
    client.post(
        "/api/v1/auth/register",
        data={"name": "Ana", "email": "ana@example.com"},
        files=_files(),
    )
    response = client.post(
        "/api/v1/auth/login",
        data={"email": "ana@example.com"},
        files=_files(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str)
    assert body["access_token"]


def test_login_unknown_email_and_bad_face_share_message(client):
    client.app.state.face_script.extend([[1.0, 0.0], [0.0, 1.0]])
    client.post(
        "/api/v1/auth/register",
        data={"name": "Ana", "email": "ana@example.com"},
        files=_files(),
    )
    unknown = client.post(
        "/api/v1/auth/login",
        data={"email": "missing@example.com"},
        files=_files(),
    )
    mismatch = client.post(
        "/api/v1/auth/login",
        data={"email": "ana@example.com"},
        files=_files(),
    )
    assert unknown.status_code == 401
    assert mismatch.status_code == 401
    assert unknown.json()["detail"] == "Credenciais inválidas"
    assert mismatch.json()["detail"] == "Credenciais inválidas"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_auth.py -v`

Expected: FAIL with `ModuleNotFoundError` for `app.main` or `app.schemas`.

- [ ] **Step 3: Write minimal implementation**

Create empty `app/schemas/__init__.py`, `app/api/__init__.py`, `app/api/v1/__init__.py`, and `app/api/v1/endpoints/__init__.py`.

Create `app/schemas/auth.py`:

```python
from pydantic import BaseModel


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
```

Create `app/schemas/user.py`:

```python
from datetime import datetime

from pydantic import BaseModel

from app.models.user import User


class UserPublic(BaseModel):
    id: str
    name: str
    email: str
    role: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_user(cls, user: User) -> "UserPublic":
        return cls(
            id=user.id,
            name=user.name,
            email=user.email,
            role=user.role,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )
```

Create `app/api/deps.py`:

```python
import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token
from app.models.user import EmailAlreadyExists, InvalidCredentials, User, UserNotFound
from app.services.face import FaceError
from app.services.users import UserService

bearer_scheme = HTTPBearer(auto_error=False)


def raise_domain(exc: Exception) -> None:
    if isinstance(exc, FaceError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, InvalidCredentials):
        raise HTTPException(status_code=401, detail="Credenciais inválidas") from exc
    if isinstance(exc, UserNotFound):
        raise HTTPException(status_code=404, detail="Usuário não encontrado") from exc
    if isinstance(exc, EmailAlreadyExists):
        raise HTTPException(status_code=409, detail="E-mail já cadastrado") from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Não autenticado")
    settings = request.app.state.settings
    service: UserService = request.app.state.user_service
    try:
        payload = decode_access_token(
            credentials.credentials,
            settings.jwt_secret_key,
            settings.jwt_algorithm,
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Não autenticado") from exc
    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=401, detail="Não autenticado")
    try:
        return await service.get_user(user_id)
    except UserNotFound as exc:
        raise HTTPException(status_code=401, detail="Não autenticado") from exc


async def require_admin(current: User = Depends(get_current_user)) -> User:
    if current.role != "admin":
        raise HTTPException(status_code=403, detail="Permissão insuficiente")
    return current
```

Create `app/api/v1/endpoints/auth.py`:

```python
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.api.deps import raise_domain
from app.core.security import create_access_token
from app.schemas.auth import TokenResponse
from app.schemas.user import UserPublic
from app.services.users import UserService

router = APIRouter()


async def _read_required_image(image: UploadFile | None) -> tuple[bytes, str | None]:
    if image is None:
        raise HTTPException(status_code=400, detail="Arquivo de imagem ausente")
    content = await image.read()
    return content, image.content_type


@router.post("/register", response_model=UserPublic, status_code=201)
async def register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    image: UploadFile | None = File(None),
) -> UserPublic:
    content, content_type = await _read_required_image(image)
    service: UserService = request.app.state.user_service
    try:
        user = await service.register(
            name=name,
            email=email,
            image=content,
            content_type=content_type,
        )
    except Exception as exc:
        raise_domain(exc)
    return UserPublic.from_user(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    email: str = Form(...),
    image: UploadFile | None = File(None),
) -> TokenResponse:
    content, content_type = await _read_required_image(image)
    service: UserService = request.app.state.user_service
    settings = request.app.state.settings
    try:
        user = await service.authenticate(
            email=email,
            image=content,
            content_type=content_type,
        )
    except Exception as exc:
        raise_domain(exc)
    token = create_access_token(
        user_id=user.id,
        role=user.role,
        secret=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        expires_minutes=settings.access_token_expire_minutes,
    )
    return TokenResponse(access_token=token)
```

`raise_domain` either raises `HTTPException` or re-raises. The type checker may think `user` is unset; keep the call as written because every path raises. If the checker flags it, add `raise AssertionError("erro de domínio não traduzido")` immediately after `raise_domain(exc)`.

Create `app/api/v1/router.py`:

```python
from fastapi import APIRouter

from app.api.v1.endpoints import auth

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
```

Create `app/main.py`:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import Settings, load_settings
from app.services.users import UserService


@asynccontextmanager
async def lifespan(app: FastAPI):
    if app.state.skip_database:
        yield
        return
    raise RuntimeError("Banco não configurado")


def create_app(*, settings: Settings | None = None, user_service: UserService | None = None) -> FastAPI:
    resolved = settings or load_settings()
    app = FastAPI(title="FaceCheck", lifespan=lifespan)
    app.state.settings = resolved
    app.state.user_service = user_service
    app.state.skip_database = user_service is not None
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_auth.py tests/test_config.py -v`

Expected: PASS. `test_config.py` still passes with the environment values set by `conftest.py` because constructor arguments win over the environment.

- [ ] **Step 5: Commit**

```powershell
git add app/schemas app/api app/main.py tests/conftest.py tests/test_auth.py
git commit -m "feat: expose register and facial login"
```

### Task 6: Profile and admin user routes

**Files:**
- Modify: `app/schemas/user.py`
- Create: `app/api/v1/endpoints/users.py`
- Modify: `app/api/v1/router.py`
- Create: `tests/test_users.py`

**Interfaces:**
- Consumes: `get_current_user`, `require_admin`, `raise_domain`, `UserPublic.from_user`, `UserService.update_user`, `UserService.get_user`, `UserService.delete_user`, `UserService.list_users`, `UserService.register`, `create_access_token`, `InMemoryUserRepository.seed`, `User`.
- Produces: `UserList(items: list[UserPublic], total: int)`. Routes under `/api/v1/users`: `GET /me`, `PATCH /me`, `GET ""`, `POST ""`, `GET /{user_id}`, `PATCH /{user_id}`, `DELETE /{user_id}`. `DELETE` returns `204` with no body. `GET ""` uses query `skip: int = 0` (`ge=0`) and `limit: int = 50` (`ge=1`, `le=100`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_users.py`:

```python
from datetime import datetime, timezone

from app.core.security import create_access_token
from app.models.user import User
from tests.images import PNG_1X1


def _files():
    return {"image": ("face.png", PNG_1X1, "image/png")}


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _register(client, email: str, embedding: list[float]) -> dict:
    client.app.state.face_script.append(embedding)
    response = client.post(
        "/api/v1/auth/register",
        data={"name": "Ana", "email": email},
        files=_files(),
    )
    assert response.status_code == 201
    return response.json()


def _token(user_id: str, role: str = "user") -> str:
    return create_access_token(
        user_id=user_id,
        role=role,
        secret="test-secret",
        algorithm="HS256",
        expires_minutes=60,
    )


def test_user_updates_own_name_and_photo(client):
    created = _register(client, "ana@example.com", [1.0, 0.0])
    client.app.state.face_script.append([0.0, 1.0])
    response = client.patch(
        "/api/v1/users/me",
        data={"name": "Ana Souza"},
        files=_files(),
        headers=_auth(_token(created["id"])),
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Ana Souza"
    assert "embedding" not in response.json()


def test_user_is_forbidden_on_admin_routes(client):
    created = _register(client, "ana@example.com", [1.0, 0.0])
    headers = _auth(_token(created["id"], role="admin"))
    assert client.get("/api/v1/users", headers=headers).status_code == 403
    assert client.get(f"/api/v1/users/{created['id']}", headers=headers).status_code == 403
    denied = client.post(
        "/api/v1/users",
        data={"name": "Bia", "email": "bia@example.com"},
        files=_files(),
        headers=headers,
    )
    assert denied.status_code == 403
    assert denied.json()["detail"] == "Permissão insuficiente"
    assert client.patch(
        f"/api/v1/users/{created['id']}",
        data={"name": "Outra"},
        headers=headers,
    ).status_code == 403
    assert client.delete(f"/api/v1/users/{created['id']}", headers=headers).status_code == 403


def test_missing_token_is_unauthenticated(client):
    response = client.get("/api/v1/users/me")
    assert response.status_code == 401
    assert response.json()["detail"] == "Não autenticado"


def test_admin_lists_creates_reads_updates_and_deletes(client, repository):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    repository.seed(
        User(
            id="admin-1",
            name="Admin",
            email="admin@example.com",
            role="admin",
            embedding=[1.0, 0.0],
            created_at=now,
            updated_at=now,
        )
    )
    headers = _auth(_token("admin-1", role="user"))
    listed = client.get("/api/v1/users", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    client.app.state.face_script.append([0.2, 0.8])
    created = client.post(
        "/api/v1/users",
        data={"name": "Bia", "email": "bia@example.com"},
        files=_files(),
        headers=headers,
    )
    assert created.status_code == 201
    assert created.json()["role"] == "user"
    user_id = created.json()["id"]

    fetched = client.get(f"/api/v1/users/{user_id}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["email"] == "bia@example.com"

    renamed = client.patch(
        f"/api/v1/users/{user_id}",
        data={"name": "Bia Souza"},
        headers=headers,
    )
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Bia Souza"

    deleted = client.delete(f"/api/v1/users/{user_id}", headers=headers)
    assert deleted.status_code == 204
    assert deleted.content == b""
    missing = client.get(f"/api/v1/users/{user_id}", headers=headers)
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Usuário não encontrado"
    self_deleted = client.delete("/api/v1/users/admin-1", headers=headers)
    assert self_deleted.status_code == 204


def test_deleted_user_token_is_rejected(client):
    created = _register(client, "ana@example.com", [1.0, 0.0])
    headers = _auth(_token(created["id"]))
    repository_users = client.app.state.user_service.repository.users
    repository_users.pop(created["id"])
    response = client.get("/api/v1/users/me", headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"] == "Não autenticado"
```

The forbidden test sends `role="admin"` inside the JWT while the stored user remains `user`. Admin routes must still return `403`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_users.py -v`

Expected: FAIL with `404` because `/api/v1/users/me` is not registered.

- [ ] **Step 3: Write minimal implementation**

Append to `app/schemas/user.py`:

```python
class UserList(BaseModel):
    items: list[UserPublic]
    total: int
```

Create `app/api/v1/endpoints/users.py`:

```python
from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile

from app.api.deps import get_current_user, raise_domain, require_admin
from app.models.user import User
from app.schemas.user import UserList, UserPublic
from app.services.users import UserService

router = APIRouter()


async def _optional_image(image: UploadFile | None) -> tuple[bytes | None, str | None]:
    if image is None:
        return None, None
    return await image.read(), image.content_type


@router.get("/me", response_model=UserPublic)
async def read_me(current: User = Depends(get_current_user)) -> UserPublic:
    return UserPublic.from_user(current)


@router.patch("/me", response_model=UserPublic)
async def update_me(
    request: Request,
    name: str | None = Form(None),
    email: str | None = Form(None),
    image: UploadFile | None = File(None),
    current: User = Depends(get_current_user),
) -> UserPublic:
    content, content_type = await _optional_image(image)
    service: UserService = request.app.state.user_service
    try:
        updated = await service.update_user(
            current.id,
            name=name,
            email=email,
            image=content,
            content_type=content_type,
        )
    except Exception as exc:
        raise_domain(exc)
    return UserPublic.from_user(updated)


@router.get("", response_model=UserList)
async def list_users(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    _admin: User = Depends(require_admin),
) -> UserList:
    service: UserService = request.app.state.user_service
    items, total = await service.list_users(skip=skip, limit=limit)
    return UserList(items=[UserPublic.from_user(user) for user in items], total=total)


@router.post("", response_model=UserPublic, status_code=201)
async def create_user(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    image: UploadFile | None = File(None),
    _admin: User = Depends(require_admin),
) -> UserPublic:
    from app.api.v1.endpoints.auth import _read_required_image

    content, content_type = await _read_required_image(image)
    service: UserService = request.app.state.user_service
    try:
        user = await service.register(
            name=name,
            email=email,
            image=content,
            content_type=content_type,
        )
    except Exception as exc:
        raise_domain(exc)
    return UserPublic.from_user(user)


@router.get("/{user_id}", response_model=UserPublic)
async def read_user(
    user_id: str,
    request: Request,
    _admin: User = Depends(require_admin),
) -> UserPublic:
    service: UserService = request.app.state.user_service
    try:
        user = await service.get_user(user_id)
    except Exception as exc:
        raise_domain(exc)
    return UserPublic.from_user(user)


@router.patch("/{user_id}", response_model=UserPublic)
async def update_user(
    user_id: str,
    request: Request,
    name: str | None = Form(None),
    email: str | None = Form(None),
    image: UploadFile | None = File(None),
    _admin: User = Depends(require_admin),
) -> UserPublic:
    content, content_type = await _optional_image(image)
    service: UserService = request.app.state.user_service
    try:
        updated = await service.update_user(
            user_id,
            name=name,
            email=email,
            image=content,
            content_type=content_type,
        )
    except Exception as exc:
        raise_domain(exc)
    return UserPublic.from_user(updated)


@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: str,
    request: Request,
    _admin: User = Depends(require_admin),
) -> None:
    service: UserService = request.app.state.user_service
    try:
        await service.delete_user(user_id)
    except Exception as exc:
        raise_domain(exc)
```

Declare `/me` before `/{user_id}`. Replace `app/api/v1/router.py` with:

```python
from fastapi import APIRouter

from app.api.v1.endpoints import auth, users

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_auth.py tests/test_users.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/schemas/user.py app/api/v1/endpoints/users.py app/api/v1/router.py tests/test_users.py
git commit -m "feat: add profile and admin user routes"
```

### Task 7: MongoDB persistence, health, and README

**Files:**
- Modify: `app/models/user.py`
- Create: `app/db/__init__.py`
- Create: `app/db/mongodb.py`
- Modify: `app/main.py`
- Create: `tests/test_mongo.py`
- Create: `README.md`

**Interfaces:**
- Consumes: `User`, `UserRepository` method names, `EmailAlreadyExists`, `UserService`, `FaceService`, `Settings`, `create_app`.
- Produces: `UserDocument` Beanie model, collection name `users`, unique index on `email`. `MongoUserRepository` implements `UserRepository`. `get_by_id("not-an-id")` returns `None` without querying. Duplicate key on `add` raises `EmailAlreadyExists`. `open_database(settings: Settings) -> AsyncIOMotorClient` pings, calls `init_beanie`, and returns the client. Lifespan stores that client on `app.state.mongo_client` and builds `UserService` with `MongoUserRepository` and `FaceService` only when `skip_database` is false. `GET /health` returns `200` `{"status": "ok"}` or `503` `Banco indisponível`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mongo.py`:

```python
import pytest
from pymongo.errors import DuplicateKeyError

from app.core.config import Settings
from app.main import create_app
from app.services.users import UserService
from tests.fakes import InMemoryUserRepository


class FakeAdmin:
    def __init__(self, ok: bool) -> None:
        self.ok = ok
        self.admin = self

    async def command(self, _name: str) -> dict:
        if not self.ok:
            raise ConnectionError("down")
        return {"ok": 1}


def test_invalid_object_id_does_not_query():
    from app.db.mongodb import MongoUserRepository

    repo = MongoUserRepository()
    assert _run(repo.get_by_id("not-an-id")) is None


def test_duplicate_key_becomes_email_already_exists():
    from app.db.mongodb import translate_write_error
    from app.models.user import EmailAlreadyExists

    with pytest.raises(EmailAlreadyExists):
        raise translate_write_error(DuplicateKeyError("dup"))


def test_health_ok_and_unavailable():
    from fastapi.testclient import TestClient

    settings = Settings(
        _env_file=None,
        mongodb_uri="mongodb://localhost:27017",
        jwt_secret_key="test-secret",
    )
    service = UserService(InMemoryUserRepository(), _UnusedFace(), distance_threshold=0.30)
    app = create_app(settings=settings, user_service=service)
    app.state.mongo_client = FakeAdmin(ok=True)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").json() == {"status": "ok"}
        client.app.state.mongo_client = FakeAdmin(ok=False)
        failed = client.get("/health")
        assert failed.status_code == 503
        assert failed.json()["detail"] == "Banco indisponível"


class _UnusedFace:
    def extract_embedding(self, content: bytes, content_type: str | None) -> list[float]:
        return [1.0]


def _run(awaitable):
    import asyncio

    return asyncio.run(awaitable)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_mongo.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.db'`.

- [ ] **Step 3: Write minimal implementation**

Create empty `app/db/__init__.py`.

Replace `app/models/user.py` with:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from beanie import Document, Indexed


class EmailAlreadyExists(Exception):
    pass


class InvalidCredentials(Exception):
    pass


class UserNotFound(Exception):
    pass


@dataclass(frozen=True)
class User:
    id: str
    name: str
    email: str
    role: str
    embedding: list[float]
    created_at: datetime
    updated_at: datetime


class UserRepository(Protocol):
    async def get_by_email(self, email: str) -> User | None:
        pass

    async def get_by_id(self, user_id: str) -> User | None:
        pass

    async def add(
        self,
        *,
        name: str,
        email: str,
        role: str,
        embedding: list[float],
        created_at: datetime,
        updated_at: datetime,
    ) -> User:
        pass

    async def save(self, user: User) -> User:
        pass

    async def remove(self, user_id: str) -> bool:
        pass

    async def list_page(self, skip: int, limit: int) -> tuple[list[User], int]:
        pass


class UserDocument(Document):
    name: str
    email: Indexed(str, unique=True)
    role: str
    embedding: list[float]
    created_at: datetime
    updated_at: datetime

    class Settings:
        name = "users"
```

Create `app/db/mongodb.py`:

```python
from beanie import PydanticObjectId, init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError

from app.core.config import Settings
from app.models.user import EmailAlreadyExists, User, UserDocument


def translate_write_error(exc: Exception) -> Exception:
    if isinstance(exc, DuplicateKeyError):
        return EmailAlreadyExists()
    return exc


def _parse_id(user_id: str) -> PydanticObjectId | None:
    try:
        return PydanticObjectId(user_id)
    except Exception:
        return None


def to_user(document: UserDocument) -> User:
    return User(
        id=str(document.id),
        name=document.name,
        email=document.email,
        role=document.role,
        embedding=list(document.embedding),
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


class MongoUserRepository:
    async def get_by_email(self, email: str) -> User | None:
        document = await UserDocument.find_one(UserDocument.email == email)
        if document is None:
            return None
        return to_user(document)

    async def get_by_id(self, user_id: str) -> User | None:
        parsed = _parse_id(user_id)
        if parsed is None:
            return None
        document = await UserDocument.get(parsed)
        if document is None:
            return None
        return to_user(document)

    async def add(
        self,
        *,
        name: str,
        email: str,
        role: str,
        embedding: list[float],
        created_at,
        updated_at,
    ) -> User:
        document = UserDocument(
            name=name,
            email=email,
            role=role,
            embedding=list(embedding),
            created_at=created_at,
            updated_at=updated_at,
        )
        try:
            await document.insert()
        except Exception as exc:
            raise translate_write_error(exc) from exc
        return to_user(document)

    async def save(self, user: User) -> User:
        parsed = _parse_id(user.id)
        if parsed is None:
            raise ValueError("id inválido")
        document = await UserDocument.get(parsed)
        if document is None:
            raise ValueError("id inválido")
        document.name = user.name
        document.email = user.email
        document.role = user.role
        document.embedding = list(user.embedding)
        document.updated_at = user.updated_at
        try:
            await document.save()
        except Exception as exc:
            raise translate_write_error(exc) from exc
        return to_user(document)

    async def remove(self, user_id: str) -> bool:
        parsed = _parse_id(user_id)
        if parsed is None:
            return False
        document = await UserDocument.get(parsed)
        if document is None:
            return False
        await document.delete()
        return True

    async def list_page(self, skip: int, limit: int) -> tuple[list[User], int]:
        total = await UserDocument.find_all().count()
        documents = (
            await UserDocument.find_all()
            .sort("-created_at")
            .skip(skip)
            .limit(limit)
            .to_list()
        )
        return [to_user(document) for document in documents], total


async def open_database(settings: Settings) -> AsyncIOMotorClient:
    client = AsyncIOMotorClient(settings.mongodb_uri)
    try:
        await client.admin.command("ping")
        await init_beanie(database=client[settings.mongodb_db_name], document_models=[UserDocument])
    except Exception:
        client.close()
        raise
    return client
```

Replace `app/main.py` with:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.api.v1.router import api_router
from app.core.config import Settings, load_settings
from app.db.mongodb import MongoUserRepository, open_database
from app.services.face import FaceService
from app.services.users import UserService


@asynccontextmanager
async def lifespan(app: FastAPI):
    if app.state.skip_database:
        yield
        return
    settings: Settings = app.state.settings
    client = await open_database(settings)
    app.state.mongo_client = client
    app.state.user_service = UserService(
        MongoUserRepository(),
        FaceService(
            model_name=settings.face_model_name,
            detector_backend=settings.face_detector_backend,
        ),
        distance_threshold=settings.face_distance_threshold,
    )
    try:
        yield
    finally:
        client.close()


def create_app(*, settings: Settings | None = None, user_service: UserService | None = None) -> FastAPI:
    resolved = settings or load_settings()
    app = FastAPI(title="FaceCheck", lifespan=lifespan)
    app.state.settings = resolved
    app.state.user_service = user_service
    app.state.skip_database = user_service is not None
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/health")
    async def health(request: Request) -> dict[str, str]:
        client = getattr(request.app.state, "mongo_client", None)
        if client is None:
            raise HTTPException(status_code=503, detail="Banco indisponível")
        try:
            await client.admin.command("ping")
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Banco indisponível") from exc
        return {"status": "ok"}

    return app


app = create_app()
```

Create `README.md`:

```markdown
# FaceCheck

API FastAPI com cadastro e login por reconhecimento facial. A senha não existe. O DeepFace gera um embedding na entrada e o MongoDB guarda esse vetor.

## Requisitos

- Python 3.11 ou mais novo
- MongoDB acessível por uma URI

## Configuração

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

Preencha no `.env`:

- `MONGODB_URI` — URI do MongoDB
- `JWT_SECRET_KEY` — segredo usado para assinar o token

Os outros valores já funcionam. `MONGODB_DB_NAME` padrão é `facecheck`. Sem `MONGODB_URI` ou `JWT_SECRET_KEY`, o processo recusa subir e mostra o nome da variável.

`FACE_MODEL_NAME` padrão é `Facenet512` e `FACE_DISTANCE_METRIC` precisa continuar `cosine`. Não troque o modelo depois que existirem usuários: embeddings de modelos diferentes não são comparáveis. O limiar padrão `0.30` é o corte de cosseno do Facenet512.

A primeira foto baixa os pesos do Facenet512. Isso exige rede e pode demorar.

## Executar

Na raiz do projeto:

```powershell
.venv\Scripts\python -m uvicorn app.main:app --reload
```

- `GET /health` responde `{"status": "ok"}` quando o MongoDB responde ao ping.
- Documentação interativa: `http://127.0.0.1:8000/docs`

## Rotas

Prefixo `/api/v1`. A foto vai no campo `image` em `multipart/form-data`.

- `POST /auth/register` — `name`, `email`, `image`. Cria `role=user`.
- `POST /auth/login` — `email`, `image`. Devolve `access_token` e `token_type`.
- `GET /users/me` e `PATCH /users/me` — perfil de quem está no token. O patch aceita `name`, `email` e `image`; envie ao menos um.
- `GET /users`, `POST /users`, `GET /users/{id}`, `PATCH /users/{id}`, `DELETE /users/{id}` — só `admin`.

As rotas protegidas usam o header `Authorization: Bearer <token>`.

## Promover um admin

O cadastro sempre grava `user`. No banco `facecheck`, coleção `users`:

```javascript
db.users.updateOne({ email: "voce@empresa.com" }, { $set: { role: "admin" } })
```

Saia e entre de novo. A permissão é lida do banco a cada request.

## Testes

```powershell
.venv\Scripts\python -m pytest -v
```

Os testes usam repositório em memória e um representer falso. Eles não abrem o MongoDB e não baixam o modelo.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest -v`

Expected: PASS for every test file. `tests/test_mongo.py` does not open a real MongoDB connection.

- [ ] **Step 5: Commit**

```powershell
git add app/models/user.py app/db/__init__.py app/db/mongodb.py app/main.py tests/test_mongo.py README.md
git commit -m "feat: persist users in MongoDB"
```
