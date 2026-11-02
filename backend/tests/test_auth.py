import jwt

from app.services.face import MAX_IMAGE_BYTES
from tests.images import PNG_1X1

TEST_SECRET = "test-secret-key-for-facecheck-0123456789"


def _files():
    return {"image": ("face.png", PNG_1X1, "image/png")}


def test_register_hides_embedding(client):
    client.app.state.face_script.append([1.0, 0.0])
    response = client.post(
        "/api/auth/register",
        data={"name": "Ana", "email": "Ana@Example.com"},
        files=_files(),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "Ana@Example.com"
    assert body["role"] == "GERAL PERMISSIONS"
    assert "embedding" not in body
    assert set(body) == {"id", "name", "email", "role", "created_at", "updated_at"}


def test_register_duplicate_email_returns_409(client):
    client.app.state.face_script.extend([[1.0, 0.0], [1.0, 0.0]])
    payload = {"name": "Ana", "email": "ana@example.com"}
    assert client.post("/api/auth/register", data=payload, files=_files()).status_code == 201
    response = client.post("/api/auth/register", data=payload, files=_files())
    assert response.status_code == 409
    assert response.json()["detail"] == "E-mail já cadastrado"


def test_register_without_one_face_returns_400(client):
    client.app.state.face_script.append(ValueError("Face could not be detected"))
    response = client.post(
        "/api/auth/register",
        data={"name": "Ana", "email": "ana@example.com"},
        files=_files(),
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Nenhum rosto encontrado"


def test_register_missing_image_returns_400(client):
    response = client.post(
        "/api/auth/register",
        data={"name": "Ana", "email": "ana@example.com"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Arquivo de imagem ausente"


def test_login_returns_bearer_token(client):
    client.app.state.face_script.extend([[1.0, 0.0], [1.0, 0.0]])
    registered = client.post(
        "/api/auth/register",
        data={"name": "Ana", "email": "ana@example.com"},
        files=_files(),
    )
    response = client.post(
        "/api/auth/login",
        data={"email": "ana@example.com"},
        files=_files(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str)
    assert body["access_token"]
    payload = jwt.decode(body["access_token"], TEST_SECRET, algorithms=["HS256"])
    assert payload["sub"] == registered.json()["id"]
    assert payload["role"] == "GERAL PERMISSIONS"


def test_login_unknown_email_and_bad_face_share_message(client):
    client.app.state.face_script.extend([[1.0, 0.0], [0.0, 1.0]])
    client.post(
        "/api/auth/register",
        data={"name": "Ana", "email": "ana@example.com"},
        files=_files(),
    )
    unknown = client.post(
        "/api/auth/login",
        data={"email": "missing@example.com"},
        files=_files(),
    )
    mismatch = client.post(
        "/api/auth/login",
        data={"email": "ana@example.com"},
        files=_files(),
    )
    assert unknown.status_code == 401
    assert mismatch.status_code == 401
    assert unknown.json()["detail"] == "Credenciais inválidas"
    assert mismatch.json()["detail"] == "Credenciais inválidas"


def test_register_oversized_upload_returns_400_without_calling_deepface(client):
    calls = []
    client.app.state.user_service.face_service._represent = lambda *a, **k: calls.append(1)
    oversized = b"a" * (MAX_IMAGE_BYTES + 1)
    response = client.post(
        "/api/auth/register",
        data={"name": "Ana", "email": "ana@example.com"},
        files={"image": ("face.png", oversized, "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Imagem maior que 5 MB"
    assert calls == []
