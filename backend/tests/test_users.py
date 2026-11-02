import uuid
from datetime import datetime, timezone

import pytest

from app.models.user import Role, User
from app.security import create_access_token
from tests.images import PNG_1X1


def _files():
    return {"image": ("face.png", PNG_1X1, "image/png")}


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _register(client, email: str, embedding: list[float]) -> dict:
    client.app.state.face_script.append(embedding)
    response = client.post(
        "/api/auth/register",
        data={"name": "Ana", "email": email},
        files=_files(),
    )
    assert response.status_code == 201
    return response.json()


def _token(user_id: str, role: str = Role.GERAL.value) -> str:
    return create_access_token(
        user_id=user_id,
        role=role,
        secret="test-secret-key-for-facecheck-0123456789",
        algorithm="HS256",
        expires_minutes=60,
    )


def test_user_updates_own_name_and_photo(client):
    created = _register(client, "ana@example.com", [1.0, 0.0])
    client.app.state.face_script.append([0.0, 1.0])
    response = client.patch(
        "/api/users/me",
        data={"name": "Ana Souza"},
        files=_files(),
        headers=_auth(_token(created["id"])),
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Ana Souza"
    assert "embedding" not in response.json()


def test_user_is_forbidden_on_admin_routes(client):
    created = _register(client, "ana@example.com", [1.0, 0.0])
    # token diz ministry, mas o banco diz geral. vale o banco
    headers = _auth(_token(created["id"], role=Role.MINISTRY.value))
    assert client.get("/api/users", headers=headers).status_code == 403
    assert client.get(f"/api/users/{created['id']}", headers=headers).status_code == 403
    denied = client.post(
        "/api/users",
        data={"name": "Bia", "email": "bia@example.com"},
        files=_files(),
        headers=headers,
    )
    assert denied.status_code == 403
    assert denied.json()["detail"] == "Permissão insuficiente"
    assert client.patch(
        f"/api/users/{created['id']}",
        data={"name": "Outra"},
        headers=headers,
    ).status_code == 403
    assert client.delete(f"/api/users/{created['id']}", headers=headers).status_code == 403


def test_missing_token_is_unauthenticated(client):
    response = client.get("/api/users/me")
    assert response.status_code == 401
    assert response.json()["detail"] == "Não autenticado"


def _seed_admin(db, role: Role = Role.MINISTRY) -> str:
    admin_id = uuid.uuid4()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db.add(
        User(
            id=admin_id,
            name="Admin",
            email="admin@example.com",
            role=role,
            embedding=[1.0, 0.0],
            created_at=now,
            updated_at=now,
        )
    )
    db.commit()
    return str(admin_id)


@pytest.mark.parametrize("role", [Role.DIRECTOR, Role.MINISTRY])
def test_manager_lists_creates_reads_updates_and_deletes(client, db, role):
    admin_id = _seed_admin(db, role)
    # token diz geral, mas o banco diz director/ministry. vale o banco
    headers = _auth(_token(admin_id, role=Role.GERAL.value))
    listed = client.get("/api/users", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    client.app.state.face_script.append([0.2, 0.8])
    created = client.post(
        "/api/users",
        data={"name": "Bia", "email": "bia@example.com"},
        files=_files(),
        headers=headers,
    )
    assert created.status_code == 201
    assert created.json()["role"] == Role.GERAL.value
    user_id = created.json()["id"]

    fetched = client.get(f"/api/users/{user_id}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["email"] == "bia@example.com"

    renamed = client.patch(
        f"/api/users/{user_id}",
        data={"name": "Bia Souza"},
        headers=headers,
    )
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Bia Souza"

    deleted = client.delete(f"/api/users/{user_id}", headers=headers)
    assert deleted.status_code == 204
    assert deleted.content == b""
    missing = client.get(f"/api/users/{user_id}", headers=headers)
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Usuário não encontrado"
    self_deleted = client.delete(f"/api/users/{admin_id}", headers=headers)
    assert self_deleted.status_code == 204


def test_deleted_user_token_is_rejected(client, db):
    created = _register(client, "ana@example.com", [1.0, 0.0])
    headers = _auth(_token(created["id"]))
    user = db.get(User, uuid.UUID(created["id"]))
    db.delete(user)
    db.commit()
    response = client.get("/api/users/me", headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"] == "Não autenticado"


def test_me_returns_own_profile_for_login_token(client):
    created = _register(client, "ana@example.com", [1.0, 0.0])
    client.app.state.face_script.append([1.0, 0.0])
    login = client.post(
        "/api/auth/login",
        data={"email": "ana@example.com"},
        files=_files(),
    )
    assert login.status_code == 200
    response = client.get("/api/users/me", headers=_auth(login.json()["access_token"]))
    assert response.status_code == 200
    assert response.json()["email"] == "ana@example.com"
    assert response.json()["id"] == created["id"]


def test_list_users_limit_above_100_returns_422(client, db):
    admin_id = _seed_admin(db)
    response = client.get("/api/users?limit=101", headers=_auth(_token(admin_id, role=Role.MINISTRY.value)))
    assert response.status_code == 422


def test_patch_oversized_upload_returns_400_without_calling_deepface(client):
    from app.services.face import MAX_IMAGE_BYTES

    created = _register(client, "ana@example.com", [1.0, 0.0])
    calls = []
    client.app.state.user_service.face_service._represent = lambda *a, **k: calls.append(1)
    response = client.patch(
        "/api/users/me",
        files={"image": ("face.png", b"a" * (MAX_IMAGE_BYTES + 1), "image/png")},
        headers=_auth(_token(created["id"])),
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Imagem maior que 5 MB"
    assert calls == []
