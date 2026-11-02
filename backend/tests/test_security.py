from datetime import datetime, timedelta, timezone

import jwt
import pytest


def test_token_round_trip_contains_sub_and_role():
    from app.security import create_access_token, decode_access_token

    now = datetime.now(timezone.utc)
    token = create_access_token(
        user_id="abc123",
        role="GERAL PERMISSIONS",
        secret="test-secret-key-for-facecheck-0123456789",
        algorithm="HS256",
        expires_minutes=60,
        now=now,
    )
    payload = decode_access_token(token, "test-secret-key-for-facecheck-0123456789", "HS256")
    assert payload["sub"] == "abc123"
    assert payload["role"] == "GERAL PERMISSIONS"
    assert payload["exp"] == int((now + timedelta(minutes=60)).timestamp())


def test_rejects_expired_token():
    from app.security import create_access_token, decode_access_token

    token = create_access_token(
        user_id="abc123",
        role="MINISTRY PERMISSIONS",
        secret="test-secret-key-for-facecheck-0123456789",
        algorithm="HS256",
        expires_minutes=1,
        now=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(token, "test-secret-key-for-facecheck-0123456789", "HS256")


def test_rejects_wrong_secret():
    from app.security import create_access_token, decode_access_token

    token = create_access_token(
        user_id="abc123",
        role="GERAL PERMISSIONS",
        secret="test-secret-key-for-facecheck-0123456789",
        algorithm="HS256",
        expires_minutes=60,
    )
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(token, "other-secret-key-for-facecheck-0123456789", "HS256")
