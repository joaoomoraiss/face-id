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
    # sub é o id. role vai junto mas a api n confia só nisso
    payload = {
        "sub": user_id,
        "role": role,
        "exp": issued + timedelta(minutes=expires_minutes),
    }
    token = jwt.encode(payload, secret, algorithm=algorithm)
    # pyjwt antigo devolve bytes
    if isinstance(token, bytes):
        return token.decode("utf-8")
    return token


def decode_access_token(token: str, secret: str, algorithm: str) -> dict:
    return jwt.decode(token, secret, algorithms=[algorithm])
