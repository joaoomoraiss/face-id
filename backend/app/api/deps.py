import jwt
from fastapi import Depends, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.exceptions.auth import ForbiddenException, UnauthenticatedException
from app.exceptions.face import InvalidImageException
from app.exceptions.user import UserNotFoundException
from app.database import get_db
from app.models.user import Role, User, role_value
from app.security import decode_access_token
from app.services.face import MAX_IMAGE_BYTES
from app.services.user import UserService

bearer_scheme = HTTPBearer(auto_error=False)


async def read_limited_upload(image: UploadFile) -> bytes:
    # le 1 byte a mais pra saber se passou de 5mb sem puxar o arquivo inteiro
    content = await image.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise InvalidImageException("Imagem maior que 5 MB")
    return content


async def read_required_image(image: UploadFile | None) -> tuple[bytes, str | None]:
    if image is None:
        raise InvalidImageException("Arquivo de imagem ausente")
    content = await read_limited_upload(image)
    return content, image.content_type


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise UnauthenticatedException()
    settings = request.app.state.settings
    service: UserService = request.app.state.user_service
    try:
        payload = decode_access_token(
            credentials.credentials,
            settings.jwt_secret_key,
            settings.jwt_algorithm,
        )
    except jwt.PyJWTError as exc:
        # token ruim e user apagado respondem igual, 401
        raise UnauthenticatedException() from exc
    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise UnauthenticatedException()
    try:
        return await service.get_user(db, user_id)
    except UserNotFoundException as exc:
        raise UnauthenticatedException() from exc


# director e ministry gerenciam usuarios. geral so mexe no proprio perfil
MANAGER_ROLES = frozenset({Role.DIRECTOR.value, Role.MINISTRY.value})


async def require_admin(current: User = Depends(get_current_user)) -> User:
    # verificar role do user no banco, n só a q veio no token
    if role_value(current.role) not in MANAGER_ROLES:
        raise ForbiddenException()
    return current
