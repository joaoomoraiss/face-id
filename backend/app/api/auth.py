from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import read_required_image
from app.database import get_db
from app.security import create_access_token
from app.models.user import role_value
from app.schemas.auth_request import LoginRequest, RegisterRequest
from app.schemas.auth_response import LoginResponse, RegisterResponse
from app.services.auth import AuthService

router = APIRouter()


@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    image: UploadFile | None = File(None),
    db: Session = Depends(get_db),
) -> RegisterResponse:
    # foto vai no forms junto com nome e email
    payload = RegisterRequest(name=name, email=email)
    content, content_type = await read_required_image(image)
    service: AuthService = request.app.state.auth_service
    user = await service.register(
        db,
        name=payload.name,
        email=payload.email,
        image=content,
        content_type=content_type,
    )
    return RegisterResponse.from_user(user)


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    email: str = Form(...),
    image: UploadFile | None = File(None),
    db: Session = Depends(get_db),
) -> LoginResponse:
    payload = LoginRequest(email=email)
    content, content_type = await read_required_image(image)
    service: AuthService = request.app.state.auth_service
    settings = request.app.state.settings
    user = await service.authenticate(
        db,
        email=payload.email,
        image=content,
        content_type=content_type,
    )
    # manda a role no token tbm, mas as rotas conferem de novo no banco
    token = create_access_token(
        user_id=str(user.id),
        role=role_value(user.role),
        secret=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        expires_minutes=settings.access_token_expire_minutes,
    )
    return LoginResponse(access_token=token)
