from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, read_limited_upload, read_required_image, require_admin
from app.database import get_db
from app.models.user import User
from app.schemas.user_request import CreateUserRequest, ListUsersRequest, UpdateUserRequest
from app.schemas.user_response import UserListResponse, UserResponse
from app.services.auth import AuthService
from app.services.user import UserService

router = APIRouter()


# no patch a foto é opcional, no cadastro n
async def _optional_image(image: UploadFile | None) -> tuple[bytes | None, str | None]:
    if image is None:
        return None, None
    return await read_limited_upload(image), image.content_type


@router.get("/me", response_model=UserResponse)
async def read_me(current: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.from_user(current)


@router.patch("/me", response_model=UserResponse)
async def update_me(
    request: Request,
    name: str | None = Form(None),
    email: str | None = Form(None),
    image: UploadFile | None = File(None),
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserResponse:
    payload = UpdateUserRequest(name=name, email=email)
    content, content_type = await _optional_image(image)
    service: UserService = request.app.state.user_service
    updated = await service.update_user(
        db,
        str(current.id),
        name=payload.name,
        email=payload.email,
        image=content,
        content_type=content_type,
    )
    return UserResponse.from_user(updated)


@router.get("", response_model=UserListResponse)
async def list_users(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> UserListResponse:
    params = ListUsersRequest(skip=skip, limit=limit)
    service: UserService = request.app.state.user_service
    items, total = await service.list_users(db, skip=params.skip, limit=params.limit)
    return UserListResponse(items=[UserResponse.from_user(user) for user in items], total=total)


@router.post("", response_model=UserResponse, status_code=201)
async def create_user(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    image: UploadFile | None = File(None),
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> UserResponse:
    payload = CreateUserRequest(name=name, email=email)
    content, content_type = await read_required_image(image)
    # mesmo fluxo do register, ent entra como user tbm
    service: AuthService = request.app.state.auth_service
    user = await service.register(
        db,
        name=payload.name,
        email=payload.email,
        image=content,
        content_type=content_type,
    )
    return UserResponse.from_user(user)


@router.get("/{user_id}", response_model=UserResponse)
async def read_user(
    user_id: str,
    request: Request,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> UserResponse:
    service: UserService = request.app.state.user_service
    user = await service.get_user(db, user_id)
    return UserResponse.from_user(user)


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    request: Request,
    name: str | None = Form(None),
    email: str | None = Form(None),
    image: UploadFile | None = File(None),
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> UserResponse:
    payload = UpdateUserRequest(name=name, email=email)
    content, content_type = await _optional_image(image)
    service: UserService = request.app.state.user_service
    updated = await service.update_user(
        db,
        user_id,
        name=payload.name,
        email=payload.email,
        image=content,
        content_type=content_type,
    )
    return UserResponse.from_user(updated)


@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: str,
    request: Request,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> None:
    service: UserService = request.app.state.user_service
    await service.delete_user(db, user_id)
