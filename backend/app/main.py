from contextlib import asynccontextmanager
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.user import router as user_router
from app.config import load_settings
from app.database import prepare_database
from app.services.auth import AuthService
from app.services.face import FaceService
from app.services.user import UserService

load_dotenv()
settings = load_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # sobe o banco antes de atender request
    prepare_database(settings.face_model_name)
    yield


app = FastAPI(title="FaceCheck", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


face = FaceService(
    model_name=settings.face_model_name,
    detector_backend=settings.face_detector_backend,
)
app.state.settings = settings
app.state.auth_service = AuthService(
    face,
    distance_threshold=settings.face_distance_threshold,
)
app.state.user_service = UserService(face)


app.include_router(router=auth_router, prefix="/api/auth", tags=["auth"])
# POST /api/auth/register
# POST /api/auth/login

app.include_router(router=user_router, prefix="/api/users", tags=["users"])
# GET /api/users/me
# PATCH /api/users/me
# GET /api/users
# POST /api/users
# GET /api/users/{user_id}
# PATCH /api/users/{user_id}
# DELETE /api/users/{user_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
