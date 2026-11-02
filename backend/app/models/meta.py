from sqlalchemy import String, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from app.database import Base
from app.models.user import User

FACE_MODEL_KEY = "face_model"


def decide_face_model(*, stored_model: str | None, configured_model: str, user_count: int) -> str:
    if user_count > 0 and stored_model is not None and stored_model != configured_model:
        raise RuntimeError(
            f"FACE_MODEL_NAME não pode mudar depois que existirem usuários "
            f"(gravado: {stored_model!r}, configurado: {configured_model!r})"
        )
    return configured_model


class AppMeta(Base):
    __tablename__ = "app_meta"

    key: Mapped[str] = mapped_column(String(50), primary_key=True)
    face_model_name: Mapped[str] = mapped_column(String(100), nullable=False)


# se ja tem user, n deixa trocar o modelo. embedding de um n bate com o outro
def ensure_recorded_face_model(db: Session, configured_model: str) -> None:
    count = db.scalar(select(func.count()).select_from(User)) or 0
    meta = db.get(AppMeta, FACE_MODEL_KEY)
    stored = meta.face_model_name if meta is not None else None
    model = decide_face_model(
        stored_model=stored,
        configured_model=configured_model,
        user_count=count,
    )
    if meta is None:
        db.add(AppMeta(key=FACE_MODEL_KEY, face_model_name=model))
    else:
        meta.face_model_name = model
    db.commit()
