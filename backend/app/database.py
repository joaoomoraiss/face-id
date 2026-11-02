from collections.abc import Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config import load_settings

Base = declarative_base()

_engine = None
_SessionLocal = None


def _connect():
    global _engine, _SessionLocal
    if _engine is None:
        load_dotenv()
        url = load_settings().database_url
        # pre_ping descarta conexao morta se o container do postgres reiniciar
        _engine = create_engine(url, pool_pre_ping=True)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def __getattr__(name: str):
    if name == "engine":
        return _connect()
    if name == "SessionLocal":
        _connect()
        return _SessionLocal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# cria as tabelas se n existirem e grava qual modelo de rosto ta valendo
def prepare_database(face_model_name: str) -> None:
    from app.models.meta import AppMeta, ensure_recorded_face_model
    from app.models.user import User

    _connect()
    Base.metadata.create_all(bind=_engine, tables=[User.__table__, AppMeta.__table__])
    db = _SessionLocal()
    try:
        ensure_recorded_face_model(db, face_model_name)
    finally:
        db.close()


# uma session por request, fecha mesmo se der erro
def get_db() -> Iterator[Session]:
    _connect()
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
