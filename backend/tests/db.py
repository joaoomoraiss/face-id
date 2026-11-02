import os

from sqlalchemy import create_engine, make_url, text
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.models.meta import AppMeta
from app.models.user import User

_engine = None


# cria o banco de teste no mesmo postgres do compose, se ainda n existir
def _ensure_database(url: str) -> None:
    target = make_url(url)
    admin = create_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            exists = connection.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": target.database},
            )
            if not exists:
                connection.execute(text(f'CREATE DATABASE "{target.database}"'))
    finally:
        admin.dispose()


def _test_engine():
    global _engine
    if _engine is None:
        # conftest já trocou DATABASE_URL pelo banco facecheck_test
        url = os.environ["DATABASE_URL"]
        _ensure_database(url)
        _engine = create_engine(url, pool_pre_ping=True)
        Base.metadata.create_all(_engine, tables=[User.__table__, AppMeta.__table__])
    return _engine


# cada teste começa com as tabelas vazias. DELETE e n TRUNCATE: truncate pede lock
# exclusivo e trava se algum teste anterior deixou session aberta lendo a tabela
def open_test_session() -> Session:
    engine = _test_engine()
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM users"))
        connection.execute(text("DELETE FROM app_meta"))
    return sessionmaker(bind=engine, expire_on_commit=False)()
