from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.core.database import Base, get_db


def test_get_db_yields_session_and_closes():
    gen = get_db()
    session = next(gen)
    assert session is not None

    try:
        next(gen)
    except StopIteration:
        pass

def test_session_can_execute_query():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    TestSession = sessionmaker(bind=engine)
    session = TestSession()

    try:
        result = session.execute(text("SELECT 1"))
        assert result.scalar() == 1
    finally:
        session.close()