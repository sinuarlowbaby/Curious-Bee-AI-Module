import os
import psycopg2
import psycopg2.pool
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=2,
    maxconn=10,
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", 5432),
    dbname=os.getenv("DB_NAME", "yourdb"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", ""),
)


@contextmanager
def get_conn():
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def get_connection():
    """FastAPI dependency."""
    with get_conn() as conn:
        yield conn