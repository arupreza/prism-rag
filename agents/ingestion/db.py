"""Postgres connection helpers.

We register pgvector on each connection so `vector` columns serialize cleanly
to/from numpy arrays without manual casting.
"""
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from .config import PG_DSN


@contextmanager
def connect():
    """Yield a psycopg connection with pgvector types registered.

    Usage:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    with psycopg.connect(PG_DSN) as conn:
        register_vector(conn)
        yield conn