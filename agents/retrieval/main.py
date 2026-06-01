"""Phase 4 — FastAPI retrieval service.

POST /retrieve   → top-k leaf chunks + tree path
GET  /healthz    → liveness
GET  /readyz     → DB ping + encoder warm check

Design notes
------------
- One TreeSearcher (one BGE-M3 in GPU/CPU memory) is shared across requests.
  The encoder is thread-safe for forward passes; FastAPI's default sync route
  serializes through the process anyway. For real concurrency, run multiple
  uvicorn workers — each gets its own encoder copy. Don't share a model across
  forks: CUDA contexts won't survive os.fork().
- DB connections are short-lived per-request (psycopg's context manager). If
  retrieval QPS gets high enough to matter, swap to `psycopg_pool.ConnectionPool`
  with `register_vector` in `configure`. Premature today.
- The HNSW `ef_search` is set per-txn inside tree_search, not here, so it never
  leaks to other queries on a pooled connection.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.ingestion.db import connect
from agents.retrieval.tree_search import (
    DEFAULT_ALPHA,
    DEFAULT_BEAM,
    DEFAULT_FANOUT,
    DEFAULT_K,
    TreeSearcher,
)


# ── lifespan: build searcher once ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.searcher = TreeSearcher()
    # Cheap warmup so the first user request doesn't pay the JIT/CUDA tax.
    app.state.searcher.encode_query("warmup")
    yield
    # nothing to dispose; psycopg connections are per-request


app = FastAPI(title="PRISM-RAG retrieval", version="0.4", lifespan=lifespan)


# ── schemas ─────────────────────────────────────────────────────────────────
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field("top_down", description="'top_down' or 'collapsed'")
    domain: str | None = None
    source: str | None = None
    k: int = Field(DEFAULT_K, ge=1, le=50)
    beam: int = Field(DEFAULT_BEAM, ge=1, le=32)
    fanout: int = Field(DEFAULT_FANOUT, ge=1, le=500)
    alpha: float = Field(DEFAULT_ALPHA, ge=0.0, le=2.0)


class RetrieveResponse(BaseModel):
    query: str
    mode: str
    domain: str | None
    source: str | None
    k: int
    latency_ms: float
    leaves: list[dict]
    path: list[list[dict]]
    extras: dict


# ── endpoints ───────────────────────────────────────────────────────────────
@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/readyz")
def readyz() -> dict:
    # encoder ready?
    try:
        _ = app.state.searcher.encode_query("ping")
    except Exception as e:                            # pragma: no cover
        raise HTTPException(503, f"encoder not ready: {e}") from e
    # DB reachable + schema present?
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM tree_nodes WHERE level = 0 LIMIT 1")
            n_leaves = cur.fetchone()[0]
    except Exception as e:                            # pragma: no cover
        raise HTTPException(503, f"db not ready: {e}") from e
    return {"ok": True, "n_leaves": n_leaves}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    if req.mode not in {"top_down", "collapsed"}:
        raise HTTPException(400, f"bad mode: {req.mode!r}")
    t0 = time.perf_counter()
    try:
        res = app.state.searcher.retrieve(
            req.query,
            mode=req.mode,
            domain=req.domain,
            source=req.source,
            k=req.k,
            beam=req.beam,
            fanout=req.fanout,
            alpha=req.alpha,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:                            # pragma: no cover
        raise HTTPException(500, f"retrieval failed: {e}") from e
    dt = (time.perf_counter() - t0) * 1000.0

    return RetrieveResponse(
        query=res.query,
        mode=res.mode,
        domain=res.domain,
        source=res.source,
        k=res.k,
        latency_ms=round(dt, 2),
        leaves=[h.to_dict() for h in res.leaves],
        path=[[h.to_dict() for h in lvl] for lvl in res.path],
        extras=res.extras,
    )


# ── direct-run convenience: `python -m agents.retrieval.main` ───────────────
if __name__ == "__main__":                            # pragma: no cover
    import uvicorn
    uvicorn.run(
        "agents.retrieval.main:app",
        host=os.getenv("RETRIEVAL_HOST", "0.0.0.0"),
        port=int(os.getenv("RETRIEVAL_PORT", "8001")),
        workers=int(os.getenv("RETRIEVAL_WORKERS", "1")),
    )