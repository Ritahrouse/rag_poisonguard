# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  app.py
#  FastAPI wrapper around the audit pipeline.
#  Run locally:  uvicorn app:app --reload
# ─────────────────────────────────────────────

import logging
import sys

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
import service

logging.basicConfig(
    level=config.LOG_LEVEL,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG-PoisonGuard", version="0.1.0")


class AuditRequest(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    top_k: int | None = Field(default=None, ge=5, le=50)
    z_threshold: float | None = Field(default=None, ge=-5.0, le=0.0)
    inject_poison: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/audit")
def audit(req: AuditRequest):
    if req.inject_poison:
        # Wired up once PoisonInjectorAgent is integrated (next step).
        raise HTTPException(status_code=501, detail="inject_poison is not implemented yet")
    try:
        return service.run_audit(req.query, req.top_k, req.z_threshold)
    except service.Busy:
        raise HTTPException(status_code=429, detail="All audit slots are busy, retry shortly")
    except requests.RequestException:
        logger.exception("arXiv request failed")
        raise HTTPException(status_code=502, detail="Could not reach arXiv")
