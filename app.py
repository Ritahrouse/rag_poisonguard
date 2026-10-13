# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  app.py
#  FastAPI wrapper around the audit pipeline.
#  Run locally:  uvicorn app:app --reload
# ─────────────────────────────────────────────

import logging
import sys
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
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
    query: str = Field(min_length=3, max_length=300, examples=["RAG poisoning attacks defense"])
    top_k: Optional[int] = Field(default=None, ge=5, le=50, examples=[20])
    z_threshold: Optional[float] = Field(default=None, ge=-5.0, le=0.0, examples=[-1.7])
    inject_poison: bool = False


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/audit")
def audit(req: AuditRequest):
    try:
        return service.run_audit(req.query, req.top_k, req.z_threshold, req.inject_poison)
    except service.Busy:
        raise HTTPException(status_code=429, detail="All audit slots are busy, retry shortly")
    except requests.RequestException:
        logger.exception("arXiv request failed")
        raise HTTPException(status_code=502, detail="Could not reach arXiv")
