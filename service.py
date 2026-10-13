# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  service.py
#  Pipeline logic for the API: search -> retrieve -> audit.
#  No FastAPI code here, so it can be tested and reused.
#
#  Design notes:
#   - Each request gets its own throw-away FAISS index in a temp dir,
#     so requests never see each other's papers and nothing persists.
#   - arXiv results are cached briefly to avoid hammering the arXiv API.
#   - SummaryAgent / Ollama is deliberately not used here.
# ─────────────────────────────────────────────

import copy
import logging
import tempfile
import threading
import time
from typing import Optional

import config
from agents.auditor_agent import AuditorAgent

logger = logging.getLogger(__name__)

_search_cache: dict[tuple[str, int], tuple[float, list[dict]]] = {}
_cache_lock = threading.Lock()
_slots = threading.BoundedSemaphore(config.MAX_CONCURRENT_AUDITS)


class Busy(Exception):
    """Raised when all audit slots are in use."""


def _make_search_agent(top_k: int):
    from agents.search_agent import SearchAgent
    return SearchAgent(top_k=top_k)


def _make_retriever(corpus_dir: str):
    from agents.retriever_agent import RetrieverAgent
    return RetrieverAgent(corpus_dir=corpus_dir, embedding_model=config.EMBEDDING_MODEL)


def _search(query: str, top_k: int) -> list[dict]:
    key = (query.strip().lower(), top_k)
    now = time.time()
    with _cache_lock:
        hit = _search_cache.get(key)
        if hit and now - hit[0] < config.SEARCH_CACHE_TTL_SECONDS:
            return copy.deepcopy(hit[1])

    papers = _make_search_agent(top_k).search(query)

    if papers:
        with _cache_lock:
            if len(_search_cache) >= config.SEARCH_CACHE_MAX_ENTRIES:
                _search_cache.pop(next(iter(_search_cache)))
            _search_cache[key] = (now, copy.deepcopy(papers))
    return papers


def _public(paper: dict) -> dict:
    out = {
        "id": paper.get("id"),
        "title": paper.get("title"),
        "authors": paper.get("authors", []),
        "published": paper.get("published"),
        "url": paper.get("url"),
        "score": paper.get("score"),
        "similarity_z_score": paper.get("similarity_z_score"),
        "metadata_flags": paper.get("metadata_flags", []),
        "flagged": paper.get("flagged", False),
        "flag_reason": paper.get("flag_reason"),
    }
    # Ground truth only exists in evaluation runs with injected poison
    if "poisoned" in paper:
        out["poisoned"] = bool(paper["poisoned"])
    return out


def run_audit(query: str, top_k: Optional[int] = None, z_threshold: Optional[float] = None) -> dict:
    top_k = top_k or config.TOP_K
    z = config.Z_SCORE_THRESHOLD if z_threshold is None else z_threshold

    if not _slots.acquire(blocking=False):
        raise Busy()
    try:
        t0 = time.perf_counter()
        papers = _search(query, top_k)
        if not papers:
            return {"query": query, "clean": [], "quarantined": [],
                    "timing_ms": {"retrieval": 0.0, "audit": 0.0}}

        with tempfile.TemporaryDirectory() as tmp:
            retriever = _make_retriever(tmp)
            retriever.add_papers(papers)
            retrieved = retriever.retrieve(query, top_k=top_k)
        t1 = time.perf_counter()

        clean, quarantined = AuditorAgent(z_score_threshold=z).audit(retrieved)
        t2 = time.perf_counter()

        logger.info("audit done query=%r clean=%d quarantined=%d", query, len(clean), len(quarantined))
        return {
            "query": query,
            "z_threshold": z,
            "clean": [_public(p) for p in clean],
            "quarantined": [_public(p) for p in quarantined],
            "timing_ms": {
                "retrieval": round((t1 - t0) * 1000, 1),
                "audit": round((t2 - t1) * 1000, 1),
            },
        }
    finally:
        _slots.release()
