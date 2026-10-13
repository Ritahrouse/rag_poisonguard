import threading

import pytest
from fastapi.testclient import TestClient

import app as app_module
import service
from helpers import make_result_set

client = TestClient(app_module.app)


class FakeSearch:
    calls = 0

    def __init__(self, top_k):
        self.top_k = top_k

    def search(self, query):
        FakeSearch.calls += 1
        return make_result_set()


class FakeRetriever:
    def __init__(self, corpus_dir):
        self.papers = []

    def add_papers(self, papers):
        self.papers = papers

    def retrieve(self, query, top_k):
        return self.papers[:top_k]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeSearch.calls = 0
    service._search_cache.clear()
    monkeypatch.setattr(service, "_make_search_agent", lambda top_k: FakeSearch(top_k))
    monkeypatch.setattr(service, "_make_retriever", lambda d: FakeRetriever(d))
    monkeypatch.setattr(service, "_slots", threading.BoundedSemaphore(1))


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_audit_quarantines_poison():
    r = client.post("/audit", json={"query": "rag poisoning defense"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["quarantined"]) == 1
    assert body["quarantined"][0]["poisoned"] is True
    assert len(body["clean"]) == 19
    assert set(body["timing_ms"]) == {"retrieval", "audit"}


def test_search_results_are_cached():
    for _ in range(2):
        client.post("/audit", json={"query": "rag poisoning defense"})
    assert FakeSearch.calls == 1


def test_short_query_rejected():
    assert client.post("/audit", json={"query": "a"}).status_code == 422


def test_inject_poison_not_implemented_yet():
    r = client.post("/audit", json={"query": "rag poisoning", "inject_poison": True})
    assert r.status_code == 501


def test_busy_returns_429():
    assert service._slots.acquire(blocking=False)
    r = client.post("/audit", json={"query": "rag poisoning defense"})
    assert r.status_code == 429
