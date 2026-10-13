import threading

import pytest
from fastapi.testclient import TestClient

import app as app_module
import service
from helpers import make_legit_set, make_paper, make_result_set

client = TestClient(app_module.app)


class FakeSearch:
    calls = 0
    papers = staticmethod(make_result_set)

    def __init__(self, top_k):
        self.top_k = top_k

    def search(self, query):
        FakeSearch.calls += 1
        return FakeSearch.papers()


class FakeRetriever:
    def __init__(self, corpus_dir):
        self.papers = []

    def add_papers(self, papers):
        self.papers = list(papers)

    def retrieve(self, query, top_k):
        # like FAISS L2: lowest distance first
        return [dict(p) for p in sorted(self.papers, key=lambda p: p["score"])[:top_k]]


class FakeInjector:
    poison_score = 0.4

    def __init__(self, retriever):
        self.retriever = retriever

    def inject(self, query, attacker_goal):
        poison = make_paper(
            77, score=FakeInjector.poison_score, poisoned=True, id="2503.77777",
            title="Injected poison", abstract="fake abstract",
        )
        self.retriever.papers.append(poison)
        return poison


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeSearch.calls = 0
    FakeSearch.papers = staticmethod(make_result_set)
    FakeInjector.poison_score = 0.4
    service._search_cache.clear()
    monkeypatch.setattr(service, "_make_search_agent", lambda top_k: FakeSearch(top_k))
    monkeypatch.setattr(service, "_make_retriever", lambda d: FakeRetriever(d))
    monkeypatch.setattr(service, "_make_injector", lambda r: FakeInjector(r))
    monkeypatch.setattr(service, "_slots", threading.BoundedSemaphore(1))


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_root_redirects_to_docs():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"


def test_audit_quarantines_poison():
    r = client.post("/audit", json={"query": "rag poisoning defense"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["quarantined"]) == 1
    assert body["quarantined"][0]["poisoned"] is True
    assert len(body["clean"]) == 19
    assert body["attack"] is None
    assert set(body["timing_ms"]) == {"retrieval", "audit"}


def test_search_results_are_cached():
    for _ in range(2):
        client.post("/audit", json={"query": "rag poisoning defense"})
    assert FakeSearch.calls == 1


def test_short_query_rejected():
    assert client.post("/audit", json={"query": "a"}).status_code == 422


def test_inject_poison_is_retrieved_and_detected():
    FakeSearch.papers = staticmethod(make_legit_set)
    r = client.post("/audit", json={"query": "rag poisoning defense", "inject_poison": True})
    assert r.status_code == 200
    attack = r.json()["attack"]
    assert attack["injected"] is True
    assert attack["poison_id"] == "2503.77777"
    assert attack["retrieved"] is True
    assert attack["rank"] == 1
    assert attack["detected"] is True


def test_inject_poison_not_retrieved_is_reported():
    FakeSearch.papers = staticmethod(make_legit_set)
    FakeInjector.poison_score = 9.0  # far from the query, falls outside top_k
    r = client.post(
        "/audit", json={"query": "rag poisoning defense", "top_k": 5, "inject_poison": True}
    )
    attack = r.json()["attack"]
    assert attack["retrieved"] is False
    assert attack["rank"] is None
    assert attack["detected"] is False


def test_swagger_defaults_are_sensible():
    props = client.get("/openapi.json").json()["components"]["schemas"]["AuditRequest"]["properties"]
    assert props["top_k"]["examples"] == [20]
    assert props["z_threshold"]["examples"] == [-1.7]


def test_busy_returns_429():
    assert service._slots.acquire(blocking=False)
    r = client.post("/audit", json={"query": "rag poisoning defense"})
    assert r.status_code == 429
