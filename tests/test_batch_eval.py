import batch_eval
import eval_analysis
from helpers import make_paper


def fake_search(query):
    return [make_paper(i, score=1.0 + 0.01 * i, published="2024-01-15") for i in range(10)]


def fake_poison(query, papers):
    return {"id": "2508.22586", "title": "Poison", "abstract": "fake " + eval_analysis.FALLBACK_MARKER,
            "authors": ["X"], "published": "2025-03-19", "poisoned": True}


def fake_retrieve(papers, query, top_k, poison):
    items = [dict(p) for p in papers]
    if poison is not None:
        items.append(dict(poison, score=0.5))
    return sorted(items, key=lambda p: p["score"])[:top_k]


def test_collect_builds_clean_and_poison_runs():
    runs = batch_eval.collect(["q1", "q2"], fake_search, fake_poison, fake_retrieve, top_k=5)
    assert len(runs) == 2
    run = runs[0]
    assert len(run["clean"]) == 5
    assert run["poison"]["rank"] == 1
    assert run["poison"]["is_fallback"] is True
    assert run["poison"]["retrieved"][0]["id"] == "2508.22586"


def test_collect_reports_poison_outside_top_k():
    def far_retrieve(papers, query, top_k, poison):
        items = [dict(p) for p in papers]
        if poison is not None:
            items.append(dict(poison, score=99.0))
        return sorted(items, key=lambda p: p["score"])[:top_k]

    runs = batch_eval.collect(["q"], fake_search, fake_poison, far_retrieve, top_k=5)
    assert runs[0]["poison"]["rank"] is None


def test_collect_skips_queries_without_results():
    runs = batch_eval.collect(["q"], lambda q: [], fake_poison, fake_retrieve, top_k=5)
    assert runs == []


def test_cache_calls_producer_once(tmp_path):
    calls = []

    def produce():
        calls.append(1)
        return {"a": 1}

    path = str(tmp_path / "sub" / "x.json")
    assert batch_eval._cached(path, produce) == {"a": 1}
    assert batch_eval._cached(path, produce) == {"a": 1}
    assert len(calls) == 1


def test_empty_results_are_not_cached(tmp_path):
    calls = []

    def produce():
        calls.append(1)
        return []

    path = str(tmp_path / "x.json")
    batch_eval._cached(path, produce)
    batch_eval._cached(path, produce)
    assert len(calls) == 2


def test_default_queries_are_unique_and_enough():
    qs = batch_eval.DEFAULT_QUERIES
    assert len(qs) == len(set(qs)) >= 30
