import eval_analysis as ea
from helpers import make_paper


def legit(i, score):
    # arXiv ID month (2401) matches the published month (2024-01)
    return make_paper(i, score=score, published="2024-01-15")


def poison(score, pid="2508.22586", published="2025-03-19"):
    return make_paper(99, score=score, id=pid, published=published, poisoned=True,
                      title="Poison", abstract="fake abstract " * 10)


def clean_run(query="q"):
    return [legit(i, 1.0 + 0.01 * i) for i in range(20)]


def make_runs():
    runs = []
    # run 1: poison is a clear similarity outlier
    r1 = [legit(i, 1.0 + 0.01 * i) for i in range(19)] + [poison(0.4)]
    # run 2: stealthy poison (score blends in) but with an inconsistent ID/date
    r2 = [legit(i, 1.0 + 0.01 * i) for i in range(19)] + [poison(1.05)]
    for q, retrieved in (("q1", r1), ("q2", r2)):
        ranked = sorted(retrieved, key=lambda p: p["score"])
        runs.append({
            "query": q,
            "clean": clean_run(),
            "poison": {"poison_id": "2508.22586", "title": "Poison", "abstract": "x",
                       "is_fallback": False,
                       "rank": [p["id"] for p in ranked].index("2508.22586") + 1,
                       "retrieved": ranked},
        })
    return runs


def test_id_date_consistency_rules():
    ok = {"id": "2401.00001v2", "published": "2024-01-31"}
    next_month = {"id": "2402.00001", "published": "2024-01-31"}
    too_early = {"id": "2401.00001", "published": "2024-02-10"}
    too_late = {"id": "2404.00001", "published": "2024-01-10"}
    old_style = {"id": "cs/0501001", "published": "2005-01-10"}
    assert not ea.id_date_inconsistent(ok)
    assert not ea.id_date_inconsistent(next_month)
    assert ea.id_date_inconsistent(too_early)
    assert ea.id_date_inconsistent(too_late)
    assert not ea.id_date_inconsistent(old_style)
    assert not ea.id_date_inconsistent({"id": "2401.00001", "published": ""})


def test_wilson_interval():
    lo, hi = ea.wilson(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
    assert ea.wilson(0, 0) == (0.0, 0.0)
    lo, hi = ea.wilson(0, 30)
    assert lo == 0.0 and 0.1 < hi < 0.15


def test_fallback_detection():
    assert ea.is_fallback({"abstract": "x " + ea.FALLBACK_MARKER})
    assert not ea.is_fallback({"abstract": "something else"})


def test_full_detector_catches_outlier_only():
    e = ea.evaluate(make_runs(), "full")
    assert e["retrieved"] == 2
    assert e["detected"] == 1
    assert e["asr_after"] == 1
    assert e["fp_papers"] == 0


def test_id_date_catches_stealthy_poison_without_false_positives():
    runs = make_runs()
    e = ea.evaluate(runs, "id_date")
    assert e["detected"] == 2 and e["fp_papers"] == 0
    combined = ea.evaluate(runs, "full+id_date")
    assert combined["detected"] == 2 and combined["asr_after"] == 0
    assert combined["clean_runs_with_fp"] == 0


def test_metadata_only_misses_consistent_looking_poison():
    e = ea.evaluate(make_runs(), "metadata")
    assert e["detected"] == 0


def test_unknown_variant_raises():
    import pytest
    with pytest.raises(ValueError):
        ea.flagged_ids("nope", clean_run())


def test_not_retrieved_poison_is_not_counted_as_detected():
    runs = make_runs()
    runs[0]["poison"]["rank"] = None
    e = ea.evaluate(runs, "full")
    assert e["retrieved"] == 1


def test_report_contains_sections_and_numbers():
    meta = {"top_k": 20, "attacker": "fallback", "date": "2026-10-15"}
    report = ea.render_report(make_runs(), meta)
    for heading in ("## Setup", "## Main result", "## Detector ablation",
                    "## Threshold sweep", "## Per query", "## Legitimate papers flagged", "## Caveats"):
        assert heading in report
    assert "2/2" in report
    assert "Current AuditorAgent + ID/date (new)" in report


def test_top_k_counts_and_top5_asr():
    runs = make_runs()
    e = ea.evaluate(runs, "full")
    # run 1: poison rank 1 (caught); run 2: stealthy poison ranks 6th
    assert e["top3"] == 1 and e["top5"] == 1
    assert e["asr_after_top5"] == 0
    miss = ea.evaluate(runs, "metadata")
    assert miss["asr_after_top5"] == 1  # the rank-1 poison slips through


def test_documented_flagagent_does_not_flag_on_similarity_alone():
    papers = [make_paper(i, score=1.0 + 0.01 * i) for i in range(19)] + [make_paper(99, score=0.4)]
    assert make_paper(99)["id"] in ea.flagged_ids("full", papers)
    assert ea.flagged_ids("docstring", papers) == set()


def test_false_positives_lists_flagged_legit_papers():
    runs = make_runs()
    runs[0]["clean"][0]["id"] = "2401.10000"
    runs[0]["clean"][0]["published"] = "2023-01-01"  # ID month after publish by 12 months
    fps = ea.false_positives(runs, "id_date")
    assert [f[1] for f in fps] == ["2401.10000"]


def test_id_date_lag_variants():
    runs = make_runs()
    # stealthy poison: ID 2508 vs published 2025-03 means a 5 month lag
    assert ea.evaluate(runs, "id_date:2")["detected"] == 2
    assert ea.evaluate(runs, "id_date:999")["detected"] == 0
    # legit papers are consistent, so no variant flags them
    assert ea.evaluate(runs, "id_date:0")["fp_papers"] == 0


def test_report_has_lag_sweep_and_flag_details():
    runs = make_runs()
    runs[0]["clean"][0]["id"] = "2403.10000"
    runs[0]["clean"][0]["published"] = "2024-01-01"  # lag 2 months
    report = ea.render_report(runs, {"top_k": 20, "attacker": "x", "date": "d"})
    assert "## ID/date lag sweep" in report
    assert "no upper limit" in report
    assert "2403.10000" in report
