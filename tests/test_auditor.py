from datetime import datetime, timezone

from agents.auditor_agent import AuditorAgent, MetadataInspector
from helpers import make_paper, make_result_set


def test_outlier_similarity_is_quarantined_and_legit_papers_pass():
    clean, quarantined = AuditorAgent().audit(make_result_set())
    assert len(quarantined) == 1
    assert quarantined[0]["poisoned"] is True
    assert len(clean) == 19


def test_no_false_positives_on_uniform_clean_set():
    papers = [make_paper(i, score=1.0 + 0.01 * i) for i in range(20)]
    clean, quarantined = AuditorAgent().audit(papers)
    assert quarantined == []
    assert len(clean) == 20


def test_fewer_than_three_papers_skips_similarity_check():
    clean, quarantined = AuditorAgent().audit([make_paper(1), make_paper(2)])
    assert quarantined == []


def test_critical_metadata_flag_quarantines_paper():
    bad = make_paper(5, id="not-an-arxiv-id", authors=[])
    papers = [make_paper(i, score=1.0 + 0.01 * i) for i in range(10)] + [bad]
    _, quarantined = AuditorAgent().audit(papers)
    assert [p["id"] for p in quarantined] == ["not-an-arxiv-id"]
    assert "Critical metadata" in quarantined[0]["flag_reason"]


def test_similarity_alone_flags_paper():
    # Documents current behaviour: FlagAgent flags on similarity alone,
    # even though its class docstring says it does not.
    clean, quarantined = AuditorAgent().audit(make_result_set())
    assert quarantined[0]["metadata_flags"] == []
    assert "Anomalous similarity" in quarantined[0]["flag_reason"]


def test_future_date_is_relative_to_current_year():
    year = datetime.now(timezone.utc).year
    papers = [
        make_paper(1, published=f"{year}-06-01"),
        make_paper(2, published=f"{year + 1}-01-01"),
    ]
    out = MetadataInspector().check(papers)
    assert out[0]["metadata_flags"] == []
    assert any("Future date" in f for f in out[1]["metadata_flags"])
