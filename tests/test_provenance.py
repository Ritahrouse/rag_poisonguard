import pytest
import requests

import eval_analysis as ea
import provenance as prov
from helpers import make_paper

FEED = """<feed xmlns="http://www.w3.org/2005/Atom">
 <entry><id>http://arxiv.org/abs/2401.10000v2</id>
  <title>Legitimate   paper
   number 0</title><summary>Some   abstract text.</summary></entry>
 <entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_x</id>
  <title>Error</title><summary>incorrect id format</summary></entry>
</feed>"""


def record_for(paper):
    return {"title": paper["title"], "abstract": paper["abstract"]}


def test_base_id_strips_version_only():
    assert prov.base_id("2401.10000v3") == "2401.10000"
    assert prov.base_id("2401.10000") == "2401.10000"
    assert prov.base_id("hep-th/9901001v1") == "hep-th/9901001"


def test_parse_feed_skips_error_entries_and_normalises_whitespace():
    parsed = prov.parse_feed(FEED)
    assert list(parsed) == ["2401.10000"]
    assert parsed["2401.10000"]["title"] == "Legitimate paper number 0"
    assert parsed["2401.10000"]["abstract"] == "Some abstract text."


def test_check_paper_outcomes():
    p = make_paper(1)
    assert prov.check_paper(p, record_for(p)) == "verified"
    assert prov.check_paper(p, None) == "not_found"
    assert prov.check_paper(p, dict(record_for(p), title="A completely different title")) == "mismatch"
    assert prov.check_paper(p, dict(record_for(p), abstract="totally unrelated words here")) == "mismatch"
    # harmless formatting differences still verify
    shouted = dict(record_for(p), title=p["title"].upper() + "  ")
    assert prov.check_paper(p, shouted) == "verified"


class FakeArxiv:
    def __init__(self, db, bad=()):
        self.db, self.bad, self.calls = db, set(bad), []

    def __call__(self, params):
        ids = params["id_list"].split(",")
        self.calls.append(ids)
        if self.bad & set(ids):
            raise requests.RequestException("boom")
        entries = "".join(
            f"<entry><id>http://arxiv.org/abs/{i}v1</id><title>{self.db[i]['title']}</title>"
            f"<summary>{self.db[i]['abstract']}</summary></entry>"
            for i in ids if i in self.db
        )
        return f'<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'


def test_lookup_batches_and_marks_missing_ids_as_none(monkeypatch):
    db = {"2401.00001": {"title": "A", "abstract": "x"}, "2401.00002": {"title": "B", "abstract": "y"}}
    fake = FakeArxiv(db)
    monkeypatch.setattr(prov, "_http_get", fake)
    records, failed = prov.lookup(["2401.00001v2", "2401.00002", "2501.99999"], batch_size=2, delay=0)
    assert records["2401.00001"]["title"] == "A"
    assert records["2501.99999"] is None
    assert failed == set()
    assert len(fake.calls) == 2


def test_failing_id_is_isolated_and_reported_unavailable(monkeypatch):
    db = {f"2401.0000{i}": {"title": str(i), "abstract": "z"} for i in range(1, 5)}
    monkeypatch.setattr(prov, "_http_get", FakeArxiv(db, bad=["2401.00003"]))
    records, failed = prov.lookup(list(db), batch_size=4, delay=0, retries=1)
    assert failed == {"2401.00003"}
    assert set(records) == {"2401.00001", "2401.00002", "2401.00004"}


def make_provenance(papers_ok, poison_id="2508.22586", poison_record=None):
    data = {prov.base_id(p["id"]): record_for(p) for p in papers_ok}
    data[prov.base_id(poison_id)] = poison_record
    return data


def runs_with_provenance():
    legit = [make_paper(i, score=1.0 + 0.01 * i, published="2024-01-15") for i in range(20)]
    poison = make_paper(99, score=1.05, id="2508.22586", published="2025-03-19",
                        title="Poison", abstract="fake abstract " * 10, poisoned=True)
    ranked = sorted(legit[:19] + [poison], key=lambda p: p["score"])
    runs = [{"query": "q", "clean": legit,
             "poison": {"poison_id": "2508.22586", "title": "Poison", "abstract": poison["abstract"],
                        "is_fallback": False, "rank": [p["id"] for p in ranked].index("2508.22586") + 1,
                        "retrieved": ranked}}]
    return runs, make_provenance(legit)


def test_provenance_variant_flags_unknown_ids_and_no_legit_paper():
    runs, data = runs_with_provenance()
    e = ea.evaluate(runs, "provenance", provenance=data)
    assert e["detected"] == 1 and e["fp_papers"] == 0


def test_unavailable_lookups_are_not_flagged():
    runs, data = runs_with_provenance()
    data["2508.22586"] = "unavailable"
    e = ea.evaluate(runs, "provenance", provenance=data)
    assert e["detected"] == 0 and e["fp_papers"] == 0


def test_mismatching_text_for_a_real_id_is_flagged():
    runs, data = runs_with_provenance()
    data["2508.22586"] = {"title": "A real, different paper", "abstract": "completely other words"}
    assert ea.evaluate(runs, "provenance", provenance=data)["detected"] == 1


def test_provenance_variant_needs_data():
    runs, _ = runs_with_provenance()
    with pytest.raises(ValueError):
        ea.evaluate(runs, "provenance")


def test_report_includes_provenance_sections_only_when_given():
    runs, data = runs_with_provenance()
    meta = {"top_k": 20, "attacker": "x", "date": "d"}
    with_prov = ea.render_report(runs, meta, data)
    assert "## arXiv provenance lookup" in with_prov
    assert "arXiv provenance check only (new)" in with_prov
    without = ea.render_report(runs, meta)
    assert "provenance lookup" not in without


def test_identity_copy_is_caught_only_when_the_abstract_is_compared():
    runs, data = runs_with_provenance()
    real = make_paper(50, published="2024-01-15")
    data["2508.22586"] = record_for(real)  # arXiv holds a real paper under the poison's ID
    poison_run = runs[0]["poison"]
    poison_run["title"] = real["title"]
    for p in poison_run["retrieved"]:
        if p["id"] == "2508.22586":
            p["title"] = real["title"]  # attacker copied the real title, kept its own abstract
    assert ea.evaluate(runs, "provenance", provenance=data)["detected"] == 1
    assert ea.evaluate(runs, "provenance_title", provenance=data)["detected"] == 0
