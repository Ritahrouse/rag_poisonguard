import json
import random

import eval_analysis as ea
import round2_consistent_metadata as r2


def test_generated_ids_are_consistent_with_the_date():
    rng = random.Random(1)
    for published in ("2025-03-19", "2025-12-30", "2025-01-02"):
        pid = r2.id_for_date(published, rng)
        assert not ea.id_date_inconsistent({"id": pid, "published": published})


def test_main_rewrites_files_and_backs_up(tmp_path):
    src, backup = tmp_path / "poison_llm", tmp_path / "backup"
    src.mkdir()
    original = {"id": "2508.22586", "title": "T", "abstract": "A", "published": "2025-03-19",
                "url": "https://arxiv.org/abs/2508.22586"}
    (src / "a.json").write_text(json.dumps(original))
    assert r2.main(str(src), str(backup)) == 0
    new = json.loads((src / "a.json").read_text())
    assert new["abstract"] == "A" and new["id"] != "2508.22586"
    assert new["url"].endswith(new["id"])
    assert json.loads((backup / "a.json").read_text())["id"] == "2508.22586"
    # running again keeps the backup and gives the same IDs
    assert r2.main(str(src), str(backup)) == 0
    assert json.loads((src / "a.json").read_text())["id"] == new["id"]
    assert json.loads((backup / "a.json").read_text())["id"] == "2508.22586"


def test_main_without_files_returns_error(tmp_path):
    assert r2.main(str(tmp_path / "none"), str(tmp_path / "b")) == 1
