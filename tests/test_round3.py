import json
import os

import batch_eval
import round3_real_identity as r3


def build_cache(tmp_path, queries, top_k=5):
    cache = tmp_path / "eval_cache"
    (cache / "search").mkdir(parents=True)
    (cache / "poison_llm").mkdir(parents=True)
    for qi, q in enumerate(queries):
        papers = [{"id": f"24{qi:02d}.{i:05d}", "title": f"Real {qi}-{i}", "authors": [f"A{qi}"],
                   "published": f"2024-{qi + 1:02d}-10", "url": f"https://arxiv.org/abs/24{qi:02d}.{i:05d}",
                   "abstract": "real abstract"} for i in range(1, 4)]
        (cache / "search" / (batch_eval._key(q, top_k) + ".json")).write_text(json.dumps(papers))
        poison = {"id": "2501.99999", "title": "Fake", "authors": ["X"], "published": "2025-02-02",
                  "url": "https://arxiv.org/abs/2501.99999", "abstract": "poison text", "poisoned": True}
        (cache / "poison_llm" / (batch_eval._key(q) + ".json")).write_text(json.dumps(poison))
    return cache


def load_poison(cache, q):
    return json.loads((cache / "poison_llm" / (batch_eval._key(q) + ".json")).read_text())


def test_poison_takes_a_real_identity_from_another_query(tmp_path):
    queries = ["q one", "q two", "q three"]
    cache = build_cache(tmp_path, queries)
    assert r3.main(str(cache), queries, top_k=5, offset=1) == 0
    for qi, q in enumerate(queries):
        poison = load_poison(cache, q)
        own_ids = {f"24{qi:02d}.{i:05d}" for i in range(1, 4)}
        assert poison["id"].startswith("24") and poison["id"] not in own_ids
        assert poison["abstract"] == "poison text" and poison["poisoned"] is True
        assert poison["title"].startswith("Real ")
        assert poison["url"].endswith(poison["id"])


def test_backup_is_kept_and_rerun_is_stable(tmp_path):
    queries = ["q one", "q two", "q three"]
    cache = build_cache(tmp_path, queries)
    r3.main(str(cache), queries, top_k=5, offset=1)
    first = load_poison(cache, "q one")
    r3.main(str(cache), queries, top_k=5, offset=1)
    assert load_poison(cache, "q one") == first
    backup = cache / "poison_llm_round2" / (batch_eval._key("q one") + ".json")
    assert json.loads(backup.read_text())["id"] == "2501.99999"


def test_missing_poison_directory_is_an_error(tmp_path):
    assert r3.main(str(tmp_path / "nothing"), ["q"], top_k=5) == 1
    assert not os.path.exists(tmp_path / "nothing")
