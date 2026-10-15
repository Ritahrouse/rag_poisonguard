#!/usr/bin/env python
"""
Red/blue round 3: the attacker copies the identity of a REAL arXiv paper (ID, title,
authors, date, URL) and attaches its own poison abstract. The donor paper comes from
another query's results, so it is not in the current query's result set.

  python round3_real_identity.py
  python batch_eval.py --attacker llm --out eval_results_round3 --provenance

Round-2 poison files are backed up to eval_cache/poison_llm_round2 first.
Needs the search and poison caches from the earlier runs.
"""
import json
import os
import shutil
import sys

import batch_eval
import config


def main(cache_dir: str = "eval_cache", queries=None, top_k: int = config.TOP_K, offset: int = 7) -> int:
    queries = queries or batch_eval.DEFAULT_QUERIES
    poison_dir = os.path.join(cache_dir, "poison_llm")
    backup = os.path.join(cache_dir, "poison_llm_round2")

    def search(q):
        path = os.path.join(cache_dir, "search", batch_eval._key(q, top_k) + ".json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    if not os.path.isdir(poison_dir):
        print(f"No poison files in {poison_dir}. Run the earlier rounds first.")
        return 1
    if not os.path.exists(backup):
        shutil.copytree(poison_dir, backup)
        print(f"Backed up current poison papers to {backup}")

    done = 0
    for i, query in enumerate(queries):
        own = search(query)
        poison_path = os.path.join(poison_dir, batch_eval._key(query) + ".json")
        if not own or not os.path.exists(poison_path):
            print(f"skip (no cache): {query}")
            continue
        own_ids = {p["id"] for p in own}
        donor = None
        for step in range(1, len(queries)):
            others = search(queries[(i + offset * step) % len(queries)]) or []
            candidates = [p for p in others if p["id"] not in own_ids]
            if candidates:
                donor = candidates[i % len(candidates)]
                break
        if donor is None:
            print(f"skip (no donor): {query}")
            continue
        with open(poison_path) as f:
            poison = json.load(f)
        for field in ("id", "title", "authors", "published", "url"):
            poison[field] = donor[field]  # abstract stays the attacker's
        with open(poison_path, "w") as f:
            json.dump(poison, f)
        done += 1
    print(f"Gave {done} poison papers a real paper's identity.")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
