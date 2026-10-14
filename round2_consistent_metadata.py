#!/usr/bin/env python
"""
Red/blue round 2: keep the cached poison papers (same title and abstract) but give
each one an arXiv ID whose month matches its published date, which is what the
injector should do. Round-1 poison files are backed up first.

  python round2_consistent_metadata.py
  python batch_eval.py --attacker llm --out eval_results_round2

The second command reuses the arXiv and poison caches, so it takes minutes and
does not need Ollama.
"""
import glob
import json
import os
import random
import shutil
import sys

SRC = "eval_cache/poison_llm"
BACKUP = "eval_cache/poison_llm_round1"


def id_for_date(published: str, rng: random.Random) -> str:
    year, month = int(published[:4]), int(published[5:7])
    month += rng.choice([0, 0, 1])  # same month, sometimes the next one
    if month > 12:
        month -= 12
        year += 1
    return f"{year % 100:02d}{month:02d}.{rng.randint(10000, 29999)}"


def main(src: str = SRC, backup: str = BACKUP) -> int:
    files = sorted(glob.glob(os.path.join(src, "*.json")))
    if not files:
        print(f"No poison files in {src}. Run the full evaluation first.")
        return 1
    if not os.path.exists(backup):
        shutil.copytree(src, backup)
        print(f"Backed up round-1 poison papers to {backup}")
    for path in files:
        with open(path) as f:
            poison = json.load(f)
        rng = random.Random(os.path.basename(path))  # same result on every run
        poison["id"] = id_for_date(poison["published"], rng)
        poison["url"] = f"https://arxiv.org/abs/{poison['id']}"
        with open(path, "w") as f:
            json.dump(poison, f)
    print(f"Rewrote {len(files)} poison papers with consistent ID and date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
