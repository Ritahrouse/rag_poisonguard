# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  build_corpus.py
#  Step 1 — Corpus Builder
#
#  Fetches 300+ legitimate papers from arXiv
#  across multiple topic areas and indexes them
#  into the local FAISS vector store.
#
#  Run ONCE to build the evaluation corpus.
#  Usage: python build_corpus.py
# ─────────────────────────────────────────────

import os
import json
import time
import logging
import config

os.makedirs("logs", exist_ok=True)
os.makedirs("evaluation", exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.FileHandler(config.LOG_FILE), logging.StreamHandler()],
)

from agents.search_agent import SearchAgent
from agents.retriever_agent import RetrieverAgent

# ── Query bank — diverse topics for a realistic corpus ───────────────────────
# Mix of: core target topic + adjacent areas (general NLP, cybersecurity, ML)
# This mirrors how a real scholarly corpus works — not all papers are on topic.

# Fewer queries, more papers each — stays within rate limits
QUERY_BANK = [
    "retrieval augmented generation poisoning attacks defense security",
    "RAG systems adversarial robustness LLM knowledge base",
    "large language models retrieval security evaluation benchmark",
    "dense retrieval NLP question answering information retrieval",
]

PAPERS_PER_QUERY = 100  # arXiv allows up to 30,000 — 100 is safe
TARGET_TOTAL     = 300


def main():
    print("\n" + "="*62)
    print("  RAG-PoisonGuard — Corpus Builder")
    print("="*62)
    print(f"  Queries     : {len(QUERY_BANK)}")
    print(f"  Per query   : {PAPERS_PER_QUERY} papers")
    print(f"  Target total: {TARGET_TOTAL} unique papers")
    print("="*62)

    searcher  = SearchAgent(top_k=PAPERS_PER_QUERY)
    retriever = RetrieverAgent(
        corpus_dir=config.CORPUS_DIR,
        embedding_model=config.EMBEDDING_MODEL
    )

    # Reset corpus for clean build
    print("\n  Resetting corpus for clean build...")
    retriever.reset()

    total_added = 0
    seen_ids    = set()

    for i, query in enumerate(QUERY_BANK):
        if total_added >= TARGET_TOTAL:
            print(f"\n  Target of {TARGET_TOTAL} papers reached. Stopping early.")
            break

        print(f"\n  [{i+1}/{len(QUERY_BANK)}] Query: '{query}'")
        papers = searcher.search(query)

        # Deduplicate across queries
        new_papers = [p for p in papers if p["id"] not in seen_ids]
        for p in new_papers:
            seen_ids.add(p["id"])

        if new_papers:
            retriever.add_papers(new_papers)
            total_added += len(new_papers)
            print(f"    → Added {len(new_papers)} new papers "
                  f"({total_added} total, {len(papers)-len(new_papers)} duplicates skipped)")
        else:
            print("    → No new papers (all duplicates)")

        # Be polite to arXiv API — 3 second delay between requests
        if i < len(QUERY_BANK) - 1:
            time.sleep(15)

    # Save corpus manifest
    manifest = {
        "total_papers": total_added,
        "queries_used": len(QUERY_BANK),
        "papers_per_query": PAPERS_PER_QUERY,
        "corpus_dir": config.CORPUS_DIR,
        "paper_ids": list(seen_ids),
    }
    manifest_path = os.path.join("evaluation", "corpus_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*62}")
    print(f"  Corpus built: {total_added} legitimate papers")
    print(f"  Saved to    : {config.CORPUS_DIR}/")
    print(f"  Manifest    : {manifest_path}")
    print(f"{'='*62}\n")
    print("  Next step: python generate_poison_set.py")


if __name__ == "__main__":
    main()
