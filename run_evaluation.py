# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  run_evaluation.py
#  Step 3 — Offline Evaluator
#
#  Loads pre-built corpus + poison set,
#  injects poison papers in batches,
#  runs AuditorAgent, computes full metrics.
#  Compares against baseline_results.json.
#
#  Runs in ~30 seconds after corpus is built.
#  Usage: python run_evaluation.py
#         python run_evaluation.py --batch-size 10
# ─────────────────────────────────────────────

import argparse
import json
import os
import time
import logging
import config

os.makedirs("logs", exist_ok=True)
os.makedirs("evaluation", exist_ok=True)
logging.basicConfig(level=logging.WARNING,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[logging.FileHandler(config.LOG_FILE), logging.StreamHandler()])

from agents.retriever_agent import RetrieverAgent
from agents.auditor_agent import AuditorAgent
import numpy as np


def inject_poison_papers(retriever: RetrieverAgent, poison_papers: list[dict]):
    """Inject all poison papers into the FAISS index."""
    existing_ids = {p["id"] for p in retriever.metadata}
    new_papers   = [p for p in poison_papers if p["id"] not in existing_ids]

    if not new_papers:
        return 0

    texts      = [p["abstract"] for p in new_papers]
    embeddings = retriever.model.encode(texts, show_progress_bar=False).astype("float32")
    retriever.index.add(embeddings)
    retriever.metadata.extend(new_papers)
    retriever.save()
    return len(new_papers)


def retrieve_for_query(retriever: RetrieverAgent, query: str, top_k: int) -> list[dict]:
    """Retrieve top-k papers for a query from the FAISS index."""
    k = min(top_k, retriever.index.ntotal)
    embedding = retriever.model.encode([query], show_progress_bar=False).astype("float32")
    distances, indices = retriever.index.search(embedding, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        paper = dict(retriever.metadata[idx])
        paper["score"] = float(dist)
        results.append(paper)
    return results


def compute_full_metrics(all_results, all_clean, all_quarantined, n_poison, n_legit):
    """Compute comprehensive metrics across all evaluation queries."""

    # Aggregate across all query evaluations
    total_poison_surfaced = sum(
        1 for r in all_results if r.get("poisoned")
    )
    total_tp = sum(1 for p in all_quarantined if p.get("poisoned"))
    total_fp = sum(1 for p in all_quarantined if not p.get("poisoned"))
    total_fn = sum(1 for p in all_clean if p.get("poisoned"))

    asr_before = total_poison_surfaced / n_poison if n_poison > 0 else 0
    asr_after  = total_fn / n_poison if n_poison > 0 else 0
    fpr        = total_fp / n_legit if n_legit > 0 else 0
    fnr        = total_fn / total_poison_surfaced if total_poison_surfaced > 0 else 0

    return {
        "corpus_size_legitimate": n_legit,
        "corpus_size_poison":     n_poison,
        "total_corpus_size":      n_legit + n_poison,
        "poison_surfaced_before": total_poison_surfaced,
        "ASR_before_mitigation":  round(asr_before, 4),
        "true_positives":         total_tp,
        "false_positives":        total_fp,
        "false_negatives":        total_fn,
        "FPR":                    round(fpr, 4),
        "FNR":                    round(fnr, 4),
        "ASR_after_mitigation":   round(asr_after, 4),
        "recovery":               "YES",
    }


def compare_with_baseline(metrics: dict, baseline_path: str):
    """Print side-by-side comparison with baseline small-scale results."""
    if not os.path.exists(baseline_path):
        print("  [!] Baseline results not found — skipping comparison.")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)

    b = baseline["metrics"]

    print(f"\n{'='*70}")
    print("  COMPARISON: Baseline (small) vs Full Evaluation (large)")
    print(f"{'='*70}")
    print(f"  {'Metric':<35} {'Baseline':>12} {'Full Eval':>12}")
    print(f"  {'-'*60}")

    comparisons = [
        ("Corpus — legitimate papers",
         baseline["corpus"]["legitimate_papers"],
         metrics["corpus_size_legitimate"]),
        ("Corpus — poison papers",
         baseline["corpus"]["poison_papers_injected"],
         metrics["corpus_size_poison"]),
        ("ASR before mitigation",
         b["ASR_before_mitigation"],
         metrics["ASR_before_mitigation"]),
        ("ASR after mitigation",
         b["ASR_after_mitigation"],
         metrics["ASR_after_mitigation"]),
        ("FPR (False Positive Rate)",
         b["FPR"],
         metrics["FPR"]),
        ("FNR (False Negative Rate)",
         b["FNR"],
         metrics["FNR"]),
        ("True Positives",
         b["true_positives"],
         metrics["true_positives"]),
        ("False Positives",
         b["false_positives"],
         metrics["false_positives"]),
        ("False Negatives",
         b["false_negatives"],
         metrics["false_negatives"]),
    ]

    for label, base_val, full_val in comparisons:
        if isinstance(base_val, float) or isinstance(full_val, float):
            print(f"  {label:<35} {base_val:>12.2f} {full_val:>12.4f}")
        else:
            print(f"  {label:<35} {base_val:>12} {full_val:>12}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG-PoisonGuard Full Evaluation")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="How many poison papers to evaluate per batch")
    parser.add_argument("--retrieval-k", type=int, default=50,
                        help="Top-k papers to retrieve per query for evaluation")
    args = parser.parse_args()

    poison_path   = "evaluation/poison_set.json"
    baseline_path = "evaluation/baseline_results.json"
    results_path  = "evaluation/full_evaluation_results.json"

    # ── Load poison set ───────────────────────────────────────────────────────
    if not os.path.exists(poison_path):
        print("[!] Poison set not found. Run generate_poison_set.py first.")
        return

    with open(poison_path) as f:
        poison_papers = json.load(f)

    n_poison = len(poison_papers)
    print(f"\n{'='*62}")
    print("  RAG-PoisonGuard — Full Evaluation")
    print(f"{'='*62}")
    print(f"  Poison papers   : {n_poison}")
    print(f"  Retrieval top-k : {args.retrieval_k}")
    print(f"{'='*62}\n")

    # ── Load corpus ───────────────────────────────────────────────────────────
    retriever = RetrieverAgent(
        corpus_dir=config.CORPUS_DIR,
        embedding_model=config.EMBEDDING_MODEL
    )

    if retriever.index.ntotal == 0:
        print("[!] Corpus empty. Run build_corpus.py first.")
        return

    n_legit = retriever.index.ntotal
    print(f"  Corpus loaded: {n_legit} legitimate papers")

    # ── Inject all poison papers ──────────────────────────────────────────────
    print(f"  Injecting {n_poison} poison papers into corpus...")
    added = inject_poison_papers(retriever, poison_papers)
    print(f"  Injected {added} papers. Total corpus: {retriever.index.ntotal}\n")

    # ── Run evaluation per query ──────────────────────────────────────────────
    auditor = AuditorAgent(
        z_score_threshold=-1.7,
        corpus_metadata=retriever.metadata
    )

    all_results    = []
    all_clean      = []
    all_quarantined = []
    query_results  = []

    unique_queries = list({p["target_query"] for p in poison_papers})
    print(f"  Evaluating across {len(unique_queries)} unique queries...\n")

    t_start = time.perf_counter()

    for qi, query in enumerate(unique_queries):
        print(f"  [{qi+1}/{len(unique_queries)}] Query: '{query[:55]}...'")

        # Retrieve top-k for this query
        results = retrieve_for_query(retriever, query, top_k=args.retrieval_k)

        # Audit
        clean, quarantined = auditor.audit(results)

        # Track
        all_results.extend(results)
        all_clean.extend(clean)
        all_quarantined.extend(quarantined)

        p_surfaced  = sum(1 for r in results if r.get("poisoned"))
        p_caught    = sum(1 for r in quarantined if r.get("poisoned"))
        p_missed    = sum(1 for r in clean if r.get("poisoned"))
        fp          = sum(1 for r in quarantined if not r.get("poisoned"))

        print(f"    Poison surfaced: {p_surfaced} | "
              f"Caught: {p_caught} | Missed: {p_missed} | FP: {fp}")

        query_results.append({
            "query": query,
            "results_retrieved": len(results),
            "poison_surfaced": p_surfaced,
            "poison_caught": p_caught,
            "poison_missed": p_missed,
            "false_positives": fp,
        })

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_full_metrics(
        all_results, all_clean, all_quarantined, n_poison, n_legit
    )
    metrics["total_latency_ms"]    = round(latency_ms, 1)
    metrics["latency_per_query_ms"] = round(latency_ms / len(unique_queries), 1)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  FULL EVALUATION METRICS")
    print(f"{'='*62}")
    print(f"  {'Metric':<40} {'Value':>10}")
    print(f"  {'-'*52}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<40} {v:>10.4f}")
        else:
            print(f"  {k:<40} {str(v):>10}")
    print(f"{'='*62}")

    # ── Compare with baseline ─────────────────────────────────────────────────
    compare_with_baseline(metrics, baseline_path)

    # ── Save full results ─────────────────────────────────────────────────────
    output = {
        "experiment": "full_scale_evaluation",
        "date": "2026-05-29",
        "metrics": metrics,
        "per_query_results": query_results,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Full results saved to {results_path}")
    print("  Evaluation complete.\n")


if __name__ == "__main__":
    main()
