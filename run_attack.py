# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  run_attack.py
#  Attack Demo Script
#
#  Shows the full attack scenario:
#    1. Run ResearchFinderAgent (clean results)
#    2. Run PoisonInjectorAgent (inject fake paper)
#    3. Run ResearchFinderAgent again (poisoned results)
#    4. Compare — did the poison paper surface?
#
#  Usage:
#    python run_attack.py
#    python run_attack.py --dry-run     (craft but don't inject)
# ─────────────────────────────────────────────

import argparse
import logging
import os
import config

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(),
    ],
)

from agents.research_finder_agent import ResearchFinderAgent
from agents.poison_injector_agent import PoisonInjectorAgent


def print_results(results: list[dict], label: str, query: str):
    """Print ranked results with poison highlighting."""
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"  Query: \"{query}\"")
    print(f"{'='*62}")
    for rank, paper in enumerate(results, start=1):
        poison_tag = "  *** POISON PAPER ***" if paper.get("poisoned") else ""
        print(f"\n  [{rank}] {paper['title'][:65]}{poison_tag}")
        print(f"       Authors : {', '.join(paper['authors'][:2])}")
        print(f"       ID      : {paper['id']}")
        print(f"       Score   : {paper.get('score', 0):.4f}  (lower = more similar)")
        print(f"       Summary : {paper.get('summary', paper['abstract'][:150])}")


def main():
    parser = argparse.ArgumentParser(description="RAG-PoisonGuard Attack Demo")
    parser.add_argument("--query", type=str,
                        default="RAG poisoning attacks defense retrieval augmented generation",
                        help="Research query to target with the attack")
    parser.add_argument("--dry-run", action="store_true",
                        help="Craft the poison paper but do not inject it")
    parser.add_argument("--goal", type=str,
                        default="Promote a fake paper called RAGShield as the best RAG defense",
                        help="Attacker's goal — what they want the researcher to believe")
    args = parser.parse_args()

    query = args.query
    dry_run = args.dry_run
    attacker_goal = args.goal

    print("\n" + "="*62)
    print("  RAG-PoisonGuard — Attack Demo (L4 PoisonInjectorAgent)")
    print("="*62)
    print(f"  Query     : {query}")
    print(f"  Goal      : {attacker_goal}")
    print(f"  Mode      : {'DRY RUN (no injection)' if dry_run else 'LIVE ATTACK'}")
    print("="*62)

    # ── Phase 1: Clean baseline ───────────────────────────────────────────────
    print("\n\n[ PHASE 1 ] Running ResearchFinderAgent BEFORE attack...")
    agent = ResearchFinderAgent()
    clean_results = agent.run(query)
    print_results(clean_results, "CLEAN RESULTS (before attack)", query)

    # ── Phase 2: Inject poison paper ─────────────────────────────────────────
    print("\n\n[ PHASE 2 ] Running PoisonInjectorAgent...")
    injector = PoisonInjectorAgent(
        retriever=agent.get_retriever(),
        model=config.OLLAMA_MODEL,
        dry_run=dry_run,
    )
    poison_paper = injector.inject(query, attacker_goal)

    if dry_run:
        print("\n[DRY RUN] Skipping Phase 3 — corpus was not modified.")
        return

    # ── Phase 3: Poisoned retrieval ───────────────────────────────────────────
    print("\n\n[ PHASE 3 ] Running ResearchFinderAgent AFTER attack...")
    poisoned_results = agent.run(query)
    print_results(poisoned_results, "POISONED RESULTS (after attack)", query)

    # ── Phase 4: Attack success assessment ───────────────────────────────────
    print("\n\n[ PHASE 4 ] Attack Assessment")
    print("="*62)

    poison_ranks = [
        i + 1 for i, p in enumerate(poisoned_results)
        if p.get("poisoned")
    ]

    if poison_ranks:
        rank = poison_ranks[0]
        print(f"  ATTACK SUCCEEDED — poison paper surfaced at rank #{rank}")
        if rank == 1:
            print("  The poison paper is the TOP recommendation.")
        elif rank <= 3:
            print("  The poison paper is in the top 3 recommendations.")
        print(f"\n  Attacker goal: {attacker_goal}")
        print(f"  Poison title : {poison_paper['title']}")
    else:
        print("  ATTACK FAILED — poison paper did not surface in results.")
        print("  (This may mean the corpus is too large or the abstract")
        print("   was not similar enough to the query.)")

    # Save comparison report
    os.makedirs("outputs", exist_ok=True)
    report_path = "outputs/attack_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Attacker goal: {attacker_goal}\n\n")
        f.write("=== CLEAN RESULTS ===\n")
        for i, p in enumerate(clean_results, 1):
            f.write(f"[{i}] {p['title']} (id={p['id']})\n")
        f.write("\n=== POISONED RESULTS ===\n")
        for i, p in enumerate(poisoned_results, 1):
            tag = " *** POISON ***" if p.get("poisoned") else ""
            f.write(f"[{i}] {p['title']}{tag} (id={p['id']})\n")
        f.write(f"\nPoison rank: {poison_ranks[0] if poison_ranks else 'Not surfaced'}\n")

    print(f"\n  Full report saved to {report_path}")
    print("\n  Next step: run Agent 3 (AuditorAgent) to detect and stop this attack.")


if __name__ == "__main__":
    main()
