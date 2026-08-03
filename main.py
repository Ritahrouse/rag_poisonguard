# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  main.py
#  Entry point — run ResearchFinderAgent
#  from the terminal.
#
#  Usage:
#    python main.py
#    python main.py --query "RAG poisoning attacks"
# ─────────────────────────────────────────────

import argparse
import logging
import os
import config

# ── Logging setup ─────────────────────────────────────────────────────────────
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


def main():
    parser = argparse.ArgumentParser(description="RAG-PoisonGuard: Research Paper Discovery Agent")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Research query string. If omitted, you will be prompted interactively.",
    )
    args = parser.parse_args()

    query = args.query
    if not query:
        print("\n" + "="*60)
        print("  RAG-PoisonGuard — Research Paper Discovery Agent")
        print("="*60)
        query = input("\nEnter your research query: ").strip()
        if not query:
            print("[!] No query provided. Exiting.")
            return

    agent = ResearchFinderAgent()
    results = agent.run(query)

    if not results:
        print("\n[!] No results to display.")
        return

    # Display ranked recommendations
    output = agent.summary_agent.format_recommendations(results, query)
    print(output)

    # Save output to file
    os.makedirs("outputs", exist_ok=True)
    output_file = os.path.join("outputs", "recommendations.txt")
    with open(output_file, "w") as f:
        f.write(output)
    print(f"\n[✓] Results saved to {output_file}")


if __name__ == "__main__":
    main()
