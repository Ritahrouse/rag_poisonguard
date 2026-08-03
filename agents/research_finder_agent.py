# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  agents/research_finder_agent.py
#  ResearchFinderAgent (L3 — Defender)
#  Orchestrates SearchAgent → RetrieverAgent
#  → SummaryAgent into one cohesive workflow.
# ─────────────────────────────────────────────

import logging
from agents.search_agent import SearchAgent
from agents.retriever_agent import RetrieverAgent
from agents.summary_agent import SummaryAgent
import config

logger = logging.getLogger(__name__)


class ResearchFinderAgent:
    """
    L3 Defender agent.

    Workflow:
      1. SearchAgent  — fetches papers from arXiv matching the query
      2. RetrieverAgent — embeds and indexes them in a local FAISS store,
                          then retrieves the top-k most similar to the query
      3. SummaryAgent — summarises each paper using Ollama and ranks them

    Returns a ranked list of paper dicts ready for display (or auditing).
    """

    def __init__(self):
        logger.info("[ResearchFinderAgent] Initialising sub-agents...")
        self.search_agent = SearchAgent(top_k=config.TOP_K)
        self.retriever_agent = RetrieverAgent(
            corpus_dir=config.CORPUS_DIR,
            embedding_model=config.EMBEDDING_MODEL,
        )
        self.summary_agent = SummaryAgent(model=config.OLLAMA_MODEL)
        logger.info("[ResearchFinderAgent] Ready.")

    def run(self, query: str) -> list[dict]:
        """
        Execute the full discovery pipeline for a given query.

        Args:
            query: natural-language research question

        Returns:
            Ranked list of paper dicts, each containing:
            id, title, abstract, authors, published, url,
            score, summary, relevance, poisoned (flag)
        """
        print(f"\n[ResearchFinderAgent] Running pipeline for query: '{query}'")

        # Step 1 — Search arXiv
        print("  Step 1/3 — SearchAgent: querying arXiv...")
        papers = self.search_agent.search(query)
        if not papers:
            print("  [!] No papers returned from arXiv. Check your connection.")
            return []
        print(f"  → Found {len(papers)} papers on arXiv.")

        # Step 2 — Embed and index, then retrieve
        print("  Step 2/3 — RetrieverAgent: embedding and indexing...")
        self.retriever_agent.add_papers(papers)
        retrieved = self.retriever_agent.retrieve(query, top_k=config.TOP_K)
        print(f"  → Retrieved top {len(retrieved)} papers from local index.")

        # Step 3 — Summarise and rank
        print("  Step 3/3 — SummaryAgent: summarising with Ollama...")
        ranked = self.summary_agent.summarise(retrieved, query)
        print(f"  → Summarisation complete. {len(ranked)} papers ranked.")

        return ranked

    def get_retriever(self) -> RetrieverAgent:
        """Expose the retriever for direct access by PoisonInjectorAgent."""
        return self.retriever_agent
