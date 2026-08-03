# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  agents/summary_agent.py
#  SummaryAgent: calls Ollama to summarise
#  each retrieved paper and rank them.
# ─────────────────────────────────────────────

import logging
import ollama

logger = logging.getLogger(__name__)


class SummaryAgent:
    """
    Uses a local Ollama LLM to:
      1. Summarise each retrieved paper in 2-3 sentences.
      2. Produce a brief relevance explanation for the query.

    Falls back to the raw abstract if Ollama is unavailable.
    """

    def __init__(self, model: str = "llama3"):
        self.model = model

    def summarise(self, papers: list[dict], query: str) -> list[dict]:
        """
        Summarise each paper and add a 'summary' and 'relevance' field.

        Args:
            papers: list of paper dicts from RetrieverAgent
            query:  the original researcher query

        Returns:
            Same list with 'summary' and 'relevance' fields added,
            sorted by retrieval score (ascending = most similar first).
        """
        logger.info(f"[SummaryAgent] Summarising {len(papers)} papers.")
        summarised = []

        for i, paper in enumerate(papers):
            logger.info(f"[SummaryAgent] Processing paper {i+1}/{len(papers)}: {paper['title'][:60]}")
            summary, relevance = self._call_ollama(paper, query)
            enriched = dict(paper)
            enriched["summary"] = summary
            enriched["relevance"] = relevance
            summarised.append(enriched)

        # Sort by retrieval score — lower L2 distance = more relevant
        summarised.sort(key=lambda p: p.get("score", 9999))
        return summarised

    def _call_ollama(self, paper: dict, query: str) -> tuple[str, str]:
        """
        Call Ollama with a structured prompt to summarise the paper.
        Returns (summary, relevance_explanation).
        """
        prompt = f"""You are a research assistant. A researcher asked: "{query}"

Here is a paper they may be interested in:

Title: {paper['title']}
Authors: {', '.join(paper['authors'][:3])}
Published: {paper['published']}
Abstract: {paper['abstract']}

Please respond with exactly two sections:
SUMMARY: A 2-3 sentence plain-English summary of what this paper does and finds.
RELEVANCE: One sentence explaining why this paper is or is not relevant to the researcher's query.
"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response["message"]["content"].strip()
            summary, relevance = self._parse_response(text)
            return summary, relevance

        except Exception as e:
            logger.warning(f"[SummaryAgent] Ollama call failed: {e}. Using raw abstract.")
            return paper["abstract"][:300] + "...", "Relevance could not be assessed."

    def _parse_response(self, text: str) -> tuple[str, str]:
        """Extract SUMMARY and RELEVANCE sections from LLM response."""
        summary = ""
        relevance = ""

        for line in text.splitlines():
            if line.upper().startswith("SUMMARY:"):
                summary = line.split(":", 1)[-1].strip()
            elif line.upper().startswith("RELEVANCE:"):
                relevance = line.split(":", 1)[-1].strip()

        # Fallback: if the LLM didn't follow the format, use the whole response
        if not summary:
            summary = text[:300]
        if not relevance:
            relevance = "See summary above."

        return summary, relevance

    def format_recommendations(self, papers: list[dict], query: str) -> str:
        """
        Format the ranked recommendations as a readable string
        for display in the terminal.
        """
        lines = [
            f"\n{'='*60}",
            f"  RANKED RECOMMENDATIONS",
            f"  Query: \"{query}\"",
            f"{'='*60}\n",
        ]

        for rank, paper in enumerate(papers, start=1):
            status = "⚠ FLAGGED AS SUSPICIOUS" if paper.get("flagged") else ""
            lines += [
                f"[{rank}] {paper['title']} {status}",
                f"    Authors : {', '.join(paper['authors'][:3])}",
                f"    Published: {paper['published']}",
                f"    arXiv ID : {paper['id']}",
                f"    Score    : {paper.get('score', 'N/A'):.4f}",
                f"    Summary  : {paper.get('summary', 'N/A')}",
                f"    Relevance: {paper.get('relevance', 'N/A')}",
                "",
            ]

        return "\n".join(lines)
