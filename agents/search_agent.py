# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  agents/search_agent.py
#  SearchAgent: queries the arXiv API and
#  returns a list of paper metadata dicts.
# ─────────────────────────────────────────────

import requests
import xml.etree.ElementTree as ET
import logging
import time

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
NAMESPACE = "{http://www.w3.org/2005/Atom}"


class SearchAgent:
    """
    Queries the arXiv API for papers matching a natural-language query.

    Returns a list of dicts, each with:
        - id         : arXiv paper ID (e.g. "2401.12345")
        - title      : paper title
        - abstract   : full abstract text
        - authors    : list of author names
        - published  : publication date string
        - url        : link to the arXiv abstract page
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def search(self, query: str) -> list[dict]:
        """
        Search arXiv for papers matching the query.

        Args:
            query: natural-language research query

        Returns:
            List of paper metadata dicts (up to top_k results)
        """
        logger.info(f"[SearchAgent] Querying arXiv: '{query}'")

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": self.top_k,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        for attempt in range(4):
            try:
                response = requests.get(ARXIV_API_URL, params=params, timeout=15)
                if response.status_code == 429:
                    wait = 30 * (attempt + 1)
                    print(f"    [Rate limited] Waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                break
            except requests.RequestException as e:
                logger.error(f"[SearchAgent] arXiv API request failed: {e}")
                return []
        else:
            logger.error("[SearchAgent] All retries failed.")
            return []

        papers = self._parse_response(response.text)
        logger.info(f"[SearchAgent] Retrieved {len(papers)} papers from arXiv.")
        return papers

    def _parse_response(self, xml_text: str) -> list[dict]:
        """Parse arXiv Atom XML response into a list of paper dicts."""
        root = ET.fromstring(xml_text)
        papers = []

        for entry in root.findall(f"{NAMESPACE}entry"):
            paper_id_url = entry.findtext(f"{NAMESPACE}id", default="")
            paper_id = paper_id_url.split("/abs/")[-1].strip()

            title = entry.findtext(f"{NAMESPACE}title", default="").strip()
            title = " ".join(title.split())  # normalise whitespace

            abstract = entry.findtext(f"{NAMESPACE}summary", default="").strip()
            abstract = " ".join(abstract.split())

            published = entry.findtext(f"{NAMESPACE}published", default="")[:10]

            authors = [
                author.findtext(f"{NAMESPACE}name", default="")
                for author in entry.findall(f"{NAMESPACE}author")
            ]

            papers.append({
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published": published,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "poisoned": False,   # flag used later by AuditorAgent
            })

        return papers
