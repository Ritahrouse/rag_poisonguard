# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  agents/retriever_agent.py
#  RetrieverAgent: embeds paper abstracts into
#  a local FAISS vector store and retrieves
#  the most relevant papers for a query.
# ─────────────────────────────────────────────

import os
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """
    Embeds paper abstracts using a local sentence-transformer model
    and stores them in a FAISS index for similarity search.

    The index and metadata are saved to disk under CORPUS_DIR so
    they persist across runs and can be inspected / poisoned.
    """

    def __init__(self, corpus_dir: str = "corpus", embedding_model: str = "all-MiniLM-L6-v2"):
        self.corpus_dir = corpus_dir
        self.index_path = os.path.join(corpus_dir, "faiss.index")
        self.metadata_path = os.path.join(corpus_dir, "metadata.json")

        os.makedirs(corpus_dir, exist_ok=True)

        logger.info(f"[RetrieverAgent] Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()

        # Load existing index + metadata, or create fresh ones
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self._load()
        else:
            self._create_fresh()

    # ── Index management ─────────────────────────────────────────────────────

    def _create_fresh(self):
        """Initialise an empty flat L2 FAISS index."""
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata: list[dict] = []
        logger.info("[RetrieverAgent] Created fresh FAISS index.")

    def _load(self):
        """Load persisted FAISS index and metadata from disk."""
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
        logger.info(f"[RetrieverAgent] Loaded index with {self.index.ntotal} vectors.")

    def save(self):
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"[RetrieverAgent] Saved index ({self.index.ntotal} vectors) to disk.")

    def reset(self):
        """Wipe the index and metadata — useful between evaluation runs."""
        self._create_fresh()
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        logger.info("[RetrieverAgent] Index reset.")

    # ── Ingest ───────────────────────────────────────────────────────────────

    def add_papers(self, papers: list[dict]):
        """
        Embed and add a list of paper dicts to the index.
        Skips papers whose ID already exists in metadata.

        Args:
            papers: list of dicts from SearchAgent (or PoisonInjectorAgent)
        """
        new_papers = [p for p in papers if not self._already_indexed(p["id"])]
        if not new_papers:
            logger.info("[RetrieverAgent] All papers already indexed — nothing to add.")
            return

        texts = [p["abstract"] for p in new_papers]
        embeddings = self.model.encode(texts, show_progress_bar=False).astype("float32")

        self.index.add(embeddings)
        self.metadata.extend(new_papers)
        self.save()
        logger.info(f"[RetrieverAgent] Added {len(new_papers)} new papers to index.")

    def _already_indexed(self, paper_id: str) -> bool:
        return any(p["id"] == paper_id for p in self.metadata)

    # ── Retrieve ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embed the query and return the top-k most similar papers.

        Args:
            query:  natural-language search query
            top_k:  number of results to return

        Returns:
            List of paper dicts, each with an added 'score' field
            (L2 distance — lower is more similar).
        """
        if self.index.ntotal == 0:
            logger.warning("[RetrieverAgent] Index is empty — no papers to retrieve.")
            return []

        k = min(top_k, self.index.ntotal)
        query_embedding = self.model.encode([query], show_progress_bar=False).astype("float32")
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            paper = dict(self.metadata[idx])
            paper["score"] = float(dist)
            results.append(paper)

        logger.info(f"[RetrieverAgent] Retrieved {len(results)} papers for query.")
        return results
