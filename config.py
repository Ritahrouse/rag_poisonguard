# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  config.py
#  Central configuration — edit values here
# ─────────────────────────────────────────────

# Ollama model to use (must be pulled locally)
# Run: ollama pull llama3 — or swap for mistral, phi3, etc.
OLLAMA_MODEL = "llama3"

# How many papers to retrieve from arXiv per query
TOP_K = 20

# Embedding model for the vector store (runs locally via sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Where to store the local vector index and paper metadata
CORPUS_DIR = "corpus"

# Logging
LOG_FILE = "logs/run.log"
