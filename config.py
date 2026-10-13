# ─────────────────────────────────────────────
#  RAG-PoisonGuard  |  config.py
#  Central configuration. Every value can be
#  overridden with an environment variable, so
#  the same code runs locally and in a container.
# ─────────────────────────────────────────────

import os

# Ollama model for SummaryAgent (CLI only; the API does not use it)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# How many papers to retrieve from arXiv per query
TOP_K = int(os.getenv("TOP_K", "20"))

# Embedding model for the vector store (sentence-transformers, runs on CPU)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Where the CLI stores its local vector index and paper metadata
CORPUS_DIR = os.getenv("CORPUS_DIR", "corpus")

# Logging (CLI writes to this file; the API logs to stdout only)
LOG_FILE = os.getenv("LOG_FILE", "logs/run.log")

# AuditorAgent similarity threshold (z-score)
Z_SCORE_THRESHOLD = float(os.getenv("Z_SCORE_THRESHOLD", "-1.7"))

# API settings
MAX_CONCURRENT_AUDITS = int(os.getenv("MAX_CONCURRENT_AUDITS", "2"))
SEARCH_CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "3600"))
SEARCH_CACHE_MAX_ENTRIES = int(os.getenv("SEARCH_CACHE_MAX_ENTRIES", "128"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Poison injection (API). The attacker goal is fixed server-side on purpose, so a
# public endpoint cannot be used to generate arbitrary fake abstracts.
ATTACKER_GOAL = os.getenv(
    "ATTACKER_GOAL",
    "Present the unverified defense described here as the most reliable option available.",
)
# Use an Ollama LLM to craft the poison paper. Off by default: a cloud container has
# no Ollama server, so the agent's built-in fallback paper is used instead.
ENABLE_LLM_ATTACKER = os.getenv("ENABLE_LLM_ATTACKER", "false").lower() == "true"
