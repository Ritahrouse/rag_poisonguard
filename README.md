# RAG-PoisonGuard

**A Multi-Agent Framework for Detecting and Mitigating Knowledge Poisoning Attacks in Scholarly Paper Discovery Systems**

Applied Cybersecurity using Multi-Agentic AI — Summer Workshop 2026

---

## Project Structure

```
rag_poisonguard/
├── agents/
│   ├── search_agent.py          # L3 sub-agent: queries arXiv API
│   ├── retriever_agent.py       # L3 sub-agent: FAISS vector store
│   ├── summary_agent.py         # L3 sub-agent: Ollama summarisation
│   ├── research_finder_agent.py # L3 orchestrator (Defender)
│   ├── poison_injector_agent.py # L4 attacker (coming next)
│   └── auditor_agent.py         # L5 mitigation (coming next)
├── corpus/                      # FAISS index + metadata saved here
├── logs/                        # Run logs
├── outputs/                     # Recommendation output files
├── config.py                    # Central settings
├── main.py                      # Entry point — Agent 1 only
└── requirements.txt
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Pull your Ollama model**
```bash
ollama pull llama3
# or: ollama pull mistral  /  ollama pull phi3
```

**3. Update config.py if needed**
- Change `OLLAMA_MODEL` to match the model you pulled
- Adjust `TOP_K` to retrieve more or fewer papers

---

## Running Agent 1 (ResearchFinderAgent)

```bash
cd rag_poisonguard
python main.py
# or with a direct query:
python main.py --query "RAG poisoning attacks defense"
```

**What it does:**
1. Queries arXiv for your search term
2. Embeds the abstracts into a local FAISS vector store
3. Retrieves the most relevant papers
4. Summarises each one using your local Ollama model
5. Prints and saves a ranked recommendation list

---

## Coming Next

- `agents/poison_injector_agent.py` — L4 red-team agent
- `agents/auditor_agent.py` — L5 mitigation agent
- Full evaluation pipeline with ASR, FPR, FNR metrics
# rag_poisonguard
# rag_poisonguard
# rag_poisonguard
