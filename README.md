# Research Intelligence System

Multi-agent system that autonomously discovers, analyses, and connects research
knowledge using free API providers — no rate limit issues.

## Architecture

```
User Query
    │
    ▼
Supervisor (DeepSeek R1)   ← decomposes query into scoped tasks
    │
    ├─ Pros Agent  (Groq Llama 3.3 70B)      ← finds advantages
    │
    ├─ Cons Agent  (Cerebras Llama 3.1 70B)  ← critiques pros draft  ← REFLECTIVE
    │
    └─ Future Agent (DeepSeek R1)            ← roadmap from debate   ← REFLECTIVE
    │
    ▼
Supervisor synthesis → Final Report

Each agent uses RAG: ArXiv papers → ChromaDB (local embeddings, no API cost)
```

## Quick Setup

### 1. Clone and install
```bash
git clone <your-repo>
cd research_intel
pip install -r requirements.txt
```

### 2. Get free API keys (takes ~10 min total)
| Provider  | URL                              | Free Tier         |
|-----------|----------------------------------|-------------------|
| Groq      | https://console.groq.com         | 30 req/min        |
| Cerebras  | https://cloud.cerebras.ai        | 60 req/min        |
| DeepSeek  | https://platform.deepseek.com    | 500k tokens/min   |
| Mistral   | https://console.mistral.ai       | 1 req/sec         |

### 3. Configure
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 4. Run

**Web UI (recommended for demo):**
```bash
python main.py
# Open http://localhost:8000
```

**CLI (for testing):**
```bash
python cli.py "transformer attention mechanisms"
python cli.py "CRISPR gene editing" --pdf mypaper.pdf --output report.md
```

## Project Structure
```
research_intel/
├── main.py                    # Entry point (web server)
├── cli.py                     # Command-line interface
├── requirements.txt
├── .env.example               # Copy to .env and fill keys
│
├── core/
│   ├── llm_router.py          # LiteLLM multi-provider routing + failover
│   ├── paper_fetcher.py       # ArXiv fetch, scoring, PDF download
│   ├── vector_store.py        # ChromaDB + local sentence-transformer embeddings
│   └── pipeline.py            # Main orchestration (wires all agents)
│
├── agents/
│   ├── supervisor.py          # Query decomposition + final synthesis
│   ├── pros_agent.py          # Advantages + breakthroughs
│   ├── cons_agent.py          # Limitations + critique (reflective)
│   └── future_agent.py        # Roadmap + opportunities (reflective)
│
├── api/
│   └── server.py              # FastAPI + SSE streaming
│
├── ui/
│   └── templates/
│       └── index.html         # Frontend
│
└── data/
    ├── papers/                # Downloaded PDFs (auto-created)
    └── chroma_db/             # Vector DB (auto-created, persists across runs)
```

## Key Design Decisions

### Why no Gemini?
Gemini's free tier caps too quickly for 3 agents × N calls. This system
distributes load across providers that are optimised for different tasks.

### Why local embeddings?
`sentence-transformers/all-MiniLM-L6-v2` runs on CPU, is fast, and costs $0.
No API calls for embedding = no rate limits on the RAG layer.

### Why sequential (reflective) pipeline?
The Cons agent receives the Pros draft — it critiques *specific claims*,
not generic cons. The Future agent sees the full debate. This produces
dramatically better output quality than running all agents in parallel.

### Why ChromaDB persists?
Re-running the same domain reuses cached embeddings — subsequent runs are
much faster and don't re-download papers.
