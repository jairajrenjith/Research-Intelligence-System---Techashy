"""
agents/pros_agent.py
--------------------
Pros Agent — finds advantages, breakthroughs, validated benefits.
Provider: Groq (Llama 3.3 70B) with Cerebras / Mistral fallback.

Implements the Reflective Pipeline:
  - Receives its specific task from the Supervisor
  - Uses RAG context from the vector store
  - Produces a structured draft of pros/advantages
"""

import logging
from core.llm_router import call_llm_async
from core.vector_store import ResearchVectorStore

logger = logging.getLogger(__name__)

PROS_SYSTEM = """\
You are a research analyst specialising in identifying advantages, breakthroughs,
and positive evidence in academic research. You read paper excerpts and extract
concrete, specific benefits — not vague generalities.
Be direct. Use bullet points. Cite paper titles when possible.
"""

PROS_PROMPT = """\
TASK: {task}

RELEVANT RESEARCH CONTEXT (from {n_papers} papers):
{context}

Based on this research context, provide a structured analysis of:

## Validated Advantages
- List 4–6 concrete, evidence-backed advantages found in the research
- Include specific metrics, results, or claims where available

## Key Breakthroughs
- List 2–3 significant recent breakthroughs or novel findings

## Strongest Evidence
- What does the weight of evidence most strongly support?

## Confidence Assessment
- How strong is the evidence overall? (High / Medium / Low) and why

Be specific. Avoid generic statements. Reference paper findings directly.
"""


async def run_pros_agent(
    task: str,
    vector_store: ResearchVectorStore,
    n_context_chunks: int = 6,
) -> str:
    """
    Run the Pros agent. Returns a markdown string of findings.
    """
    context = vector_store.get_context_string(task, n_results=n_context_chunks)
    stats   = vector_store.stats()

    messages = [
        {"role": "system", "content": PROS_SYSTEM},
        {
            "role": "user",
            "content": PROS_PROMPT.format(
                task=task,
                context=context,
                n_papers=stats["total_chunks"],
            ),
        },
    ]

    result = await call_llm_async("pros", messages, max_tokens=1500, temperature=0.6)
    logger.info("Pros agent completed")
    return result
