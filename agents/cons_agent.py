"""
agents/cons_agent.py
--------------------
Cons Agent — critiques, challenges, and finds limitations.
Provider: Cerebras (Llama 3.1 70B) with Groq / Mistral fallback.

Reflective Pipeline: receives the Pros agent's draft as input context
so it critiques specific claims rather than generating generic cons.
"""

import logging
from core.llm_router import call_llm_async
from core.vector_store import ResearchVectorStore

logger = logging.getLogger(__name__)

CONS_SYSTEM = """\
You are a critical research analyst — a constructive sceptic. Your job is to
find limitations, challenges, contradictions, and risks in research claims.
You are not negative for the sake of it — you surface real, evidence-backed concerns.
Be specific. Cite papers. Challenge weak claims directly.
"""

CONS_PROMPT = """\
TASK: {task}

PROS AGENT DRAFT (critique these specific claims):
{pros_draft}

RELEVANT RESEARCH CONTEXT (additional papers):
{context}

Provide a structured critical analysis:

## Key Limitations
- List 4–6 concrete limitations or challenges found in the research
- Directly counter or qualify specific claims from the Pros draft where evidence warrants

## Methodological Concerns
- Are there gaps, biases, or reproducibility issues in the research?

## Risks & Downsides
- What are the practical risks or negative outcomes documented?

## Contradictions in the Literature
- Where do papers disagree or produce conflicting results?

## What the Pros Draft Overstates
- Specifically call out any claims that are unsupported or exaggerated

Be rigorous and fair — acknowledge strong evidence even while critiquing.
"""


async def run_cons_agent(
    task: str,
    pros_draft: str,
    vector_store: ResearchVectorStore,
    n_context_chunks: int = 6,
) -> str:
    """
    Run the Cons agent.
    pros_draft: output from the Pros agent (reflective pipeline context)
    Returns a markdown string of critical findings.
    """
    context = vector_store.get_context_string(task, n_results=n_context_chunks)

    messages = [
        {"role": "system", "content": CONS_SYSTEM},
        {
            "role": "user",
            "content": CONS_PROMPT.format(
                task=task,
                pros_draft=pros_draft,
                context=context,
            ),
        },
    ]

    result = await call_llm_async("cons", messages, max_tokens=1500, temperature=0.6)
    logger.info("Cons agent completed")
    return result
