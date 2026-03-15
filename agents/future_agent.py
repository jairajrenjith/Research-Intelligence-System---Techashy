"""
agents/future_agent.py
----------------------
Future Agent — opportunities, roadmap, emerging directions.
Provider: DeepSeek R1 (chain-of-thought reasoning) with Groq fallback.

Reflective Pipeline: receives both Pros and Cons outputs so it can
build a roadmap that directly addresses the tensions found in the debate.
"""

import logging
from core.llm_router import call_llm_async
from core.vector_store import ResearchVectorStore

logger = logging.getLogger(__name__)

FUTURE_SYSTEM = """\
You are a research foresight analyst. You read current research debates and
map what comes next: open problems, emerging opportunities, and concrete roadmaps.
You think in 1-year, 3-year, and 5-year horizons.
Be concrete. Avoid buzzwords. Ground predictions in the research evidence.
"""

FUTURE_PROMPT = """\
TASK: {task}

THE RESEARCH DEBATE SO FAR:

── PROS (validated advantages) ──────────────────────────
{pros_draft}

── CONS (critical challenges) ───────────────────────────
{cons_draft}

ADDITIONAL RESEARCH CONTEXT:
{context}

Based on this refined debate and the research evidence, provide:

## Open Research Questions
- What are the 3–5 most important unanswered questions in this field?

## Emerging Opportunities
- What directions show the most promise given current limitations?
- What gaps in the cons represent the biggest opportunity?

## Research Roadmap
### Near-term (0–1 year)
- Concrete steps researchers/practitioners can take now

### Mid-term (1–3 years)
- What breakthroughs or milestones are plausible?

### Long-term (3–5 years)
- What does mature success look like in this domain?

## Highest-Leverage Interventions
- If you had to pick ONE area to focus on to move the field forward fastest, what would it be and why?

## Technologies & Methods to Watch
- What tools, frameworks, or methods are emerging that could unlock progress?

Ground everything in the evidence. Be bold but realistic.
"""


async def run_future_agent(
    task: str,
    pros_draft: str,
    cons_draft: str,
    vector_store: ResearchVectorStore,
    n_context_chunks: int = 6,
) -> str:
    """
    Run the Future agent.
    Receives pros_draft AND cons_draft — builds roadmap from the full debate.
    Returns a markdown string with the future roadmap.
    """
    context = vector_store.get_context_string(task, n_results=n_context_chunks)

    messages = [
        {"role": "system", "content": FUTURE_SYSTEM},
        {
            "role": "user",
            "content": FUTURE_PROMPT.format(
                task=task,
                pros_draft=pros_draft,
                cons_draft=cons_draft,
                context=context,
            ),
        },
    ]

    result = await call_llm_async("future", messages, max_tokens=2000, temperature=0.5)
    logger.info("Future agent completed")
    return result
