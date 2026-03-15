"""
agents/supervisor.py
--------------------
Supervisor agent (DeepSeek R1 / Gemini fallback).
Responsibilities:
  1. Decompose the user's research domain into scoped sub-tasks
     for the Pros, Cons, and Future agents.
  2. After all three agents finish, synthesise their outputs into
     a final coherent research intelligence report.
"""

import logging
from core.llm_router import call_llm_async

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """\
You are a research orchestration supervisor. A user wants a comprehensive
analysis of the following research domain:

DOMAIN: {domain}

Your job is to produce precise, scoped instructions for three specialist agents:

1. PROS AGENT — must find concrete advantages, breakthroughs, and validated
   benefits found in recent research on this domain.

2. CONS AGENT — must critically evaluate limitations, challenges, risks, and
   contradictions found in recent research on this domain.

3. FUTURE AGENT — must identify emerging opportunities, open research questions,
   and a forward-looking roadmap for this domain over the next 3–5 years.

Return ONLY a JSON object in this exact format (no markdown, no extra text):
{{
  "pros_task":   "...",
  "cons_task":   "...",
  "future_task": "...",
  "context_summary": "One sentence summarising the domain for context"
}}
"""

SYNTHESISE_PROMPT = """\
You are a senior research intelligence analyst. Three specialist agents have
analysed research papers on the domain: "{domain}".

Their findings:

── PROS AGENT OUTPUT ──────────────────────────────────────────────────
{pros_output}

── CONS AGENT OUTPUT ──────────────────────────────────────────────────
{cons_output}

── FUTURE AGENT OUTPUT ────────────────────────────────────────────────
{future_output}

Your task: Synthesise these into a single, coherent Research Intelligence Report.
Structure it as follows:

# Research Intelligence Report: {domain}

## Executive Summary
(3–4 sentences capturing the overall state of the field)

## Key Advantages & Breakthroughs
(Synthesised from pros — remove duplicates, keep the strongest points)

## Critical Challenges & Limitations
(Synthesised from cons — prioritised by severity)

## Future Opportunities & Roadmap
(Synthesised from future — concrete, actionable, time-bound where possible)

## Consensus Assessment
(Where do pros and cons converge? What does the research collectively suggest?)

## Recommended Next Steps
(3–5 concrete actions for a researcher or practitioner in this domain)

Be specific, cite paper titles where agents mentioned them, and be direct.
"""


async def decompose_query(domain: str) -> dict:
    """
    Ask the supervisor to break the domain into agent-specific tasks.
    Returns dict with keys: pros_task, cons_task, future_task, context_summary
    """
    import json

    messages = [
        {"role": "system", "content": "You are a precise research orchestration supervisor. Always respond with valid JSON only."},
        {"role": "user",   "content": DECOMPOSE_PROMPT.format(domain=domain)},
    ]

    raw = await call_llm_async("supervisor", messages, max_tokens=800, temperature=0.3)

    # Strip markdown fences if model adds them
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        tasks = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Supervisor returned non-JSON, using default tasks")
        tasks = {
            "pros_task":        f"Find key advantages and validated benefits of {domain} from recent research papers.",
            "cons_task":        f"Identify critical limitations, challenges, and risks of {domain} from recent research.",
            "future_task":      f"Map emerging opportunities and a 3-5 year research roadmap for {domain}.",
            "context_summary":  f"Research domain: {domain}",
        }

    logger.info(f"Supervisor decomposed query for: {domain}")
    return tasks


async def synthesise_report(
    domain: str,
    pros_output: str,
    cons_output: str,
    future_output: str,
) -> str:
    """
    Combine the three agent outputs into a final research report.
    """
    messages = [
        {"role": "system", "content": "You are a senior research intelligence analyst. Write clear, structured reports."},
        {
            "role": "user",
            "content": SYNTHESISE_PROMPT.format(
                domain=domain,
                pros_output=pros_output,
                cons_output=cons_output,
                future_output=future_output,
            ),
        },
    ]

    report = await call_llm_async("supervisor", messages, max_tokens=3000, temperature=0.4)
    logger.info("Supervisor synthesised final report")
    return report
