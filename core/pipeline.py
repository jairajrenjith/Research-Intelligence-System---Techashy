"""
core/pipeline.py
----------------
Main orchestration pipeline. Wires together:
  Supervisor → [Pros, Cons, Future agents] (sequential / reflective) → Synthesis

Usage:
  from core.pipeline import run_research_pipeline
  result = await run_research_pipeline("quantum computing", user_pdfs=[...])
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from core.paper_fetcher import fetch_arxiv_papers, ingest_user_pdf, Paper
from core.vector_store import ResearchVectorStore
from agents.supervisor import decompose_query, synthesise_report
from agents.pros_agent import run_pros_agent
from agents.cons_agent import run_cons_agent
from agents.future_agent import run_future_agent

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    domain:        str
    papers:        list[Paper]
    pros_output:   str
    cons_output:   str
    future_output: str
    final_report:  str
    elapsed_sec:   float
    vector_stats:  dict = field(default_factory=dict)


async def run_research_pipeline(
    domain: str,
    user_pdf_bytes: Optional[list[tuple[bytes, str]]] = None,  # [(bytes, filename), ...]
    progress_cb: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """
    Full end-to-end research intelligence pipeline.

    Args:
        domain:         Research topic/domain string
        user_pdf_bytes: Optional list of (pdf_bytes, filename) tuples
        progress_cb:    Optional callback(message) for streaming status updates

    Returns:
        PipelineResult with all agent outputs and final report
    """
    def _progress(msg: str):
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    start = time.time()

    # ── Step 1: Fetch papers ────────────────────────────────────────────────
    _progress("📄 Fetching research papers from ArXiv...")
    papers = fetch_arxiv_papers(domain)

    # Ingest user-uploaded PDFs (always included, high priority)
    if user_pdf_bytes:
        for pdf_bytes, filename in user_pdf_bytes:
            _progress(f"📎 Processing uploaded paper: {filename}")
            user_paper = ingest_user_pdf(pdf_bytes, filename)
            papers.insert(0, user_paper)   # prepend so they're always included

    _progress(f"✅ Working with {len(papers)} papers")

    # ── Step 2: Build vector store ──────────────────────────────────────────
    _progress("🔍 Chunking and embedding papers (local, no API)...")
    collection_name = domain[:40]
    store = ResearchVectorStore(collection_name)
    chunks_added = store.add_papers(papers)
    _progress(f"✅ Vector store ready: {store.stats()['total_chunks']} chunks")

    # ── Step 3: Supervisor decomposes the query ─────────────────────────────
    _progress("🧠 Supervisor decomposing research query...")
    tasks = await decompose_query(domain)
    _progress(f"✅ Tasks assigned — {tasks.get('context_summary', '')}")

    # ── Step 4: Pros agent ──────────────────────────────────────────────────
    _progress("🟢 Pros agent analysing advantages...")
    pros_output = await run_pros_agent(
        task=tasks["pros_task"],
        vector_store=store,
    )
    _progress("✅ Pros agent done")

    # ── Step 5: Cons agent (receives pros draft — reflective pipeline) ──────
    _progress("🔴 Cons agent critiquing pros draft...")
    cons_output = await run_cons_agent(
        task=tasks["cons_task"],
        pros_draft=pros_output,        # ← reflective: cons sees pros first
        vector_store=store,
    )
    _progress("✅ Cons agent done")

    # ── Step 6: Future agent (sees full debate) ─────────────────────────────
    _progress("🔵 Future agent building roadmap from debate...")
    future_output = await run_future_agent(
        task=tasks["future_task"],
        pros_draft=pros_output,
        cons_draft=cons_output,        # ← reflective: future sees both
        vector_store=store,
    )
    _progress("✅ Future agent done")

    # ── Step 7: Supervisor synthesises final report ─────────────────────────
    _progress("📋 Supervisor synthesising final report...")
    final_report = await synthesise_report(
        domain=domain,
        pros_output=pros_output,
        cons_output=cons_output,
        future_output=future_output,
    )
    _progress("✅ Final report ready!")

    elapsed = round(time.time() - start, 1)
    _progress(f"🎉 Pipeline complete in {elapsed}s")

    return PipelineResult(
        domain=domain,
        papers=papers,
        pros_output=pros_output,
        cons_output=cons_output,
        future_output=future_output,
        final_report=final_report,
        elapsed_sec=elapsed,
        vector_stats=store.stats(),
    )
