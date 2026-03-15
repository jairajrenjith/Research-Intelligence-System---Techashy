"""
core/paper_fetcher.py
---------------------
Fetches research papers from ArXiv.
- Pulls up to MAX_PAPERS_FETCH papers for a domain query
- Scores them by citation signals, recency, and title relevance
- Returns the top TOP_PAPERS_SELECT papers
- Also accepts user-uploaded PDFs
"""

import os
import re
import asyncio
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone

import arxiv
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MAX_FETCH   = int(os.getenv("MAX_PAPERS_FETCH", 50))
TOP_SELECT  = int(os.getenv("TOP_PAPERS_SELECT", 10))
PAPERS_DIR  = Path(os.getenv("PAPERS_DIR", "./data/papers"))
PAPERS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Paper:
    paper_id:    str
    title:       str
    authors:     list[str]
    abstract:    str
    full_text:   str = ""
    published:   str = ""
    url:         str = ""
    source:      str = "arxiv"       # "arxiv" | "user_upload"
    score:       float = 0.0
    categories:  list[str] = field(default_factory=list)


# ── ArXiv fetching ────────────────────────────────────────────────────────────

def fetch_arxiv_papers(domain: str) -> list[Paper]:
    """
    Search ArXiv for `domain`, fetch up to MAX_FETCH results,
    score them, and return the top TOP_SELECT.
    """
    logger.info(f"Fetching ArXiv papers for: '{domain}'")

    client = arxiv.Client(page_size=MAX_FETCH, delay_seconds=1.0, num_retries=3)
    search = arxiv.Search(
        query=domain,
        max_results=MAX_FETCH,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[Paper] = []
    for result in client.results(search):
        paper = Paper(
            paper_id=result.entry_id.split("/")[-1],
            title=result.title,
            authors=[a.name for a in result.authors[:5]],
            abstract=result.summary,
            published=result.published.strftime("%Y-%m-%d") if result.published else "",
            url=result.pdf_url or result.entry_id,
            categories=result.categories,
        )
        paper.score = _score_paper(paper, domain)
        papers.append(paper)

    papers.sort(key=lambda p: p.score, reverse=True)
    top = papers[:TOP_SELECT]

    # Download PDFs for top papers (best-effort, skip on failure)
    for paper in top:
        _try_download_pdf(paper)

    logger.info(f"Selected {len(top)} top papers from {len(papers)} fetched")
    return top


def _score_paper(paper: Paper, query: str) -> float:
    """
    Simple relevance score:
    - Keyword overlap with query in title (weighted high)
    - Keyword overlap with query in abstract
    - Recency bonus (newer = better)
    """
    query_words = set(re.sub(r"[^\w\s]", "", query.lower()).split())

    title_words    = set(re.sub(r"[^\w\s]", "", paper.title.lower()).split())
    abstract_words = set(re.sub(r"[^\w\s]", "", paper.abstract.lower()).split())

    title_overlap    = len(query_words & title_words) / max(len(query_words), 1)
    abstract_overlap = len(query_words & abstract_words) / max(len(query_words), 1)

    recency = 0.0
    if paper.published:
        try:
            pub_year = int(paper.published[:4])
            recency = max(0.0, (pub_year - 2018) / 7.0)   # 0→1 from 2018→2025
        except ValueError:
            pass

    return (title_overlap * 3.0) + (abstract_overlap * 1.5) + (recency * 0.5)


def _try_download_pdf(paper: Paper) -> None:
    """Download PDF to local cache. Skip silently if it fails."""
    if not paper.url:
        return
    dest = PAPERS_DIR / f"{paper.paper_id}.pdf"
    if dest.exists():
        paper.full_text = _extract_pdf_text(dest)
        return
    try:
        import urllib.request
        urllib.request.urlretrieve(paper.url, dest)
        paper.full_text = _extract_pdf_text(dest)
        logger.info(f"Downloaded: {paper.paper_id}")
    except Exception as e:
        logger.warning(f"Could not download {paper.paper_id}: {e}")
        paper.full_text = paper.abstract   # fall back to abstract


# ── User-uploaded PDF ─────────────────────────────────────────────────────────

def ingest_user_pdf(file_bytes: bytes, filename: str) -> Paper:
    """
    Accept a user-uploaded PDF, extract text, return a Paper object.
    """
    paper_id = hashlib.md5(file_bytes).hexdigest()[:12]
    dest = PAPERS_DIR / f"user_{paper_id}.pdf"
    dest.write_bytes(file_bytes)

    text = _extract_pdf_text(dest)
    # Use first 600 chars as abstract proxy
    abstract = text[:600].replace("\n", " ").strip() if text else "User-provided paper"
    title = Path(filename).stem.replace("_", " ").replace("-", " ").title()

    paper = Paper(
        paper_id=f"user_{paper_id}",
        title=title,
        authors=["User upload"],
        abstract=abstract,
        full_text=text,
        published=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        url="",
        source="user_upload",
        score=999.0,   # always included
    )
    logger.info(f"Ingested user PDF: {filename} ({len(text)} chars)")
    return paper


def _extract_pdf_text(path: Path) -> str:
    if not HAS_FITZ:
        return ""
    try:
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        logger.warning(f"PDF text extraction failed for {path}: {e}")
        return ""