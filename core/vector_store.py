"""
core/vector_store.py
--------------------
ChromaDB-backed vector store.
- Chunks paper text with overlap
- Embeds locally using sentence-transformers (no API calls, works offline)
- Retrieves relevant chunks for each agent query
- Persists to disk so re-runs are instant
"""

import os
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from core.paper_fetcher import Paper

load_dotenv()
logger = logging.getLogger(__name__)

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
EMBED_MODEL   = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
PERSIST_DIR   = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

# ── Singleton pattern — load model once ───────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


class ResearchVectorStore:
    """
    One collection per research session (keyed by domain slug).
    Supports incremental adds — won't re-embed already-stored papers.
    """

    def __init__(self, collection_name: str):
        # Sanitise name for ChromaDB (alphanumeric + underscores only)
        safe = "".join(c if c.isalnum() else "_" for c in collection_name)
        self.collection_name = safe[:50]
        self._embedder = _get_embedder()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )
        client = _get_client()
        self._col = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"VectorStore ready: '{self.collection_name}' "
                    f"({self._col.count()} chunks already stored)")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_papers(self, papers: list[Paper]) -> int:
        """Chunk, embed, and store papers. Skips already-stored ones."""
        added = 0
        for paper in papers:
            if self._paper_already_stored(paper.paper_id):
                logger.debug(f"Skipping already-stored: {paper.paper_id}")
                continue
            chunks = self._chunk_paper(paper)
            if not chunks:
                continue
            self._embed_and_store(paper, chunks)
            added += len(chunks)
        logger.info(f"Added {added} new chunks to '{self.collection_name}'")
        return added

    def _paper_already_stored(self, paper_id: str) -> bool:
        existing = self._col.get(where={"paper_id": paper_id}, limit=1)
        return len(existing["ids"]) > 0

    def _chunk_paper(self, paper: Paper) -> list[str]:
        """Use full text if available, else fall back to abstract."""
        text = paper.full_text.strip() if paper.full_text else paper.abstract
        if not text:
            return []
        return self._splitter.split_text(text)

    def _embed_and_store(self, paper: Paper, chunks: list[str]) -> None:
        embeddings = self._embedder.encode(chunks, batch_size=32, show_progress_bar=False)
        ids        = [f"{paper.paper_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas  = [
            {
                "paper_id":  paper.paper_id,
                "title":     paper.title[:200],
                "authors":   ", ".join(paper.authors[:3]),
                "published": paper.published,
                "source":    paper.source,
                "chunk_idx": i,
            }
            for i in range(len(chunks))
        ]
        self._col.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def query(self, query_text: str, n_results: int = 8) -> list[dict]:
        """
        Return top-n relevant chunks for a query.
        Each result: {text, title, authors, published, source, distance}
        """
        if self._col.count() == 0:
            return []
        embedding = self._embedder.encode([query_text])[0].tolist()
        results = self._col.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self._col.count()),
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text":      doc,
                "title":     meta.get("title", ""),
                "authors":   meta.get("authors", ""),
                "published": meta.get("published", ""),
                "source":    meta.get("source", "arxiv"),
                "relevance": round(1 - dist, 3),   # cosine similarity
            })
        return output

    def get_context_string(self, query_text: str, n_results: int = 6) -> str:
        """Convenience: returns a formatted string ready to inject into a prompt."""
        chunks = self.query(query_text, n_results)
        if not chunks:
            return "No relevant research context available."
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[{i}] \"{c['title']}\" ({c['published']}) — {c['authors']}\n{c['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def stats(self) -> dict:
        return {
            "collection": self.collection_name,
            "total_chunks": self._col.count(),
        }
