"""
cli.py
------
Run the pipeline from the command line (no web server needed).
Usage:
  python cli.py "transformer attention mechanisms"
  python cli.py "CRISPR gene editing" --pdf paper1.pdf paper2.pdf
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)  # suppress verbose logs in CLI

from core.pipeline import run_research_pipeline


def progress(msg: str):
    print(msg, flush=True)


async def main():
    parser = argparse.ArgumentParser(description="Research Intelligence CLI")
    parser.add_argument("domain", help="Research domain/topic")
    parser.add_argument("--pdf", nargs="*", help="Paths to user PDF papers", default=[])
    parser.add_argument("--output", help="Save report to this file", default=None)
    args = parser.parse_args()

    user_pdfs = []
    for pdf_path in (args.pdf or []):
        p = Path(pdf_path)
        if not p.exists():
            print(f"Warning: PDF not found: {pdf_path}")
            continue
        user_pdfs.append((p.read_bytes(), p.name))

    print(f"\n🔬 Research Intelligence System")
    print(f"   Domain: {args.domain}")
    print(f"   User PDFs: {len(user_pdfs)}\n")

    result = await run_research_pipeline(
        domain=args.domain,
        user_pdf_bytes=user_pdfs if user_pdfs else None,
        progress_cb=progress,
    )

    print("\n" + "═" * 70)
    print(result.final_report)
    print("═" * 70)
    print(f"\nPapers used: {len(result.papers)} | "
          f"Chunks: {result.vector_stats.get('total_chunks','?')} | "
          f"Time: {result.elapsed_sec}s")

    if args.output:
        out = Path(args.output)
        out.write_text(result.final_report, encoding="utf-8")
        print(f"Report saved to: {out}")


if __name__ == "__main__":
    asyncio.run(main())
