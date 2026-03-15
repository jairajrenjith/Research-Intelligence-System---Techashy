"""
api/server.py
-------------
FastAPI server with Level 1 + Level 2 caching and JWT auth.

Routes:
  GET  /                        → serve frontend
  POST /api/research            → start pipeline (checks L1 cache first)
  GET  /api/stream/{id}         → SSE live progress
  GET  /api/result/{id}         → fetch completed result
  GET  /api/cached-domains      → list all L1 cached domains

  POST /api/register            → create account (L2)
  POST /api/login               → login, get token (L2)
  GET  /api/my-history          → user's saved reports (L2, auth required)
  GET  /api/my-report/{domain}  → load a saved report (L2, auth required)
  DELETE /api/my-report/{domain}→ delete a saved report (L2, auth required)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from core.pipeline import run_research_pipeline, PipelineResult
from core.cache import (
    get_cached_report, save_cached_report, list_cached_domains,
    register_user, verify_login, create_token, verify_token,
    save_user_report, get_user_report, get_user_history, delete_user_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Research Intelligence System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent.parent / "ui" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

_jobs: dict[str, dict] = {}
_progress_queues: dict[str, asyncio.Queue] = {}


def get_current_user(authorization: Optional[str] = None) -> Optional[str]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.replace("Bearer ", "").strip()
    return verify_token(token)


def require_user(authorization: Optional[str] = None) -> str:
    username = get_current_user(authorization)
    if not username:
        raise HTTPException(status_code=401, detail="Login required")
    return username


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    template = Path(__file__).parent.parent / "ui" / "templates" / "index.html"
    if template.exists():
        return HTMLResponse(template.read_text(encoding="utf-8"))
    return HTMLResponse("""
<!DOCTYPE html><html><head><title>RIS</title></head>
<body style="background:#050508;color:#e8e8f0;font-family:sans-serif;padding:2rem">
<h2>UI file not found</h2><p>Expected at: ui/templates/index.html</p>
</body></html>
""")


@app.post("/api/register")
async def register(username: str = Form(...), password: str = Form(...)):
    if len(username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    if len(password) < 4:
        raise HTTPException(400, "Password must be at least 4 characters")
    if not register_user(username, password):
        raise HTTPException(400, "Username already taken")
    token = create_token(username)
    return JSONResponse({"token": token, "username": username})


@app.post("/api/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if not verify_login(username, password):
        raise HTTPException(401, "Invalid username or password")
    token = create_token(username)
    return JSONResponse({"token": token, "username": username})


@app.post("/api/research")
async def start_research(
    domain: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    authorization: Optional[str] = Header(None),
):
    username = get_current_user(authorization)
    domain   = domain.strip()

    # L2 — user personal cache
    if username:
        user_cached = get_user_report(username, domain)
        if user_cached:
            job_id = "l2_" + str(uuid.uuid4())[:6]
            _jobs[job_id] = {"status": "done", "from_cache": "user", **user_cached}
            logger.info(f"L2 cache HIT: user='{username}' domain='{domain}'")
            return JSONResponse({"job_id": job_id, "cached": "user", "domain": domain})

    # L1 — shared domain cache (skip if user uploaded custom PDFs)
    has_uploads = any(f.filename and f.filename.lower().endswith(".pdf") for f in files)
    if not has_uploads:
        l1_cached = get_cached_report(domain)
        if l1_cached:
            job_id = "l1_" + str(uuid.uuid4())[:6]
            _jobs[job_id] = {"status": "done", "from_cache": "domain", **l1_cached}
            logger.info(f"L1 cache HIT: domain='{domain}'")
            if username:
                save_user_report(username, domain, l1_cached)
            return JSONResponse({"job_id": job_id, "cached": "domain", "domain": domain})

    # Cache miss — run full pipeline
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "running", "domain": domain, "started": time.time()}
    _progress_queues[job_id] = asyncio.Queue()

    user_pdfs = []
    for f in files:
        if f.filename and f.filename.lower().endswith(".pdf"):
            content = await f.read()
            user_pdfs.append((content, f.filename))

    asyncio.create_task(_run_job(job_id, domain, user_pdfs, username))
    return JSONResponse({"job_id": job_id, "cached": False, "domain": domain})


async def _run_job(job_id: str, domain: str, user_pdfs: list, username: Optional[str] = None):
    q = _progress_queues.get(job_id)

    def progress(msg: str):
        if q:
            try:
                q.put_nowait({"type": "progress", "message": msg})
            except asyncio.QueueFull:
                pass

    try:
        result: PipelineResult = await run_research_pipeline(
            domain=domain,
            user_pdf_bytes=user_pdfs if user_pdfs else None,
            progress_cb=progress,
        )

        report_data = {
            "status":       "done",
            "domain":       domain,
            "final_report": result.final_report,
            "pros":         result.pros_output,
            "cons":         result.cons_output,
            "future":       result.future_output,
            "papers": [
                {"title": p.title, "url": p.url, "published": p.published, "source": p.source}
                for p in result.papers
            ],
            "elapsed":      result.elapsed_sec,
            "vector_stats": result.vector_stats,
        }

        _jobs[job_id] = report_data

        # Save to L1 (only for non-custom-PDF runs)
        if not user_pdfs:
            save_cached_report(domain, report_data)

        # Save to L2 (always, if logged in)
        if username:
            save_user_report(username, domain, report_data)

        if q:
            q.put_nowait({"type": "done", "job_id": job_id})

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        _jobs[job_id] = {"status": "error", "error": str(e)}
        if q:
            q.put_nowait({"type": "error", "message": str(e)})


@app.get("/api/stream/{job_id}")
async def stream_progress(job_id: str):
    # Cached jobs — no queue, send instant done
    if job_id not in _progress_queues:
        job = _jobs.get(job_id)
        if job and job.get("status") == "done":
            async def instant():
                yield f"data: {json.dumps({'type': 'done', 'job_id': job_id})}\n\n"
            return StreamingResponse(instant(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache"})
        raise HTTPException(404, "Job not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        q = _progress_queues[job_id]
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                yield 'data: {"type":"ping"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JSONResponse(job)


@app.get("/api/cached-domains")
async def cached_domains():
    return JSONResponse({"domains": list_cached_domains()})


@app.get("/api/my-history")
async def my_history(authorization: Optional[str] = Header(None)):
    username = require_user(authorization)
    return JSONResponse({"history": get_user_history(username)})


@app.get("/api/my-report/{domain}")
async def my_report(domain: str, authorization: Optional[str] = Header(None)):
    username = require_user(authorization)
    report = get_user_report(username, domain)
    if not report:
        raise HTTPException(404, "Report not found in your history")
    job_id = "hist_" + str(uuid.uuid4())[:6]
    _jobs[job_id] = {"status": "done", "from_cache": "user_history", **report}
    return JSONResponse({"job_id": job_id})


@app.delete("/api/my-report/{domain}")
async def delete_report(domain: str, authorization: Optional[str] = Header(None)):
    username = require_user(authorization)
    if not delete_user_report(username, domain):
        raise HTTPException(404, "Report not found")
    return JSONResponse({"deleted": domain})


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "active_jobs": len([j for j in _jobs.values() if j.get("status") == "running"]),
        "cached_domains": len(list_cached_domains()),
    }