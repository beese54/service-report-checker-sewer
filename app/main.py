import asyncio
import json
import os
import queue
import secrets
import shutil
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

# Allow importing from both the project root (pipeline.py) and app/ (checker.py)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_APP_DIR)
for _p in (_ROOT, _APP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from checker import check_all_folders  # legacy Level-1 check
from verification.verify_folders import (
    build_folder_context,
    get_ground_truth,
    get_next_folder,
    get_progress,
    save_label,
    save_skip,
)

app = FastAPI(title="WRN Service Report Checker")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# ── Upload size limit ─────────────────────────────────────────────────────────
_MAX_UPLOAD_MB    = int(os.environ.get("MAX_UPLOAD_MB", 200))
_MAX_UPLOAD_BYTES = _MAX_UPLOAD_MB * 1024 * 1024

# ── Simple password auth ──────────────────────────────────────────────────────
_APP_PASSWORD = os.environ.get("APP_PASSWORD", "wrnchecker")
_AUTH_COOKIE  = "wrn_session"
# Session token loaded from / persisted to disk so restarts don't log users out.
# _SESSION_TOKEN is assigned after OUTPUT_BASE is created (see startup section).

_PUBLIC_PATHS = {"/login", "/health"}
# Prefix-based public paths — polling endpoints must be accessible post-restart
# without a valid session so that Fix-B polling can detect job completion even
# when the server restarts and the session cookie is invalidated.
_PUBLIC_PREFIXES = ("/status/",)


def _is_authenticated(request: Request) -> bool:
    return request.cookies.get(_AUTH_COOKIE) == _SESSION_TOKEN


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if (path in _PUBLIC_PATHS
            or path.startswith("/static")
            or any(path.startswith(p) for p in _PUBLIC_PREFIXES)):
        return await call_next(request)
    if not _is_authenticated(request):
        return RedirectResponse(url="/login", status_code=302)
    return await call_next(request)


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    if _is_authenticated(request):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, password: str = Form(...)):
    if password == _APP_PASSWORD:
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key=_AUTH_COOKIE,
            value=_SESSION_TOKEN,
            httponly=True,
            samesite="lax",
        )
        return response
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Incorrect password. Please try again."},
        status_code=401,
    )


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(_AUTH_COOKIE)
    return response


# ── Persistent output directory (can be overridden by OUTPUT_DIR env var) ────
OUTPUT_BASE = os.environ.get("OUTPUT_DIR", os.path.join(_ROOT, "pipeline_output"))
os.makedirs(OUTPUT_BASE, exist_ok=True)


# ── Startup helpers ───────────────────────────────────────────────────────────

def _load_or_create_session_token() -> str:
    """Load existing session token from disk, or create and persist a new one."""
    token_path = os.path.join(OUTPUT_BASE, ".session_token")
    try:
        with open(token_path) as fh:
            tok = fh.read().strip()
            if len(tok) >= 32:
                return tok
    except FileNotFoundError:
        pass
    tok = secrets.token_urlsafe(32)
    try:
        with open(token_path, "w") as fh:
            fh.write(tok)
    except Exception:
        pass
    return tok


def _recover_crashed_jobs():
    """Mark any jobs that were 'running' at last shutdown as errored."""
    try:
        for entry in os.scandir(OUTPUT_BASE):
            if not entry.is_dir():
                continue
            jpath = os.path.join(entry.path, "job.json")
            if not os.path.isfile(jpath):
                continue
            try:
                with open(jpath, encoding="utf-8") as fh:
                    snap = json.load(fh)
                if snap.get("status") == "running":
                    snap["status"] = "error"
                    snap["error"]  = "Job interrupted by server restart"
                    with open(jpath, "w", encoding="utf-8") as fh:
                        json.dump(snap, fh)
            except Exception:
                pass
    except Exception:
        pass


JOB_MAX_AGE_SECONDS = 24 * 60 * 60  # 24 hours


def _cleanup_old_jobs():
    """Remove job directories older than JOB_MAX_AGE_SECONDS to free disk space."""
    cutoff = time.time() - JOB_MAX_AGE_SECONDS
    try:
        for entry in os.scandir(OUTPUT_BASE):
            if not entry.is_dir() or len(entry.name) != 36 or "-" not in entry.name:
                continue
            try:
                if os.path.getmtime(entry.path) < cutoff:
                    shutil.rmtree(entry.path, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


_recover_crashed_jobs()
_cleanup_old_jobs()
_SESSION_TOKEN = _load_or_create_session_token()

# Mount static assets (JS files — CSP-compliant external scripts)
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Mount output dir so browser can fetch result images
app.mount("/reports", StaticFiles(directory=OUTPUT_BASE, html=False), name="reports")

# Mount image directories for the verify UI (override via env-vars in Docker)
_ORIG_DIR     = os.environ.get("ORIGINAL_IMAGES_DIR", os.path.join(_ROOT, "original_images"))
_ANALYSIS_DIR = os.environ.get("ANALYSIS_DIR", os.path.join(_ROOT, "adjusted_images", "no_obstruction", "difference_analysis"))
if os.path.isdir(_ORIG_DIR):
    app.mount("/originals", StaticFiles(directory=_ORIG_DIR, html=False), name="originals")
if os.path.isdir(_ANALYSIS_DIR):
    app.mount("/analysis", StaticFiles(directory=_ANALYSIS_DIR, html=False), name="analysis")

# ── In-memory job registry ────────────────────────────────────────────────────
# { job_id: { "status": "running"|"done"|"error",
#             "result": dict | None,
#             "queue":  queue.Queue | None,
#             "contact_sheets": dict | None } }
_JOBS: dict[str, dict] = {}

# ── One-job-at-a-time concurrency guard ───────────────────────────────────────
_JOB_LOCK = threading.Lock()
_ACTIVE_JOB_ID: str | None = None


# ── Job state persistence ─────────────────────────────────────────────────────

def _save_job_state(job_id: str) -> None:
    """Persist minimal job metadata to disk so results survive a restart."""
    job = _JOBS.get(job_id)
    if not job:
        return
    snapshot = {
        "job_id":         job_id,
        "status":         job.get("status"),
        "error":          job.get("error"),
        "result_path":    job.get("result_path"),
        "excel_path":     job.get("excel_path"),
        "contact_sheets": job.get("contact_sheets"),
    }
    try:
        path = os.path.join(OUTPUT_BASE, job_id, "job.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh)
    except Exception:
        pass


def _load_job_from_disk(job_id: str) -> dict | None:
    """Reconstruct a job entry from its persisted job.json (post-restart)."""
    path = os.path.join(OUTPUT_BASE, job_id, "job.json")
    try:
        with open(path, encoding="utf-8") as fh:
            snap = json.load(fh)
    except Exception:
        return None
    result = None
    if snap.get("result_path"):
        try:
            with open(snap["result_path"], encoding="utf-8") as fh:
                result = json.load(fh)
        except Exception:
            pass
    return {
        "status":         snap.get("status", "error"),
        "result":         result,
        "queue":          None,   # no SSE possible after restart
        "contact_sheets": snap.get("contact_sheets") or {},
        "logs":           [],
        "error":          snap.get("error"),
        "result_path":    snap.get("result_path"),
        "excel_path":     snap.get("excel_path"),
    }


# =============================================================================
# Health check
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "ok"}


# =============================================================================
# Legacy Level-1 check (backward compat)
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "results": None, "error": None,
                        "max_upload_mb": _MAX_UPLOAD_MB}
    )


@app.post("/check", response_class=HTMLResponse)
async def check(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "results": None,
             "error": "Please upload a .zip file.", "max_upload_mb": _MAX_UPLOAD_MB},
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "upload.zip")

        # Stream to disk in 1 MB chunks — avoids loading entire ZIP into RAM
        CHUNK_SIZE = 1 * 1024 * 1024
        bytes_written = 0
        try:
            with open(zip_path, "wb") as out_f:
                while True:
                    chunk = await file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > _MAX_UPLOAD_BYTES:
                        out_f.close()
                        os.unlink(zip_path)
                        return templates.TemplateResponse(
                            "index.html",
                            {"request": request, "results": None,
                             "error": (f"ZIP exceeds {_MAX_UPLOAD_MB} MB limit. "
                                       "Split into two ZIPs and upload separately."),
                             "max_upload_mb": _MAX_UPLOAD_MB},
                        )
                    out_f.write(chunk)
        except Exception as exc:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "results": None,
                 "error": f"Upload error: {exc}", "max_upload_mb": _MAX_UPLOAD_MB},
            )

        # ZIP bomb / extraction size validation
        MAX_UNCOMPRESSED_BYTES = 600 * 1024 * 1024  # 600 MB
        extract_dir = os.path.join(tmpdir, "extracted")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                total = sum(info.file_size for info in z.infolist())
                if total > MAX_UNCOMPRESSED_BYTES:
                    return templates.TemplateResponse(
                        "index.html",
                        {"request": request, "results": None,
                         "error": f"ZIP unpacks to {total // 1024 // 1024} MB — exceeds 600 MB limit.",
                         "max_upload_mb": _MAX_UPLOAD_MB},
                    )
                compressed = os.path.getsize(zip_path)
                if compressed > 0 and (total / compressed) > 100:
                    return templates.TemplateResponse(
                        "index.html",
                        {"request": request, "results": None,
                         "error": f"Suspicious compression ratio ({total // compressed}:1). Rejected.",
                         "max_upload_mb": _MAX_UPLOAD_MB},
                    )
                z.extractall(extract_dir)
        except zipfile.BadZipFile:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "results": None,
                 "error": "Invalid ZIP file.", "max_upload_mb": _MAX_UPLOAD_MB},
            )

        entries  = [e for e in os.scandir(extract_dir) if e.is_dir()]
        base_dir = entries[0].path if len(entries) == 1 else extract_dir
        results  = check_all_folders(base_dir)

    summary = {
        "total":          len(results),
        "no_obstruction": sum(1 for r in results if r["result"] == "no_obstruction"),
        "obstruction":    sum(1 for r in results if r["result"] == "obstruction"),
    }
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": results, "summary": summary, "error": None,
         "max_upload_mb": _MAX_UPLOAD_MB},
    )


# =============================================================================
# Full pipeline — POST /run
# =============================================================================

@app.post("/run")
async def run_pipeline_endpoint(file: UploadFile = File(...)):
    """Accept a ZIP, start the full pipeline in a background thread."""
    global _ACTIVE_JOB_ID

    if not file.filename.endswith(".zip"):
        return JSONResponse(status_code=400, content={"error": "Please upload a .zip file."})

    # One-job-at-a-time guard — prevents OOM on concurrent pipeline runs
    if not _JOB_LOCK.acquire(blocking=False):
        return JSONResponse(status_code=503, content={
            "error": "A pipeline job is already running. Please wait for it to finish.",
            "active_job_id": _ACTIVE_JOB_ID,
        })

    job_id  = str(uuid.uuid4())
    _ACTIVE_JOB_ID = job_id
    job_dir = os.path.join(OUTPUT_BASE, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Stream ZIP to disk in 1 MB chunks — avoids loading entire ZIP into RAM
    zip_path = os.path.join(job_dir, "upload.zip")
    CHUNK_SIZE = 1 * 1024 * 1024
    bytes_written = 0
    try:
        with open(zip_path, "wb") as out_f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > _MAX_UPLOAD_BYTES:
                    out_f.close()
                    os.unlink(zip_path)
                    _ACTIVE_JOB_ID = None
                    _JOB_LOCK.release()
                    return JSONResponse(
                        status_code=400,
                        content={"error": (f"ZIP exceeds {_MAX_UPLOAD_MB} MB limit. "
                                           "Split into two ZIPs and upload separately.")},
                    )
                out_f.write(chunk)
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        _ACTIVE_JOB_ID = None
        _JOB_LOCK.release()
        raise exc

    # Prepare job entry
    q: queue.Queue = queue.Queue()
    _JOBS[job_id] = {
        "status": "running", "result": None, "queue": q, "contact_sheets": None,
        "logs": [],   # all progress messages stored for post-mortem retrieval
    }

    # Launch background thread
    t = threading.Thread(
        target=_pipeline_worker,
        args=(job_id, zip_path, job_dir),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id}


def _pipeline_worker(job_id: str, zip_path: str, job_dir: str):
    """Background thread: extract ZIP, run pipeline in subprocess, push SSE events.

    The pipeline runs in a subprocess so that cv2/skimage/numpy/PyTorch memory is
    fully returned to the OS when the subprocess exits — preventing accumulation
    across multiple jobs within the same FastAPI process.
    """
    import glob as _glob
    import subprocess
    import traceback
    global _ACTIVE_JOB_ID
    q = _JOBS[job_id]["queue"]

    def push(msg: str):
        _JOBS[job_id]["logs"].append(msg)
        q.put(msg)

    # 30-minute timeout — marks job as error and releases lock so new jobs can start
    PIPELINE_TIMEOUT_SECONDS = 30 * 60
    _proc = None   # subprocess handle — used by timeout to kill it

    def _timeout_handler():
        if _JOBS.get(job_id, {}).get("status") == "running":
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["error"]  = "Pipeline timed out after 30 minutes"
            _save_job_state(job_id)
            q.put("ERROR:Pipeline timed out after 30 minutes")
            if _proc is not None:
                try:
                    _proc.kill()
                except Exception:
                    pass

    _timeout_timer = threading.Timer(PIPELINE_TIMEOUT_SECONDS, _timeout_handler)
    _timeout_timer.daemon = True
    _timeout_timer.start()

    # Write initial job.json so crash recovery can mark it as error on restart
    _save_job_state(job_id)

    try:
        # ZIP bomb / extraction size validation
        MAX_UNCOMPRESSED_BYTES = 600 * 1024 * 1024  # 600 MB
        with zipfile.ZipFile(zip_path, "r") as z:
            total = sum(info.file_size for info in z.infolist())
            if total > MAX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"ZIP unpacks to {total // 1024 // 1024} MB — exceeds 600 MB limit."
                )
            compressed = os.path.getsize(zip_path)
            if compressed > 0 and (total / compressed) > 100:
                raise ValueError(
                    f"Suspicious compression ratio ({total // compressed}:1). Rejected."
                )
            extract_dir = os.path.join(job_dir, "input")
            z.extractall(extract_dir)

        # Resolve base_dir:
        # - Multiple dirs at root  → batch of job folders, use extract_dir
        # - Single dir that itself contains subdirs → batch wrapper, step in
        # - Single dir that contains only files (images) → treat extract_dir as job folder
        # - No dirs at root (images at top level) → treat extract_dir itself as job folder
        entries = [e for e in os.scandir(extract_dir) if e.is_dir()]
        if len(entries) == 1:
            inner_dirs = [e for e in os.scandir(entries[0].path) if e.is_dir()]
            base_dir = entries[0].path if inner_dirs else extract_dir
        else:
            base_dir = extract_dir

        output_dir = os.path.join(job_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        push("STATUS:Classifying folders...")

        # ── Run pipeline in a subprocess ──────────────────────────────────────
        # Heavy libs (cv2, skimage, numpy, PyTorch) are loaded inside the
        # subprocess.  When it exits the OS reclaims all of its memory,
        # keeping the FastAPI process lean across multiple jobs.
        runner = os.path.join(_ROOT, "pipeline.py")
        cmd    = [sys.executable, runner, "--input", base_dir, "--output", output_dir]
        _proc  = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr so errors appear in SSE log
            text=True,
            env=os.environ.copy(),
        )

        # Relay each stdout line to the SSE queue in real time
        for line in _proc.stdout:
            line = line.rstrip("\n")
            if line:
                push(line)

        _proc.wait()
        if _proc.returncode != 0:
            raise RuntimeError(
                f"Pipeline subprocess exited with code {_proc.returncode}. "
                "See logs for details."
            )

        # Find result JSON written by pipeline.py
        result_files = sorted(_glob.glob(os.path.join(output_dir, "pipeline_run_*.json")))
        if not result_files:
            raise RuntimeError("Pipeline completed but no result JSON was written.")

        result_path = result_files[-1]
        with open(result_path, encoding="utf-8") as fh:
            result = json.load(fh)
        result["report_path"] = result_path

        _JOBS[job_id]["result"]      = result
        _JOBS[job_id]["status"]      = "done"
        _JOBS[job_id]["result_path"] = result_path
        _save_job_state(job_id)

        # Contact sheets in main process (PIL only — much lighter)
        try:
            def _mem():
                try:
                    for line in open("/proc/self/status"):
                        if line.startswith("VmRSS:"):
                            return f"{int(line.split()[1]) // 1024} MB"
                except Exception:
                    pass
                return "?"
            push(f"STATUS:Generating contact sheets...  mem={_mem()}")
            from app.contact_sheets import generate_contact_sheets
            sheet_paths = generate_contact_sheets(result, job_dir, base_dir)
            _JOBS[job_id]["contact_sheets"] = sheet_paths
            _save_job_state(job_id)
            push(f"STATUS:Contact sheets done  mem={_mem()}")
        except Exception as sheet_exc:
            push(f"STATUS:Contact sheets skipped -- {sheet_exc}")

        # Clean up input dir + ZIP after contact sheets complete
        try:
            if os.path.isdir(os.path.join(job_dir, "input")):
                shutil.rmtree(os.path.join(job_dir, "input"), ignore_errors=True)
            if os.path.isfile(zip_path):
                os.unlink(zip_path)
        except Exception:
            pass

        _timeout_timer.cancel()
        push(f"DONE:{job_id}")

    except BaseException as exc:
        _timeout_timer.cancel()
        tb = traceback.format_exc()
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"]  = f"{type(exc).__name__}: {exc}"
        _JOBS[job_id]["logs"].append(f"TRACEBACK:\n{tb}")
        _save_job_state(job_id)
        push(f"ERROR:{type(exc).__name__}: {exc}")

    finally:
        _ACTIVE_JOB_ID = None
        try:
            _JOB_LOCK.release()
        except RuntimeError:
            pass
        _JOBS.pop(job_id, None)
        import gc
        gc.collect()


# =============================================================================
# SSE stream — GET /stream/{job_id}
# =============================================================================

@app.get("/stream/{job_id}")
async def stream(request: Request, job_id: str):
    """Server-Sent Events stream — one event per pipeline progress message."""
    if job_id not in _JOBS:
        disk_job = _load_job_from_disk(job_id)
        if disk_job is None:
            return JSONResponse(status_code=404, content={"error": "Job not found"})
        _JOBS[job_id] = disk_job

    job = _JOBS[job_id]
    # Job was loaded from disk after restart — SSE is no longer possible
    if job.get("queue") is None:
        return JSONResponse(status_code=410, content={
            "error": "Job already completed. View results at /results/" + job_id
        })

    async def generator():
        q = _JOBS[job_id]["queue"]
        idle_ticks = 0  # counts 0.5 s ticks with no message
        while True:
            if await request.is_disconnected():
                break
            try:
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: q.get(timeout=0.5)
                )
                idle_ticks = 0
                if msg.startswith("PROGRESS "):
                    yield {"event": "progress", "data": msg[len("PROGRESS "):]}
                elif msg.startswith("DONE:"):
                    yield {"event": "done", "data": msg[len("DONE:"):]}
                    break
                elif msg.startswith("ERROR:"):
                    yield {"event": "error", "data": msg[len("ERROR:"):]}
                    break
                elif msg.startswith("STATUS:"):
                    yield {"event": "status", "data": msg[len("STATUS:"):]}
                elif msg.startswith("INIT "):
                    yield {"event": "init", "data": msg[len("INIT "):]}
                else:
                    yield {"event": "log", "data": msg}
            except Exception:
                # queue.get timeout — check job status
                status = _JOBS[job_id].get("status")
                if status == "done":
                    yield {"event": "done", "data": job_id}
                    break
                elif status == "error":
                    yield {"event": "error",
                           "data": _JOBS[job_id].get("error", "Unknown error")}
                    break
                else:
                    # Still running — send a keepalive ping every 10 s so
                    # any upstream proxy does not close the idle connection.
                    idle_ticks += 1
                    if idle_ticks % 20 == 0:   # 20 × 0.5 s = 10 s
                        yield {"event": "ping", "data": ""}

    return EventSourceResponse(generator())


# =============================================================================
# Results dashboard — GET /results/{job_id}
# =============================================================================

@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results(request: Request, job_id: str):
    if job_id not in _JOBS:
        disk_job = _load_job_from_disk(job_id)
        if disk_job is None:
            return HTMLResponse("<h2>Job not found.</h2>", status_code=404)
        _JOBS[job_id] = disk_job

    job = _JOBS[job_id]
    if job["status"] == "running":
        logs_html = "".join(f"<div>{l}</div>" for l in job.get("logs", [])[-30:])
        return HTMLResponse(
            f"<meta http-equiv='refresh' content='5'>"
            f"<h2>Job still running — auto-refreshing…</h2>"
            f"<p><a href='/logs/{job_id}'>View raw logs (JSON)</a></p>"
            f"<pre style='background:#111;color:#ccc;padding:1rem;'>{logs_html}</pre>"
        )
    if job["status"] == "error":
        return HTMLResponse(
            f"<h2>Pipeline error: {job.get('error', 'unknown')}</h2>"
            f"<p><a href='/logs/{job_id}'>Full logs</a></p>",
            status_code=200,
        )

    result         = job["result"]
    contact_sheets = job.get("contact_sheets") or {}

    if result is None:
        return HTMLResponse(
            f"<h2>Results unavailable for job {job_id}.</h2>"
            f"<p>The result file may have been cleaned up. "
            f"Check <a href='/logs/{job_id}'>logs</a> for details.</p>",
            status_code=404,
        )

    # Build URL-friendly paths relative to /reports mount
    def _sheet_url(abs_path: str | None) -> str | None:
        if not abs_path:
            return None
        try:
            rel = os.path.relpath(abs_path, OUTPUT_BASE).replace("\\", "/")
            return f"/reports/{rel}"
        except Exception:
            return None

    sheet_urls = {k: _sheet_url(v) for k, v in contact_sheets.items()}

    try:
        return templates.TemplateResponse(
            "pipeline_results.html",
            {
                "request":    request,
                "job_id":     job_id,
                "result":     result,
                "sheet_urls": sheet_urls,
            },
        )
    except Exception as exc:
        return HTMLResponse(
            f"<h2>Error rendering results for job {job_id}.</h2>"
            f"<pre>{type(exc).__name__}: {exc}</pre>"
            f"<p><a href='/logs/{job_id}'>View raw logs</a></p>",
            status_code=500,
        )


# =============================================================================
# Job status JSON — GET /status/{job_id}
# =============================================================================

@app.get("/status/{job_id}")
async def job_status(job_id: str):
    if job_id not in _JOBS:
        disk_job = _load_job_from_disk(job_id)
        if disk_job is None:
            return JSONResponse(status_code=404, content={"error": "Job not found"})
        _JOBS[job_id] = disk_job
    job = _JOBS[job_id]
    return {
        "job_id":     job_id,
        "status":     job["status"],
        "has_result": job["result"] is not None,
    }


@app.get("/logs/{job_id}")
async def job_logs(job_id: str):
    """Return all stored log lines for a job — useful after SSE disconnects."""
    if job_id not in _JOBS:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    job = _JOBS[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "error":  job.get("error"),
        "logs":   job.get("logs", []),
    }


# =============================================================================
# Job lock status — GET /job-lock-status
# =============================================================================

@app.get("/job-lock-status")
async def job_lock_status():
    locked = not _JOB_LOCK.acquire(blocking=False)
    if not locked:
        _JOB_LOCK.release()
    return {"busy": locked, "active_job_id": _ACTIVE_JOB_ID if locked else None}


# =============================================================================
# Human verification UI — GET /verify, POST /verify/submit, GET /verify/progress
# =============================================================================

@app.get("/verify", response_class=HTMLResponse)
async def verify_ui(request: Request, folder: str | None = None):
    """Render the human-verification page for one folder at a time."""
    gt  = get_ground_truth()
    fid = folder or get_next_folder(gt)
    if fid is None:
        return HTMLResponse(
            "<h2 style='font-family:sans-serif;padding:2rem;'>All folders reviewed!</h2>"
        )
    ctx  = build_folder_context(fid)
    prog = get_progress()
    return templates.TemplateResponse(
        "verify.html",
        {"request": request, "ctx": ctx, "progress": prog},
    )


@app.post("/verify/submit")
async def verify_submit(
    request:      Request,
    folder:       str = Form(...),
    action:       str = Form(...),         # "agree" | "disagree" | "skip"
    true_verdict: str = Form(""),
    failed_stage: str = Form(""),
    notes:        str = Form(""),
):
    """Save a labelling decision and redirect to the next folder."""
    if action == "skip":
        save_skip(folder)
    else:
        agreed = action == "agree"
        save_label(
            folder_id    = folder,
            agreed       = agreed,
            true_verdict = true_verdict or None,
            failed_stage = failed_stage or None,
            notes        = notes,
        )

    gt       = get_ground_truth()
    next_fid = get_next_folder(gt)
    if next_fid:
        return RedirectResponse(url=f"/verify?folder={next_fid}", status_code=303)
    return RedirectResponse(url="/verify", status_code=303)


@app.get("/verify/progress")
async def verify_progress():
    """Return labelling progress as JSON."""
    return get_progress()
