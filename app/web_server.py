# app/web_server.py
# -*- coding: utf-8 -*-
"""
TRIM Web Server — FastAPI backend
Fully aligned with the latest trim_gui.py (PipelineRunner class, new Options fields).
Enhanced with improved error handling, time filters, and resume capability.

Install:
    pip install fastapi uvicorn sse-starlette

Run:
    python run.py          ← recommended (auto-opens browser)
    python app/web_server.py   ← backend only
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, Generator, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

# ── Project root in sys.path ──────────────────────────────
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Reuse existing pipeline code ──────────────────────────
from app.bindings import build_entrypoints  # noqa: E402
from app.trim_gui import (  # noqa: E402
    AppState, AppRunStatus, Inputs, Options,
    StepRunStatus, Step, PipelineRunner,
    MAINLINE_STEPS, OPTIONAL_STEPS,
    build_pipeline, ensure_out_dirs, step_out_dir,
    manifest_append, sanitize_prefix, cleanup_intermediate_files,
    EventType, Event,
)

SCRIPT_ENTRYPOINTS = build_entrypoints()

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(title="TRIM Web API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebRunner: wraps PipelineRunner for SSE streaming ────
class WebRunner:
    def __init__(self):
        self.state = AppState()
        self.q: queue.Queue[Event] = queue.Queue()
        self._runner: Optional[PipelineRunner] = None

    def is_running(self) -> bool:
        return self._runner is not None and self._runner.is_running()

    def can_resume(self) -> bool:
        """Check if there's a paused pipeline that can be resumed."""
        return (
                self.state.status == AppRunStatus.PAUSED
                and len(self.state.last_pipeline) > 0
                and self.state.resume_from_index < len(self.state.last_pipeline)
        )

    def stop(self) -> None:
        if self._runner:
            self._runner.stop()

    def start(self, steps: List[Step], selected_only: Optional[List[str]]) -> None:
        if self.is_running():
            return

        print("\n[DEBUG] WebRunner.start() called")
        print(f"  Number of steps: {len(steps)}")
        print(f"  Selected only: {selected_only}")

        # Don't reset state - it's already populated by populate_state()!
        # self.state = AppState()  ← 删除这行
        self.q = queue.Queue()
        self.state.stop_requested = False

        print("[DEBUG] Creating PipelineRunner...")
        print(f"[DEBUG] State.inputs.out_dir = {self.state.inputs.out_dir}")
        self._runner = PipelineRunner(self.state, self.q)

        print("[DEBUG] Calling runner.run()...")
        try:
            self._runner.run(steps, selected_only)
            print("[DEBUG] runner.run() returned (thread started)")
        except Exception as e:
            print(f"[ERROR] Exception in runner.run(): {e}")
            import traceback
            traceback.print_exc()

    def resume(self) -> None:
        """Resume from last paused position."""
        if not self.can_resume() or self.is_running():
            return
        if not self._runner:
            self._runner = PipelineRunner(self.state, self.q)
        self._runner.resume()


_runner = WebRunner()

# ── Token store: POST params → short UUID token for SSE ──
_pending: Dict[str, dict] = {}


# ── Populate AppState from parsed JSON params ─────────────
def populate_state(state: AppState, p: dict) -> None:
    """Populate AppState from request parameters."""
    # ── Inputs ───────────────────────────────────────────
    state.inputs.net_xml = Path(p["net_xml"]) if p.get("net_xml") else None
    state.inputs.fcd_input = Path(p["fcd_input"]) if p.get("fcd_input") else None
    state.inputs.out_dir = Path(p["out_dir"]) if p.get("out_dir") else None

    state.inputs.import_gt = bool(p.get("import_gt", False))
    state.inputs.gt_csv = Path(p["gt_csv"]) if p.get("gt_csv") else None

    state.inputs.raw_prefix = p.get("prefix", "fcd")
    state.inputs.safe_prefix = sanitize_prefix(state.inputs.raw_prefix, fallback="fcd")

    # ── Options ──────────────────────────────────────────
    pols = p.get("pollutants", ["NOx"])
    state.options.pollutants = {k: k in pols for k in ["NOx", "PM2.5", "CO", "CO2", "THC"]}

    state.options.enable_plots = bool(p.get("enable_plots", False))
    state.options.plot_limit = int(p.get("plot_limit", 10))

    # Spatial options
    state.options.spatial_axis = p.get("spatial_axis", "both")
    state.options.spatial_enable_roi = bool(p.get("spatial_enable_roi", False))
    state.options.spatial_roi_mode = p.get("spatial_roi_mode", "shared")
    state.options.spatial_roi_sim = p.get("spatial_roi_sim", "")
    state.options.spatial_roi_gt = p.get("spatial_roi_gt", "")
    # backward-compat alias
    state.options.spatial_roi_polygon = state.options.spatial_roi_sim

    try:
        state.options.axis_bin_size = max(0.1, float(p.get("axis_bin_size", 10.0)))
    except Exception:
        state.options.axis_bin_size = 10.0
    state.options.grid_cell_m = state.options.axis_bin_size

    # Time filter options (NEW)
    state.options.enable_sumo_time_filter = bool(p.get("enable_sumo_time_filter", False))
    state.options.enable_gt_time_filter = bool(p.get("enable_gt_time_filter", False))

    # Parse time values - handle None/empty strings
    try:
        val = p.get("sumo_time_start")
        state.options.sumo_time_start = float(val) if val not in (None, "", "None") else None
    except (ValueError, TypeError):
        state.options.sumo_time_start = None

    try:
        val = p.get("sumo_time_end")
        state.options.sumo_time_end = float(val) if val not in (None, "", "None") else None
    except (ValueError, TypeError):
        state.options.sumo_time_end = None

    try:
        val = p.get("gt_time_start")
        state.options.gt_time_start = float(val) if val not in (None, "", "None") else None
    except (ValueError, TypeError):
        state.options.gt_time_start = None

    try:
        val = p.get("gt_time_end")
        state.options.gt_time_end = float(val) if val not in (None, "", "None") else None
    except (ValueError, TypeError):
        state.options.gt_time_end = None

    # ROI text (if used elsewhere)
    state.options.roi_text = p.get("roi_text", "")


# ── Validate populated state, return list of error strings
def validate_state(state: AppState) -> List[str]:
    """Validate AppState and return list of validation errors."""
    errors = []

    # Debug: Print validation checks
    print("\n[DEBUG] Starting validation...")

    # Required inputs
    if not state.inputs.net_xml:
        errors.append("net.xml path is not set")
        print(f"  [FAIL] net.xml path is not set")
    elif not Path(state.inputs.net_xml).exists():
        errors.append(f"net.xml file does not exist: {state.inputs.net_xml}")
        print(f"  [FAIL] net.xml file does not exist: {state.inputs.net_xml}")
        print(f"        Resolved path: {Path(state.inputs.net_xml).resolve()}")
    else:
        print(f"  [PASS] net.xml exists: {state.inputs.net_xml}")

    if not state.inputs.fcd_input:
        errors.append("fcd.xml path is not set")
        print(f"  [FAIL] fcd.xml path is not set")
    elif not Path(state.inputs.fcd_input).exists():
        errors.append(f"FCD input file does not exist: {state.inputs.fcd_input}")
        print(f"  [FAIL] FCD input file does not exist: {state.inputs.fcd_input}")
        print(f"        Resolved path: {Path(state.inputs.fcd_input).resolve()}")
    else:
        print(f"  [PASS] fcd.xml exists: {state.inputs.fcd_input}")

    if not state.inputs.out_dir:
        errors.append("output directory is not set")
        print(f"  [FAIL] output directory is not set")
    else:
        print(f"  [PASS] output directory set: {state.inputs.out_dir}")

    # Ground truth validation
    if state.inputs.import_gt:
        if not state.inputs.gt_csv:
            errors.append("Ground Truth CSV path is not set")
            print(f"  [FAIL] GT CSV path is not set")
        elif not Path(state.inputs.gt_csv).exists():
            errors.append(f"Ground Truth CSV file does not exist: {state.inputs.gt_csv}")
            print(f"  [FAIL] GT CSV file does not exist: {state.inputs.gt_csv}")
        else:
            print(f"  [PASS] GT CSV exists: {state.inputs.gt_csv}")
    else:
        print(f"  [SKIP] GT validation (not enabled)")

    # Time filter validation
    if state.options.enable_sumo_time_filter:
        start = state.options.sumo_time_start
        end = state.options.sumo_time_end
        if start is not None and end is not None and start >= end:
            errors.append("SUMO time filter: start time must be less than end time")
            print(f"  [FAIL] SUMO time filter invalid: {start} >= {end}")
        else:
            print(f"  [PASS] SUMO time filter valid: {start} -> {end}")
    else:
        print(f"  [SKIP] SUMO time filter (not enabled)")

    if state.options.enable_gt_time_filter:
        start = state.options.gt_time_start
        end = state.options.gt_time_end
        if start is not None and end is not None and start >= end:
            errors.append("GT time filter: start time must be less than end time")
            print(f"  [FAIL] GT time filter invalid: {start} >= {end}")
        else:
            print(f"  [PASS] GT time filter valid: {start} -> {end}")
    else:
        print(f"  [SKIP] GT time filter (not enabled)")

    # Pollutants validation
    if not any(state.options.pollutants.values()):
        errors.append("At least one pollutant must be selected")
        print(f"  [FAIL] No pollutants selected")
    else:
        selected = [k for k, v in state.options.pollutants.items() if v]
        print(f"  [PASS] Pollutants selected: {selected}")

    print(f"[DEBUG] Validation complete. Errors: {len(errors)}")
    if errors:
        for i, err in enumerate(errors, 1):
            print(f"  Error {i}: {err}")

    return errors


# ── SSE generator: drain the event queue ──────────────────
def _event_stream(runner: WebRunner) -> Generator[str, None, None]:
    """Generate Server-Sent Events from the runner's event queue."""
    print("[DEBUG] _event_stream started")
    done = False
    heartbeat_counter = 0
    event_count = 0
    events_buffer = []

    while not done:
        try:
            ev: Event = runner.q.get(timeout=0.2)
            event_count += 1
            print(
                f"[DEBUG] Event #{event_count}: type={ev.type.value}, step_id={ev.step_id}, message={ev.message[:100] if ev.message else 'None'}...")

            # If this is an error log, print it to terminal
            if ev.type == EventType.LOG and "Traceback" in (ev.message or ""):
                print(f"\n{'=' * 60}")
                print(f"[ERROR] Pipeline execution error:")
                print(f"{'=' * 60}")
                print(ev.message)
                print(f"{'=' * 60}\n")

            data = {
                "type": ev.type.value,
                "step_id": ev.step_id,
                "message": ev.message,
                "path": ev.path,
            }

            # Buffer event
            events_buffer.append((ev, data))

            # If pipeline_done, wait a moment for any trailing LOG events
            if ev.type in (EventType.PIPELINE_DONE, EventType.PIPELINE_STOPPED):
                print("[DEBUG] Pipeline done/stopped event received, waiting for trailing events...")
                time.sleep(0.5)  # Wait for any LOG events

                # Drain remaining events
                while True:
                    try:
                        trailing_ev = runner.q.get(timeout=0.1)
                        event_count += 1
                        print(
                            f"[DEBUG] Trailing event #{event_count}: type={trailing_ev.type.value}, message={trailing_ev.message[:100] if trailing_ev.message else 'None'}...")

                        if trailing_ev.type == EventType.LOG and "Traceback" in (trailing_ev.message or ""):
                            print(f"\n{'=' * 60}")
                            print(f"[ERROR] Pipeline execution error (trailing):")
                            print(f"{'=' * 60}")
                            print(trailing_ev.message)
                            print(f"{'=' * 60}\n")

                        trailing_data = {
                            "type": trailing_ev.type.value,
                            "step_id": trailing_ev.step_id,
                            "message": trailing_ev.message,
                            "path": trailing_ev.path,
                        }
                        events_buffer.append((trailing_ev, trailing_data))
                    except queue.Empty:
                        break

                done = True

        except queue.Empty:
            # Send periodic heartbeat to keep connection alive
            heartbeat_counter += 1
            if heartbeat_counter % 5 == 0:  # Every ~1 second
                yield ": heartbeat\n\n"

            # Debug: Check if runner is still alive
            if heartbeat_counter % 50 == 0:  # Every ~10 seconds
                is_running = runner.is_running()
                print(f"[DEBUG] Heartbeat check (10s): runner.is_running()={is_running}, events_received={event_count}")

    # Send all buffered events
    print(f"[DEBUG] Sending {len(events_buffer)} buffered events to client...")
    for ev, data in events_buffer:
        yield f"data: {json.dumps(data)}\n\n"

    print(f"[DEBUG] Pipeline finished. Total events: {event_count}")
    print("[DEBUG] _event_stream finished")


# ── File/Directory dialog API ─────────────────────────────
# Uses tkinter (bundled with Python on Windows/Linux) to open
# a native OS file-picker dialog when the user clicks Browse.

def _pick_file(filetypes: list) -> str:
    """Open file picker dialog and return selected path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(filetypes=filetypes, parent=root)
        root.destroy()
        return path or ""
    except Exception as e:
        print(f"[ERROR] File picker failed: {e}")
        return ""


def _pick_directory() -> str:
    """Open directory picker dialog and return selected path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(parent=root)
        root.destroy()
        return path or ""
    except Exception as e:
        print(f"[ERROR] Directory picker failed: {e}")
        return ""


import asyncio


@app.get("/api/browse/net_xml")
async def browse_net_xml():
    """Browse for net.xml file."""
    path = await asyncio.get_event_loop().run_in_executor(
        None, _pick_file, [("SUMO Network", "*.xml"), ("All files", "*.*")]
    )
    return {"path": path}


@app.get("/api/browse/fcd_xml")
async def browse_fcd_xml():
    """Browse for FCD XML file."""
    path = await asyncio.get_event_loop().run_in_executor(
        None, _pick_file, [("SUMO FCD", "*.xml"), ("All files", "*.*")]
    )
    return {"path": path}


@app.get("/api/browse/gt_csv")
async def browse_gt_csv():
    """Browse for Ground Truth CSV file."""
    path = await asyncio.get_event_loop().run_in_executor(
        None, _pick_file, [("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return {"path": path}


@app.get("/api/browse/output_dir")
async def browse_output_dir():
    """Browse for output directory."""
    path = await asyncio.get_event_loop().run_in_executor(
        None, _pick_directory
    )
    return {"path": path}


# ── Pipeline API ──────────────────────────────────────────

@app.post("/api/prepare")
async def prepare(request: Request):
    """
    Accept full pipeline params as JSON body (POST avoids URL-length
    limits on long Windows paths) and return a short UUID token.
    """
    try:
        p = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    # Debug: print to terminal so you can verify paths are received
    print("\n[DEBUG] /api/prepare received:")
    print(f"  net_xml   = {repr(p.get('net_xml'))}")
    print(f"  fcd_input = {repr(p.get('fcd_input'))}")
    print(f"  out_dir   = {repr(p.get('out_dir'))}")
    print(f"  prefix    = {repr(p.get('prefix'))}")
    print(f"  pollutants = {p.get('pollutants', [])}")
    if p.get('enable_sumo_time_filter'):
        print(f"  SUMO time filter: {p.get('sumo_time_start')} → {p.get('sumo_time_end')}")
    if p.get('enable_gt_time_filter'):
        print(f"  GT time filter: {p.get('gt_time_start')} → {p.get('gt_time_end')}")

    token = str(uuid.uuid4())
    _pending[token] = p
    return {"token": token}


@app.get("/api/run")
async def run(token: str):
    """Start the pipeline using a pre-stored token, stream SSE events."""
    if _runner.is_running():
        return JSONResponse(
            content={"error": "Pipeline already running"},
            status_code=409
        )

    p = _pending.pop(token, None)
    if p is None:
        return JSONResponse(
            content={"error": "Invalid or expired token — please try again"},
            status_code=400
        )

    # Preserve existing state if resuming
    if p.get("_resume"):
        # Skip re-population for resume
        pass
    else:
        populate_state(_runner.state, p)

    # Validate before starting
    errors = validate_state(_runner.state)
    if errors:
        # Return a one-shot SSE stream with the validation errors
        def _err_stream():
            for msg in errors:
                yield f"data: {json.dumps({'type': 'step_error', 'step_id': '', 'message': f'[Validation] {msg}', 'path': None})}\n\n"
            yield f"data: {json.dumps({'type': 'pipeline_done', 'step_id': '', 'message': '=== Pipeline error (validation failed) ===', 'path': None})}\n\n"

        return StreamingResponse(
            _err_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    use_sg = bool(p.get("use_sg", False))

    print(f"\n[DEBUG] Building pipeline (enable_sg={use_sg})...")
    try:
        steps = build_pipeline(_runner.state, enable_sg=use_sg)
        print(f"[DEBUG] Pipeline built successfully. Steps: {[s.step_id for s in steps]}")
    except Exception as e:
        print(f"[ERROR] Failed to build pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise

    selected_only = p.get("selected_only") or None
    print(f"[DEBUG] Selected steps: {selected_only if selected_only else 'All'}")

    print("[DEBUG] Starting pipeline execution...")
    _runner.start(steps, selected_only)
    print("[DEBUG] Pipeline execution thread started")

    print("[DEBUG] Returning SSE event stream...")
    return StreamingResponse(
        _event_stream(_runner),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/resume")
async def resume():
    """Resume pipeline from last paused position."""
    if _runner.is_running():
        return JSONResponse(
            content={"error": "Pipeline is already running"},
            status_code=409
        )

    if not _runner.can_resume():
        return JSONResponse(
            content={"error": "No paused pipeline to resume"},
            status_code=400
        )

    print(f"\n[DEBUG] Resuming pipeline from index {_runner.state.resume_from_index}")

    _runner.resume()

    return StreamingResponse(
        _event_stream(_runner),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/stop")
async def stop():
    """Stop the currently running pipeline."""
    _runner.stop()
    return {"ok": True, "message": "Stop request sent"}


@app.get("/api/status")
async def status():
    """Get current pipeline status."""
    return {
        "app_status": _runner.state.status.value,
        "step_status": {k: v.value for k, v in _runner.state.step_status.items()},
        "is_running": _runner.is_running(),
        "can_resume": _runner.can_resume(),
        "resume_from": _runner.state.resume_from_index if _runner.can_resume() else None,
        "last_error": _runner.state.last_error if _runner.state.last_error else None,
    }


@app.get("/api/config")
async def get_config():
    """Get current configuration state."""
    return {
        "inputs": {
            "net_xml": str(_runner.state.inputs.net_xml) if _runner.state.inputs.net_xml else None,
            "fcd_input": str(_runner.state.inputs.fcd_input) if _runner.state.inputs.fcd_input else None,
            "out_dir": str(_runner.state.inputs.out_dir) if _runner.state.inputs.out_dir else None,
            "import_gt": _runner.state.inputs.import_gt,
            "gt_csv": str(_runner.state.inputs.gt_csv) if _runner.state.inputs.gt_csv else None,
            "prefix": _runner.state.inputs.raw_prefix,
        },
        "options": {
            "pollutants": _runner.state.options.pollutants,
            "enable_plots": _runner.state.options.enable_plots,
            "plot_limit": _runner.state.options.plot_limit,
            "spatial_axis": _runner.state.options.spatial_axis,
            "axis_bin_size": _runner.state.options.axis_bin_size,
            "enable_sumo_time_filter": _runner.state.options.enable_sumo_time_filter,
            "sumo_time_start": _runner.state.options.sumo_time_start,
            "sumo_time_end": _runner.state.options.sumo_time_end,
            "enable_gt_time_filter": _runner.state.options.enable_gt_time_filter,
            "gt_time_start": _runner.state.options.gt_time_start,
            "gt_time_end": _runner.state.options.gt_time_end,
        }
    }


# ── Health check ──────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0"}


# ── Serve trim.html (single-file frontend) ───────────────
HTML_FILE = PROJECT_ROOT / "trim.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML interface."""
    if HTML_FILE.exists():
        return HTMLResponse(content=HTML_FILE.read_text(encoding="utf-8"))
    return HTMLResponse(content="""
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TRIM Web</title></head>
<body style="background:#f4f7f9;color:#0f2d3d;font-family:sans-serif;padding:40px">
  <h2>TRIM Web API is running.</h2>
  <p>Place <code>trim.html</code> in the project root (<code>TRIM_GUI/trim.html</code>)
     and restart the server to see the UI.</p>
  <p>API docs: <a href="/docs">/docs</a></p>
  <p>Health check: <a href="/api/health">/api/health</a></p>
</body></html>
""", status_code=200)


if __name__ == "__main__":
    import uvicorn
    import os
    import logging
    import warnings

    # Suppress connection timeout warnings
    warnings.filterwarnings("ignore", message=".*No events received.*")
    warnings.filterwarnings("ignore", message=".*connection may be stalled.*")

    # Configure logging to suppress verbose uvicorn connection warnings
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Get port from environment variable or use default
    port = int(os.environ.get("TRIM_PORT", 8000))

    print("\n" + "=" * 60)
    print("  TRIM Web Server")
    print("=" * 60)
    print(f"  Starting at: http://0.0.0.0:{port}")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Python: {sys.version.split()[0]}")
    print("=" * 60 + "\n")

    try:
        uvicorn.run(
            "app.web_server:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="warning",  # Changed from "info" to "warning"
            access_log=False  # Disable access log to reduce noise
        )
    except Exception as e:
        print(f"\n[ERROR] Server failed to start: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)