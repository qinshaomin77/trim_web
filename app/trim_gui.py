# app/trim_gui.py
# -*- coding: utf-8 -*-
"""
TRIM GUI - Main Entry

This file keeps GUI + runner only.
Step bindings are centralized in: app/bindings.py

Run:
    python app/trim_gui.py
"""

from __future__ import annotations

import os
import sys
import time
import json
import queue
import threading
import traceback
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Suppress connection timeout warnings from event polling
warnings.filterwarnings("ignore", message=".*No events received.*")
warnings.filterwarnings("ignore", message=".*connection may be stalled.*")

# =========================================================
# 0) Ensure project root in sys.path (so `import scripts.xxx` works)
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent  # app/ -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================
# 1) Import bindings
# =========================================================
try:
    from app.bindings import build_entrypoints
except Exception as e:
    raise RuntimeError(
        "Failed to import app.bindings.\n"
        "Please create app/bindings.py and app/adapters.py (Method A), "
        "and ensure app/__init__.py exists.\n\n"
        f"{e}"
    ) from e

SCRIPT_ENTRYPOINTS = build_entrypoints()


# =========================================================
# 2) Utilities
# =========================================================

def open_path(p: Path) -> None:
    p = Path(p)
    if not p.exists():
        messagebox.showwarning("Not Found", f"Path does not exist:\n{p}")
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{p}"')
        else:
            os.system(f'xdg-open "{p}"')
    except Exception as e:
        messagebox.showerror("Open Failed", f"Failed to open:\n{p}\n\n{e}")


# =========================================================
# 3) State / Status
# =========================================================

class Light(str, Enum):
    RED = "red"
    GREEN = "green"
    GRAY = "gray"


class AppRunStatus(str, Enum):
    READY = "Ready"
    RUNNING = "Running"
    DONE = "Done"
    ERROR = "Error"
    STOPPED = "Stopped"
    PAUSED = "Paused"  # New: distinguishes clean stop with resume capability


class StepRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class Options:
    pollutants: Dict[str, bool] = field(default_factory=lambda: {
        "NOx": True,
        "PM2.5": False,
        "CO": False,
        "CO2": False,
        "THC": False,
    })
    enable_plots: bool = False
    plot_limit: int = 10
    grid_cell_m: float = 1.0
    random_seed: int = 42
    roi_text: str = ""

    # Spatial emission map options
    spatial_axis: str = "both"  # "x", "y", or "both"
    spatial_enable_roi: bool = False
    spatial_roi_mode: str = "shared"  # "shared" or "independent"
    spatial_roi_sim: str = ""  # ROI polygon for SUMO/TRIM/SG data ("x1,y1;x2,y2;..." format)
    spatial_roi_gt: str = ""  # ROI polygon for GT data (used only in "independent" mode)
    spatial_roi_polygon: str = ""  # backward compatibility alias (mapped from spatial_roi_sim)
    axis_bin_size: float = 10.0  # for axis binning in spatial analysis

    # Time range filtering options
    enable_sumo_time_filter: bool = False  # Enable SUMO simulation time filtering
    sumo_time_start: Optional[float] = None  # SUMO start time (seconds), None = from beginning
    sumo_time_end: Optional[float] = None  # SUMO end time (seconds), None = to end
    enable_gt_time_filter: bool = False  # Enable GT time filtering
    gt_time_start: Optional[float] = None  # GT start time (seconds)
    gt_time_end: Optional[float] = None  # GT end time (seconds)


@dataclass
class Inputs:
    net_xml: Optional[Path] = None
    fcd_input: Optional[Path] = None  # .xml or .csv
    out_dir: Optional[Path] = None

    import_gt: bool = False
    gt_csv: Optional[Path] = None

    raw_prefix: str = "fcd"
    safe_prefix: str = "fcd"


@dataclass
class AppState:
    inputs: Inputs = field(default_factory=Inputs)
    options: Options = field(default_factory=Options)

    status: AppRunStatus = AppRunStatus.READY
    step_status: Dict[str, StepRunStatus] = field(default_factory=dict)
    step_outputs: Dict[str, Path] = field(default_factory=dict)
    last_error: str = ""

    stop_requested: bool = False

    # Resume capability: track last pipeline and position
    last_pipeline: List[Step] = field(default_factory=list)
    last_selected: Optional[List[str]] = None
    resume_from_index: int = 0  # Index in last_pipeline to resume from


# =========================================================
# 4) Naming (prefix sanitizer)
# =========================================================
import re

_ALLOWED_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_prefix(raw: str | None, fallback: str = "fcd", max_len: int = 64) -> str:
    s = (raw or "").strip()
    if not s:
        return fallback
    s = re.sub(r"\s+", "_", s)
    s = _ALLOWED_CHARS_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s:
        return fallback
    if len(s) > max_len:
        s = s[:max_len].rstrip("._-") or fallback
    return s


# =========================================================
# 5) Output dirs plan (fixed, GUI-friendly)
# =========================================================

STEP_DIRS = {
    "net_topology": "00_net_topology",
    "xml2csv_fcd": "01_xml2csv",
    "identify_neighbors": "02_identify_neighbors",
    "trip_split": "03_trip_split",
    "trim_postprocess": "04_trim",
    "fill_xy": "05_fill_xy",
    "ef_match": "06_ef_match",
    "emission_spatial_map": "07_emission_spatial_map",
    "emission_compare": "08_emission_compare",
    # optional
    "sg_smooth": "optional/sg_smooth",
    "gt_standardize": "optional/gt_standardize",
}


def step_out_dir(out_root: Path, step_id: str) -> Path:
    return out_root / STEP_DIRS[step_id]


def logs_dir(out_root: Path) -> Path:
    return out_root / "logs"


def manifest_path(out_root: Path) -> Path:
    return out_root / "manifest.json"


def ensure_out_dirs(out_root: Path) -> None:
    for sid in STEP_DIRS:
        step_out_dir(out_root, sid).mkdir(parents=True, exist_ok=True)
    logs_dir(out_root).mkdir(parents=True, exist_ok=True)


def manifest_append(out_root: Path, record: dict) -> None:
    mp = manifest_path(out_root)
    if mp.exists():
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []
    data.append(record)
    mp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def cleanup_intermediate_files(out_root: Path) -> None:
    """
    Clean up intermediate step outputs, keeping only the final emission_spatial_map results.

    This function:
    1. Creates a 'result' folder
    2. Copies emission_spatial_map outputs to 'result'
    3. **Closes all log file handlers before deletion**
    4. Deletes all intermediate step directories
    5. Deletes logs directory
    6. Deletes empty optional directory
    7. Keeps only: result/ and manifest.json
    """
    import shutil
    import time
    import logging
    import gc

    result_dir = out_root / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Copy emission_spatial_map outputs to result folder
    spatial_map_dir = step_out_dir(out_root, "emission_spatial_map")
    if spatial_map_dir.exists():
        for item in spatial_map_dir.iterdir():
            dest = result_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest)
            elif item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)

    # ================================================================
    # CRITICAL: Close all logging handlers BEFORE attempting deletion
    # ================================================================

    print("[Cleanup] Closing all logging handlers to release file locks...")

    # Method 1: Use logging.shutdown() - closes all handlers cleanly
    logging.shutdown()

    # Small delay to ensure Windows releases file locks
    time.sleep(0.3)

    # Method 2: Manually close any remaining FileHandlers (backup safety)
    all_loggers = [logging.getLogger()] + [
        logging.getLogger(name)
        for name in list(logging.Logger.manager.loggerDict.keys())
    ]

    closed_count = 0
    for logger in all_loggers:
        handlers = logger.handlers[:]  # Copy to avoid modification during iteration
        for handler in handlers:
            if isinstance(handler, logging.FileHandler):
                try:
                    handler.close()
                    logger.removeHandler(handler)
                    closed_count += 1
                except Exception:
                    pass

    if closed_count > 0:
        print(f"[Cleanup] Closed {closed_count} additional log file handler(s)")

    # Force garbage collection to release any remaining file handles
    gc.collect()

    # Additional delay for Windows file system
    time.sleep(0.5)

    # ================================================================
    # Delete all step directories
    # ================================================================

    successfully_deleted = 0
    failed_deletions = []

    # Delete step directories from STEP_DIRS
    for step_id in STEP_DIRS:
        step_dir = step_out_dir(out_root, step_id)
        if not step_dir.exists():
            continue

        # Try multiple times with increasing delays
        deleted = False
        for attempt in range(3):
            try:
                shutil.rmtree(step_dir)
                successfully_deleted += 1
                deleted = True
                break
            except (PermissionError, OSError) as e:
                if attempt < 2:  # Not last attempt
                    # Wait progressively longer and try again
                    time.sleep(0.3 * (attempt + 1))
                    gc.collect()
                else:
                    # Last attempt failed
                    failed_deletions.append((step_dir.name, str(e)))

        if not deleted and step_dir.exists():
            # Try partial cleanup as last resort
            partial = _cleanup_directory_partially(step_dir)
            if partial:
                # Update the failure message
                if failed_deletions and failed_deletions[-1][0] == step_dir.name:
                    failed_deletions[-1] = (step_dir.name, "Partially deleted (some files remain locked)")

    # ================================================================
    # Delete logs directory (not in STEP_DIRS but should be removed)
    # ================================================================

    logs_path = logs_dir(out_root)
    if logs_path.exists():
        deleted = False
        for attempt in range(3):
            try:
                shutil.rmtree(logs_path)
                successfully_deleted += 1
                deleted = True
                print(f"[Cleanup] Deleted logs directory")
                break
            except (PermissionError, OSError) as e:
                if attempt < 2:
                    time.sleep(0.3 * (attempt + 1))
                    gc.collect()
                else:
                    failed_deletions.append(("logs", str(e)))

        if not deleted and logs_path.exists():
            partial = _cleanup_directory_partially(logs_path)
            if partial and failed_deletions and failed_deletions[-1][0] == "logs":
                failed_deletions[-1] = ("logs", "Partially deleted (some files remain locked)")

    # ================================================================
    # Delete optional directory if it exists and is empty
    # ================================================================

    optional_path = out_root / "optional"
    if optional_path.exists():
        # Try to remove if empty (after deleting sg_smooth and gt_standardize)
        try:
            # First try to remove if already empty
            optional_path.rmdir()
            successfully_deleted += 1
            print(f"[Cleanup] Deleted empty optional directory")
        except OSError:
            # Not empty, try to delete remaining contents
            try:
                shutil.rmtree(optional_path)
                successfully_deleted += 1
                print(f"[Cleanup] Deleted optional directory with remaining contents")
            except (PermissionError, OSError) as e:
                # Try partial cleanup
                partial = _cleanup_directory_partially(optional_path)
                if partial:
                    # Try to remove again if now empty
                    try:
                        optional_path.rmdir()
                        successfully_deleted += 1
                    except:
                        failed_deletions.append(("optional", "Partially deleted (some files remain)"))
                else:
                    failed_deletions.append(("optional", str(e)))

    # ================================================================
    # Report results
    # ================================================================

    total_targets = len(STEP_DIRS) + 2  # STEP_DIRS + logs + optional

    print(f"[Cleanup] Final results saved to: {result_dir}")
    print(f"[Cleanup] Successfully deleted {successfully_deleted}/{total_targets} directories")

    if failed_deletions:
        print(f"[Cleanup] Warning: {len(failed_deletions)} directories could not be fully deleted:")
        for dirname, reason in failed_deletions:
            # Extract just the filename from the full error message
            if "另一个程序正在使用此文件" in reason or "WinError 32" in reason:
                reason = "File locked by another process"
            print(f"          - {dirname}: {reason}")
        print(f"[Cleanup] Note: Locked files can be safely deleted manually after program exit.")
    else:
        print(f"[Cleanup] All intermediate files successfully removed")
        print(f"[Cleanup] Output directory now contains only: result/ and manifest.json")

    print(f"[Cleanup] Intermediate files cleanup complete")


def _cleanup_directory_partially(directory: Path) -> bool:
    """
    Attempt to delete as much as possible from a directory, skipping locked files.

    Returns:
        True if at least some files were deleted, False otherwise
    """
    import os

    if not directory.exists():
        return False

    deleted_anything = False

    # Walk the directory tree bottom-up
    for root, dirs, files in os.walk(directory, topdown=False):
        root_path = Path(root)

        # Try to delete files
        for name in files:
            file_path = root_path / name
            try:
                file_path.unlink()
                deleted_anything = True
            except (PermissionError, OSError):
                # File is locked, skip it
                pass

        # Try to delete empty directories
        for name in dirs:
            dir_path = root_path / name
            try:
                dir_path.rmdir()  # Only removes if empty
                deleted_anything = True
            except (PermissionError, OSError):
                # Directory not empty or locked
                pass

    # Try to remove the root directory itself
    try:
        directory.rmdir()
        deleted_anything = True
    except (PermissionError, OSError):
        # Root directory not empty or locked
        pass

    return deleted_anything


# =========================================================
# 6) Validators (red/green/gray lights + readiness)
# =========================================================

def light_required_file(p: Optional[Path], suffixes: Tuple[str, ...]) -> Light:
    if p is None:
        return Light.RED
    if not p.exists() or not p.is_file():
        return Light.RED
    if suffixes and p.suffix.lower() not in suffixes:
        return Light.RED
    return Light.GREEN


def light_out_dir(p: Optional[Path]) -> Light:
    if p is None:
        return Light.RED
    try:
        p.mkdir(parents=True, exist_ok=True)
        testfile = p / ".__write_test__"
        testfile.write_text("ok", encoding="utf-8")
        testfile.unlink(missing_ok=True)
        return Light.GREEN
    except Exception:
        return Light.RED


def light_optional_gt(import_gt: bool, gt_csv: Optional[Path]) -> Light:
    if not import_gt:
        return Light.GRAY
    if gt_csv is None:
        return Light.RED
    if (not gt_csv.exists()) or (not gt_csv.is_file()) or gt_csv.suffix.lower() != ".csv":
        return Light.RED
    return Light.GREEN


def is_ready_to_run(state: AppState) -> bool:
    net_ok = light_required_file(state.inputs.net_xml, (".xml",)) == Light.GREEN
    fcd_ok = light_required_file(state.inputs.fcd_input, (".xml")) == Light.GREEN
    out_ok = light_out_dir(state.inputs.out_dir) == Light.GREEN
    gt_ok = light_optional_gt(state.inputs.import_gt, state.inputs.gt_csv)
    if gt_ok == Light.RED:
        return False
    return net_ok and fcd_ok and out_ok


# =========================================================
# 7) Pipeline definition (IDs must match STEP_DIRS and bindings)
# =========================================================

@dataclass
class Step:
    step_id: str
    title: str
    depends_on: List[str] = field(default_factory=list)
    is_optional: bool = False


MAINLINE_STEPS: List[Step] = [
    Step("net_topology", "NET: net.xml -> topology tables", [], False),
    Step("xml2csv_fcd", "STEP-00: fcd.xml -> CSV", ["net_topology"], False),
    Step("identify_neighbors", "STEP-01: identify neighbors", ["xml2csv_fcd"], False),
    Step("trip_split", "STEP-02: trip split (trip_id)", ["identify_neighbors"], False),
    Step("trim_postprocess", "TRIM: optimize & postprocess trajectories", ["trip_split"], False),
    Step("fill_xy", "STEP-03: merge/fill XY", ["trim_postprocess"], False),
    Step("ef_match", "STEP-04: emission factor match", ["fill_xy"], False),
    Step("emission_spatial_map", "STEP-05: spatial emission map", ["ef_match"], False),
    Step("emission_compare", "STEP-06: emission compare (optional GT)", ["emission_spatial_map"], False),
]

OPTIONAL_STEPS: List[Step] = [
    Step("sg_smooth", "OPTIONAL: SG smooth", ["trip_split"], True),
    Step("gt_standardize", "OPTIONAL: standardize ground truth", [], True),
]


def build_pipeline(state: AppState, enable_sg: bool) -> List[Step]:
    steps = list(MAINLINE_STEPS)

    # Optional GT standardize (prepend)
    if state.inputs.import_gt:
        steps = [OPTIONAL_STEPS[1]] + steps  # gt_standardize

    # Optional SG smooth (insert after trip_split)
    if enable_sg:
        sg = OPTIONAL_STEPS[0]
        out: List[Step] = []
        for s in steps:
            out.append(s)
            if s.step_id == "trip_split":
                out.append(sg)
        steps = out

    # Deduplicate by step_id (safe)
    seen = set()
    uniq = []
    for s in steps:
        if s.step_id not in seen:
            uniq.append(s)
            seen.add(s.step_id)
    return uniq


# =========================================================
# 8) Runner (background thread + event queue)
# =========================================================

class EventType(str, Enum):
    LOG = "log"
    STEP_START = "step_start"
    STEP_DONE = "step_done"
    STEP_ERROR = "step_error"
    PIPELINE_DONE = "pipeline_done"
    PIPELINE_STOPPED = "pipeline_stopped"


@dataclass
class Event:
    type: EventType
    step_id: str = ""
    message: str = ""
    path: Optional[str] = None


class PipelineRunner:
    def __init__(self, state: AppState, q: "queue.Queue[Event]"):
        self.state = state
        self.q = q
        self._thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def can_resume(self) -> bool:
        """Check if pipeline can be resumed."""
        return (self.state.status == AppRunStatus.PAUSED and
                len(self.state.last_pipeline) > 0 and
                self.state.resume_from_index < len(self.state.last_pipeline))

    def stop(self) -> None:
        self.state.stop_requested = True
        self.q.put(Event(EventType.LOG, message="Stop requested by user."))

    def resume(self) -> None:
        """Resume from last stopped position."""
        if not self.can_resume():
            self.q.put(Event(EventType.LOG, message="Cannot resume: no paused pipeline found."))
            return

        if self.is_running():
            self.q.put(Event(EventType.LOG, message="Runner is already running."))
            return

        self.q.put(Event(EventType.LOG, message=f"Resuming from step {self.state.resume_from_index}..."))
        self.state.stop_requested = False
        self._thread = threading.Thread(
            target=self._run_impl,
            args=(self.state.last_pipeline, self.state.last_selected, self.state.resume_from_index),
            daemon=True
        )
        self._thread.start()

    def run(self, steps: List[Step], selected_only: Optional[List[str]] = None) -> None:
        if self.is_running():
            self.q.put(Event(EventType.LOG, message="Runner is already running."))
            return

        # Save pipeline for potential resume
        self.state.last_pipeline = steps
        self.state.last_selected = selected_only
        self.state.resume_from_index = 0
        self.state.stop_requested = False

        self._thread = threading.Thread(
            target=self._run_impl, args=(steps, selected_only, 0), daemon=True
        )
        self._thread.start()

    def _run_impl(self, steps: List[Step], selected_only: Optional[List[str]], start_index: int = 0) -> None:
        try:
            self.state.status = AppRunStatus.RUNNING
            out_root = self.state.inputs.out_dir
            assert out_root is not None
            ensure_out_dirs(out_root)

            # Suppress tkinter polling timeout warnings
            import warnings
            warnings.filterwarnings("ignore", message=".*No events received.*")

            # Configure TRIM root logger so all scripts using
            # logging.getLogger("trim.xxx") can output to terminal
            import logging as _logging
            _trim_logger = _logging.getLogger("trim")
            if not _trim_logger.handlers:
                _trim_logger.setLevel(_logging.INFO)
                _sh = _logging.StreamHandler()
                _sh.setFormatter(_logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                ))
                _trim_logger.addHandler(_sh)

            # init step statuses
            for i, s in enumerate(steps):
                # Skip steps before start_index if resuming
                if i < start_index:
                    continue

                if selected_only is not None and s.step_id not in selected_only:
                    self.state.step_status[s.step_id] = StepRunStatus.SKIPPED
                else:
                    self.state.step_status[s.step_id] = StepRunStatus.PENDING

            self.q.put(Event(EventType.LOG, message="=== TRIM GUI pipeline started ==="))

            for i, s in enumerate(steps):
                # Skip steps before start_index if resuming
                if i < start_index:
                    continue

                if selected_only is not None and s.step_id not in selected_only:
                    continue

                if self.state.stop_requested:
                    # Save resume point
                    self.state.resume_from_index = i
                    self.state.step_status[s.step_id] = StepRunStatus.SKIPPED
                    # Mark remaining steps as pending for clarity
                    for j in range(i + 1, len(steps)):
                        if selected_only is None or steps[j].step_id in selected_only:
                            self.state.step_status[steps[j].step_id] = StepRunStatus.PENDING
                    continue

                # dependency check
                missing = [dep for dep in s.depends_on if dep not in self.state.step_outputs]
                if missing:
                    msg = f"[{s.step_id}] dependency missing: {missing}"
                    self.state.step_status[s.step_id] = StepRunStatus.ERROR
                    self.state.last_error = msg
                    self.q.put(Event(EventType.STEP_ERROR, step_id=s.step_id, message=msg))
                    raise RuntimeError(msg)

                self.state.step_status[s.step_id] = StepRunStatus.RUNNING
                self.q.put(Event(EventType.STEP_START, step_id=s.step_id, message=s.title))
                t0 = time.time()

                step_dir = step_out_dir(out_root, s.step_id)
                upstream = dict(self.state.step_outputs)

                fn = SCRIPT_ENTRYPOINTS.get(s.step_id)
                if fn is None:
                    raise RuntimeError(
                        f"[{s.step_id}] No binding found. Please add it in app/bindings.py."
                    )

                out_path = fn(state=self.state, out_root=out_root, step_dir=step_dir, upstream=upstream)
                if out_path is None:
                    out_path = step_dir
                out_path = Path(out_path)

                self.state.step_outputs[s.step_id] = out_path
                self.state.step_status[s.step_id] = StepRunStatus.DONE

                dt = time.time() - t0
                self.q.put(Event(EventType.STEP_DONE, step_id=s.step_id,
                                 message=f"Done in {dt:.2f}s", path=str(out_path)))

                manifest_append(out_root, {
                    "step_id": s.step_id,
                    "title": s.title,
                    "time_start": t0,
                    "time_end": t0 + dt,
                    "output": str(out_path),
                })

            if self.state.stop_requested:
                self.state.status = AppRunStatus.PAUSED  # Changed from STOPPED to PAUSED
                self.q.put(Event(EventType.PIPELINE_STOPPED,
                                 message=f"=== Pipeline paused at step {self.state.resume_from_index} ==="))
            else:
                self.state.status = AppRunStatus.DONE
                self.state.resume_from_index = len(steps)  # Reset resume index

                # Clean up intermediate files if pipeline completed successfully
                try:
                    cleanup_intermediate_files(out_root)
                    self.q.put(Event(EventType.LOG,
                                     message="[Cleanup] Intermediate files cleaned up, final results in 'result' folder"))
                except Exception as e:
                    self.q.put(Event(EventType.LOG, message=f"[Warning] Cleanup failed: {e}"))

                self.q.put(Event(EventType.PIPELINE_DONE, message="=== Pipeline done ==="))

        except Exception:
            self.state.status = AppRunStatus.ERROR
            err = traceback.format_exc()
            self.state.last_error = err
            self.q.put(Event(EventType.PIPELINE_DONE, message="=== Pipeline error ==="))
            self.q.put(Event(EventType.LOG, message=err))


# =========================================================
# 9) GUI
# =========================================================

class TRIMGuiApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TRIM GUI")
        self.root.geometry("1100x640")

        self.state = AppState()
        self.event_q: "queue.Queue[Event]" = queue.Queue()
        self.runner = PipelineRunner(self.state, self.event_q)

        # Tk variables
        self.net_var = tk.StringVar()
        self.fcd_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.gt_var = tk.StringVar()

        self.import_gt_var = tk.BooleanVar(value=False)
        self.use_sg_var = tk.BooleanVar(value=False)
        self.prefix_var = tk.StringVar(value="fcd")

        self.enable_plots_var = tk.BooleanVar(value=self.state.options.enable_plots)
        self.plot_limit_var = tk.StringVar(value=str(self.state.options.plot_limit))

        # Spatial emission map variables
        self.spatial_axis_var = tk.StringVar(value=self.state.options.spatial_axis)
        self.spatial_enable_roi_var = tk.BooleanVar(value=self.state.options.spatial_enable_roi)
        self.spatial_roi_mode_var = tk.StringVar(value=self.state.options.spatial_roi_mode)
        self.spatial_roi_sim_var = tk.StringVar(value=self.state.options.spatial_roi_sim)
        self.spatial_roi_gt_var = tk.StringVar(value=self.state.options.spatial_roi_gt)
        self.axis_bin_size_var = tk.StringVar(value=str(self.state.options.axis_bin_size))

        # Time range filtering variables
        self.enable_sumo_time_var = tk.BooleanVar(value=self.state.options.enable_sumo_time_filter)
        self.sumo_time_start_var = tk.StringVar(value="")
        self.sumo_time_end_var = tk.StringVar(value="")
        self.enable_gt_time_var = tk.BooleanVar(value=self.state.options.enable_gt_time_filter)
        self.gt_time_start_var = tk.StringVar(value="")
        self.gt_time_end_var = tk.StringVar(value="")

        # Initialize time values from state if they exist
        if self.state.options.sumo_time_start is not None:
            self.sumo_time_start_var.set(str(self.state.options.sumo_time_start))
        if self.state.options.sumo_time_end is not None:
            self.sumo_time_end_var.set(str(self.state.options.sumo_time_end))
        if self.state.options.gt_time_start is not None:
            self.gt_time_start_var.set(str(self.state.options.gt_time_start))
        if self.state.options.gt_time_end is not None:
            self.gt_time_end_var.set(str(self.state.options.gt_time_end))

        self.pol_vars: Dict[str, tk.BooleanVar] = {}

        # lights
        self.light_net: Optional[tk.Label] = None
        self.light_fcd: Optional[tk.Label] = None
        self.light_out: Optional[tk.Label] = None
        self.light_gt: Optional[tk.Label] = None

        # steps UI
        self.step_rows: Dict[str, dict] = {}

        # console
        self.console_text: Optional[tk.Text] = None

        self._nb: Optional[ttk.Notebook] = None  # type: ignore[assignment]

        # results page
        self._results_inner: Optional[ttk.Frame] = None  # type: ignore[assignment]
        self._results_canvas = None
        self._results_canvas_window = None
        self.lbl_result_summary: Optional[ttk.Label] = None  # type: ignore[assignment]

        self._build_ui()
        self._refresh_lights()
        self._poll_events()

        # status style: text + colors
        self._step_status_style = {
            StepRunStatus.PENDING: ("pending", "#FFFFFF", "#ECF0F1"),  # dark text, light gray bg
            StepRunStatus.RUNNING: ("running", "#FFFFFF", "#3498DB"),  # white on blue
            StepRunStatus.DONE: ("done", "#FFFFFF", "#2ECC71"),  # white on green
            StepRunStatus.ERROR: ("error", "#FFFFFF", "#E74C3C"),  # white on red
            StepRunStatus.SKIPPED: ("skipped", "#FFFFFF", "#95A5A6"),  # white on gray
        }

    # ---------------- UI build ----------------

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        page_inputs = ttk.Frame(nb)
        page_pipeline = ttk.Frame(nb)
        page_console = ttk.Frame(nb)

        page_results = ttk.Frame(nb)

        nb.add(page_inputs, text="Inputs & Options")
        nb.add(page_pipeline, text="Pipeline")
        nb.add(page_console, text="Console")
        nb.add(page_results, text="Results")

        self._nb = nb  # keep reference for tab switching

        self._build_inputs_page(page_inputs)
        self._build_pipeline_page(page_pipeline)
        self._build_console_page(page_console)
        self._build_results_page(page_results)

    def _build_inputs_page(self, parent: ttk.Frame):
        canvas = tk.Canvas(parent, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_canvas_resize(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", _on_canvas_resize)

        def _on_inner_resize(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _on_inner_resize)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows/macOS
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))  # Linux

        inputs_box = ttk.LabelFrame(inner, text="Inputs")
        inputs_box.pack(fill=tk.X, padx=10, pady=(10, 8))

        frm = ttk.Frame(inputs_box)
        frm.pack(fill=tk.X, expand=False, padx=10, pady=10)

        def add_path_row(r, title, var, browse_fn, light_attr):
            ttk.Label(frm, text=title, width=18).grid(row=r, column=0, sticky="w", pady=6)
            ent = ttk.Entry(frm, textvariable=var, width=90)
            ent.grid(row=r, column=1, sticky="we", padx=(0, 8))

            btn_text = "Import" if title.lower().endswith(".xml") or title.lower().endswith(".csv") else "Select"
            ttk.Button(frm, text=btn_text, command=browse_fn, width=10).grid(row=r, column=2, sticky="e", padx=(0, 6))

            def clear_fn():
                var.set("")
                self._sync_state_from_ui()
                self._refresh_lights()

            ttk.Button(frm, text="Clear", command=clear_fn, width=10).grid(row=r, column=3, sticky="e")

            light = tk.Label(frm, width=2, height=1, bg="#B0B0B0", relief=tk.SOLID, bd=1)
            light.grid(row=r, column=4, padx=(10, 0))
            setattr(self, light_attr, light)

        frm.columnconfigure(1, weight=1)

        add_path_row(0, "net.xml", self.net_var, self._browse_net, "light_net")
        add_path_row(1, "fcd.xml", self.fcd_var, self._browse_fcd, "light_fcd")
        add_path_row(2, "output dir", self.out_var, self._browse_outdir, "light_out")

        # optional GT toggle
        gt_row = ttk.Frame(frm)
        gt_row.grid(row=3, column=0, columnspan=5, sticky="we", pady=(10, 6))
        ttk.Checkbutton(gt_row, text="Ground Truth (optional)",
                        variable=self.import_gt_var, command=self._on_toggle_gt).pack(side=tk.LEFT)

        gt_frame = ttk.Frame(frm)
        gt_frame.grid(row=4, column=0, columnspan=5, sticky="we")
        ttk.Label(gt_frame, text="GT csv", width=18).grid(row=0, column=0, sticky="w")
        ttk.Entry(gt_frame, textvariable=self.gt_var, width=90).grid(row=0, column=1, sticky="we", padx=(0, 8))
        ttk.Button(
            gt_frame, text="Import", command=self._browse_gt, width=10
        ).grid(row=0, column=2, sticky="e", padx=(0, 6))

        def clear_gt():
            self.gt_var.set("")
            self._sync_state_from_ui()
            self._refresh_lights()

        ttk.Button(
            gt_frame, text="Clear", command=clear_gt, width=10
        ).grid(row=0, column=3, sticky="e")

        self.light_gt = tk.Label(gt_frame, width=2, height=1, bg="#B0B0B0", relief=tk.SOLID, bd=1)
        self.light_gt.grid(row=0, column=4, padx=(10, 0))

        gt_frame.columnconfigure(1, weight=1)

        # Options
        opt = ttk.LabelFrame(frm, text="Options")
        opt.grid(row=5, column=0, columnspan=5, sticky="we", pady=(16, 0))
        opt.columnconfigure(1, weight=1)

        ttk.Label(opt, text="prefix").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(opt, textvariable=self.prefix_var, width=20).grid(row=0, column=1, sticky="w", pady=6)

        ttk.Checkbutton(opt, text="Enable SG smooth (optional)",
                        variable=self.use_sg_var,
                        command=self._on_toggle_sg).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(opt, text="Enable plots (optional)",
                        variable=self.enable_plots_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        # pollutants
        pol = ttk.Frame(opt)
        pol.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        ttk.Label(pol, text="pollutants:").pack(side=tk.LEFT)

        for k in ["NOx", "PM2.5", "CO", "CO2", "THC"]:
            v = tk.BooleanVar(value=self.state.options.pollutants.get(k, False))
            self.pol_vars[k] = v
            ttk.Checkbutton(pol, text=k, variable=v).pack(side=tk.LEFT, padx=4)

        ttk.Label(opt, text="plot limit (vehicle-wise)").grid(row=4, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(opt, textvariable=self.plot_limit_var, width=10).grid(row=4, column=1, sticky="w", pady=6)

        # Spatial Emission Map Options
        spatial_frame = ttk.LabelFrame(opt, text="Spatial Emission Map Settings")
        spatial_frame.grid(row=5, column=0, columnspan=2, sticky="we", padx=8, pady=(16, 8))
        spatial_frame.columnconfigure(1, weight=1)

        # Axis selection
        ttk.Label(spatial_frame, text="Analysis axis:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        axis_frame = ttk.Frame(spatial_frame)
        axis_frame.grid(row=0, column=1, sticky="w", padx=8, pady=6)

        for i, axis_option in enumerate(["x", "y", "both"]):
            ttk.Radiobutton(
                axis_frame, text=axis_option,
                variable=self.spatial_axis_var,
                value=axis_option
            ).grid(row=0, column=i, sticky="w", padx=(0, 10))

        # Axis bin size
        ttk.Label(spatial_frame, text="Axis bin size (m):").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(spatial_frame, textvariable=self.axis_bin_size_var, width=10).grid(row=1, column=1, sticky="w",
                                                                                     padx=8, pady=6)

        # ROI enable checkbox
        ttk.Checkbutton(
            spatial_frame, text="Enable ROI polygon filter",
            variable=self.spatial_enable_roi_var,
            command=self._on_toggle_spatial_roi
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=6)

        # ROI mode selection (shared / independent)
        roi_mode_frame = ttk.Frame(spatial_frame)
        roi_mode_frame.grid(row=3, column=0, columnspan=2, sticky="w", padx=24, pady=(0, 6))
        ttk.Label(roi_mode_frame, text="ROI mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            roi_mode_frame, text="Shared (same ROI for all data)",
            variable=self.spatial_roi_mode_var, value="shared",
            command=self._on_toggle_spatial_roi
        ).pack(side=tk.LEFT, padx=(8, 16))
        ttk.Radiobutton(
            roi_mode_frame, text="Independent (separate ROI for Sim & GT)",
            variable=self.spatial_roi_mode_var, value="independent",
            command=self._on_toggle_spatial_roi
        ).pack(side=tk.LEFT)

        # Simulation ROI (SUMO/TRIM/SG) - label is dynamic based on mode
        self.lbl_roi_sim = ttk.Label(spatial_frame, text="Simulation ROI (SUMO/TRIM/SG):")
        self.lbl_roi_sim.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 2))
        roi_sim_text = tk.Text(spatial_frame, height=2, width=60, wrap=tk.WORD)
        roi_sim_text.grid(row=5, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 4))
        roi_sim_text.insert("1.0", self.spatial_roi_sim_var.get())

        def update_roi_sim_text(event=None):
            self.spatial_roi_sim_var.set(roi_sim_text.get("1.0", "end-1c"))
            self._sync_state_from_ui()

        roi_sim_text.bind("<KeyRelease>", update_roi_sim_text)
        roi_sim_text.bind("<ButtonRelease>", update_roi_sim_text)
        self.spatial_roi_sim_text_widget = roi_sim_text

        # GT ROI (Ground Truth) - label is dynamic based on mode
        self.lbl_roi_gt = ttk.Label(spatial_frame, text="GT ROI (Ground Truth):")
        self.lbl_roi_gt.grid(row=6, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 2))
        roi_gt_text = tk.Text(spatial_frame, height=2, width=60, wrap=tk.WORD)
        roi_gt_text.grid(row=7, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 4))
        roi_gt_text.insert("1.0", self.spatial_roi_gt_var.get())

        def update_roi_gt_text(event=None):
            self.spatial_roi_gt_var.set(roi_gt_text.get("1.0", "end-1c"))
            self._sync_state_from_ui()

        roi_gt_text.bind("<KeyRelease>", update_roi_gt_text)
        roi_gt_text.bind("<ButtonRelease>", update_roi_gt_text)
        self.spatial_roi_gt_text_widget = roi_gt_text

        # Help text
        help_label = ttk.Label(
            spatial_frame,
            text="Format: x1,y1;x2,y2;x3,y3;... (polygon vertices). "
                 "Leave empty = no filter for that group.",
            font=("TkDefaultFont", 8, "italic")
        )
        help_label.grid(row=8, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

        # Apply initial enabled/disabled state
        self._on_toggle_spatial_roi()

        # ===== Time Range Settings =====
        time_frame = ttk.LabelFrame(opt, text="Time Range Settings (Optional)")
        time_frame.grid(row=6, column=0, columnspan=2, sticky="we", padx=8, pady=(16, 8))
        time_frame.columnconfigure(0, weight=1)
        time_frame.columnconfigure(1, weight=1)

        # Left column: SUMO simulation time range
        sumo_frame = ttk.Frame(time_frame)
        sumo_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        ttk.Checkbutton(
            sumo_frame,
            text="Filter SUMO simulation time range",
            variable=self.enable_sumo_time_var,
            command=self._on_toggle_sumo_time
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.lbl_sumo_start = ttk.Label(sumo_frame, text="Start time (sec):")
        self.lbl_sumo_start.grid(row=1, column=0, sticky="w", pady=4)
        self.entry_sumo_start = ttk.Entry(sumo_frame, textvariable=self.sumo_time_start_var, width=15)
        self.entry_sumo_start.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)

        self.lbl_sumo_end = ttk.Label(sumo_frame, text="End time (sec):")
        self.lbl_sumo_end.grid(row=2, column=0, sticky="w", pady=4)
        self.entry_sumo_end = ttk.Entry(sumo_frame, textvariable=self.sumo_time_end_var, width=15)
        self.entry_sumo_end.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=4)

        ttk.Label(
            sumo_frame,
            text="(Leave empty for no limit.\nE.g., 0-1800 for first 30 min)",
            font=("TkDefaultFont", 8, "italic"),
            foreground="#666666"
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # Right column: GT time range
        gt_frame = ttk.Frame(time_frame)
        gt_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)

        ttk.Checkbutton(
            gt_frame,
            text="Filter GT (Ground Truth) time range",
            variable=self.enable_gt_time_var,
            command=self._on_toggle_gt_time
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.lbl_gt_start = ttk.Label(gt_frame, text="Start time (sec):")
        self.lbl_gt_start.grid(row=1, column=0, sticky="w", pady=4)
        self.entry_gt_start = ttk.Entry(gt_frame, textvariable=self.gt_time_start_var, width=15)
        self.entry_gt_start.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)

        self.lbl_gt_end = ttk.Label(gt_frame, text="End time (sec):")
        self.lbl_gt_end.grid(row=2, column=0, sticky="w", pady=4)
        self.entry_gt_end = ttk.Entry(gt_frame, textvariable=self.gt_time_end_var, width=15)
        self.entry_gt_end.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=4)

        ttk.Label(
            gt_frame,
            text="(Leave empty for no limit.\nMust match SUMO if comparing)",
            font=("TkDefaultFont", 8, "italic"),
            foreground="#666666"
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # Initialize time filter enabled/disabled state
        self._on_toggle_sumo_time()
        self._on_toggle_gt_time()

    def _build_pipeline_page(self, parent: ttk.Frame):
        top = ttk.Frame(parent)
        top.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ctrl = ttk.LabelFrame(top, text="Run Control")
        ctrl.pack(fill=tk.X)

        self.btn_run_all = ttk.Button(ctrl, text="Run All", command=self._on_run_all)
        self.btn_run_sel = ttk.Button(ctrl, text="Run Selected", command=self._on_run_selected)
        self.btn_resume = ttk.Button(ctrl, text="Resume", command=self._on_resume)  # NEW
        self.btn_stop = ttk.Button(ctrl, text="Stop", command=self._on_stop)
        self.btn_open_out = ttk.Button(ctrl, text="Open Output Dir", command=self._on_open_outdir)

        self.btn_run_all.pack(side=tk.LEFT, padx=6, pady=8)
        self.btn_run_sel.pack(side=tk.LEFT, padx=6, pady=8)
        self.btn_resume.pack(side=tk.LEFT, padx=6, pady=8)  # NEW
        self.btn_stop.pack(side=tk.LEFT, padx=6, pady=8)
        self.btn_open_out.pack(side=tk.LEFT, padx=6, pady=8)

        self.lbl_app_status = ttk.Label(ctrl, text="Status: Ready")
        self.lbl_app_status.pack(side=tk.RIGHT, padx=10)

        steps_frame = ttk.LabelFrame(top, text="Steps")
        steps_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        header = ttk.Frame(steps_frame)
        header.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(header, text="Use", width=6).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Task", width=52).grid(row=0, column=1, sticky="w")
        ttk.Label(header, text="Status", width=12).grid(row=0, column=2, sticky="w")
        ttk.Label(header, text="Open", width=10).grid(row=0, column=3, sticky="w")

        body = ttk.Frame(steps_frame)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # --- NEW: enforce consistent columns between header and body ---
        for f in (header, body):
            f.columnconfigure(0, minsize=60)  # Use
            f.columnconfigure(1, minsize=520)  # Step
            f.columnconfigure(2, minsize=120)  # Status
            f.columnconfigure(3, minsize=90)  # Open

        # For UI, show all potential steps (main + optional)
        show_steps = MAINLINE_STEPS + OPTIONAL_STEPS

        for i, s in enumerate(show_steps):
            use_default = (not s.is_optional)
            use_var = tk.BooleanVar(value=use_default)

            lbl_step = ttk.Label(body, text=s.title, width=52)
            lbl_status = tk.Label(body, text=StepRunStatus.PENDING.value, width=12, anchor="center", relief=tk.SOLID,
                                  bd=1,
                                  padx=6, pady=2, fg="#34495E", bg="#ECF0F1", )

            btn_open = ttk.Button(
                body, text="Open", width=10,
                command=lambda sid=s.step_id: self._on_open_step(sid)
            )
            btn_open.state(["disabled"])

            ttk.Checkbutton(body, variable=use_var).grid(row=i, column=0, sticky="w")
            lbl_step.grid(row=i, column=1, sticky="w")
            lbl_status.grid(row=i, column=2, sticky="w")
            btn_open.grid(row=i, column=3, sticky="e", padx=4)  # <-- key: align right

            self.step_rows[s.step_id] = {
                "use_var": use_var,
                "lbl_status": lbl_status,
                "btn_open": btn_open,
                "title": s.title,
            }

    def _build_console_page(self, parent: ttk.Frame):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text = tk.Text(frm, height=30, wrap=tk.WORD)
        sb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=sb.set)

        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.console_text = text

        btns = ttk.Frame(parent)
        btns.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(btns, text="Clear Console", command=self._clear_console).pack(side=tk.LEFT)

    def _build_results_page(self, parent: ttk.Frame):
        """Build the Results tab: shows per-step output summary after pipeline finishes."""
        # Top toolbar
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 4))

        ttk.Button(toolbar, text="🔄 Refresh Results", command=self._refresh_results).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="📂 Open Output Dir", command=self._on_open_outdir).pack(side=tk.LEFT)

        self.lbl_result_summary = ttk.Label(toolbar, text="Run the pipeline to see results here.",
                                            foreground="#888888")
        self.lbl_result_summary.pack(side=tk.LEFT, padx=16)

        # Scrollable canvas for step result cards
        canvas = tk.Canvas(parent, highlightthickness=0, bg="#F5F5F5")
        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._results_inner = ttk.Frame(canvas)
        self._results_canvas_window = canvas.create_window((0, 0), window=self._results_inner, anchor="nw")

        def _on_canvas_resize(event):
            canvas.itemconfig(self._results_canvas_window, width=event.width)

        def _on_inner_resize(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.bind("<Configure>", _on_canvas_resize)
        self._results_inner.bind("<Configure>", _on_inner_resize)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self._results_canvas = canvas

    def _refresh_results(self):
        """Scan step_outputs and populate the Results page with cards."""
        from PIL import Image, ImageTk  # optional; graceful fallback if not installed
        _pil_available = True
        try:
            pass
        except ImportError:
            _pil_available = False

        # Clear existing cards
        for w in self._results_inner.winfo_children():
            w.destroy()

        if not self.state.step_outputs:
            ttk.Label(self._results_inner, text="No results yet. Run the pipeline first.",
                      foreground="#888888").pack(pady=20)
            self.lbl_result_summary.configure(text="No results yet.")
            return

        done_count = sum(1 for s in self.state.step_status.values() if s == StepRunStatus.DONE)
        error_count = sum(1 for s in self.state.step_status.values() if s == StepRunStatus.ERROR)
        total_count = len(self.state.step_status)

        status_color = "#27AE60" if error_count == 0 else "#E74C3C"
        self.lbl_result_summary.configure(
            text=f"✅ {done_count} done   ❌ {error_count} error   / {total_count} total",
            foreground=status_color
        )

        # Step order for display
        ordered_steps = [s.step_id for s in (MAINLINE_STEPS + OPTIONAL_STEPS)]
        displayed = [sid for sid in ordered_steps if sid in self.state.step_outputs]
        # Append any extra outputs not in ordered list
        for sid in self.state.step_outputs:
            if sid not in displayed:
                displayed.append(sid)

        row_idx = 0
        for sid in displayed:
            out_path = Path(self.state.step_outputs[sid])
            step_status = self.state.step_status.get(sid, StepRunStatus.PENDING)

            # Status color
            status_bg = {
                StepRunStatus.DONE: "#D5F5E3",
                StepRunStatus.ERROR: "#FADBD8",
                StepRunStatus.RUNNING: "#D6EAF8",
                StepRunStatus.SKIPPED: "#EAECEE",
            }.get(step_status, "#FDFEFE")
            status_icon = {
                StepRunStatus.DONE: "✅",
                StepRunStatus.ERROR: "❌",
                StepRunStatus.RUNNING: "⏳",
                StepRunStatus.SKIPPED: "⏭",
            }.get(step_status, "⬜")

            # Card frame
            card = tk.Frame(self._results_inner, bg=status_bg, relief=tk.RIDGE, bd=1)
            card.pack(fill=tk.X, padx=10, pady=5, ipadx=6, ipady=6)

            # Header row inside card
            hdr = tk.Frame(card, bg=status_bg)
            hdr.pack(fill=tk.X)

            title_text = self.step_rows.get(sid, {}).get("title", sid)
            tk.Label(hdr, text=f"{status_icon}  {title_text}",
                     font=("TkDefaultFont", 10, "bold"), bg=status_bg,
                     anchor="w").pack(side=tk.LEFT, padx=4)

            tk.Button(hdr, text="📂 Open", relief=tk.FLAT, cursor="hand2",
                      bg="#3498DB", fg="white", padx=8, pady=2,
                      command=lambda p=out_path: open_path(p)).pack(side=tk.RIGHT, padx=4)

            # Output path label
            tk.Label(card, text=f"Output: {out_path}", fg="#555555", bg=status_bg,
                     anchor="w", wraplength=900, justify="left").pack(fill=tk.X, padx=4, pady=(0, 4))

            # File listing (if path is a directory)
            if out_path.is_dir():
                files = sorted(out_path.iterdir())
                if not files:
                    tk.Label(card, text="  (empty directory)", fg="#999999", bg=status_bg).pack(anchor="w", padx=8)
                else:
                    files_frame = tk.Frame(card, bg=status_bg)
                    files_frame.pack(fill=tk.X, padx=8, pady=(0, 4))

                    img_refs = []  # keep references to avoid GC
                    img_col, img_row = 0, 0

                    for f in files[:50]:  # limit to 50 files per step
                        ext = f.suffix.lower()
                        is_img = ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp")

                        if is_img:
                            # Try to show thumbnail
                            thumb_shown = False
                            try:
                                from PIL import Image as PILImage, ImageTk as PILImageTk
                                img = PILImage.open(f)
                                img.thumbnail((160, 120))
                                photo = PILImageTk.PhotoImage(img)
                                img_refs.append(photo)

                                cell = tk.Frame(files_frame, bg=status_bg)
                                cell.grid(row=img_row, column=img_col, padx=6, pady=4, sticky="nw")

                                tk.Label(cell, image=photo, bg=status_bg,
                                         cursor="hand2",
                                         relief=tk.SOLID, bd=1).pack()
                                tk.Label(cell, text=f.name, bg=status_bg,
                                         fg="#333333", wraplength=160,
                                         font=("TkDefaultFont", 8)).pack()

                                # Click to open
                                def _open_img(path=f):
                                    open_path(path)

                                cell.bind("<Button-1>", lambda e, path=f: open_path(path))

                                img_col += 1
                                if img_col >= 5:
                                    img_col = 0
                                    img_row += 1
                                thumb_shown = True
                            except Exception:
                                pass

                            if not thumb_shown:
                                # Fallback: show as link
                                lnk = tk.Label(files_frame, text=f"🖼 {f.name}", fg="#2980B9",
                                               bg=status_bg, cursor="hand2", anchor="w")
                                lnk.pack(anchor="w")
                                lnk.bind("<Button-1>", lambda e, path=f: open_path(path))

                        else:
                            # Non-image: show as clickable file link
                            icon = "📊" if ext == ".csv" else ("📋" if ext == ".json" else "📄")
                            lnk = tk.Label(files_frame, text=f"{icon} {f.name}",
                                           fg="#2980B9", bg=status_bg,
                                           cursor="hand2", anchor="w",
                                           font=("TkDefaultFont", 9, "underline"))
                            lnk.pack(anchor="w", pady=1)
                            lnk.bind("<Button-1>", lambda e, path=f: open_path(path))

                    # Store img_refs on the card to prevent GC
                    card._img_refs = img_refs  # type: ignore[attr-defined]

                    if len(files) > 50:
                        tk.Label(card, text=f"  ... and {len(files) - 50} more files",
                                 fg="#999999", bg=status_bg).pack(anchor="w", padx=8)

            elif out_path.is_file():
                # Single file output
                ext = out_path.suffix.lower()
                is_img = ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp")
                if is_img:
                    try:
                        from PIL import Image as PILImage, ImageTk as PILImageTk
                        img = PILImage.open(out_path)
                        img.thumbnail((300, 200))
                        photo = PILImageTk.PhotoImage(img)
                        img_lbl = tk.Label(card, image=photo, bg=status_bg,
                                           cursor="hand2", relief=tk.SOLID, bd=1)
                        img_lbl.pack(padx=8, pady=4, anchor="w")
                        img_lbl._photo = photo  # type: ignore[attr-defined]
                        img_lbl.bind("<Button-1>", lambda e, p=out_path: open_path(p))
                    except Exception:
                        pass

            row_idx += 1

        if row_idx == 0:
            ttk.Label(self._results_inner, text="No output paths recorded.",
                      foreground="#888888").pack(pady=20)

    # ---------------- Browsers ----------------

    def _browse_net(self):
        p = filedialog.askopenfilename(title="Select net.xml", filetypes=[("XML", "*.xml")])
        if p:
            self.net_var.set(p)
            self._sync_state_from_ui()
            self._refresh_lights()

    def _browse_fcd(self):
        p = filedialog.askopenfilename(
            title="Select fcd.xml",
            filetypes=[("XML", "*.xml")]
        )
        if p:
            self.fcd_var.set(p)
            self._sync_state_from_ui()
            self._refresh_lights()

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output directory")
        if p:
            self.out_var.set(p)
            self._sync_state_from_ui()
            self._refresh_lights()

    def _browse_gt(self):
        p = filedialog.askopenfilename(title="Select GT CSV", filetypes=[("CSV", "*.csv")])
        if p:
            self.gt_var.set(p)
            self._sync_state_from_ui()
            self._refresh_lights()

    def _on_toggle_gt(self):
        self._sync_state_from_ui()
        self._refresh_lights()

    def _on_toggle_sg(self):
        """Called when 'Enable SG smooth' checkbox is toggled on Inputs page."""
        self._sync_state_from_ui()
        self._refresh_lights()

    def _on_toggle_spatial_roi(self, *_args):
        """Handle spatial ROI enable/disable and mode toggle.
        Dynamically updates labels and widget states to avoid ambiguity."""
        self._sync_state_from_ui()
        enabled = self.spatial_enable_roi_var.get()
        mode = self.spatial_roi_mode_var.get()

        if not enabled:
            # All ROI widgets disabled
            self.lbl_roi_sim.config(text="ROI polygon (disabled):")
            self.lbl_roi_gt.config(text="GT ROI (disabled):")
            self.spatial_roi_sim_text_widget.config(state=tk.DISABLED, bg="#E0E0E0")
            self.spatial_roi_gt_text_widget.config(state=tk.DISABLED, bg="#E0E0E0")
        elif mode == "shared":
            # Shared: one ROI for all data, GT box locked
            self.lbl_roi_sim.config(text="\u25b6 ROI polygon (applied to ALL data):")
            self.lbl_roi_gt.config(text="GT ROI (using shared ROI above):")
            self.spatial_roi_sim_text_widget.config(state=tk.NORMAL, bg="white")
            self.spatial_roi_gt_text_widget.config(state=tk.DISABLED, bg="#E0E0E0")
        else:
            # Independent: separate ROI for each
            self.lbl_roi_sim.config(text="Simulation ROI (SUMO/TRIM/SG):")
            self.lbl_roi_gt.config(text="GT ROI (Ground Truth):")
            self.spatial_roi_sim_text_widget.config(state=tk.NORMAL, bg="white")
            self.spatial_roi_gt_text_widget.config(state=tk.NORMAL, bg="white")

    def _on_toggle_sumo_time(self):
        """Enable/disable SUMO time range inputs based on checkbox."""
        enabled = self.enable_sumo_time_var.get()
        state = "normal" if enabled else "disabled"

        self.lbl_sumo_start.config(state=state)
        self.entry_sumo_start.config(state=state)
        self.lbl_sumo_end.config(state=state)
        self.entry_sumo_end.config(state=state)

        if not enabled:
            self.sumo_time_start_var.set("")
            self.sumo_time_end_var.set("")

        self._sync_state_from_ui()

    def _on_toggle_gt_time(self):
        """Enable/disable GT time range inputs based on checkbox."""
        enabled = self.enable_gt_time_var.get()
        state = "normal" if enabled else "disabled"

        self.lbl_gt_start.config(state=state)
        self.entry_gt_start.config(state=state)
        self.lbl_gt_end.config(state=state)
        self.entry_gt_end.config(state=state)

        if not enabled:
            self.gt_time_start_var.set("")
            self.gt_time_end_var.set("")

        self._sync_state_from_ui()

    # ---------------- Sync & Lights ----------------

    def _sync_state_from_ui(self):
        self.state.inputs.net_xml = Path(self.net_var.get()) if self.net_var.get().strip() else None
        self.state.inputs.fcd_input = Path(self.fcd_var.get()) if self.fcd_var.get().strip() else None
        self.state.inputs.out_dir = Path(self.out_var.get()) if self.out_var.get().strip() else None

        self.state.inputs.import_gt = bool(self.import_gt_var.get())
        self.state.inputs.gt_csv = Path(self.gt_var.get()) if self.gt_var.get().strip() else None

        self.state.inputs.raw_prefix = self.prefix_var.get().strip() or "fcd"
        self.state.inputs.safe_prefix = sanitize_prefix(self.state.inputs.raw_prefix, fallback="fcd")

        for k, v in self.pol_vars.items():
            self.state.options.pollutants[k] = bool(v.get())

        try:
            self.state.options.plot_limit = max(0, int(self.plot_limit_var.get().strip()))
        except Exception:
            self.state.options.plot_limit = 10

        # Spatial emission map options
        self.state.options.spatial_axis = self.spatial_axis_var.get()
        self.state.options.spatial_enable_roi = bool(self.spatial_enable_roi_var.get())
        self.state.options.spatial_roi_mode = self.spatial_roi_mode_var.get()
        self.state.options.spatial_roi_sim = self.spatial_roi_sim_var.get().strip()
        self.state.options.spatial_roi_gt = self.spatial_roi_gt_var.get().strip()
        # Backward compatibility: spatial_roi_polygon mirrors spatial_roi_sim
        self.state.options.spatial_roi_polygon = self.state.options.spatial_roi_sim

        try:
            self.state.options.axis_bin_size = max(0.1, float(self.axis_bin_size_var.get().strip()))
        except Exception:
            self.state.options.axis_bin_size = 1.0

        # Update grid_cell_m to match axis_bin_size for backward compatibility
        self.state.options.grid_cell_m = self.state.options.axis_bin_size

        # Time range filtering options
        self.state.options.enable_sumo_time_filter = bool(self.enable_sumo_time_var.get())
        self.state.options.sumo_time_start = self._parse_time_value(self.sumo_time_start_var.get())
        self.state.options.sumo_time_end = self._parse_time_value(self.sumo_time_end_var.get())
        self.state.options.enable_gt_time_filter = bool(self.enable_gt_time_var.get())
        self.state.options.gt_time_start = self._parse_time_value(self.gt_time_start_var.get())
        self.state.options.gt_time_end = self._parse_time_value(self.gt_time_end_var.get())

    def _parse_time_value(self, value: str) -> Optional[float]:
        """
        Parse time value from string input.

        Returns:
            float if valid number, None if empty or invalid
        """
        value = value.strip()
        if not value:
            return None

        try:
            t = float(value)
            if t < 0:
                return None  # Negative time doesn't make sense
            return t
        except ValueError:
            return None  # Invalid input

    def _set_light(self, label: tk.Label, light: Light):
        color_map = {
            Light.RED: "#E74C3C",
            Light.GREEN: "#2ECC71",
            Light.GRAY: "#B0B0B0",
        }
        label.configure(bg=color_map[light])

    def _sync_optional_steps(self):
        """Auto-check/uncheck optional pipeline steps based on Inputs & Options.

        When the user enables GT import (and provides a valid CSV) or enables
        SG smooth on the Inputs & Options page, the corresponding optional
        step checkboxes on the Pipeline page are automatically synchronised
        so the user does not have to tick them manually.
        """
        # SG smooth ↔ use_sg_var
        if "sg_smooth" in self.step_rows:
            self.step_rows["sg_smooth"]["use_var"].set(bool(self.use_sg_var.get()))

        # GT standardize ↔ import_gt_var (only when GT CSV is also provided)
        if "gt_standardize" in self.step_rows:
            gt_enabled = bool(self.import_gt_var.get()) and bool(self.gt_var.get().strip())
            self.step_rows["gt_standardize"]["use_var"].set(gt_enabled)

    def _refresh_lights(self):
        self._sync_state_from_ui()

        ln = light_required_file(self.state.inputs.net_xml, (".xml",))
        lf = light_required_file(self.state.inputs.fcd_input, (".xml"))
        lo = light_out_dir(self.state.inputs.out_dir)
        lg = light_optional_gt(self.state.inputs.import_gt, self.state.inputs.gt_csv)

        if self.light_net: self._set_light(self.light_net, ln)
        if self.light_fcd: self._set_light(self.light_fcd, lf)
        if self.light_out: self._set_light(self.light_out, lo)
        if self.light_gt: self._set_light(self.light_gt, lg)

        # ---- Auto-sync optional step checkboxes in Pipeline page ----
        self._sync_optional_steps()

        ready = is_ready_to_run(self.state)
        can_resume = self.runner.can_resume()

        if ready and not self.runner.is_running():
            self.btn_run_all.state(["!disabled"])
            self.btn_run_sel.state(["!disabled"])
        else:
            self.btn_run_all.state(["disabled"])
            self.btn_run_sel.state(["disabled"])

        # Resume button: enabled only when paused and can resume
        if can_resume and not self.runner.is_running():
            self.btn_resume.state(["!disabled"])
        else:
            self.btn_resume.state(["disabled"])

        if self.runner.is_running():
            self.btn_stop.state(["!disabled"])
        else:
            self.btn_stop.state(["disabled"])

        self.lbl_app_status.configure(text=f"Status: {self.state.status.value}")

    # ---------------- Console ----------------

    def _log(self, msg: str):
        if not self.console_text:
            return
        ts = time.strftime("%H:%M:%S")
        self.console_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.console_text.see(tk.END)

    def _clear_console(self):
        if self.console_text:
            self.console_text.delete("1.0", tk.END)

    # ---------------- Run callbacks ----------------

    def _on_run_all(self):
        self._refresh_lights()
        if not is_ready_to_run(self.state):
            messagebox.showwarning("Not Ready", "Required inputs are not ready (red). Please complete inputs.")
            return

        enable_sg = bool(self.use_sg_var.get())
        steps = build_pipeline(self.state, enable_sg=enable_sg)

        self._log("Run All clicked.")
        self._log(f"safe_prefix = {self.state.inputs.safe_prefix}")
        self._disable_run_buttons_while_running()
        self.runner.run(steps, selected_only=None)

    def _on_run_selected(self):
        self._refresh_lights()
        if not is_ready_to_run(self.state):
            messagebox.showwarning("Not Ready", "Required inputs are not ready (red). Please complete inputs.")
            return

        enable_sg = bool(self.use_sg_var.get())
        steps = build_pipeline(self.state, enable_sg=enable_sg)

        selected = []
        for sid, row in self.step_rows.items():
            # Only allow selecting optional steps if checkbox is on
            if row["use_var"].get():
                selected.append(sid)

        if not selected:
            messagebox.showwarning("No Steps", "No steps selected.")
            return

        # Optional steps sanity: if user selected sg_smooth, ensure enable_sg is True
        if "sg_smooth" in selected and not enable_sg:
            self._log("Note: sg_smooth selected but 'Enable SG smooth' is OFF. It will still run if bound.")

        # Optional GT sanity
        if "gt_standardize" in selected and not self.state.inputs.import_gt:
            self._log("Warning: gt_standardize selected but Import GT is OFF. It may fail due to missing GT path.")

        self._log(f"Run Selected: {selected}")
        self._disable_run_buttons_while_running()
        self.runner.run(steps, selected_only=selected)

    def _disable_run_buttons_while_running(self):
        self.btn_run_all.state(["disabled"])
        self.btn_run_sel.state(["disabled"])
        self.btn_resume.state(["disabled"])  # Also disable resume while running
        self.btn_stop.state(["!disabled"])

    def _on_resume(self):
        """Resume from last paused position."""
        if not self.runner.can_resume():
            messagebox.showinfo("Cannot Resume",
                                "No paused pipeline to resume. Please use 'Run All' or 'Run Selected'.")
            return

        self._log(f"Resuming pipeline from step index {self.state.resume_from_index}...")
        self._disable_run_buttons_while_running()
        self.runner.resume()

    def _on_stop(self):
        if self.runner.is_running():
            self.runner.stop()

    def _on_open_outdir(self):
        self._sync_state_from_ui()
        if self.state.inputs.out_dir:
            open_path(self.state.inputs.out_dir)

    def _on_open_step(self, step_id: str):
        p = self.state.step_outputs.get(step_id)
        if p:
            open_path(p)

    def _apply_step_status_style(self, lbl: tk.Label, status: StepRunStatus) -> None:
        text, fg, bg = self._step_status_style.get(
            status, (status.value, "#34495E", "#ECF0F1")
        )
        lbl.configure(text=text, fg=fg, bg=bg)

    # ---------------- Events polling ----------------

    def _poll_events(self):
        try:
            while True:
                ev = self.event_q.get_nowait()
                if ev.type == EventType.LOG:
                    self._log(ev.message)

                elif ev.type == EventType.STEP_START:
                    self._log(f"[{ev.step_id}] START: {ev.message}")
                    self._set_step_status(ev.step_id, StepRunStatus.RUNNING)

                elif ev.type == EventType.STEP_DONE:
                    outp = ev.path or ""
                    self._log(f"[{ev.step_id}] DONE: {ev.message} -> {outp}")
                    self._set_step_status(ev.step_id, StepRunStatus.DONE)
                    self._enable_open_if_available(ev.step_id)

                elif ev.type == EventType.STEP_ERROR:
                    self._log(f"[{ev.step_id}] ERROR: {ev.message}")
                    self._set_step_status(ev.step_id, StepRunStatus.ERROR)

                elif ev.type == EventType.PIPELINE_STOPPED:
                    self._log(ev.message)
                    self._refresh_lights()

                elif ev.type == EventType.PIPELINE_DONE:
                    self._log(ev.message)
                    if self.state.status == AppRunStatus.ERROR and self.state.last_error:
                        self._log(self.state.last_error)
                    self._refresh_lights()
                    # Auto-populate Results tab and switch to it
                    self._refresh_results()
                    # Switch to Results tab (index 3)
                    try:
                        self._nb.select(3)
                    except Exception:
                        pass

        except queue.Empty:
            pass

        self.root.after(120, self._poll_events)

    def _set_step_status(self, step_id: str, status: StepRunStatus):
        self.state.step_status[step_id] = status
        row = self.step_rows.get(step_id)
        if row:
            lbl = row["lbl_status"]
            # lbl is tk.Label (not ttk.Label) after our change
            self._apply_step_status_style(lbl, status)

    def _enable_open_if_available(self, step_id: str):
        row = self.step_rows.get(step_id)
        if not row:
            return
        p = self.state.step_outputs.get(step_id)
        if p and Path(p).exists():
            row["btn_open"].state(["!disabled"])
        else:
            row["btn_open"].state(["disabled"])


# =========================================================
# 10) Main entry
# =========================================================

def main():
    root = tk.Tk()
    app = TRIMGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()