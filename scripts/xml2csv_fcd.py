# -*- coding: utf-8 -*-
r"""
xml2csv_fcd.py (GUI-friendly)

Purpose:
1) Call external xml2csv.py tool: FCD.xml -> semicolon-separated CSV
2) Convert semicolon CSV -> comma CSV (chunked)
3) Produce a standardized FCD CSV for downstream steps (GUI pipeline)

Standardized output MUST contain (or will abort):
- timestep_time, vehicle_id, vehicle_speed, vehicle_type, vehicle_x, vehicle_y, vehicle_lane, vehicle_pos

If vehicle_accel / vehicle_odometer missing:
- vehicle_accel: broad mapping, else computed by diff(speed)/diff(time) per vehicle; NaN filled with 0
- vehicle_odometer: broad mapping, else integrate ds=v*dt+0.5*a*dt^2 and cumulative sum per vehicle

On missing required columns:
- logger.error + raise ValueError
- Message includes "Please check Help → FCD data specification"

Outputs:
- {out_dir}/{prefix}_fcd_semicolon.csv         (raw)
- {out_dir}/{prefix}_fcd.csv                   (raw comma)
- {out_dir}/{prefix}_fcd_standardized.csv      (for pipeline)
- {out_dir}/{prefix}_xml2csv_report.csv        (report)

CLI usage:
  python xml2csv_fcd.py --xml2csv_py ".../xml2csv.py" --fcd_xml ".../fcd.xml" --out_dir ".../out" --prefix "fcd"

Note:
- This module does NOT add logging handlers; GUI should configure handlers.
- CLI main() will add basicConfig if no handlers exist.
"""

from __future__ import annotations

import os
import sys
import re
import time
import subprocess
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd


LOGGER_NAME = "trim.xml2csv_fcd"


def get_logger() -> logging.Logger:
    """
    Return a logger. Do NOT add handlers here (GUI should configure handlers).
    CLI main() will add basicConfig if needed.
    """
    return logging.getLogger(LOGGER_NAME)


def _ensure_cli_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )


# ------------------------- Config -------------------------

@dataclass
class Xml2CsvConfig:
    xml2csv_py: str
    fcd_xml: str
    out_dir: str
    prefix: str = "fcd"

    # conversion control
    make_comma_csv: bool = True
    make_parquet: bool = False  # optional; kept for compatibility

    # chunk size for semicolon->comma
    chunksize: int = 500_000

    # cwd for running xml2csv tool (default = directory containing xml2csv_py)
    tool_cwd: Optional[str] = None

    # output encoding
    csv_encoding: str = "utf-8-sig"

    # standardization control
    standardized_name_suffix: str = "_fcd_standardized"
    strict_lane_pos: bool = True  # require vehicle_lane & vehicle_pos
    dt_fallback: float = 1.0      # used when dt invalid or missing
    accel_check_report: bool = True  # if accel exists, compute diff-accel stats for report


# ------------------------- Utilities -------------------------

def sanitize_filename(s: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", str(s))


def _ensure_file(path: str, label: str) -> None:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _safe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{name}] missing required columns: {miss}. Existing: {list(df.columns)}")


def _to_numeric_inplace(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _coalesce_series(df: pd.DataFrame, candidates: List[str]) -> Tuple[pd.Series, Optional[str]]:
    """
    Return first candidate column that exists; else all-NaN series.
    Also returns the chosen source column name.
    """
    for c in candidates:
        if c in df.columns:
            return df[c], c
    return pd.Series([np.nan] * len(df), index=df.index), None


def _estimate_dt(df: pd.DataFrame, logger: logging.Logger) -> float:
    """
    Estimate dt from positive diffs of timestep_time.
    Prefer within vehicle_id if present.
    """
    if "timestep_time" not in df.columns:
        return 1.0
    t = pd.to_numeric(df["timestep_time"], errors="coerce")
    if t.notna().sum() == 0:
        return 1.0

    diffs = None
    if "vehicle_id" in df.columns:
        sub = df[["vehicle_id", "timestep_time"]].copy()
        sub["timestep_time"] = pd.to_numeric(sub["timestep_time"], errors="coerce")
        sub = sub.dropna(subset=["timestep_time"])
        if not sub.empty:
            sub["vehicle_id"] = sub["vehicle_id"].astype(str)
            sub = sub.sort_values(["vehicle_id", "timestep_time"], kind="mergesort")
            d = sub.groupby("vehicle_id", sort=False)["timestep_time"].diff()
            d = pd.to_numeric(d, errors="coerce")
            d = d[(d > 0) & np.isfinite(d)]
            if len(d):
                diffs = d.to_numpy(dtype=float)

    if diffs is None or diffs.size == 0:
        t_sorted = np.sort(t.dropna().to_numpy(dtype=float))
        if t_sorted.size >= 2:
            d = np.diff(t_sorted)
            d = d[(d > 0) & np.isfinite(d)]
            if d.size:
                diffs = d

    if diffs is None or diffs.size == 0:
        return 1.0

    med = float(np.median(diffs))
    rounded = np.round(diffs, 6)
    vals, cnts = np.unique(rounded, return_counts=True)
    mode_val = float(vals[int(np.argmax(cnts))])
    dt = mode_val if abs(mode_val - med) <= 1e-6 else med
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0
    logger.info("[dt:auto] estimated dt=%.6f (median=%.6f, mode=%.6f, n=%d)", dt, med, mode_val, diffs.size)
    return dt


def _raise_missing_required(missing: List[str], existing: List[str], logger: logging.Logger) -> None:
    msg = (
        "Missing required columns for standardized FCD output: "
        + ", ".join(missing)
        + ".\n"
        + "Please check Help → FCD data specification and ensure your FCD CSV contains the required fields "
        + "(or configure the xml2csv export accordingly).\n"
        + "Current columns are: "
        + ", ".join(existing)
    )
    logger.error(msg)
    raise ValueError(msg)


# ------------------------- Step 1: xml2csv tool -------------------------

def _run_xml2csv_tool(cfg: Xml2CsvConfig, logger: logging.Logger) -> str:
    """
    Call xml2csv.py to output semicolon-separated CSV.
    Return generated semicolon CSV path (..._fcd_semicolon.csv).
    """
    _ensure_file(cfg.xml2csv_py, "xml2csv_py")
    _ensure_file(cfg.fcd_xml, "fcd_xml")
    _safe_mkdir(cfg.out_dir)

    prefix_safe = sanitize_filename(cfg.prefix)
    out_base = os.path.join(cfg.out_dir, f"{prefix_safe}_fcd_semicolon")
    out_csv = out_base + ".csv"

    tool_cwd = cfg.tool_cwd or os.path.dirname(cfg.xml2csv_py) or None

    cmd = [
        sys.executable,
        cfg.xml2csv_py,
        "-o", out_base,
        "-s", ";",
        "-q", "",
        cfg.fcd_xml
    ]

    logger.info("=" * 80)
    logger.info("[xml2csv_fcd] xml2csv: FCD.xml -> semicolon CSV")
    logger.info("CMD: %s", " ".join(cmd))
    logger.info("CWD: %s", tool_cwd or "(default)")
    logger.info("=" * 80)

    t0 = time.time()
    subprocess.run(cmd, check=True, text=True, encoding="utf-8", cwd=tool_cwd)
    dt = time.time() - t0

    if not os.path.exists(out_csv):
        raise FileNotFoundError(f"xml2csv did not generate expected file: {out_csv}")

    logger.info("[OK] semicolon CSV: %s (time=%.1fs)", out_csv, dt)
    return out_csv


# ------------------------- Step 2: semicolon -> comma -------------------------

def semicolon_to_comma_csv(
    in_semicolon_csv: str,
    out_comma_csv: str,
    chunksize: int,
    encoding: str,
    logger: logging.Logger
) -> str:
    """
    Convert semicolon-separated CSV to comma-separated CSV (chunked).
    """
    _ensure_file(in_semicolon_csv, "in_semicolon_csv")
    _safe_mkdir(os.path.dirname(out_comma_csv))

    logger.info("=" * 80)
    logger.info("[xml2csv_fcd] semicolon CSV -> comma CSV (chunked)")
    logger.info("IN : %s", in_semicolon_csv)
    logger.info("OUT: %s", out_comma_csv)
    logger.info("chunksize=%d", chunksize)
    logger.info("=" * 80)

    first = True
    total_rows = 0
    t0 = time.time()

    reader = pd.read_csv(in_semicolon_csv, sep=";", engine="c", chunksize=chunksize)

    for chunk in reader:
        total_rows += len(chunk)
        chunk.to_csv(
            out_comma_csv,
            index=False,
            encoding=encoding,
            mode=("w" if first else "a"),
            header=first
        )
        first = False

    dt = time.time() - t0
    logger.info("[OK] comma CSV: %s rows=%d (time=%.1fs)", out_comma_csv, total_rows, dt)
    return out_comma_csv


# ------------------------- Step 3: standardize FCD CSV -------------------------

STD_REQUIRED_BASE = ["timestep_time", "vehicle_id", "vehicle_speed", "vehicle_type", "vehicle_x", "vehicle_y"]
STD_REQUIRED_LANE_POS = ["vehicle_lane", "vehicle_pos"]

CANDIDATES: Dict[str, List[str]] = {
    "timestep_time": ["timestep_time", "time", "t", "sim_time", "step"],
    "vehicle_id": ["vehicle_id", "id", "veh_id", "vehicleID", "vehID"],
    "vehicle_speed": ["vehicle_speed", "speed", "veh_speed", "v", "speed_mps", "vehicle_speed_mps"],
    "vehicle_type": ["vehicle_type", "type", "vtype", "vType", "vclass", "vClass"],
    "vehicle_x": ["vehicle_x", "x", "pos_x", "coord_x", "coordinate_x"],
    "vehicle_y": ["vehicle_y", "y", "pos_y", "coord_y", "coordinate_y"],
    "vehicle_lane": ["vehicle_lane", "lane", "lane_id", "laneID", "laneId"],
    "vehicle_pos": ["vehicle_pos", "pos", "lane_pos", "lanePos", "vehicle_position", "position"],
    # optional
    "vehicle_accel": ["vehicle_accel", "accel", "acceleration", "vehicle_acceleration", "vehicle_acceleration_mps2", "accel_ms2"],
    "vehicle_odometer": ["vehicle_odometer", "odometer", "distance", "mileage", "cumulative_distance", "vehicle_mileage", "mileage_pos"],
}


def standardize_fcd_csv(
    in_csv: str,
    out_csv: str,
    strict_lane_pos: bool,
    dt_fallback: float,
    csv_encoding: str,
    accel_check_report: bool,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Read comma CSV, produce standardized CSV.
    Return a report dict (mapping info + stats).
    """
    _ensure_file(in_csv, "in_csv")
    _safe_mkdir(os.path.dirname(out_csv))

    logger.info("=" * 80)
    logger.info("[xml2csv_fcd] standardize FCD CSV")
    logger.info("IN : %s", in_csv)
    logger.info("OUT: %s", out_csv)
    logger.info("strict_lane_pos=%s", strict_lane_pos)
    logger.info("=" * 80)

    t0 = time.time()
    df = pd.read_csv(in_csv, encoding=csv_encoding)

    report: Dict[str, Any] = {
        "in_csv": in_csv,
        "out_csv": out_csv,
        "rows_in": int(len(df)),
        "strict_lane_pos": bool(strict_lane_pos),
        "mapping": {},
        "computed": {},
        "warnings": [],
    }

    # Map/coalesce columns
    std = pd.DataFrame(index=df.index)

    chosen_sources: Dict[str, Optional[str]] = {}
    for std_col, cand in CANDIDATES.items():
        s, src = _coalesce_series(df, cand)
        chosen_sources[std_col] = src
        std[std_col] = s

    report["mapping"] = {k: v for k, v in chosen_sources.items() if v is not None}

    # Required checks (existence in standardized frame)
    missing = []
    for c in STD_REQUIRED_BASE:
        if std[c].isna().all():
            missing.append(c)

    if strict_lane_pos:
        for c in STD_REQUIRED_LANE_POS:
            if std[c].isna().all():
                missing.append(c)

    if missing:
        _raise_missing_required(missing, list(df.columns), logger)

    # Normalize types
    std["vehicle_id"] = std["vehicle_id"].astype(str)
    std["vehicle_type"] = std["vehicle_type"].astype(str)

    # Numeric conversions
    _to_numeric_inplace(std, ["timestep_time", "vehicle_speed", "vehicle_x", "vehicle_y", "vehicle_pos"])
    # lane may be str
    std["vehicle_lane"] = std["vehicle_lane"].astype(str)

    # Sort for per-vehicle ops
    std = std.sort_values(["vehicle_id", "timestep_time"], kind="mergesort").reset_index(drop=True)

    # dt estimation (used for odometer integration fallback)
    dt_auto = _estimate_dt(std, logger)
    dt_used = dt_auto if np.isfinite(dt_auto) and dt_auto > 0 else float(dt_fallback)
    if not np.isfinite(dt_used) or dt_used <= 0:
        dt_used = 1.0
    report["dt_auto"] = float(dt_auto)
    report["dt_used_for_fallback"] = float(dt_used)

    # ---------- vehicle_accel handling ----------
    accel_src = chosen_sources.get("vehicle_accel")
    if accel_src is None or std["vehicle_accel"].isna().all():
        logger.warning("[std] vehicle_accel missing -> computing by diff(speed)/diff(time).")
        report["computed"]["vehicle_accel"] = "diff(speed)/diff(time)"
        g = std.groupby("vehicle_id", sort=False)
        dv = g["vehicle_speed"].diff()
        dt = g["timestep_time"].diff()
        dt = pd.to_numeric(dt, errors="coerce")
        acc = dv / dt
        acc = acc.replace([np.inf, -np.inf], np.nan)
        # invalid dt -> 0
        acc = acc.where((dt > 0) & np.isfinite(dt), 0.0)
        std["vehicle_accel"] = acc.fillna(0.0)
    else:
        # numeric + fill missing 0
        std["vehicle_accel"] = pd.to_numeric(std["vehicle_accel"], errors="coerce")
        # Optional check vs diff-based accel
        if accel_check_report:
            g = std.groupby("vehicle_id", sort=False)
            dv = g["vehicle_speed"].diff()
            dt = g["timestep_time"].diff()
            dt = pd.to_numeric(dt, errors="coerce")
            acc_diff = (dv / dt).replace([np.inf, -np.inf], np.nan)
            valid = (dt > 0) & np.isfinite(dt) & acc_diff.notna() & std["vehicle_accel"].notna()
            if valid.any():
                err = (std.loc[valid, "vehicle_accel"] - acc_diff.loc[valid]).abs()
                report["computed"]["vehicle_accel_check_mae"] = float(err.mean())
                report["computed"]["vehicle_accel_check_p95"] = float(np.nanpercentile(err.to_numpy(dtype=float), 95))
                logger.info("[std] accel check vs diff: MAE=%.4f, P95=%.4f",
                            report["computed"]["vehicle_accel_check_mae"],
                            report["computed"]["vehicle_accel_check_p95"])
        std["vehicle_accel"] = std["vehicle_accel"].fillna(0.0)

    # ---------- vehicle_odometer handling ----------
    odo_src = chosen_sources.get("vehicle_odometer")
    std["vehicle_odometer"] = pd.to_numeric(std["vehicle_odometer"], errors="coerce")
    if odo_src is None or std["vehicle_odometer"].isna().all():
        logger.warning("[std] vehicle_odometer missing -> integrating ds=v*dt+0.5*a*dt^2 and cumsum per vehicle.")
        report["computed"]["vehicle_odometer"] = "integral(v,a,dt)"
        g = std.groupby("vehicle_id", sort=False)
        dt_seg = g["timestep_time"].diff()
        dt_seg = pd.to_numeric(dt_seg, errors="coerce")
        # fallback dt for invalid segments
        dt_seg = dt_seg.where((dt_seg > 0) & np.isfinite(dt_seg), dt_used)
        v = std["vehicle_speed"].fillna(0.0)
        a = std["vehicle_accel"].fillna(0.0)
        ds = v * dt_seg + 0.5 * a * (dt_seg ** 2)
        ds = ds.fillna(0.0)
        std["vehicle_odometer"] = ds.groupby(std["vehicle_id"], sort=False).cumsum()
    else:
        # fill missing using integration only where NaN (simple fill-missing policy)
        miss_mask = std["vehicle_odometer"].isna()
        if miss_mask.any():
            logger.warning("[std] vehicle_odometer partially missing -> filling missing segments by integration policy.")
            report["computed"]["vehicle_odometer_fill_missing"] = int(miss_mask.sum())
            g = std.groupby("vehicle_id", sort=False)
            dt_seg = g["timestep_time"].diff()
            dt_seg = pd.to_numeric(dt_seg, errors="coerce")
            dt_seg = dt_seg.where((dt_seg > 0) & np.isfinite(dt_seg), dt_used)
            v = std["vehicle_speed"].fillna(0.0)
            a = std["vehicle_accel"].fillna(0.0)
            ds = v * dt_seg + 0.5 * a * (dt_seg ** 2)
            ds = ds.fillna(0.0)
            odo_int = ds.groupby(std["vehicle_id"], sort=False).cumsum()
            std.loc[miss_mask, "vehicle_odometer"] = odo_int.loc[miss_mask]
        std["vehicle_odometer"] = std["vehicle_odometer"].fillna(0.0)

    # Final required check (ensure not all NaN)
    final_required = STD_REQUIRED_BASE + (STD_REQUIRED_LANE_POS if strict_lane_pos else [])
    still_missing = [c for c in final_required if std[c].isna().all()]
    if still_missing:
        _raise_missing_required(still_missing, list(df.columns), logger)

    # Output columns (keep only what pipeline needs + accel/odometer)
    out_cols = [
        "timestep_time",
        "vehicle_id",
        "vehicle_speed",
        "vehicle_accel",
        "vehicle_odometer",
        "vehicle_type",
        "vehicle_x",
        "vehicle_y",
        "vehicle_lane",
        "vehicle_pos",
    ]
    std_out = std[out_cols].copy()
    std_out.to_csv(out_csv, index=False, encoding=csv_encoding)

    dt_cost = time.time() - t0
    report["rows_out"] = int(len(std_out))
    report["time_sec"] = float(dt_cost)

    logger.info("[OK] standardized CSV saved: %s (rows=%d, time=%.1fs)", out_csv, len(std_out), dt_cost)
    return report


# ------------------------- Optional: Parquet (kept) -------------------------

def to_parquet(in_csv: str, out_parquet: str, sep: Optional[str], chunksize: int, logger: logging.Logger) -> str:
    _ensure_file(in_csv, "in_csv")
    _safe_mkdir(os.path.dirname(out_parquet))

    logger.info("=" * 80)
    logger.info("[xml2csv_fcd] CSV -> Parquet")
    logger.info("IN : %s", in_csv)
    logger.info("OUT: %s", out_parquet)
    logger.info("chunksize=%d", chunksize)
    logger.info("=" * 80)

    t0 = time.time()
    parts = []
    total_rows = 0

    read_kwargs = {}
    if sep is not None:
        read_kwargs["sep"] = sep
        read_kwargs["engine"] = "c"

    for chunk in pd.read_csv(in_csv, chunksize=chunksize, **read_kwargs):
        total_rows += len(chunk)
        parts.append(chunk)

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.to_parquet(out_parquet, index=False)

    dt = time.time() - t0
    logger.info("[OK] parquet: %s rows=%d (time=%.1fs)", out_parquet, total_rows, dt)
    return out_parquet

# ------------------------- GUI entry wrapper -------------------------

def run_gui(
    *,
    in_path: Any,
    out_dir: Any,
    # GUI extra fields (ignore if not used)
    state=None,
    upstream=None,
    options=None,
    # allow passing tool path explicitly
    xml2csv_py: Optional[str] = None,
    prefix: str = "fcd",
    strict_lane_pos: bool = True,
    dt_fallback: float = 1.0,
    accel_check_report: bool = True,
    chunksize: int = 500_000,
    tool_cwd: Optional[str] = None,
    csv_encoding: str = "utf-8-sig",
    make_parquet: bool = False,
    **kwargs,
) -> str:
    """
    GUI-friendly entrypoint.

    Keeps original implementation route:
      - build Xml2CsvConfig
      - call run(cfg)
      - return standardized_csv
    """
    logger = get_logger()

    fcd_xml = str(in_path)
    out_dir_s = str(out_dir)

    # try infer xml2csv_py from state if not provided
    if xml2csv_py is None and state is not None:
        inp = getattr(state, "inputs", None)
        xml2csv_py = getattr(inp, "xml2csv_py", None) if inp is not None else None

    # NEW (minimal): auto-detect scripts/xml2csv.py if still missing
    if not xml2csv_py:
        try:
            here = os.path.dirname(os.path.abspath(__file__))  # .../scripts
            cand = os.path.join(here, "xml2csv.py")
            if os.path.exists(cand) and os.path.isfile(cand):
                xml2csv_py = cand
                logger.info("[xml2csv_fcd] auto-detected xml2csv.py: %s", xml2csv_py)
        except Exception:
            pass

    if not xml2csv_py:
        raise RuntimeError(
            "[xml2csv_fcd] xml2csv_py is required for GUI run.\n"
            "Tried: state.inputs.xml2csv_py and auto-detect scripts/xml2csv.py.\n"
            "Please provide state.inputs.xml2csv_py OR pass xml2csv_py in bindings.py."
        )

    cfg = Xml2CsvConfig(
        xml2csv_py=str(xml2csv_py),
        fcd_xml=fcd_xml,
        out_dir=out_dir_s,
        prefix=prefix,
        make_comma_csv=True,              # GUI pipeline requires it
        make_parquet=bool(make_parquet),
        chunksize=int(chunksize),
        tool_cwd=tool_cwd,
        csv_encoding=csv_encoding,
        strict_lane_pos=bool(strict_lane_pos),
        dt_fallback=float(dt_fallback),
        accel_check_report=bool(accel_check_report),
    )

    semicolon_csv, comma_csv, parquet_path, standardized_csv, report_csv = run(cfg)

    logger.info("[GUI] standardized_csv: %s", standardized_csv)
    return standardized_csv


# ------------------------- Public: run() -------------------------

def run(cfg: Xml2CsvConfig) -> Tuple[str, Optional[str], Optional[str], str, str]:
    """
    Execute xml2csv_fcd step.
    Return:
      (semicolon_csv, comma_csv_or_None, parquet_or_None, standardized_csv, report_csv)
    """
    logger = get_logger()

    prefix_safe = sanitize_filename(cfg.prefix)
    _safe_mkdir(cfg.out_dir)

    semicolon_csv = _run_xml2csv_tool(cfg, logger)

    comma_csv = None
    if cfg.make_comma_csv:
        comma_csv = os.path.join(cfg.out_dir, f"{prefix_safe}_fcd.csv")
        comma_csv = semicolon_to_comma_csv(
            in_semicolon_csv=semicolon_csv,
            out_comma_csv=comma_csv,
            chunksize=cfg.chunksize,
            encoding=cfg.csv_encoding,
            logger=logger
        )
    else:
        logger.warning("make_comma_csv=False is not recommended for GUI pipeline. Standardization expects comma CSV.")
        # still try to read semicolon if user insists (not implemented here)
        # enforce comma for pipeline
        raise ValueError("make_comma_csv must be True for GUI pipeline standardization.")

    # Standardized output
    standardized_csv = os.path.join(cfg.out_dir, f"{prefix_safe}{cfg.standardized_name_suffix}.csv")
    report = standardize_fcd_csv(
        in_csv=comma_csv,
        out_csv=standardized_csv,
        strict_lane_pos=bool(cfg.strict_lane_pos),
        dt_fallback=float(cfg.dt_fallback),
        csv_encoding=cfg.csv_encoding,
        accel_check_report=bool(cfg.accel_check_report),
        logger=logger
    )

    report_csv = os.path.join(cfg.out_dir, f"{prefix_safe}_xml2csv_report.csv")
    pd.DataFrame([{
        **{f"cfg.{k}": v for k, v in asdict(cfg).items()},
        **{f"report.{k}": v for k, v in report.items() if k != "mapping"},
        "report.mapping": str(report.get("mapping", {})),
    }]).to_csv(report_csv, index=False, encoding=cfg.csv_encoding)
    logger.info("[OK] report saved: %s", report_csv)

    parquet_path = None
    if cfg.make_parquet:
        parquet_path = os.path.join(cfg.out_dir, f"{prefix_safe}_fcd.parquet")
        parquet_path = to_parquet(comma_csv, parquet_path, sep=None, chunksize=cfg.chunksize, logger=logger)

    return semicolon_csv, comma_csv, parquet_path, standardized_csv, report_csv


# ------------------------- CLI -------------------------

def main() -> None:
    _ensure_cli_logging()
    logger = get_logger()

    p = argparse.ArgumentParser(description="Convert SUMO fcd.xml to CSV and produce standardized FCD CSV.")
    p.add_argument("--xml2csv_py", required=True, help="Path to xml2csv.py tool")
    p.add_argument("--fcd_xml", required=True, help="Path to SUMO fcd.xml")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--prefix", default="fcd")

    p.add_argument("--no_comma", action="store_true", help="Disable comma CSV (NOT recommended; GUI requires comma CSV)")
    p.add_argument("--parquet", action="store_true", help="Export parquet (optional)")
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--tool_cwd", default="", help="Working directory for running xml2csv.py (optional)")

    p.add_argument("--strict_lane_pos", action="store_true", help="Require vehicle_lane and vehicle_pos (recommended)")
    p.add_argument("--dt_fallback", type=float, default=1.0)
    p.add_argument("--disable_accel_check", action="store_true", help="Disable accel diff-check report")

    args = p.parse_args()

    cfg = Xml2CsvConfig(
        xml2csv_py=args.xml2csv_py,
        fcd_xml=args.fcd_xml,
        out_dir=args.out_dir,
        prefix=args.prefix,
        make_comma_csv=(not bool(args.no_comma)),
        make_parquet=bool(args.parquet),
        chunksize=int(args.chunksize),
        tool_cwd=(args.tool_cwd if args.tool_cwd else None),
        strict_lane_pos=bool(args.strict_lane_pos),
        dt_fallback=float(args.dt_fallback),
        accel_check_report=(not bool(args.disable_accel_check)),
    )

    semicolon_csv, comma_csv, parquet_path, standardized_csv, report_csv = run(cfg)

    logger.info("Done.")
    logger.info("semicolon CSV   : %s", semicolon_csv)
    logger.info("comma CSV       : %s", comma_csv)
    logger.info("standardized CSV: %s", standardized_csv)
    logger.info("report CSV      : %s", report_csv)
    logger.info("parquet         : %s", parquet_path or "(skipped)")


if __name__ == "__main__":
    main()
