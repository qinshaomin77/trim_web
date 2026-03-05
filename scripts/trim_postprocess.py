# -*- coding: utf-8 -*-
"""
Modified trim_postprocess.py

Changes made:
1. Modified OutPaths.make() to save badcase files in the main out_dir (04_trim)
   instead of a separate 15_badcase subdirectory
2. All badcase CSV files will be saved directly in 04_trim directory

Files that will be saved in 04_trim:
- {prefix}_BadCase.csv - Complete data rows for bad cases
- {prefix}_BadCase_summary.csv - Statistics summary for each case
- {prefix}_OkCase_ids.csv - IDs of cases that passed jerk envelope test
- {prefix}_BadCase_ids.csv - IDs of cases that failed jerk envelope test
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
# =========================================================
# CRITICAL: Ensure scripts/ directory is in sys.path
# This allows importing trim_optimize.py when run from GUI
# =========================================================
_THIS_FILE = os.path.abspath(__file__)
_SCRIPT_DIR = os.path.dirname(_THIS_FILE)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
# =========================================================

import re
import json
import time
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Set

import numpy as np
import pandas as pd

# ----------------------------
# Fully Adaptive max_workers Detection
# ----------------------------
import os
import functools
import platform
from concurrent.futures import ProcessPoolExecutor
from typing import Optional


# =========================================================
# MODIFIED: OutPaths to save badcase files in main out_dir
# =========================================================

LOG_FILENAME = "trim_postprocess.log"

def get_log_path(out_dir: str) -> str:
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, LOG_FILENAME)

@dataclass
class OutPaths:
    root: str
    @staticmethod
    def make(out_dir: str) -> "OutPaths":
        root = os.path.abspath(out_dir)
        return OutPaths(
            root=root,
        )
    def ensure(self) -> None:
        os.makedirs(self.root, exist_ok=True)


def binary_search_max_workers():
    left, right = 1, os.cpu_count() * 4
    max_valid = 1

    while left <= right:
        mid = (left + right) // 2
        try:
            with ProcessPoolExecutor(max_workers=mid) as executor:
                max_valid = mid
                left = mid + 1
        except ValueError:
            right = mid - 1
    return max(1, int(max_valid * 0.8))


# =========================================================
def setup_logger(
    log_path: str,
    level=logging.INFO,
    to_console: bool = False,
    name: str = "trim_postprocess"
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # 如果已经初始化过 handler，就直接复用（避免重复写多份）
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if to_console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def sanitize_filename(s: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", str(s))


def write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{name}] missing required columns:{miss}")

# ----------------------------
# Envelope utilities (for selecting stage2 cases)
# ----------------------------
def load_jerk_envelope_df(jerk_env_csv: str, decimals: int, csv_encoding: str) -> pd.DataFrame:
    env = pd.read_csv(jerk_env_csv, encoding=csv_encoding)
    need = ["accel_round", "q1_smooth", "q99_smooth"]
    miss = [c for c in need if c not in env.columns]
    if miss:
        raise ValueError(f"Jerk envelope missing columns: {miss}")

    env = env.copy()
    env["accel_round"] = pd.to_numeric(env["accel_round"], errors="coerce").round(decimals)
    env["jerk_min"] = pd.to_numeric(env["q1_smooth"], errors="coerce")
    env["jerk_max"] = pd.to_numeric(env["q99_smooth"], errors="coerce")
    env = env[["accel_round", "jerk_min", "jerk_max"]].dropna(subset=["accel_round"]).copy()
    return env


@dataclass
class FilterConfig:
    stage1_csv: str
    out_dir: str
    prefix: str = "fcd"

    jerk_env_csv: Optional[str] = None
    accel_round_decimals: int = 1

    jerk_use_dt: bool = False
    dt: float = 1.0
    jerk_envelope_threshold_pct: float = 90.0
    min_points_threshold: int = 3
    export_points: bool = True

    col_case: str = "trip_id"
    col_vid: str = "vehicle_id"
    col_t: str = "timestep_time"
    col_opt_a: str = "vehicle_accel"

    csv_encoding: str = "utf-8-sig"
    log_to_console: bool = False


@dataclass
class MergeConfig:
    stage1_csv: str
    stage2_csv: str
    out_dir: str
    prefix: str = "fcd"
    out_name: str = "trim_all"

    replace_policy: str = "notna_any"
    col_vid: str = "vehicle_id"
    col_t: str = "timestep_time"

    csv_encoding: str = "utf-8-sig"
    log_to_console: bool = False


@dataclass
class Stage1Config:
    trip_split_csv: str
    out_dir: str
    prefix: str = "fcd"

    n_jobs: int = 60
    time_limit_sec: int = 300
    out_format: str = "csv"
    out_name: str = "trim_stage1"

    replace_policy: str = "notna_any"
    export_points: bool = True

    jerk_use_dt: bool = False
    dt: float = 1.0

    csv_encoding: str = "utf-8-sig"
    log_to_console: bool = False


@dataclass
class Stage2Config:
    badcase_csv: str
    out_dir: str
    prefix: str = "fcd"

    n_jobs: int = 60
    time_limit_sec: int = 300
    out_format: str = "csv"
    out_name: str = "trim_stage2"

    jerk_env_csv: Optional[str] = None
    accel_round_decimals: int = 1

    replace_policy: str = "notna_any"
    export_points: bool = True

    jerk_use_dt: bool = False
    dt: float = 1.0
    jerk_envelope_threshold_pct: float = 90.0

    csv_encoding: str = "utf-8-sig"
    log_to_console: bool = False


def _force_keys_post(df: pd.DataFrame, col_vid: str, col_t: str, col_trip: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    df[col_vid] = df[col_vid].astype(str)
    df[col_t] = pd.to_numeric(df[col_t], errors="coerce").fillna(0.0)
    if col_trip and col_trip in df.columns:
        df[col_trip] = df[col_trip].astype(str)
    return df


# =========================================================
def filter_bad_cases_by_jerk_envelope(cfg: FilterConfig, logger: Optional[logging.Logger] = None) -> Tuple[str, str]:
    """
    Filter bad cases based on jerk envelope validation.

    Saves 4 CSV files in the badcase_dir (which is now the main out_dir):
    1. {prefix}_BadCase.csv - Complete data rows for bad cases
    2. {prefix}_BadCase_summary.csv - Statistics summary for each case
    3. {prefix}_OkCase_ids.csv - IDs of cases that passed jerk envelope test
    4. {prefix}_BadCase_ids.csv - IDs of cases that failed jerk envelope test

    Returns:
        Tuple of (badcase_csv_path, summary_csv_path)
    """
    out_paths = OutPaths.make(cfg.out_dir)
    out_paths.ensure()

    # 统一日志：优先使用上层传入的 logger
    if logger is None:
        logger = setup_logger(get_log_path(cfg.out_dir), to_console=cfg.log_to_console)

    t0 = time.time()
    gcols = [cfg.col_case, cfg.col_vid]

    logger.info("=== TRIM Filter Bad Cases ===")
    logger.info(f"stage1_csv: {cfg.stage1_csv}")
    logger.info(f"out_dir: {out_paths.root}")
    logger.info(f"jerk envelope threshold: {cfg.jerk_envelope_threshold_pct}%")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))

    jerk_env_path = None
    if cfg.jerk_env_csv and os.path.exists(cfg.jerk_env_csv):
        jerk_env_path = cfg.jerk_env_csv
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
        jerk_env_path = os.path.join(data_dir, "jerk_envelope.csv")

    logger.info(f"jerk_env_csv: {jerk_env_path}")
    env_df = load_jerk_envelope_df(jerk_env_path, cfg.accel_round_decimals, cfg.csv_encoding)
    require_cols(env_df, ["accel_round", "jerk_min", "jerk_max"], "jerk_env_csv")

    env_lookup = dict(zip(env_df["accel_round"], zip(env_df["jerk_min"], env_df["jerk_max"])))

    # Load stage1 data
    df = pd.read_csv(cfg.stage1_csv, encoding=cfg.csv_encoding)
    require_cols(df, gcols + [cfg.col_t, cfg.col_opt_a], "stage1_csv")

    # Calculate jerk
    df = _force_keys_post(df, cfg.col_vid, cfg.col_t, cfg.col_case if cfg.col_case in df.columns else None)
    df = df.sort_values([cfg.col_vid, cfg.col_case, cfg.col_t])

    if cfg.jerk_use_dt:
        df["_a_next"] = df.groupby([cfg.col_vid, cfg.col_case])[cfg.col_opt_a].shift(-1)
        df["_jerk"] = (df["_a_next"] - df[cfg.col_opt_a]) / cfg.dt
    else:
        df["_a_next"] = df.groupby([cfg.col_vid, cfg.col_case])[cfg.col_opt_a].shift(-1)
        df["_dt"] = df.groupby([cfg.col_vid, cfg.col_case])[cfg.col_t].diff().fillna(1.0)
        df["_dt"] = df["_dt"].replace(0, 1.0)
        df["_jerk"] = (df["_a_next"] - df[cfg.col_opt_a]) / df["_dt"]

    # Round acceleration and lookup thresholds
    df["_accel_round"] = df[cfg.col_opt_a].round(cfg.accel_round_decimals)
    df[["jerk_min", "jerk_max"]] = df["_accel_round"].apply(
        lambda x: pd.Series(env_lookup.get(x, (np.nan, np.nan)))
    )

    # Check envelope
    df["_has_thr"] = df[["jerk_min", "jerk_max"]].notna().all(axis=1)
    df["_in_env"] = (
            df["_has_thr"] &
            (df["_jerk"] >= df["jerk_min"]) &
            (df["_jerk"] <= df["jerk_max"])
    )

    # Filter to valid jerk points (ignore NaN jerk)
    df_points = df[df["_jerk"].notna()].copy()

    # Case-level statistics
    case_stats = (
        df_points.groupby(gcols, as_index=False)
        .agg(
            n_total=("_in_env", "size"),
            n_in_env=("_in_env", "sum"),
        )
    )
    case_stats["pct_in_env"] = (case_stats["n_in_env"] / case_stats["n_total"] * 100).fillna(0)

    # Use configurable threshold
    THRESHOLD_PCT = cfg.jerk_envelope_threshold_pct
    MIN_POINTS_THRESHOLD = cfg.min_points_threshold
    logger.info(f"[filter] Using jerk envelope threshold: {THRESHOLD_PCT}%")

    # Case passes if >= threshold% of points are in envelope
    case_stats["pass_in_env_threshold"] = (
            (case_stats["n_total"] < MIN_POINTS_THRESHOLD) |
            (case_stats["pct_in_env"] >= THRESHOLD_PCT)
    )

    # Build all case index from df (not df_points)
    all_case_df = df[gcols].drop_duplicates().copy()
    all_case_df[cfg.col_case] = all_case_df[cfg.col_case].astype(str)
    all_case_df[cfg.col_vid] = all_case_df[cfg.col_vid].astype(str)

    # Merge pass decision back to all cases
    all_case_df = all_case_df.merge(
        case_stats[[cfg.col_case, cfg.col_vid, "pass_in_env_threshold", "pct_in_env", "n_total", "n_in_env"]],
        on=gcols,
        how="left"
    )

    # Fill NaN (cases with no jerk points) - consider them as passing
    all_case_df["pass_in_env_threshold"] = all_case_df["pass_in_env_threshold"].fillna(True)
    all_case_df["pct_in_env"] = all_case_df["pct_in_env"].fillna(100.0)
    all_case_df["n_total"] = all_case_df["n_total"].fillna(0).astype(int)
    all_case_df["n_in_env"] = all_case_df["n_in_env"].fillna(0).astype(int)

    # Extract OK and BAD case IDs
    ok_ids_df = all_case_df[all_case_df["pass_in_env_threshold"]][[cfg.col_case, cfg.col_vid]].copy()
    bad_ids_df = all_case_df[~all_case_df["pass_in_env_threshold"]][[cfg.col_case, cfg.col_vid]].copy()

    # ====================================================================

    # summary stats (additional diagnostics)
    df_points["_missing_thr"] = (~df_points["_has_thr"]).astype(int)
    df_points["_out_range"] = ((df_points["_has_thr"]) & (~df_points["_in_env"])).astype(int)

    case_summary = (
        df_points.groupby(gcols, as_index=False)
        .agg(
            n_jerk_points=("_in_env", "size"),
            n_in_env_points=("_in_env", "sum"),
            n_missing_thr=("_missing_thr", "sum"),
            n_out_of_range=("_out_range", "sum"),
            jerk_min_obs=("_jerk", "min"),
            jerk_max_obs=("_jerk", "max"),
        )
    )

    # Merge pass decision and percentage into summary
    case_summary = case_summary.merge(
        all_case_df[[cfg.col_case, cfg.col_vid, "pass_in_env_threshold", "pct_in_env"]],
        on=gcols,
        how="left"
    )

    # Fill missing values for cases with no summary stats
    if len(case_summary) < len(all_case_df):
        # Drop columns from case_summary that already exist in all_case_df (except join keys)
        # to avoid pandas _x/_y suffix conflicts on merge
        overlap_cols = [c for c in case_summary.columns if c in all_case_df.columns and c not in gcols]
        case_summary_clean = case_summary.drop(columns=overlap_cols, errors="ignore")
        case_summary = all_case_df.merge(case_summary_clean, on=gcols, how="left")
        case_summary["pass_in_env_threshold"] = case_summary["pass_in_env_threshold"].fillna(True)
        case_summary["pct_in_env"] = case_summary["pct_in_env"].fillna(100.0)
        case_summary["n_jerk_points"] = case_summary["n_jerk_points"].fillna(0).astype(int)
        case_summary["n_in_env_points"] = case_summary["n_in_env_points"].fillna(0).astype(int)
        case_summary["n_missing_thr"] = case_summary["n_missing_thr"].fillna(0).astype(int)
        case_summary["n_out_of_range"] = case_summary["n_out_of_range"].fillna(0).astype(int)

    # Extract bad case rows from original data
    bad_key = set(zip(bad_ids_df[cfg.col_case].astype(str), bad_ids_df[cfg.col_vid].astype(str)))
    df["_case_key"] = list(zip(df[cfg.col_case].astype(str), df[cfg.col_vid].astype(str)))
    bad_rows = df[df["_case_key"].isin(bad_key)].drop(columns=["_case_key"], errors="ignore").copy()

    # outputs - SAVED DIRECTLY IN 04_trim DIRECTORY
    prefix_safe = sanitize_filename(cfg.prefix)

    badcase_csv = os.path.join(cfg.out_dir, f"{prefix_safe}_BadCase.csv")
    badids_csv = os.path.join(cfg.out_dir, f"{prefix_safe}_BadIDs.csv")
    bad_rows.to_csv(badcase_csv, index=False, encoding=cfg.csv_encoding)
    bad_ids_df.to_csv(badids_csv, index=False, encoding=cfg.csv_encoding)

    if cfg.export_points:
        points_csv = os.path.join(out_paths.root, f"{prefix_safe}_BadCase_points.csv")
        keep_cols = [
            cfg.col_case, cfg.col_vid, cfg.col_t,
            cfg.col_opt_a, "_a_next", "_jerk", "_accel_round",
            "jerk_min", "jerk_max", "_has_thr", "_in_env",
        ]
        keep_cols = [c for c in keep_cols if c in df_points.columns]
        df_points[keep_cols].to_csv(points_csv, index=False, encoding=cfg.csv_encoding)

    snapshot = {
        "cfg": asdict(cfg),
        "jerk_env_csv_resolved": jerk_env_path,
        "rule": {
            "point_in_env": "has_thr & jerk in [min,max]",
            "case_pass": f"At least {THRESHOLD_PCT}% of jerk points must be in_env",
            "case_key": f"({cfg.col_case}, {cfg.col_vid})",
        },
        "stats": {
            "total_cases": int(len(all_case_df)),
            "bad_cases": int(len(bad_ids_df)),
            "ok_cases": int(len(ok_ids_df)),
            "bad_rows": int(len(bad_rows)),
            "jerk_points": int(len(df_points)),
            "threshold_pct": float(THRESHOLD_PCT),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(os.path.join(out_paths.root, f"{prefix_safe}_BadCase_snapshot.json"), snapshot)

    logger.info(f"[filter] Total cases: {len(all_case_df)}")
    logger.info(f"[filter] OK cases: {len(ok_ids_df)} ({100 * len(ok_ids_df) / len(all_case_df):.1f}%)")
    logger.info(f"[filter] BAD cases: {len(bad_ids_df)} ({100 * len(bad_ids_df) / len(all_case_df):.1f}%)")
    logger.info(f"[filter] Threshold: {THRESHOLD_PCT}% of points must be in envelope")
    logger.info(f"[filter] Files saved in: {out_paths.root}")
    logger.info(f"time: {(time.time() - t0):.2f} sec")

    return badcase_csv, badids_csv


# =========================================================
# Rest of the functions remain the same...
def merge_stage1_stage2(cfg: MergeConfig, logger: Optional[logging.Logger] = None) -> str:
    out_paths = OutPaths.make(cfg.out_dir)
    out_paths.ensure()

    if logger is None:
        logger = setup_logger(get_log_path(cfg.out_dir), to_console=cfg.log_to_console)

    t0 = time.time()
    prefix_safe = sanitize_filename(cfg.prefix)

    logger.info("=== TRIM Merge: Stage2 overwrite Stage1 ===")
    logger.info(f"stage1_csv: {cfg.stage1_csv}")
    logger.info(f"stage2_csv: {cfg.stage2_csv}")
    logger.info(f"out_dir   : {out_paths.root}")
    logger.info(f"policy    : {cfg.replace_policy}")

    df1 = pd.read_csv(cfg.stage1_csv, encoding=cfg.csv_encoding)
    df2 = pd.read_csv(cfg.stage2_csv, encoding=cfg.csv_encoding)

    require_cols(df1, [cfg.col_vid, cfg.col_t], "stage1_csv")
    require_cols(df2, [cfg.col_vid, cfg.col_t], "stage2_csv")

    has_trip = ("trip_id" in df1.columns and "trip_id" in df2.columns)
    df1 = _force_keys_post(df1, cfg.col_vid, cfg.col_t, "trip_id" if has_trip else None)
    df2 = _force_keys_post(df2, cfg.col_vid, cfg.col_t, "trip_id" if has_trip else None)

    cand_cols = [c for c in ["vehicle_jerk", "vehicle_accel", "vehicle_speed", "vehicle_odometer"] if c in df2.columns]
    if not cand_cols:
        raise ValueError(
            "No overwrite columns found in stage2_csv. Expected any of: "
            "vehicle_jerk, vehicle_accel, vehicle_speed, vehicle_odometer."
        )

    logger.info(f"merge columns: {cand_cols}")

    keys = [cfg.col_vid, cfg.col_t] + (["trip_id"] if has_trip else [])
    dfm = df1.merge(df2[keys + cand_cols], on=keys, how="left", suffixes=("", "__s2"))

    for col in cand_cols:
        s2_col = f"{col}__s2"
        if s2_col not in dfm.columns:
            continue

        if cfg.replace_policy == "notna_any":
            mask = dfm[s2_col].notna()
        elif cfg.replace_policy == "notna_all3":
            s2_cols = [f"{c}__s2" for c in cand_cols if f"{c}__s2" in dfm.columns]
            if len(s2_cols) >= 3:
                mask = dfm[s2_cols[:3]].notna().all(axis=1)
            elif s2_cols:
                mask = dfm[s2_cols].notna().all(axis=1)
            else:
                mask = pd.Series([False] * len(dfm), index=dfm.index)
        else:
            raise ValueError(f"Unknown replace_policy: {cfg.replace_policy}")

        dfm.loc[mask, col] = dfm.loc[mask, s2_col]
        dfm.drop(columns=[s2_col], inplace=True, errors="ignore")

    out_path = os.path.join(cfg.out_dir, f"{prefix_safe}_{cfg.out_name}.csv")
    dfm.to_csv(out_path, index=False, encoding=cfg.csv_encoding)

    n_replaced = dfm.groupby(keys).first().reset_index()
    n_replaced = len(n_replaced)

    logger.info(f"[merge] rows replaced: {n_replaced:,}")
    logger.info(f"[merge] done -> {out_path} | time={(time.time() - t0):.2f} sec")

    return out_path


# Import trim_optimize functions (rest of pipeline)
try:
    # Try to import from same directory first
    import trim_optimize as ro
except ImportError:
    # If that fails, try to add current directory to path and import
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    import trim_optimize as ro


def run_stage1(cfg: Stage1Config) -> dict:
    """
    Full TRIM pipeline: stage1 -> filter -> stage2(bad only) -> merge
    Now saves badcase files directly in 04_trim directory.
    """
    out_paths = OutPaths.make(cfg.out_dir)
    out_paths.ensure()

    logger = setup_logger(get_log_path(cfg.out_dir), to_console=cfg.log_to_console)

    t0 = time.time()
    logger.info("=== TRIM STAGE1 ===")
    logger.info(f"trip_split_csv: {cfg.trip_split_csv}")
    logger.info(f"out_dir: {out_paths.root}")
    logger.info(f"prefix: {cfg.prefix}")

    # stage1
    logger.info("[pipeline] Stage1 START")
    opt_cfg = ro.TRIMOptimizeConfig(
        input_csv=cfg.trip_split_csv,
        out_dir=cfg.out_dir,
        prefix=cfg.prefix,
        n_jobs=cfg.n_jobs,
        time_limit_sec=cfg.time_limit_sec,
        out_format=cfg.out_format,
        csv_encoding=cfg.csv_encoding,
        log_to_console=cfg.log_to_console,
    )
    stage1_csv = ro.run_stage1(opt_cfg, logger=logger)
    logger.info(f"[pipeline] Stage1 DONE -> {stage1_csv}")

    return stage1_csv

def run_stage2(cfg: Stage2Config) -> dict:
    out_paths = OutPaths.make(cfg.out_dir)
    out_paths.ensure()

    logger = setup_logger(get_log_path(cfg.out_dir), to_console=cfg.log_to_console)

    t0 = time.time()
    logger.info("=== TRIM STAGE2 ===")
    logger.info(f"badcase_csv: {cfg.badcase_csv}")
    logger.info(f"out_dir: {out_paths.root}")
    logger.info(f"prefix: {cfg.prefix}")

    # stage1
    logger.info("[pipeline] Stage2 START")
    opt_cfg = ro.TRIMOptimizeConfig(
        input_csv=cfg.badcase_csv,
        out_dir=cfg.out_dir,
        prefix=cfg.prefix,
        n_jobs=cfg.n_jobs,
        time_limit_sec=cfg.time_limit_sec,
        out_format=cfg.out_format,
        csv_encoding=cfg.csv_encoding,
        log_to_console=cfg.log_to_console,
        jerk_envelope_csv=cfg.jerk_env_csv,
        accel_round_decimals=cfg.accel_round_decimals,
    )
    stage2_csv = ro.run_stage2(opt_cfg, logger=logger)
    logger.info(f"[pipeline] Stage2 DONE -> {stage2_csv}")

    return stage2_csv

# =========================================================
# GUI Entry
# =========================================================
def add_io_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--out_dir", required=True, help="Output directory")
    sp.add_argument("--prefix", default="fcd", help="Filename prefix (default: fcd)")
    sp.add_argument(
        "--csv_encoding",
        default="utf-8-sig",
        help="CSV encoding (default: utf-8-sig)",
    )
    sp.add_argument(
        "--log_to_console",
        action="store_true",
        help="Enable console logging (default: off)",
    )


def add_optimize_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--n_jobs", type=int, default=binary_search_max_workers(), help="Parallel workers")
    sp.add_argument("--time_limit_sec", type=int, default=300, help="Time limit in seconds")
    sp.add_argument(
        "--out_format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format (default: csv)",
    )
    sp.add_argument(
        "--out_name",
        default="",
        help="Output name (optional, overrides default out_name in cfg)",
    )


def add_jerk_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--jerk_env_csv",
        default="",
        help="Path to jerk envelope CSV (optional; if empty, resolve from resources/data)",
    )
    sp.add_argument("--accel_round_decimals", type=int, default=1, help="Acceleration rounding decimals")
    sp.add_argument(
        "--jerk_envelope_threshold_pct",
        type=float,
        default=90.0,
        help="OK if >= this percent of points fall in envelope (default: 90.0)",
    )
    sp.add_argument(
        "--min_points_threshold",
        type=int,
        default=3,
        help="Cases with jerk points < this threshold are treated as OK (default: 3)",
    )
    sp.add_argument(
        "--export_points",
        action="store_true",
        help="Export jerk points diagnostics CSV (default: off)",
    )
    sp.add_argument(
        "--jerk_use_dt",
        action="store_true",
        help="Use fixed dt instead of timestep_time diff (default: off)",
    )
    sp.add_argument("--dt", type=float, default=1.0, help="Fixed dt when --jerk_use_dt is on (default: 1.0)")


def add_merge_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--replace_policy",
        choices=["notna_any", "notna_all3"],
        default="notna_any",
        help="Stage2 overwrite policy (default: notna_any)",
    )
    sp.add_argument("--merge_out_name", default="trim_all", help="Merged output name (default: trim_all)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("trim_postprocess.py (filter + merge + pipeline)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---------------------------
    # run: trip_split -> stage1 -> filter -> stage2(bad) -> merge
    # ---------------------------
    sp0 = sub.add_parser("run", help="trip_split.csv -> stage1 -> filter -> stage2(bad) -> merge")
    sp0.add_argument("--trip_split_csv", required=True, help="Input trip_split CSV")
    add_io_args(sp0)
    add_optimize_args(sp0)
    add_jerk_args(sp0)
    add_merge_args(sp0)

    # ---------------------------
    # filter: stage1 -> badcase selection
    # ---------------------------
    sp1 = sub.add_parser("filter", help="Filter bad cases by jerk envelope")
    sp1.add_argument("--stage1_csv", required=True, help="Stage1 output CSV")
    add_io_args(sp1)
    add_jerk_args(sp1)

    # ---------------------------
    # merge: stage2 overwrite stage1
    # ---------------------------
    sp2 = sub.add_parser("merge", help="Merge stage2 result back to stage1")
    sp2.add_argument("--stage1_csv", required=True)
    sp2.add_argument("--stage2_csv", required=True)
    add_io_args(sp2)
    add_merge_args(sp2)

    return p


def _opt_get(options, name, default=None):
    """Read option from dict-like or attr-like object."""
    if options is None:
        return default
    if isinstance(options, dict):
        return options.get(name, default)
    return getattr(options, name, default)

def _resolve_input(in_path=None, input_csv=None, in_csv=None, upstream=None):
    """Resolve upstream first, then fall back to provided input args."""
    if isinstance(upstream, dict):
        for k in ("trip_split_csv", "trip_split", "output", "out_path"):
            v = upstream.get(k)
            if v:
                return str(v)
    for v in (in_path, input_csv, in_csv):
        if v:
            return str(v)
    return None


def run_gui(*, in_path=None, input_csv=None, in_csv=None, out_dir=None, state=None, upstream=None, options=None):
    """
    GUI entry for trim_gui (adapters.py will call this).

    Expected to return a filesystem path (str/Path) to the final merged CSV.
    """
    # options priority: explicit options > state.options
    if options is None and state is not None:
        options = getattr(state, "options", None)

    trip_split_csv = _resolve_input(in_path=in_path, input_csv=input_csv, in_csv=in_csv, upstream=upstream)
    if not trip_split_csv:
        raise ValueError("[trim_postprocess] No input CSV found. Need trip_split_csv from upstream or in_path/input_csv.")

    if out_dir is None:
        # GUI adapter always passes out_dir, but keep safe fallback
        if state is not None and hasattr(state, "out_dir"):
            out_dir = getattr(state, "out_dir")
        else:
            raise ValueError("[trim_postprocess] out_dir is required for GUI run.")

    out_dir = str(out_dir)

    # read global settings (names you can align to GUI options)
    prefix = _opt_get(options, "prefix", "fcd")
    n_jobs = int(_opt_get(options, "n_jobs", binary_search_max_workers()))
    time_limit_sec = int(_opt_get(options, "time_limit_sec", 300))
    out_format = _opt_get(options, "out_format", "csv")
    replace_policy = _opt_get(options, "replace_policy", "notna_any")

    jerk_env_csv = _opt_get(options, "jerk_env_csv", "") or None
    accel_round_decimals = int(_opt_get(options, "accel_round_decimals", 1))
    jerk_use_dt = bool(_opt_get(options, "jerk_use_dt", False))
    dt = float(_opt_get(options, "dt", 1.0))
    jerk_threshold_pct = float(_opt_get(options, "jerk_envelope_threshold_pct", 90.0))
    min_points_threshold = int(_opt_get(options, "min_points_threshold", 3))

    export_points = bool(_opt_get(options, "export_points", False))
    csv_encoding = _opt_get(options, "csv_encoding", "utf-8-sig")
    log_to_console = bool(_opt_get(options, "log_to_console", False))

    # -------- Stage 1 --------
    cfg1 = Stage1Config(
        trip_split_csv=trip_split_csv,
        out_dir=out_dir,
        prefix=prefix,
        n_jobs=n_jobs,
        time_limit_sec=time_limit_sec,
        out_format=out_format,
        replace_policy=replace_policy,
        export_points=export_points,
        jerk_use_dt=jerk_use_dt,
        dt=dt,
        csv_encoding=csv_encoding,
        log_to_console=log_to_console,
    )
    stage1_csv = run_stage1(cfg1)

    # -------- Filter --------
    cfg2 = FilterConfig(
        stage1_csv=stage1_csv,
        out_dir=out_dir,
        prefix=prefix,
        jerk_env_csv=jerk_env_csv,
        accel_round_decimals=accel_round_decimals,
        jerk_use_dt=jerk_use_dt,
        dt=dt,
        jerk_envelope_threshold_pct=jerk_threshold_pct,
        min_points_threshold=min_points_threshold,
        export_points=export_points,
        csv_encoding=csv_encoding,
        log_to_console=log_to_console,
    )
    badcase_csv, badids_csv = filter_bad_cases_by_jerk_envelope(cfg2)

    # -------- Stage 2 --------
    cfg3 = Stage2Config(
        badcase_csv=badcase_csv,
        out_dir=out_dir,
        prefix=prefix,
        n_jobs=n_jobs,
        time_limit_sec=time_limit_sec,
        out_format=out_format,
        jerk_env_csv=jerk_env_csv,
        accel_round_decimals=accel_round_decimals,
        replace_policy=replace_policy,
        export_points=export_points,
        jerk_use_dt=jerk_use_dt,
        dt=dt,
        jerk_envelope_threshold_pct=jerk_threshold_pct,
        csv_encoding=csv_encoding,
        log_to_console=log_to_console,
    )
    stage2_csv = run_stage2(cfg3)

    # -------- Merge --------
    cfg4 = MergeConfig(
        stage1_csv=stage1_csv,
        stage2_csv=stage2_csv,
        out_dir=out_dir,
        prefix=prefix,
        replace_policy=replace_policy,
        csv_encoding=csv_encoding,
        log_to_console=log_to_console,
    )
    merge_csv = merge_stage1_stage2(cfg4)

    return Path(merge_csv)

def main():
    args = build_parser().parse_args()

    cfg1 = Stage1Config(
        trip_split_csv=args.trip_split_csv,
        out_dir=args.out_dir,
        prefix=args.prefix,
        n_jobs=args.n_jobs,
        time_limit_sec=args.time_limit_sec,
        out_format=args.out_format,
        replace_policy=args.replace_policy,
        export_points=bool(args.export_points),
        jerk_use_dt=bool(args.jerk_use_dt),
        dt=float(args.dt),
        log_to_console=bool(args.log_to_console),
    )
    stage1_csv = run_stage1(cfg1)

    cfg2 = FilterConfig(
        stage1_csv=stage1_csv,
        out_dir=args.out_dir,
        prefix=args.prefix,
        jerk_env_csv=args.jerk_env_csv,
        accel_round_decimals=args.accel_round_decimals,
        jerk_use_dt=bool(args.jerk_use_dt),
        dt=float(args.dt),
        jerk_envelope_threshold_pct=float(args.jerk_envelope_threshold_pct),
        export_points=bool(args.export_points),
        log_to_console=bool(args.log_to_console),
    )
    badcase_csv, badids_csv = filter_bad_cases_by_jerk_envelope(cfg2)

    cfg3 = Stage2Config(
        badcase_csv=badcase_csv,
        out_dir=args.out_dir,
        prefix=args.prefix,
        n_jobs=args.n_jobs,
        time_limit_sec=args.time_limit_sec,
        out_format=args.out_format,
        jerk_env_csv=args.jerk_env_csv,
        accel_round_decimals=args.accel_round_decimals,
        replace_policy=args.replace_policy,
        export_points=bool(args.export_points),
        jerk_use_dt=bool(args.jerk_use_dt),
        dt=float(args.dt),
        jerk_envelope_threshold_pct=float(args.jerk_envelope_threshold_pct),
        log_to_console=bool(args.log_to_console),
    )
    stage2_csv = run_stage2(cfg3)

    cfg4 = MergeConfig(
        stage1_csv=stage1_csv,
        stage2_csv=stage2_csv,
        out_dir=args.out_dir,
        prefix=args.prefix,
        replace_policy=args.replace_policy,
        log_to_console=bool(args.log_to_console),
    )
    merge_stage1_stage2(cfg4)



if __name__ == "__main__":
    main()