# -*- coding: utf-8 -*-
"""
ef_match.py (GUI-friendly replacement) - PICKLE FIX

Step: EF Match
- Input: trajectory CSV (recommended from fill_xy output) with standardized schema when possible:
    timestep_time, vehicle_id, vehicle_speed, vehicle_accel, vehicle_odometer,
    vehicle_type, vehicle_x, vehicle_y, data_type
  (trip_id optional; extra cols allowed)
- Optional: standardized ground truth CSV (gt_standardize output):
    timestep_time, vehicle_id, vehicle_speed, vehicle_accel, vehicle_type (+ optional x/y/odometer)

- Factor table: emission_factor.csv with columns:
    vehicle_type, pollutant, speed_kmh, accel_ms2, EmissionFactor_gs

Output:
- <out_dir>/<prefix>_emission_matched.csv (standard schema + {pollutant}_gs cols)
- <out_dir>/<prefix>_emission_match_report.csv
- <out_dir>/<prefix>_unknown_vehicle_types.csv (only if exists)
- optional plots: <out_dir>/plots_emission_speed/*.png

GUI-friendly updates:
- No excessive print; uses logger (GUI can inject its logger/handlers)
- Prefix sanitize (safe filename)
- dt can be "auto" (estimated from timestep_time)
- Robust column coalesce; outputs standardized columns
- Plotting optional; can be kept off by GUI global toggle
"""


from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

# plotting is optional; import lazily


LOGGER_NAME = "trim.ef_match"


# ------------------------- Logger -------------------------

def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def _ensure_cli_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )



# =========================================================
# Fully Adaptive max_workers Detection
# =========================================================
from concurrent.futures import ProcessPoolExecutor
def detect_max_workers():
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

# ------------------------- Config -------------------------

@dataclass
class EFMatchConfig:
    traj_csv: str
    emission_factor_csv: str
    out_dir: str
    prefix: str = "fcd"

    # Optional ground truth
    ground_truth_csv: Optional[str] = None

    # Pollutants
    pollutants: List[str] = None  # e.g., ["NOx", "PM2.5"]

    # Vehicle type handling
    default_vehicle_type: str = "sedan"

    # dt for odometer reconstruction when needed
    dt: str = "auto"  # "auto" or numeric string/float
    dt_fallback: float = 1.0

    # Output encoding
    csv_encoding: str = "utf-8-sig"

    # Plotting (GUI global control)
    enable_plot: bool = False
    plot_n_vehicles: int = 10
    plot_scope: str = "non_gt"  # non_gt | gt_only | all
    plot_workers: int = 0  # 0=auto
    random_seed: int = 42  # fixed seed for reproducibility

    n_jobs: int = 1


# ------------------------- Standard schema -------------------------

STD_BASE_COLS = [
    "timestep_time",
    "vehicle_id",
    "vehicle_speed",
    "vehicle_accel",
    "vehicle_odometer",
    "vehicle_type",
    "vehicle_x",
    "vehicle_y",
    "data_type",
]

vehicle_type_map = {
    "passenger_3": "sedan",
    "passenger_4": "sedan",
    "passenger_5": "sedan",
    "passenger_6": "sedan",
    "passenger_7": "sedan",
    "passenger_8": "sedan",
    "slow_passenger": "sedan",
    "fast_passenger": "sedan",
    "sedan": "sedan",
    "mpv_3": "MPV",
    "mpv_4": "MPV",
    "mpv_5": "MPV",
    "mpv_6": "MPV",
    "mpv_7": "MPV",
    "mpv_8": "MPV",
    "slow_mpv": "MPV",
    "fast_mpv": "MPV",
    "mpv": "MPV",
    "MPV": "MPV",
    "truck_3": "truck",
    "truck_4": "truck",
    "truck_5": "truck",
    "truck_6": "truck",
    "truck_7": "truck",
    "truck_8": "truck",
    "slow_truck": "truck",
    "fast_truck": "truck",
    "truck": "truck",
}

default_vehicletype = 'sedan'
# ------------------------- Utilities -------------------------

def sanitize_filename(s: str) -> str:
    # keep Chinese OK if FS supports, but remove invalid path chars
    return re.sub(r'[\\/*?:"<>|]', "_", str(s))


def _clean_str(s: str) -> str:
    s = str(s)
    s = s.strip().strip('"').strip("'")
    s = s.replace("\r", "").replace("\n", "").replace("\t", "")
    return s


def pollutant_to_col(pollutant: str) -> str:
    """
    Convert pollutant display name to a safe column prefix.
    Examples: "NOx" -> "NOx", "PM2.5" -> "PM25"
    """
    p = _clean_str(pollutant)
    if p.upper() == "NOX":
        return "NOx"
    if p.upper() in {"PM2.5", "PM2_5", "PM2-5", "PM2 5", "PM2,5", "PM25"}:
        return "PM25"
    p2 = p.replace(".", "")
    p2 = re.sub(r"[^A-Za-z0-9_]+", "", p2)
    return p2 if p2 else "POLLUTANT"


def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{name}] missing required columns: {miss}. Existing: {list(df.columns)}")


def _to_numeric_inplace(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def coalesce_cols(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """
    Row-wise coalesce: return first non-null among candidate columns.
    If none exists, returns all-NaN series.
    """
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            out = out.where(out.notna(), s)
    return out


def _estimate_dt_from_df(df: pd.DataFrame, logger: logging.Logger) -> float:
    """
    Estimate dt from timestep_time differences (robust).
    - diff within each (data_type, vehicle_id) if possible; else within vehicle_id; else global
    - use median of positive diffs, with mode fallback for nicer number
    """
    if "timestep_time" not in df.columns:
        return 1.0

    tmp = df[["timestep_time"]].copy()
    tmp["timestep_time"] = pd.to_numeric(tmp["timestep_time"], errors="coerce")
    tmp = tmp.dropna(subset=["timestep_time"])
    if tmp.empty:
        return 1.0

    # prefer grouping if keys exist
    keys = []
    if "data_type" in df.columns:
        keys.append("data_type")
    if "vehicle_id" in df.columns:
        keys.append("vehicle_id")

    diffs_all = []
    if keys:
        sub = df[keys + ["timestep_time"]].copy()
        sub["timestep_time"] = pd.to_numeric(sub["timestep_time"], errors="coerce")
        sub = sub.dropna(subset=["timestep_time"])
        if not sub.empty:
            sub = sub.sort_values(keys + ["timestep_time"], kind="mergesort")
            d = sub.groupby(keys, sort=False)["timestep_time"].diff()
            d = pd.to_numeric(d, errors="coerce")
            d = d[(d > 0) & np.isfinite(d)]
            if len(d):
                diffs_all.append(d.to_numpy(dtype=float))

    if not diffs_all:
        # global diff
        t = np.sort(tmp["timestep_time"].to_numpy(dtype=float))
        d = np.diff(t)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            diffs_all.append(d)

    if not diffs_all:
        return 1.0

    diffs = np.concatenate(diffs_all)
    med = float(np.median(diffs))
    rounded = np.round(diffs, 6)
    vals, cnts = np.unique(rounded, return_counts=True)
    mode_val = float(vals[int(np.argmax(cnts))])

    dt = mode_val if abs(mode_val - med) <= 1e-6 else med
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0

    logger.info("[dt:auto] estimated dt=%.6f (median=%.6f, mode=%.6f, n=%d)", dt, med, mode_val, diffs.size)
    return dt

# ------------------------- Core functions -------------------------

def load_emission_factors(ef_csv: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load emission_factor.csv and return pivoted table + vehicle types.
    Expected columns: vehicle_type, pollutant, speed_kmh, accel_ms2, EmissionFactor_gs
    """
    if not Path(ef_csv).exists():
        raise FileNotFoundError(f"Emission factor file not found: {ef_csv}")

    df_factor = pd.read_csv(ef_csv, encoding="utf-8")
    logger.info("[ef] loaded %d rows from %s", len(df_factor), ef_csv)

    req_cols = ["vehicle_type", "pollutant", "speed_kmh", "accel_ms2", "EmissionFactor_gs"]
    _require_cols(df_factor, req_cols, "emission_factor")

    _to_numeric_inplace(df_factor, ["speed_kmh", "accel_ms2", "EmissionFactor_gs"])

    df_factor["vehicle_type"] = (
        df_factor["vehicle_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    vehicle_type_map_norm = {str(k).strip().lower(): v for k, v in vehicle_type_map.items()}
    mapped = df_factor["vehicle_type"].map(vehicle_type_map_norm)
    df_factor["vehicle_type"] = mapped.fillna(default_vehicletype)

    df_factor["speed_kmh"] = pd.to_numeric(df_factor["speed_kmh"], errors="coerce")
    df_factor["accel_ms2"] = pd.to_numeric(df_factor["accel_ms2"], errors="coerce")
    df_factor["speed_kmh"] = df_factor["speed_kmh"].round(0).clip(lower=0, upper=140).astype("Int64")
    df_factor["accel_ms2"] = df_factor["accel_ms2"].round(1).clip(lower=-2.0, upper=10.0).astype(float)

    # pivot: rows = (vehicle_type, speed_kmh, accel_ms2), cols = pollutant
    factor_wide = df_factor.pivot_table(
        index=["vehicle_type", "speed_kmh", "accel_ms2"],
        columns="pollutant",
        values="EmissionFactor_gs",
        aggfunc="first"
    ).reset_index()


    # flatten column names
    factor_wide.columns.name = None
    factor_wide.columns = [str(c) for c in factor_wide.columns]

    idx_cols = {"vehicle_type", "speed_kmh", "accel_ms2"}
    factor_wide = factor_wide.rename(
        columns={c: f"{c}_gs" for c in factor_wide.columns if c not in idx_cols}
    )

    known_vehicle_types = sorted(df_factor["vehicle_type"].dropna().astype(str).unique())
    logger.info("[ef] known vehicle types: %s", known_vehicle_types)

    available_pollutants = sorted([c for c in factor_wide.columns
                                   if c not in {"vehicle_type", "speed_kmh", "accel_ms2"}])
    logger.info("[ef] available pollutants: %s", available_pollutants)

    return factor_wide


def prepare_trajectory(df_full: pd.DataFrame) -> pd.DataFrame:
    # keep numeric cols numeric (DO NOT include vehicle_type)
    numeric_cols = ["timestep_time", "vehicle_speed", "vehicle_accel", "vehicle_x",
                    "vehicle_y", "vehicle_odometer"]
    _to_numeric_inplace(df_full, numeric_cols)

    # main matching: need speed + accel + vehicle_type
    req_main = ["vehicle_speed", "vehicle_accel", "vehicle_type"]
    missing = [c for c in req_main if c not in df_full.columns]
    if missing:
        raise ValueError(f"Cannot match emissions: missing columns {missing}")

    df_main = df_full.dropna(subset=req_main).copy()

    # normalize vehicle_type + mapping
    df_main["vehicle_type"] = df_main["vehicle_type"].astype(str).str.strip().str.lower()
    default_vehicletype_norm = str(default_vehicletype).strip().lower()
    vehicle_type_map_norm = {str(k).strip().lower(): str(v).strip().lower()
                             for k, v in vehicle_type_map.items()}

    mapped = df_main["vehicle_type"].map(vehicle_type_map_norm)
    df_main["vehicle_type"] = mapped.fillna(default_vehicletype_norm)

    # speed/accel -> numeric + discretize
    v_ms = pd.to_numeric(df_main["vehicle_speed"], errors="coerce")
    a_ms2 = pd.to_numeric(df_main["vehicle_accel"], errors="coerce")

    df_main["speed_kmh"] = (v_ms * 3.6).round().clip(0, 140).astype("Int64")
    df_main["accel_ms2"] = a_ms2.round(1).clip(-2.0, 10.0).astype(float)

    # ensure usable for matching
    df_main = df_main.dropna(subset=["speed_kmh", "accel_ms2", "vehicle_type"])

    return df_main

def match_emission_factors(
    df_main: pd.DataFrame,
    factor_wide: pd.DataFrame,
    pollutants: List[str],
    logger: logging.Logger
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Match emission factors to trajectory data (by vehicle_type, speed_kmh, accel_ms2).
    Returns: (df_out, unknown_vehicle_types)
    """
    pollutants = list(pollutants or [])
    if not pollutants:
        logger.warning("[match] No pollutants selected for matching")

    # --- keys ---
    keys = ["vehicle_type", "speed_kmh", "accel_ms2"]
    for k in keys:
        if k not in df_main.columns:
            raise ValueError(f"[match] df_main missing key column: {k}")
        if k not in factor_wide.columns:
            raise ValueError(f"[match] factor_wide missing key column: {k}")

    # --- build pollutant columns to merge ---
    # allow input like "NOx" -> use "NOx_gs"
    pollutant_cols = []
    for p in pollutants:
        col = p if str(p).endswith("_gs") else f"{p}_gs"
        pollutant_cols.append(col)

    # keep only existing pollutant columns
    existing_pollutant_cols = [c for c in pollutant_cols if c in factor_wide.columns]
    missing_pollutant_cols = [c for c in pollutant_cols if c not in factor_wide.columns]
    if missing_pollutant_cols:
        logger.warning("[match] Missing pollutant columns in factor table: %s", missing_pollutant_cols)

    # --- dtype alignment to avoid merge errors / mismatches ---
    # vehicle_type: str lower
    df_left = df_main.copy()
    df_left["vehicle_type"] = df_left["vehicle_type"].astype(str).str.strip().str.lower()

    df_right = factor_wide[keys + existing_pollutant_cols].copy()
    df_right["vehicle_type"] = df_right["vehicle_type"].astype(str).str.strip().str.lower()

    # speed_kmh: use Int64 on both sides (or both float) — choose one, must match
    df_left["speed_kmh"] = pd.to_numeric(df_left["speed_kmh"], errors="coerce").round().astype("Int64")
    df_right["speed_kmh"] = pd.to_numeric(df_right["speed_kmh"], errors="coerce").round().astype("Int64")

    # accel_ms2: 1 decimal float on both sides
    df_left["accel_ms2"] = pd.to_numeric(df_left["accel_ms2"], errors="coerce").round(1).astype(float)
    df_right["accel_ms2"] = pd.to_numeric(df_right["accel_ms2"], errors="coerce").round(1).astype(float)

    # --- unknown vehicle types (in trajectory but not in EF table) ---
    known_types = set(df_right["vehicle_type"].dropna().unique())
    unknown_vehicle_types = sorted(set(df_left["vehicle_type"].dropna().unique()) - known_types)
    if unknown_vehicle_types:
        logger.warning("[match] Unknown vehicle types (mapped to no EF rows): %s", unknown_vehicle_types)

    # --- merge ---
    df_out = df_left.merge(df_right, on=keys, how="left")

    # --- optional: report match rate for the first pollutant col (if any) ---
    if existing_pollutant_cols:
        miss_rate = float(df_out[existing_pollutant_cols[0]].isna().mean()) * 100.0
        logger.info("[match] EF missing rate (based on %s): %.2f%%", existing_pollutant_cols[0], miss_rate)

    return df_out

def _build_match_report(df_out: pd.DataFrame, emission_cols: List[str]) -> pd.DataFrame:
    """Build a summary report of emission matching results."""
    report_rows = []
    for col in emission_cols:
        if col in df_out.columns:
            total = len(df_out)
            null_count = df_out[col].isna().sum()
            present = (total - null_count) > 0
            null_rate = null_count / max(total, 1)
            report_rows.append({
                "col": col,
                "total": total,
                "present": present,
                "null": null_count,
                "null_rate": null_rate
            })
        else:
            report_rows.append({
                "col": col,
                "total": 0,
                "present": False,
                "null": 0,
                "null_rate": 1.0
            })
    return pd.DataFrame(report_rows)


def _standardize_output_schema(df_full: pd.DataFrame, emission_cols: List[str], dt: float,
                               logger: logging.Logger) -> pd.DataFrame:
    """
    Standardize output to consistent schema with emission columns.
    """
    df_out = df_full.copy()

    # Ensure standard columns order
    output_cols = STD_BASE_COLS.copy()

    # Add emission columns
    for col in emission_cols:
        if col in df_out.columns:
            output_cols.append(col)

    # Add any other columns that exist
    for col in df_out.columns:
        if col not in output_cols:
            output_cols.append(col)

    # Select and reorder columns
    available_cols = [c for c in output_cols if c in df_out.columns]
    df_out = df_out[available_cols].copy()

    # Sort by data_type, vehicle_id, timestep_time
    sort_cols = []
    if "data_type" in df_out.columns:
        sort_cols.append("data_type")
    if "vehicle_id" in df_out.columns:
        sort_cols.append("vehicle_id")
    if "timestep_time" in df_out.columns:
        sort_cols.append("timestep_time")

    if sort_cols:
        df_out = df_out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    return df_out


# ------------------------- Plotting -------------------------

def plot_vehicle_emission_speed_panels(df_vehicle: pd.DataFrame, emission_col: str, pollutant_label: str,
                                       out_png: Path) -> None:
    """
    Plot emission vs speed for one vehicle, with panels by data_type.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if emission_col not in df_vehicle.columns:
        return

    # Filter valid data
    df_plot = (
        df_vehicle
        .dropna(subset=["vehicle_speed", emission_col])
        .sort_values(["data_type", "vehicle_id", "timestep_time"], kind="mergesort")
        .copy()
    )
    if df_plot.empty:
        return

    # Group by data_type
    data_types = df_plot["data_type"].unique() if "data_type" in df_plot.columns else ["data"]
    n_types = len(data_types)

    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4))
    if n_types == 1:
        axes = [axes]

    for i, dtype in enumerate(data_types):
        ax = axes[i]
        if "data_type" in df_plot.columns:
            subset = df_plot[df_plot["data_type"] == dtype]
        else:
            subset = df_plot

        if not subset.empty:
            if "timestep_time" in subset.columns:
                subset_sorted = subset.sort_values("timestep_time")
                ax.plot(subset_sorted["vehicle_speed"] * 3.6, subset_sorted[emission_col], alpha=0.7, linewidth=1.5)
            else:
                ax.plot(subset["vehicle_speed"] * 3.6, subset[emission_col], alpha=0.7, linewidth=1.5)

            ax.set_xlabel("Speed (km/h)")
            ax.set_ylabel(f"{pollutant_label} (g/s)")
            ax.set_title(f"{dtype}")
            ax.grid(True, alpha=0.3)

    plt.suptitle(f"Vehicle {df_vehicle['vehicle_id'].iloc[0]} - {pollutant_label} Emission vs Speed")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------- FIXED: Module-level worker function -------------------------

def _plot_worker(args):
    """
    Module-level worker function for multiprocessing plotting.
    This avoids the pickle error that occurs with nested functions.

    Args: (df_records, emission_col, pollutant_label, out_png_str)
    """
    df_records, emission_col, pollutant_label, out_png_str = args

    # Reconstruct DataFrame from records
    df_vehicle = pd.DataFrame(df_records)
    out_png = Path(out_png_str)

    try:
        plot_vehicle_emission_speed_panels(df_vehicle, emission_col, pollutant_label, out_png)
        return out_png_str
    except Exception as e:
        return f"ERROR: {str(e)}"


def run_plotting(df_out_std: pd.DataFrame, selected_pollutants_norm: List[str], plots_dir: Path,
                 plot_n_vehicles: int, plot_workers: int, random_seed: int, plot_scope: str,
                 logger: logging.Logger) -> None:
    """
    Generate emission vs speed plots for sample vehicles.
    """
    if not selected_pollutants_norm:
        logger.warning("[plot] No pollutants to plot")
        return

    # Filter by plot scope
    if plot_scope == "gt_only" and "data_type" in df_out_std.columns:
        df_plot = df_out_std[df_out_std["data_type"] == "ground_truth"].copy()
    elif plot_scope == "non_gt" and "data_type" in df_out_std.columns:
        df_plot = df_out_std[df_out_std["data_type"] != "ground_truth"].copy()
    else:
        df_plot = df_out_std.copy()

    if df_plot.empty:
        logger.warning("[plot] No data after scope filter: %s", plot_scope)
        return

    # Sample vehicles
    vehicle_ids = df_plot["vehicle_id"].unique()
    total = len(vehicle_ids)
    n = min(plot_n_vehicles, total)

    if n <= 0:
        logger.warning("[plot] No vehicles to plot")
        return

    np.random.seed(random_seed)
    if n >= total:
        selected = sorted(vehicle_ids)
        logger.info("[plot] plotting all %d vehicles", total)
    else:
        selected = sorted(np.random.choice(vehicle_ids, size=n, replace=False).tolist())
        logger.info("[plot] plotting random sample %d/%d (seed=%d)", n, total, random_seed)

    poll_pairs = []
    for p in selected_pollutants_norm:
        safe = pollutant_to_col(p)
        col = f"{safe}_gs"
        label = "NOx" if str(p).upper() == "NOX" else ("PM2.5" if str(p).upper() == "PM2.5" else str(p))
        poll_pairs.append((safe, col, label))

    tasks = []
    for vid in selected:
        dfv = df_plot[df_plot["vehicle_id"].astype(str) == str(vid)].copy()
        for safe_poll, col_gs, label in poll_pairs:
            if col_gs in dfv.columns:
                # FIXED: Convert to dict records instead of passing DataFrame directly
                dfv_records = dfv.to_dict('records')
                tasks.append((dfv_records, col_gs, label, str(plots_dir / f"{safe_poll}_{vid}.png")))

    if not tasks:
        logger.warning("[plot] No tasks created (missing emission cols?)")
        return

    workers = detect_max_workers()

    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[plot] tasks=%d, workers=%d, out=%s", len(tasks), workers, str(plots_dir))

    # Keep it simple: if workers==1, run in-process to keep logger stable
    if workers <= 1:
        for dfv_records, col_gs, label, out_png_str in tasks:
            try:
                df_vehicle = pd.DataFrame(dfv_records)
                plot_vehicle_emission_speed_panels(df_vehicle, col_gs, label, Path(out_png_str))
            except Exception as e:
                logger.warning("[plot] failed %s: %s", Path(out_png_str).name, str(e))
        return

    # FIXED: Use module-level worker function
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futs = [exe.submit(_plot_worker, task) for task in tasks]
        done = 0
        for fut in as_completed(futs):
            try:
                result = fut.result()
                if isinstance(result, str) and result.startswith("ERROR:"):
                    logger.warning("[plot] failed: %s", result)
            except Exception as e:
                logger.warning("[plot] failed: %s", str(e))
            done += 1
            if done % 20 == 0 or done == len(futs):
                logger.info("[plot] progress: %d/%d", done, len(futs))


# ------------------------- Main runner -------------------------

def run_ef_match(cfg: EFMatchConfig, logger: Optional[logging.Logger] = None) -> str:
    """
    Main emission factor matching function.
    """
    if logger is None:
        logger = get_logger()

    logger.info("[EF-MATCH] starting emission factor matching")
    logger.info("[config] traj_csv=%s", cfg.traj_csv)
    logger.info("[config] emission_factor_csv=%s", cfg.emission_factor_csv)
    logger.info("[config] out_dir=%s", cfg.out_dir)
    logger.info("[config] prefix=%s", cfg.prefix)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix_safe = sanitize_filename(str(cfg.prefix))

    # Load trajectory
    if not Path(cfg.traj_csv).exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {cfg.traj_csv}")

    df_full = pd.read_csv(cfg.traj_csv, encoding="utf-8")
    logger.info("[traj] loaded %d rows from %s", len(df_full), cfg.traj_csv)

    # Load ground truth if provided
    if cfg.ground_truth_csv and Path(cfg.ground_truth_csv).exists():
        df_gt = pd.read_csv(cfg.ground_truth_csv, encoding="utf-8")
        logger.info("[gt] loaded %d rows from %s", len(df_gt), cfg.ground_truth_csv)

        # Add data_type to distinguish
        if "data_type" not in df_full.columns:
            df_full["data_type"] = "trajectory"
        if "data_type" not in df_gt.columns:
            df_gt["data_type"] = "ground_truth"

        # Combine
        df_full = pd.concat([df_full, df_gt], ignore_index=True, sort=False)
        logger.info("[combined] total %d rows after merging GT", len(df_full))

    # Load emission factors
    factor_wide = load_emission_factors(cfg.emission_factor_csv, logger)

    # Validate and normalize pollutants
    available_pollutants = [c for c in factor_wide.columns if c not in {"vehicle_type", "speed_kmh", "accel_ms2"}]
    if not available_pollutants:
        raise ValueError("No pollutants found in emission factor table")

    selected_pollutants = cfg.pollutants or ["NOx"]
    selected_pollutants_norm = []
    for p in selected_pollutants:
        p_clean = _clean_str(p)
        p_col = f"{p_clean}_gs"
        if not p_clean:
            continue
        # find matching available pollutant (case-insensitive)
        found = None
        for avail in available_pollutants:
            if str(avail).strip().lower() == p_col.lower():
                found = avail
                break
        if found:
            selected_pollutants_norm.append(found)
        else:
            logger.warning("[pollutant] '%s' not found in factor table (available: %s)", p_clean, available_pollutants)

    if not selected_pollutants_norm:
        logger.warning("[pollutant] No valid pollutants selected, defaulting to first available: %s",
                       available_pollutants[0])
        selected_pollutants_norm = [available_pollutants[0]]

    logger.info("[pollutant] selected: %s", selected_pollutants_norm)

    # Auto dt estimation
    dt = cfg.dt
    if str(dt).lower() == "auto":
        dt = _estimate_dt_from_df(df_full, logger)
    else:
        try:
            dt = float(dt)
            if dt <= 0:
                dt = float(cfg.dt_fallback)
                logger.warning("[dt] invalid dt, using fallback=%.6f", dt)
        except (ValueError, TypeError):
            dt = float(cfg.dt_fallback)
            logger.warning("[dt] invalid dt format, using fallback=%.6f", dt)
        else:
            logger.info("[dt] using dt=%.6f", dt)

    # Prepare
    df_main = prepare_trajectory(df_full)

    # Match
    df_out_full = match_emission_factors(
        df_main=df_main,
        factor_wide=factor_wide,
        pollutants=selected_pollutants_norm,
        logger=logger,
    )

    # Match report
    match_cols = [f"{pollutant_to_col(p)}_gs" for p in selected_pollutants_norm]
    report_df = _build_match_report(df_out_full, match_cols)
    report_path = out_dir / f"{prefix_safe}_emission_match_report.csv"
    report_df.to_csv(report_path, index=False, encoding=cfg.csv_encoding)
    for _, r in report_df.iterrows():
        logger.info("[report] %s present=%s null=%d null_rate=%.2f%%",
                    r["col"], bool(r["present"]), int(r["null"]), 100.0 * float(r["null_rate"]))

    # Standardize output
    df_out_std = _standardize_output_schema(df_out_full, emission_cols=match_cols, dt=float(dt), logger=logger)

    out_csv = out_dir / f"{prefix_safe}_emission_matched.csv"
    df_out_std.to_csv(out_csv, index=False, encoding=cfg.csv_encoding)
    logger.info("[OK] emission matched saved: %s", str(out_csv))
    logger.info("[OK] output columns: %s", ", ".join(df_out_std.columns.tolist()))

    # Optional plotting
    if cfg.enable_plot:
        plots_dir = out_dir / "plots_emission_speed"
        run_plotting(
            df_out_std=df_out_std,
            selected_pollutants_norm=selected_pollutants_norm,
            plots_dir=plots_dir,
            plot_n_vehicles=int(cfg.plot_n_vehicles),
            plot_workers=int(cfg.plot_workers),
            random_seed=int(cfg.random_seed),
            plot_scope=str(cfg.plot_scope),
            logger=logger,
        )
        logger.info("[OK] plots saved under: %s", str(plots_dir))
    else:
        logger.info("[plot] disabled (global setting).")

    return str(out_csv)


# ------------------------- GUI Integration -------------------------

def run_gui(*, in_path, out_dir, state=None, upstream=None, options=None):
    """
    GUI-friendly wrapper for run_ef_match.

    Expected inputs:
    - in_path: Path to filled XY CSV (from fill_xy step)
    - out_dir: Output directory for this step
    - state: AppState object containing inputs and options
    - upstream: Dict of upstream step outputs
    - options: Additional options (optional)

    Returns:
    - Path to output CSV file
    """
    logger = get_logger()

    # Validate inputs
    if not in_path or not Path(in_path).exists():
        raise FileNotFoundError(f"Input trajectory CSV not found: {in_path}")

    if not state:
        raise ValueError("State object is required")

    # Setup output
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate safe prefix
    safe_prefix = getattr(state.inputs, 'safe_prefix', 'fcd')

    # Determine emission factor file path
    # Look for emission_factor.csv in project root or data directory
    project_root = Path(__file__).parent.parent if hasattr(Path(__file__), 'parent') else Path(".")
    possible_factor_paths = [
        project_root / "emission_factor.csv",
        project_root / "data" / "emission_factor.csv",
        Path("emission_factor.csv"),
        Path("data") / "emission_factor.csv"
    ]

    emission_factor_csv = None
    for path in possible_factor_paths:
        if path.exists():
            emission_factor_csv = str(path)
            break

    if not emission_factor_csv:
        raise FileNotFoundError(
            f"Emission factor file not found. Searched in: {[str(p) for p in possible_factor_paths]}\n"
            "Please place emission_factor.csv in the project root or data/ directory."
        )

    # Check for optional ground truth
    gt_csv = None
    if getattr(state.inputs, 'import_gt', False) and upstream:
        gt_path = upstream.get('gt_standardize')
        if gt_path and Path(gt_path).exists():
            gt_csv = str(gt_path)

    # Get pollutants from state options
    pollutants = []
    if hasattr(state.options, 'pollutants'):
        pollutants = [k for k, v in state.options.pollutants.items() if v]

    if not pollutants:
        pollutants = ["NOx"]  # default fallback

    # Configure plotting
    enable_plots = getattr(state.options, 'enable_plots', False)
    plot_limit = getattr(state.options, 'plot_limit', 10)

    # Build configuration
    cfg = EFMatchConfig(
        traj_csv=str(in_path),
        emission_factor_csv=emission_factor_csv,
        out_dir=str(out_dir),
        prefix=safe_prefix,

        # Optional ground truth
        ground_truth_csv=gt_csv,

        # Pollutants
        pollutants=pollutants,

        # Vehicle type handling
        default_vehicle_type="sedan",

        # Time step
        dt="auto",
        dt_fallback=1.0,

        # Plotting
        enable_plot=enable_plots,
        plot_n_vehicles=plot_limit,
        plot_scope="non_gt",  # Focus on trajectory data, not ground truth
        plot_workers=0,  # Auto-detect
        random_seed=getattr(state.options, 'random_seed', 42),
    )

    logger.info("[ef_match] Starting emission factor matching")
    logger.info(f"[ef_match] Input trajectory: {in_path}")
    logger.info(f"[ef_match] Emission factors: {emission_factor_csv}")
    logger.info(f"[ef_match] Ground truth: {gt_csv or 'Not used'}")
    logger.info(f"[ef_match] Pollutants: {pollutants}")
    logger.info(f"[ef_match] Output: {out_dir}")

    # Run the main function
    try:
        result_path = run_ef_match(cfg, logger=logger)
        logger.info(f"[ef_match] Successfully completed. Output: {result_path}")
        return result_path
    except Exception as e:
        logger.error(f"[ef_match] Failed: {e}")
        raise


# ------------------------- CLI wrapper -------------------------

def main() -> None:
    import argparse

    _ensure_cli_logging()
    logger = get_logger()

    p = argparse.ArgumentParser(description="EF match: match trajectories with emission_factor.csv (GUI-friendly).")
    p.add_argument("--traj_csv", required=True)
    p.add_argument("--emission_factor_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prefix", default="fcd")

    p.add_argument("--ground_truth_csv", default=None)

    p.add_argument("--pollutants", default="NOx", help="Comma-separated, e.g., 'NOx,PM2.5,CO'")
    p.add_argument("--default_vehicle_type", default="sedan")

    p.add_argument("--dt", default="auto", help="dt seconds or 'auto'")
    p.add_argument("--dt_fallback", type=float, default=1.0)

    p.add_argument("--enable_plot", action="store_true")
    p.add_argument("--plot_n_vehicles", type=int, default=10)
    p.add_argument("--plot_scope", choices=["non_gt", "gt_only", "all"], default="non_gt")
    p.add_argument("--plot_workers", type=int, default=0)
    p.add_argument("--random_seed", type=int, default=42)

    args = p.parse_args()

    pollutants = [_clean_str(x) for x in str(args.pollutants).split(",") if _clean_str(x)]
    if not pollutants:
        pollutants = ["NOx"]

    cfg = EFMatchConfig(
        traj_csv=args.traj_csv,
        emission_factor_csv=args.emission_factor_csv,
        out_dir=args.out_dir,
        prefix=args.prefix,
        ground_truth_csv=args.ground_truth_csv,
        pollutants=pollutants,
        default_vehicle_type=args.default_vehicle_type,
        dt=args.dt,
        dt_fallback=float(args.dt_fallback),
        enable_plot=bool(args.enable_plot),
        plot_n_vehicles=int(args.plot_n_vehicles),
        plot_scope=str(args.plot_scope),
        plot_workers=int(args.plot_workers),
        random_seed=int(args.random_seed),
    )

    out_csv = run_ef_match(cfg, logger=logger)
    logger.info("Done. Output: %s", out_csv)


if __name__ == "__main__":
    main()