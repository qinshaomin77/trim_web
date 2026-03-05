# -*- coding: utf-8 -*-
"""
sg_smooth.py

Input:
- CSV from trip_split.py
  Required columns:
    timestep_time, vehicle_id, trip_id, vehicle_speed, vehicle_odometer

Output (minimal 7 columns):
  timestep_time, vehicle_id, trip_id, vehicle_speed, vehicle_accel, vehicle_jerk, vehicle_odometer

Algorithm:
- Stage A (per trip): Savitzky-Golay smooth speed, derive accel, clip bounds, re-integrate speed for v-a consistency,
  optionally preserve trip distance using input vehicle_odometer span.
- Stage B (per vehicle): rebuild continuous vehicle_odometer by integrating smoothed speed.

Logging:
- Uses logger "trim.sg_smooth" and does NOT configure handlers.
  GUI/pipeline should configure logging handlers.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception as e:  # ImportError or runtime issues
    savgol_filter = None
    _SCIPY_IMPORT_ERROR = e
else:
    _SCIPY_IMPORT_ERROR = None


LOGGER_NAME = "trim.sg_smooth"


def _get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


@dataclass
class SGSmoothConfig:
    input_csv: str
    out_csv: str

    # columns
    col_time: str = "timestep_time"
    col_vid: str = "vehicle_id"
    col_trip: str = "trip_id"
    col_speed: str = "vehicle_speed"
    col_odo: str = "vehicle_odometer"

    # SG params
    window: int = 11
    polyorder: int = 2

    # bounds
    v_min: float = 0.0
    v_max: float = 60.0   # m/s, adjust if needed
    a_min: float = -10.0
    a_max: float =  10.0

    # behavior
    preserve_trip_distance: bool = True
    lock_endpoints_to_zero: bool = True
    zero_eps: float = 1e-6

    # parallel
    n_jobs: int = 1

    # output
    csv_encoding: str = "utf-8-sig"


def _ensure_file(path: str, label: str) -> None:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _safe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}. Existing: {list(df.columns)}")


def _auto_window(n: int, w: int, p: int) -> int:
    """
    Ensure window is odd, <= n, and >= p+2 (odd).
    If not possible, return 0 to indicate "no smoothing".
    """
    if n <= 2:
        return 0
    w = int(w)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < p + 2:
        # minimal odd >= p+2
        w2 = p + 2
        if w2 % 2 == 0:
            w2 += 1
        if w2 <= n:
            w = w2
        else:
            return 0
    return w


def _diff_over_time(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    forward difference with dt, last value copied.
    """
    dt = np.diff(t, append=t[-1])
    dt_safe = np.where(dt <= 0, np.nan, dt)
    dx = np.diff(x, append=x[-1])
    y = dx / dt_safe
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def _integrate_speed(v0: float, a: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    v[i+1] = v[i] + a[i]*dt
    last a uses dt to last (0), harmless.
    """
    dt = np.diff(t, append=t[-1])
    dt = np.where(dt < 0, 0.0, dt)
    v = np.empty_like(a, dtype=float)
    v[0] = float(v0)
    for i in range(1, len(a)):
        v[i] = v[i - 1] + float(a[i - 1]) * float(dt[i - 1])
    return v


def _scale_interior_speed_to_match_distance(v: np.ndarray, t: np.ndarray, target_dist: float) -> np.ndarray:
    """
    Scale interior points (1..n-2) so that integral(v dt) matches target_dist (roughly).
    Keep endpoints fixed.
    """
    n = len(v)
    if n <= 2:
        return v

    dt = np.diff(t, prepend=t[0])
    dt = np.where(dt < 0, 0.0, dt)

    # current distance using trapezoid
    v_prev = np.roll(v, 1)
    v_prev[0] = v[0]
    dist = np.sum(0.5 * (v + v_prev) * dt)

    if not np.isfinite(dist) or dist <= 1e-9 or (not np.isfinite(target_dist)) or target_dist <= 0:
        return v

    ratio = target_dist / dist
    if not np.isfinite(ratio) or ratio <= 0:
        return v

    v2 = v.copy()
    v2[1:-1] *= ratio
    return v2


def _process_one_trip(df_trip: pd.DataFrame, cfg: SGSmoothConfig) -> pd.DataFrame:
    """
    Input: one (vehicle_id, trip_id) group
    Output: same rows with _sm_speed, _sm_accel
    """
    if savgol_filter is None:
        raise ImportError(f"scipy is required for sg_smooth but failed to import: {_SCIPY_IMPORT_ERROR}")

    g = df_trip.sort_values(cfg.col_time).copy()

    t = pd.to_numeric(g[cfg.col_time], errors="coerce").to_numpy()
    v_raw = pd.to_numeric(g[cfg.col_speed], errors="coerce").fillna(0.0).to_numpy()

    n = len(g)
    w = _auto_window(n, cfg.window, cfg.polyorder)

    if w <= 0:
        v_sm = v_raw.copy()
    else:
        v_sm = savgol_filter(v_raw, window_length=w, polyorder=cfg.polyorder, mode="interp")

    # clip v
    v_sm = np.clip(v_sm, cfg.v_min, cfg.v_max)

    # lock endpoints to zero if original endpoints are ~0
    if cfg.lock_endpoints_to_zero:
        if abs(v_raw[0]) <= cfg.zero_eps:
            v_sm[0] = 0.0
        if abs(v_raw[-1]) <= cfg.zero_eps:
            v_sm[-1] = 0.0

    # accel from smoothed v
    a = _diff_over_time(v_sm, t)
    a = np.clip(a, cfg.a_min, cfg.a_max)

    # re-integrate v to enforce v-a consistency (keep initial v)
    v_rec = _integrate_speed(v0=v_sm[0], a=a, t=t)
    v_rec = np.clip(v_rec, cfg.v_min, cfg.v_max)

    # distance preservation within trip (using input odometer span)
    if cfg.preserve_trip_distance and cfg.col_odo in g.columns:
        odo = pd.to_numeric(g[cfg.col_odo], errors="coerce").to_numpy()
        target_dist = float(odo[-1] - odo[0]) if len(odo) >= 2 else float("nan")
        if np.isfinite(target_dist) and target_dist > 0:
            v_rec = _scale_interior_speed_to_match_distance(v_rec, t, target_dist)
            v_rec = np.clip(v_rec, cfg.v_min, cfg.v_max)
            # recompute accel after scaling
            a = _diff_over_time(v_rec, t)
            a = np.clip(a, cfg.a_min, cfg.a_max)
            v_rec = _integrate_speed(v0=v_rec[0], a=a, t=t)
            v_rec = np.clip(v_rec, cfg.v_min, cfg.v_max)

    out = g[[cfg.col_time, cfg.col_vid, cfg.col_trip]].copy()
    out["_sm_speed"] = v_rec
    out["_sm_accel"] = a
    return out


def run(cfg: SGSmoothConfig) -> str:
    logger = _get_logger()

    _ensure_file(cfg.input_csv, "input_csv")
    _safe_mkdir(os.path.dirname(cfg.out_csv))

    df = pd.read_csv(cfg.input_csv)

    _require_cols(df, [cfg.col_time, cfg.col_vid, cfg.col_trip, cfg.col_speed, cfg.col_odo], "input_csv")

    # types
    df[cfg.col_vid] = df[cfg.col_vid].astype(str)
    df[cfg.col_trip] = pd.to_numeric(df[cfg.col_trip], errors="coerce")
    df[cfg.col_time] = pd.to_numeric(df[cfg.col_time], errors="coerce")
    df[cfg.col_speed] = pd.to_numeric(df[cfg.col_speed], errors="coerce")
    df[cfg.col_odo] = pd.to_numeric(df[cfg.col_odo], errors="coerce")

    df = df.dropna(subset=[cfg.col_vid, cfg.col_trip, cfg.col_time]).copy()
    df[cfg.col_trip] = df[cfg.col_trip].astype(int)

    logger.info("=" * 80)
    logger.info("[sg_smooth] START")
    logger.info("input_csv: %s", cfg.input_csv)
    logger.info("out_csv  : %s", cfg.out_csv)
    logger.info("n_jobs   : %d", cfg.n_jobs)
    logger.info("window/polyorder: %d/%d", cfg.window, cfg.polyorder)
    logger.info("bounds v:[%s,%s] a:[%s,%s]", cfg.v_min, cfg.v_max, cfg.a_min, cfg.a_max)
    logger.info("preserve_trip_distance: %s", cfg.preserve_trip_distance)
    logger.info("=" * 80)

    groups = list(df.groupby([cfg.col_vid, cfg.col_trip], sort=False))
    logger.info("trip groups: %d", len(groups))

    results = []

    if cfg.n_jobs and cfg.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=cfg.n_jobs) as ex:
            futures = []
            for (_, _), g in groups:
                futures.append(ex.submit(_process_one_trip, g, cfg))
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for (_, _), g in groups:
            results.append(_process_one_trip(g, cfg))

    df_sm = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if df_sm.empty:
        raise RuntimeError("sg_smooth produced empty result (no trip groups).")

    # Merge smoothed speed/accel back to align with original rows (by vid, trip, time)
    df_key = df[[cfg.col_time, cfg.col_vid, cfg.col_trip]].copy()
    df_key = df_key.merge(df_sm, on=[cfg.col_time, cfg.col_vid, cfg.col_trip], how="left")

    # Stage B: rebuild continuous odometer per vehicle using smoothed speed
    df_key = df_key.sort_values([cfg.col_vid, cfg.col_time]).reset_index(drop=True)

    odo_all = []
    for vid, gv in df_key.groupby(cfg.col_vid, sort=False):
        gv = gv.sort_values(cfg.col_time).copy()
        t = gv[cfg.col_time].to_numpy(dtype=float)
        v = gv["_sm_speed"].fillna(0.0).to_numpy(dtype=float)

        dt = np.diff(t, prepend=t[0])
        dt = np.where(dt < 0, 0.0, dt)
        v_prev = np.roll(v, 1)
        v_prev[0] = v[0]
        dist = 0.5 * (v + v_prev) * dt
        odo = np.cumsum(dist)

        gv["_sm_odo"] = odo
        odo_all.append(gv)

    df_key = pd.concat(odo_all, ignore_index=True)

    # Compute jerk per vehicle from smoothed accel
    jerk_all = []
    for vid, gv in df_key.groupby(cfg.col_vid, sort=False):
        gv = gv.sort_values(cfg.col_time).copy()
        t = gv[cfg.col_time].to_numpy(dtype=float)
        a = gv["_sm_accel"].fillna(0.0).to_numpy(dtype=float)
        j = _diff_over_time(a, t)
        gv["_sm_jerk"] = j
        jerk_all.append(gv)

    df_key = pd.concat(jerk_all, ignore_index=True)

    # Final minimal output (7 cols)
    out = pd.DataFrame({
        "timestep_time": df_key[cfg.col_time].astype(float),
        "vehicle_id": df_key[cfg.col_vid].astype(str),
        "trip_id": df_key[cfg.col_trip].astype(int),
        "vehicle_speed": df_key["_sm_speed"].astype(float),
        "vehicle_accel": df_key["_sm_accel"].astype(float),
        "vehicle_jerk": df_key["_sm_jerk"].astype(float),
        "vehicle_odometer": df_key["_sm_odo"].astype(float),
    })

    out = out.sort_values(["vehicle_id", "timestep_time"]).reset_index(drop=True)
    out.to_csv(cfg.out_csv, index=False, encoding=cfg.csv_encoding)

    logger.info("[OK] Output CSV: %s", cfg.out_csv)
    logger.info("[sg_smooth] DONE")
    return cfg.out_csv


# --------------------------- GUI entry ---------------------------

def run_gui(*, in_path, out_dir, state=None, upstream=None, options=None) -> str:
    """GUI adapter entry point for sg_smooth."""
    from pathlib import Path

    logger = _get_logger()

    in_path = Path(in_path)
    out_dir = Path(out_dir)

    # Resolve input: if directory, find the trip_split CSV inside
    if in_path.is_dir():
        csvs = sorted(in_path.glob("*trip_split*.csv"))
        if not csvs:
            csvs = sorted(in_path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in: {in_path}")
        in_csv = str(csvs[0])
        logger.info("[sg_smooth] Resolved input directory -> %s", in_csv)
    else:
        in_csv = str(in_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = "fcd"
    if state and hasattr(state, "inputs"):
        safe_prefix = getattr(state.inputs, "safe_prefix", "fcd") or "fcd"

    out_csv = str(out_dir / f"{safe_prefix}_sg_smooth.csv")

    cfg = SGSmoothConfig(
        input_csv=in_csv,
        out_csv=out_csv,
    )

    result = run(cfg)
    return result


# --------------------------- CLI ---------------------------

def _ensure_cli_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )


def main() -> None:
    import argparse

    _ensure_cli_logging()
    logger = _get_logger()

    p = argparse.ArgumentParser(description="Savitzky-Golay smoothing baseline for trip-split trajectories.")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--window", type=int, default=11)
    p.add_argument("--polyorder", type=int, default=2)
    p.add_argument("--v_max", type=float, default=60.0)
    p.add_argument("--a_min", type=float, default=-10.0)
    p.add_argument("--a_max", type=float, default=10.0)
    p.add_argument("--no_preserve_dist", action="store_true")
    args = p.parse_args()

    cfg = SGSmoothConfig(
        input_csv=args.input_csv,
        out_csv=args.out_csv,
        n_jobs=args.n_jobs,
        window=args.window,
        polyorder=args.polyorder,
        v_max=args.v_max,
        a_min=args.a_min,
        a_max=args.a_max,
        preserve_trip_distance=(not args.no_preserve_dist),
    )
    out = run(cfg)
    logger.info("Finished. Output: %s", out)


if __name__ == "__main__":
    main()