# -*- coding: utf-8 -*-
"""
trip_split.py

Input:
- CSV from identify_neighbors.py (at minimum: vehicle_id, timestep_time, vehicle_speed)

Purpose:
- Split each vehicle's time series into driving trips (工况) based on stop segments
  (speed==0 or <= eps).
- ALL rows are preserved (including pure-stop segments). Every row gets a trip_id.
- Boundary stop rows (one before / one after a driving segment) are absorbed into
  the adjacent driving segment.
- "Sandwich" zero-speed points (isolated zero flanked by non-zero on both sides)
  are duplicated so they appear in both the preceding and following trip.
- Compute kinematics per vehicle:
  - vehicle_accel  (m/s²)  forward difference: v(t+1) - v(t), last point = 0
  - vehicle_jerk   (m/s³)  forward difference: a(t+1) - a(t), last point = 0
  - distance_m     (m)     per-step displacement: v + 0.5*a  (dt=1s assumed)
  - cumulative_distance (m) cumulative sum of distance_m, shifted so first row = 0
  - vehicle_odometer (m)   first vehicle_pos + cumulative_distance

Logging:
- Uses logger "trim.trip_split" and does NOT configure handlers.
  GUI/pipeline should configure logging handlers.

Core logic rewrite (v2):
- Follows the reference split_trip / adjust_condition_ids approach.
- No rows are deleted; segment labels are reassigned to absorb boundary stops.
- Pure-stop segments receive their own trip_id (not discarded).
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER_NAME = "trim.trip_split"


def _get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


@dataclass
class TripSplitConfig:
    input_csv: str
    out_dir: str
    prefix: str = "fcd"

    # required columns
    col_time: str = "timestep_time"
    col_vid: str = "vehicle_id"
    col_speed: str = "vehicle_speed"

    # optional columns (kept if exist)
    col_pos: str = "vehicle_pos"  # lane-based position, optional

    # behavior
    speed_zero_eps: float = 0.0  # speed <= eps treated as stop
    csv_encoding: str = "utf-8-sig"


# ========================= Utilities =========================

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


# ========================= Core: split_trip per vehicle =========================

def _split_trip_one_vehicle(
    car_data: pd.DataFrame,
    col_time: str,
    col_speed: str,
    eps: float,
) -> pd.DataFrame:
    """
    Segment one vehicle's time-sorted data into trips.

    Logic (following reference split_trip):
    1. Mark zero-speed rows.
    2. Create segments via cumsum on change-points of the zero-speed mask.
    3. For each driving segment (not all-zero), absorb the boundary stop rows
       (the one timestep immediately before and after) by reassigning their
       segment label to the driving segment.
    4. All rows are kept; no data is deleted.

    Returns the input DataFrame with an added 'segment' column.
    """
    car_data = car_data.sort_values(by=[col_time]).reset_index(drop=True)
    car_timestamps = car_data[col_time].tolist()

    # Identify zero-speed rows
    speed_vals = pd.to_numeric(car_data[col_speed], errors="coerce").fillna(0.0)
    if eps <= 0:
        zero_speed_mask = (speed_vals == 0.0)
    else:
        zero_speed_mask = (speed_vals <= eps)

    # Segment labeling: each consecutive run of same stop/non-stop state gets a unique id
    car_data["segment"] = (zero_speed_mask != zero_speed_mask.shift()).cumsum()

    # For each driving segment, absorb boundary stop rows
    for segment_id, group in car_data.groupby("segment"):
        # Skip pure-stop segments — they keep their own segment label
        if zero_speed_mask.loc[group.index].all():
            continue

        # This is a driving segment: try to absorb one stop row before and after
        first_time = group[col_time].iloc[0]
        last_time = group[col_time].iloc[-1]

        prev_timestamp = first_time - 1
        next_timestamp = last_time + 1

        if prev_timestamp in car_timestamps:
            mask = car_data[col_time] == prev_timestamp
            car_data.loc[mask, "segment"] = segment_id

        if next_timestamp in car_timestamps:
            mask = car_data[col_time] == next_timestamp
            car_data.loc[mask, "segment"] = segment_id

    car_data = car_data.sort_values(by=[col_time]).reset_index(drop=True)
    return car_data


def _adjust_condition_ids(
    group: pd.DataFrame,
    col_time: str,
    col_speed: str,
    eps: float,
) -> pd.DataFrame:
    """
    Handle "sandwich" zero-speed points: an isolated zero-speed row where the
    previous row is non-zero and the next row is also non-zero.

    These rows sit at the boundary of two trips. We duplicate such rows so they
    appear in both the preceding trip (with trip_id - 1) and the following trip.

    This ensures both adjacent trips have a proper stop boundary.
    """
    speed_vals = pd.to_numeric(group[col_speed], errors="coerce").fillna(0.0)
    if eps <= 0:
        zero_mask = (speed_vals == 0.0)
    else:
        zero_mask = (speed_vals <= eps)

    # Sandwich condition: current is zero, prev is non-zero, next is non-zero
    cond_sandwich = (
        (zero_mask.shift(1, fill_value=True) == False) &
        (zero_mask == True) &
        (zero_mask.shift(-1, fill_value=True) == False)
    )

    sandwich_rows = group[cond_sandwich]
    if sandwich_rows.empty:
        return group

    # Duplicate sandwich rows
    group = pd.concat([group, sandwich_rows], axis=0, ignore_index=True)

    # For duplicated timesteps, the second occurrence gets trip_id decremented by 1
    # (so it belongs to the preceding trip)
    duplicate_timestamps = group[group.duplicated(subset=[col_time], keep=False)]

    if not duplicate_timestamps.empty:
        for ts in duplicate_timestamps[col_time].unique():
            dup_rows = group[group[col_time] == ts]
            if len(dup_rows) < 2:
                continue
            # The second (and later) occurrences: decrement trip_id
            for idx in dup_rows.iloc[1:].index:
                current_tid = group.loc[idx, "trip_id"]
                if pd.notna(current_tid) and current_tid > 1:
                    group.loc[idx, "trip_id"] = int(current_tid) - 1

    return group


# ========================= Kinematics =========================

def _compute_kinematics_per_vehicle(
    df_vehicle: pd.DataFrame,
    col_time: str,
    col_speed: str,
    col_pos: str,
) -> pd.DataFrame:
    """
    Compute kinematics for one vehicle (already sorted by time):
    - vehicle_accel:  forward difference of speed (v_{t+1} - v_t), last = 0
    - vehicle_jerk:   forward difference of accel (a_{t+1} - a_t), last = 0
    - distance_m:     v + 0.5 * a  (per-step displacement, dt=1s)
    - cumulative_distance: cumsum of distance_m, shifted so first row = 0
    - vehicle_odometer: first vehicle_pos + cumulative_distance
    """
    df = df_vehicle.copy()

    speed = pd.to_numeric(df[col_speed], errors="coerce").fillna(0.0)

    # Acceleration: forward difference (next speed - current speed), last point = 0
    accel = speed.shift(-1) - speed
    accel = accel.fillna(0.0)
    df["vehicle_accel"] = accel

    # Jerk: forward difference of acceleration, last point = 0
    jerk = accel.shift(-1) - accel
    jerk = jerk.fillna(0.0)
    df["vehicle_jerk"] = jerk

    # Per-step displacement: s = v*dt + 0.5*a*dt^2, with dt=1
    df["distance_m"] = speed + 0.5 * accel

    # Cumulative distance: cumsum shifted so first row starts at 0
    cum_dist = df["distance_m"].cumsum().shift(1, fill_value=0.0)
    df["cumulative_distance"] = cum_dist

    # Odometer: first vehicle_pos + cumulative_distance
    if col_pos in df.columns:
        first_pos = pd.to_numeric(df[col_pos].iloc[0], errors="coerce")
        if pd.isna(first_pos):
            first_pos = 0.0
        df["vehicle_odometer"] = first_pos + cum_dist
    else:
        df["vehicle_odometer"] = cum_dist

    return df


# ========================= Main core =========================

def _run_core(cfg: TripSplitConfig) -> str:
    """
    Core implementation.

    Flow:
    1. Read CSV, type cleaning.
    2. Per vehicle: segment-based trip splitting (no rows deleted).
    3. Generate global trip_id via (vehicle_id, segment) ngroup.
    4. Adjust sandwich zero-speed points (duplicate to both adjacent trips).
    5. Compute kinematics per vehicle.
    6. Output CSV.
    """
    logger = _get_logger()

    _ensure_file(cfg.input_csv, "input_csv")
    _safe_mkdir(cfg.out_dir)

    logger.info("=" * 80)
    logger.info("[trip_split] START")
    logger.info("input_csv: %s", cfg.input_csv)
    logger.info("out_dir  : %s", cfg.out_dir)
    logger.info("prefix   : %s", cfg.prefix)
    logger.info("speed_zero_eps: %s", cfg.speed_zero_eps)
    logger.info("=" * 80)

    df = pd.read_csv(cfg.input_csv, encoding=cfg.csv_encoding)

    _require_cols(df, [cfg.col_time, cfg.col_vid, cfg.col_speed], "input_csv")

    # dtype hygiene
    df[cfg.col_time] = pd.to_numeric(df[cfg.col_time], errors="coerce")
    df[cfg.col_speed] = pd.to_numeric(df[cfg.col_speed], errors="coerce")
    df[cfg.col_vid] = df[cfg.col_vid].astype(str)

    df = df.dropna(subset=[cfg.col_time, cfg.col_speed])
    df = df.sort_values([cfg.col_vid, cfg.col_time]).reset_index(drop=True)

    logger.info("[trip_split] Total rows: %d, vehicles: %d", len(df), df[cfg.col_vid].nunique())

    # ---- Step 1: Segment-based trip splitting per vehicle ----
    logger.info("[trip_split] Splitting into segments...")

    vehicle_results = []
    for vid, veh_data in df.groupby(cfg.col_vid, sort=False):
        segmented = _split_trip_one_vehicle(
            veh_data, cfg.col_time, cfg.col_speed, cfg.speed_zero_eps
        )
        vehicle_results.append(segmented)

    df = pd.concat(vehicle_results, ignore_index=True)

    # ---- Step 2: Generate trip_id from (vehicle_id, segment) ----
    df["trip_id"] = df.groupby([cfg.col_vid, "segment"]).ngroup() + 1

    logger.info("[trip_split] Segments created. Unique trip_ids: %d", df["trip_id"].nunique())

    # ---- Step 3: Adjust sandwich zero-speed points ----
    logger.info("[trip_split] Adjusting sandwich zero-speed boundary points...")

    adjusted_parts = []
    for vid, veh_group in df.groupby(cfg.col_vid, sort=False):
        veh_group = veh_group.sort_values(cfg.col_time).reset_index(drop=True)
        adjusted = _adjust_condition_ids(
            veh_group, cfg.col_time, cfg.col_speed, cfg.speed_zero_eps
        )
        adjusted_parts.append(adjusted)

    df = pd.concat(adjusted_parts, ignore_index=True)

    # Drop helper column
    df = df.drop(columns=["segment"], errors="ignore")

    # Sort for kinematics computation
    df = df.sort_values([cfg.col_vid, cfg.col_time]).reset_index(drop=True)

    # ---- Step 4: Compute kinematics per vehicle ----
    logger.info("[trip_split] Computing kinematics (accel, jerk, odometer)...")

    kin_parts = []
    for vid, veh_data in df.groupby(cfg.col_vid, sort=False):
        veh_data = veh_data.sort_values(cfg.col_time).reset_index(drop=True)
        veh_kin = _compute_kinematics_per_vehicle(
            veh_data, cfg.col_time, cfg.col_speed, cfg.col_pos
        )
        kin_parts.append(veh_kin)

    df = pd.concat(kin_parts, ignore_index=True)

    # ---- Step 5: Final sort and output ----
    df = df.sort_values([cfg.col_vid, "trip_id", cfg.col_time]).reset_index(drop=True)

    # Drop internal columns that should not appear in output
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    out_path = os.path.join(cfg.out_dir, f"{cfg.prefix}_trip_split.csv")
    df.to_csv(out_path, index=False, encoding=cfg.csv_encoding)

    logger.info("[trip_split] Output rows: %d", len(df))
    logger.info("[trip_split] Unique vehicles: %d", df[cfg.col_vid].nunique())
    logger.info("[trip_split] Unique trip_ids: %d", df["trip_id"].nunique())
    logger.info("[trip_split] DONE: %s", out_path)

    return out_path


# ========================= Entry points =========================

def run_gui(
        cfg: "TripSplitConfig | None" = None,
        *,
        in_path: str | None = None,
        input_csv: str | None = None,
        out_dir: str | None = None,
        prefix: str = "fcd",
        speed_zero_eps: float = 0.0,
        csv_encoding: str | None = None,
) -> str:
    """
    Compatibility wrapper for GUI adapter.

    Supports BOTH:
      - run_gui(cfg=TripSplitConfig(...))
      - run_gui(in_path=..., out_dir=..., prefix=..., speed_zero_eps=...)

    Note:
      - GUI binding/adapters often pass `in_path` and `out_dir`.
      - This wrapper converts them into TripSplitConfig then calls core logic.
    """
    if cfg is not None:
        return _run_core(cfg)

    # accept either in_path or input_csv
    real_in = input_csv or in_path
    if not real_in:
        raise TypeError("[trip_split] run_gui() missing required argument: 'cfg' or ('in_path'/'input_csv').")
    if not out_dir:
        raise TypeError("[trip_split] run_gui() missing required argument: 'out_dir'.")

    # If in_path is a directory, find the first CSV inside
    real_in_path = str(real_in)
    if os.path.isdir(real_in_path):
        csvs = sorted([f for f in os.listdir(real_in_path) if f.endswith(".csv")])
        if not csvs:
            raise RuntimeError(f"[trip_split] No CSV found in directory: {real_in_path}")
        real_in_path = os.path.join(real_in_path, csvs[0])

    cfg2 = TripSplitConfig(
        input_csv=real_in_path,
        out_dir=str(out_dir),
        prefix=str(prefix),
        speed_zero_eps=float(speed_zero_eps),
    )

    if csv_encoding is not None and hasattr(cfg2, "csv_encoding"):
        setattr(cfg2, "csv_encoding", csv_encoding)

    return _run_core(cfg2)


# Alias for backward compatibility (adapters try "run" as candidate name)
run = run_gui


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

    p = argparse.ArgumentParser(description="Split trajectories into trips based on stop segments.")
    p.add_argument("--input_csv", required=True, help="input CSV from identify_neighbors step")
    p.add_argument("--out_dir", required=True, help="output directory")
    p.add_argument("--prefix", default="fcd", help="output prefix")
    p.add_argument("--speed_zero_eps", type=float, default=0.0, help="speed<=eps treated as stop")
    args = p.parse_args()

    cfg = TripSplitConfig(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        prefix=args.prefix,
        speed_zero_eps=args.speed_zero_eps,
    )
    out_path = run(cfg)
    logger.info("Finished. Output: %s", out_path)


if __name__ == "__main__":
    main()