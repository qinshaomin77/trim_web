# -*- coding: utf-8 -*-
"""
fill_xy（copy）.py (GUI-friendly) - MULTIPROCESSING ENABLED VERSION

Purpose
- Merge multiple trajectory datasets (SUMO baseline + TRIM + optional SG + optional Method4)
- Standardize columns to project schema
- Fill missing kinematics (accel/jerk/odometer) if missing (or all-NaN)
- Fill missing (x,y) for non-SUMO datasets using SUMO reference mapping:
    (vehicle_odometer -> vehicle_x, vehicle_y) per vehicle_id

MULTIPROCESSING FEATURES:
- Fast parallel plotting using ProcessPoolExecutor
- Module-level worker functions (properly serializable)
- Intelligent CPU core detection and utilization
- Fallback to single-threaded mode if multiprocessing fails
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

LOGGER_NAME = "trim.fill_xy"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


# ------------------------- Config -------------------------

@dataclass
class FillXYConfig:
    # Required inputs
    sumo_csv: str
    trim_csv: str
    out_csv: str

    # Optional inputs
    use_sg_smooth: bool = False
    sg_csv: Optional[str] = None

    use_method4: bool = False
    m4_csv: Optional[str] = None
    m4_type: str = "METHOD4"  # name shown in data_type

    # Global numeric settings
    dt: str = "auto"  # "auto" or numeric string/float (seconds)
    dt_min_positive: float = 1e-6

    # Fill strategy
    xy_extrapolate_mode: str = "clamp"  # "clamp" or "extrapolate"
    require_sumo_xy: bool = True  # if True, SUMO must have x/y
    require_sumo_odo: bool = False  # if True, SUMO must have odometer; otherwise compute

    # Column names (project schema)
    col_time: str = "timestep_time"
    col_vid: str = "vehicle_id"
    col_trip: str = "trip_id"  # optional
    col_speed: str = "vehicle_speed"
    col_accel: str = "vehicle_accel"
    col_jerk: str = "vehicle_jerk"
    col_odo: str = "vehicle_odometer"
    col_type: str = "vehicle_type"
    col_x: str = "vehicle_x"
    col_y: str = "vehicle_y"
    col_dtype: str = "data_type"

    # IO
    csv_encoding: str = "utf-8-sig"

    # Output columns
    keep_trip_and_jerk: bool = False

    # Plot options (GUI controlled) - MULTIPROCESSING ENABLED
    plot_xy: bool = False
    plot_saj: bool = False  # speed/accel/jerk
    plot_n: int = 10
    plot_workers: int = 0  # 0=auto-detect, >0=specific number of workers
    random_seed: int = 42
    plot_dir: Optional[str] = None  # if None, uses out_csv folder


# ------------------------- MODULE-LEVEL WORKER FUNCTIONS (SERIALIZABLE) -------------------------

def _plot_vehicle_xy_worker(args: Tuple[pd.DataFrame, str, str, str, str, str, str]) -> str:
    """
    Module-level worker function for XY plotting - can be pickled/serialized
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df_vehicle, col_x, col_y, col_dtype, vid, out_path, plot_title = args

    try:
        plt.figure(figsize=(10, 8))

        # Check if we have data_type for grouping
        has_dtype = col_dtype in df_vehicle.columns and not df_vehicle[col_dtype].isna().all()

        if has_dtype:
            # Plot by data_type
            for dtype, gg in df_vehicle.groupby(col_dtype, sort=False):
                if col_x in gg.columns and col_y in gg.columns:
                    x_vals = gg[col_x].to_numpy()
                    y_vals = gg[col_y].to_numpy()
                    valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                    if valid_mask.any():
                        plt.plot(x_vals[valid_mask], y_vals[valid_mask], 'o-',
                                 label=f"{dtype}", alpha=0.8, markersize=3)
        else:
            # Plot all together
            if col_x in df_vehicle.columns and col_y in df_vehicle.columns:
                x_vals = df_vehicle[col_x].to_numpy()
                y_vals = df_vehicle[col_y].to_numpy()
                valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                if valid_mask.any():
                    plt.plot(x_vals[valid_mask], y_vals[valid_mask], 'o-',
                             label="trajectory", alpha=0.8, markersize=3)

        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.title(f"{plot_title} | vehicle_id={vid}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

        return f"Success: {out_path}"
    except Exception as e:
        plt.close('all')  # Cleanup
        return f"Failed: {out_path} - {str(e)}"


def _plot_vehicle_speed_worker(args: Tuple[pd.DataFrame, str, str, str, str, str, str]) -> str:
    """
    Module-level worker function for speed plotting - can be pickled/serialized
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df_vehicle, col_time, col_speed, col_dtype, vid, out_path, plot_title = args

    try:
        plt.figure(figsize=(12, 4))

        # Check if we have data_type for grouping
        has_dtype = col_dtype in df_vehicle.columns and not df_vehicle[col_dtype].isna().all()

        if has_dtype:
            for dtype, gg in df_vehicle.groupby(col_dtype, sort=False):
                plt.plot(gg[col_time].to_numpy(), gg[col_speed].to_numpy(),
                         label=f"{dtype}-speed")
        else:
            plt.plot(df_vehicle[col_time].to_numpy(), df_vehicle[col_speed].to_numpy(),
                     label="speed")

        plt.xlabel("Time")
        plt.ylabel("Speed")
        plt.title(f"{plot_title} | vehicle_id={vid}")
        plt.legend()
        plt.tight_layout()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

        return f"Success: {out_path}"
    except Exception as e:
        plt.close('all')  # Cleanup
        return f"Failed: {out_path} - {str(e)}"


def _plot_vehicle_accel_worker(args: Tuple[pd.DataFrame, str, str, str, str, str, str]) -> str:
    """
    Module-level worker function for acceleration plotting - can be pickled/serialized
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df_vehicle, col_time, col_accel, col_dtype, vid, out_path, plot_title = args

    try:
        plt.figure(figsize=(12, 4))

        # Check if we have data_type for grouping
        has_dtype = col_dtype in df_vehicle.columns and not df_vehicle[col_dtype].isna().all()

        if has_dtype:
            for dtype, gg in df_vehicle.groupby(col_dtype, sort=False):
                plt.plot(gg[col_time].to_numpy(), gg[col_accel].to_numpy(),
                         label=f"{dtype}-accel")
        else:
            plt.plot(df_vehicle[col_time].to_numpy(), df_vehicle[col_accel].to_numpy(),
                     label="accel")

        plt.xlabel("Time")
        plt.ylabel("Acceleration")
        plt.title(f"{plot_title} | vehicle_id={vid}")
        plt.legend()
        plt.tight_layout()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

        return f"Success: {out_path}"
    except Exception as e:
        plt.close('all')  # Cleanup
        return f"Failed: {out_path} - {str(e)}"


# ------------------------- Utils -------------------------

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


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common alternative names -> standard schema."""
    df = df.copy()
    rename_map = {
        "car_speed": "vehicle_speed",
        "vehicle_acceleration": "vehicle_accel",
        "vehicle_acceleration_mps2": "vehicle_accel",
        "accel": "vehicle_accel",
        "odometer": "vehicle_odometer",
        "cumulative_distance": "vehicle_odometer",
        "distance_m": "vehicle_odometer",
        "coord_x": "vehicle_x",
        "coord_y": "vehicle_y",
        "coordinate_x": "vehicle_x",
        "coordinate_y": "vehicle_y",
        "x": "vehicle_x",
        "y": "vehicle_y",
        "veh_type": "vehicle_type",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)
    return df


def _to_numeric_inplace(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _estimate_dt_from_sumo(df_sumo: pd.DataFrame, col_time: str, col_vid: str, logger: logging.Logger) -> float:
    """Estimate dt from SUMO time series."""
    try:
        t = df_sumo[[col_vid, col_time]].dropna().copy()
        t[col_vid] = t[col_vid].astype(str)
        t[col_time] = pd.to_numeric(t[col_time], errors="coerce")
        t = t.dropna(subset=[col_time])

        if t.empty:
            logger.warning("[dt:auto] SUMO time is empty after cleaning. Fallback dt=1.0")
            return 1.0

        t = t.sort_values([col_vid, col_time], kind="mergesort")
        dt_list = []
        for _, g in t.groupby(col_vid, sort=False):
            arr = g[col_time].to_numpy(dtype=float)
            d = np.diff(arr)
            d = d[np.isfinite(d) & (d > 0)]
            if d.size:
                dt_list.append(d)

        if not dt_list:
            logger.warning("[dt:auto] No positive diffs found. Fallback dt=1.0")
            return 1.0

        diffs = np.concatenate(dt_list)
        med = float(np.median(diffs))

        # Simple mode estimation
        rounded = np.round(diffs, 6)
        vals, cnts = np.unique(rounded, return_counts=True)
        mode_val = float(vals[int(np.argmax(cnts))])

        dt = mode_val if abs(mode_val - med) <= 1e-6 else med
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0

        logger.info("[dt:auto] estimated dt=%.6f", dt)
        return dt
    except Exception as e:
        logger.warning("[dt:auto] Error estimating dt: %s. Using dt=1.0", str(e))
        return 1.0


def _compute_accel_jerk_odometer_per_vehicle(df: pd.DataFrame, cfg: FillXYConfig, dt: float,
                                             logger: logging.Logger) -> pd.DataFrame:
    """Compute missing kinematics if needed."""
    try:
        df = df.copy()
        need_accel = (cfg.col_accel not in df.columns) or df[cfg.col_accel].isna().all()
        need_jerk = (cfg.col_jerk not in df.columns) or df[cfg.col_jerk].isna().all()
        need_odo = (cfg.col_odo not in df.columns) or df[cfg.col_odo].isna().all()

        if not (need_accel or need_jerk or need_odo):
            return df

        # ensure columns exist
        for col in [cfg.col_accel, cfg.col_jerk, cfg.col_odo]:
            if col not in df.columns:
                df[col] = np.nan

        df[cfg.col_vid] = df[cfg.col_vid].astype(str)
        _to_numeric_inplace(df, [cfg.col_time, cfg.col_speed])
        df = df.sort_values([cfg.col_vid, cfg.col_time], kind="mergesort")

        for vid, g in df.groupby(cfg.col_vid, sort=False):
            g = g.dropna(subset=[cfg.col_time, cfg.col_speed]).copy()
            if len(g) <= 1:
                continue

            arr_v = g[cfg.col_speed].to_numpy(dtype=float)
            idx = g.index

            # Simple forward difference calculations
            if need_accel and len(arr_v) > 1:
                a = np.zeros_like(arr_v)
                dv = np.diff(arr_v)
                a[:-1] = dv / dt
                df.loc[idx, cfg.col_accel] = a

            if need_jerk and len(arr_v) > 1:
                a_now = df.loc[idx, cfg.col_accel].to_numpy(dtype=float)
                j = np.zeros_like(a_now)
                if len(a_now) > 1:
                    da = np.diff(a_now)
                    j[:-1] = da / dt
                df.loc[idx, cfg.col_jerk] = j

            if need_odo:
                if len(arr_v) == 1:
                    odo = np.array([0.0])
                else:
                    avg_speeds = (arr_v[:-1] + arr_v[1:]) / 2
                    increments = avg_speeds * dt
                    odo = np.concatenate([[0.0], np.cumsum(increments)])
                df.loc[idx, cfg.col_odo] = odo

        return df
    except Exception as e:
        logger.warning("[compute_kinematics] Error: %s", str(e))
        return df


def _read_standardize_one(csv_path: str, data_type: str, cfg: FillXYConfig, dt: float,
                          logger: logging.Logger) -> pd.DataFrame:
    """Read and standardize one dataset."""
    try:
        df = pd.read_csv(csv_path, encoding=cfg.csv_encoding)
        df = _normalize_columns(df)

        _require_cols(df, [cfg.col_time, cfg.col_vid, cfg.col_speed], f"{data_type}_csv")

        # ensure required columns exist
        required_cols = [cfg.col_trip, cfg.col_accel, cfg.col_jerk, cfg.col_odo, cfg.col_x, cfg.col_y]
        for c in required_cols:
            if c not in df.columns:
                df[c] = np.nan

        # cast types
        df[cfg.col_vid] = df[cfg.col_vid].astype(str)
        df[cfg.col_trip] = df[cfg.col_trip].astype(str)
        _to_numeric_inplace(df, [cfg.col_time, cfg.col_speed, cfg.col_accel, cfg.col_jerk, cfg.col_odo, cfg.col_x,
                                 cfg.col_y])

        # Force non-SUMO data types to have NaN x/y coordinates
        # so that _fill_xy_using_sumo_reference will re-interpolate them
        # from the SUMO reference mapping (odometer -> x, y)
        df[cfg.col_x] = np.nan
        df[cfg.col_y] = np.nan

        # compute missing kinematics
        df = _compute_accel_jerk_odometer_per_vehicle(df, cfg, dt, logger)
        df[cfg.col_dtype] = data_type

        logger.info("[read] %s: %d rows, %d vehicles", data_type, len(df), df[cfg.col_vid].nunique())
        return df
    except Exception as e:
        logger.error("[read] Failed to read %s: %s", data_type, str(e))
        raise


def _build_sumo_interpolation_map(df_sumo: pd.DataFrame, cfg: FillXYConfig, logger: logging.Logger) -> Dict[
    str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build per-vehicle interpolation mapping: odometer -> (x, y) from SUMO."""
    try:
        req_cols = [cfg.col_vid, cfg.col_odo, cfg.col_x, cfg.col_y]
        _require_cols(df_sumo, req_cols, "SUMO for XY mapping")

        maps = {}
        for vid, g in df_sumo.groupby(cfg.col_vid, sort=False):
            g = g.dropna(subset=[cfg.col_odo, cfg.col_x, cfg.col_y]).copy()
            if len(g) == 0:
                continue

            g = g.sort_values(cfg.col_odo, kind="mergesort")
            odo_vals = g[cfg.col_odo].to_numpy(dtype=float)
            x_vals = g[cfg.col_x].to_numpy(dtype=float)
            y_vals = g[cfg.col_y].to_numpy(dtype=float)

            if len(odo_vals) > 0:
                # remove duplicates
                unique_mask = np.concatenate([[True], np.diff(odo_vals) != 0])
                odo_vals = odo_vals[unique_mask]
                x_vals = x_vals[unique_mask]
                y_vals = y_vals[unique_mask]

                if len(odo_vals) > 0:
                    maps[str(vid)] = (odo_vals, x_vals, y_vals)

        logger.info("[sumo_map] Built interpolation maps for %d vehicles", len(maps))
        return maps
    except Exception as e:
        logger.warning("[sumo_map] Error building maps: %s", str(e))
        return {}


def _fill_xy_using_sumo_reference(df_all: pd.DataFrame, df_sumo: pd.DataFrame, cfg: FillXYConfig,
                                  logger: logging.Logger) -> pd.DataFrame:
    """Fill missing (x, y) using SUMO reference.

    Performance note
    ----------------
    All per-vehicle interpolation results are collected into flat numpy
    arrays and written back to the DataFrame in **one single vectorised
    operation** at the end, which avoids the severe overhead of per-row
    ``DataFrame.loc`` assignments (100-1000× faster on large datasets).
    """
    try:
        df_all = df_all.copy()

        sumo_maps = _build_sumo_interpolation_map(df_sumo, cfg, logger)
        if not sumo_maps:
            logger.warning("[fill_xy] No valid SUMO interpolation maps. XY will remain as-is.")
            return df_all

        # --- Work on the underlying numpy arrays directly ---
        x_arr = df_all[cfg.col_x].to_numpy(dtype=float).copy()
        y_arr = df_all[cfg.col_y].to_numpy(dtype=float).copy()
        odo_arr = df_all[cfg.col_odo].to_numpy(dtype=float)

        # Pre-compute masks on the whole array once
        x_nan_all = ~np.isfinite(x_arr)
        y_nan_all = ~np.isfinite(y_arr)
        odo_ok_all = np.isfinite(odo_arr)

        filled_count = 0
        total_missing = 0
        n_vehicles = 0

        for vid, g in df_all.groupby(cfg.col_vid, sort=False):
            vid_str = str(vid)
            if vid_str not in sumo_maps:
                continue

            odo_ref, x_ref, y_ref = sumo_maps[vid_str]
            # Integer position indices into the numpy arrays
            pos = g.index.to_numpy()

            x_missing = x_nan_all[pos]
            y_missing = y_nan_all[pos]
            odo_valid = odo_ok_all[pos]

            need_fill_mask = (x_missing | y_missing) & odo_valid
            if not need_fill_mask.any():
                continue

            local_idx = np.where(need_fill_mask)[0]
            global_idx = pos[local_idx]

            total_missing += len(local_idx)
            n_vehicles += 1

            odo_target = odo_arr[global_idx]

            if cfg.xy_extrapolate_mode == "clamp":
                odo_target = np.clip(odo_target, odo_ref[0], odo_ref[-1])

            x_interp = np.interp(odo_target, odo_ref, x_ref)
            y_interp = np.interp(odo_target, odo_ref, y_ref)

            # Vectorised conditional write into the flat arrays
            x_fill = x_missing[local_idx]
            y_fill = y_missing[local_idx]
            x_arr[global_idx[x_fill]] = x_interp[x_fill]
            y_arr[global_idx[y_fill]] = y_interp[y_fill]

            filled_count += len(local_idx)

        # --- Single bulk write back to DataFrame ---
        df_all[cfg.col_x] = x_arr
        df_all[cfg.col_y] = y_arr

        logger.info("[fill_xy] Filled %d/%d missing XY coords across %d vehicles",
                    filled_count, total_missing, n_vehicles)
        return df_all
    except Exception as e:
        logger.warning("[fill_xy] Error filling XY: %s", str(e))
        return df_all


def _filter_output_columns(df: pd.DataFrame, cfg: FillXYConfig, logger: logging.Logger) -> pd.DataFrame:
    """Filter output columns - INCLUDES data_type for plotting."""
    try:
        # Core required columns including data_type
        required_cols = [
            cfg.col_time,  # timestep_time
            cfg.col_vid,  # vehicle_id
            cfg.col_speed,  # vehicle_speed
            cfg.col_accel,  # vehicle_accel
            cfg.col_odo,  # vehicle_odometer
            cfg.col_type,  # vehicle_type
            cfg.col_x,  # vehicle_x
            cfg.col_y,  # vehicle_y
            cfg.col_dtype,  # data_type - ESSENTIAL for plotting
        ]

        # Add optional columns
        if cfg.keep_trip_and_jerk:
            required_cols.insert(2, cfg.col_trip)  # Insert trip_id
            if cfg.col_jerk not in required_cols:
                accel_idx = required_cols.index(cfg.col_accel)
                required_cols.insert(accel_idx + 1, cfg.col_jerk)

        # Ensure all required columns exist
        for col in required_cols:
            if col not in df.columns:
                if col == cfg.col_dtype:
                    df[col] = "unknown"
                else:
                    df[col] = np.nan

        # Filter to only required columns
        df_filtered = df[required_cols].copy()

        logger.info(f"[output] Final columns: {required_cols}")
        logger.info(f"[output] Shape: {df_filtered.shape}")

        return df_filtered
    except Exception as e:
        logger.warning("[filter_output] Error filtering columns: %s", str(e))
        return df


def _plot_samples_multiprocess(df: pd.DataFrame, cfg: FillXYConfig, logger: logging.Logger) -> None:
    """
    MULTIPROCESSING ENABLED plotting with proper error handling and fallback
    """
    if not (cfg.plot_xy or cfg.plot_saj):
        return

    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("[plot] matplotlib not available. Skipping plots.")
        return

    try:
        # Determine output directory
        if cfg.plot_dir:
            out_dir = cfg.plot_dir
        else:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(cfg.out_csv)), "plots")

        _safe_mkdir(out_dir)

        # Sample vehicles
        np.random.seed(cfg.random_seed)
        vehicles = df[cfg.col_vid].unique()
        n_sample = min(cfg.plot_n, len(vehicles))
        if n_sample == 0:
            logger.warning("[plot] No vehicles to plot")
            return

        sample_vids = np.random.choice(vehicles, size=n_sample, replace=False)

        # Determine number of workers
        if cfg.plot_workers > 0:
            max_workers = cfg.plot_workers
        else:
            cpu_count = os.cpu_count() or 4
            max_workers = max(1, min(cpu_count - 1, 8))  # Leave 1 CPU for main process

        logger.info("[plot] Starting multiprocess plotting: %d vehicles, %d workers", n_sample, max_workers)

        # Prepare plot tasks
        plot_tasks = []

        for vid in sample_vids:
            g = df[df[cfg.col_vid] == vid].copy()
            if g.empty:
                continue

            # XY trajectory plot task
            if cfg.plot_xy and cfg.col_x in g.columns and cfg.col_y in g.columns:
                xy_path = os.path.join(out_dir, f"xy_vehicle_{vid}.png")
                xy_args = (g, cfg.col_x, cfg.col_y, cfg.col_dtype, str(vid), xy_path, "Trajectory")
                plot_tasks.append((_plot_vehicle_xy_worker, xy_args))

            # Speed plot task
            if cfg.plot_saj:
                speed_path = os.path.join(out_dir, f"speed_vehicle_{vid}.png")
                speed_args = (g, cfg.col_time, cfg.col_speed, cfg.col_dtype, str(vid), speed_path, "Speed")
                plot_tasks.append((_plot_vehicle_speed_worker, speed_args))

                # Acceleration plot task (if column exists)
                if cfg.col_accel in g.columns:
                    accel_path = os.path.join(out_dir, f"accel_vehicle_{vid}.png")
                    accel_args = (g, cfg.col_time, cfg.col_accel, cfg.col_dtype, str(vid), accel_path, "Acceleration")
                    plot_tasks.append((_plot_vehicle_accel_worker, accel_args))

        if not plot_tasks:
            logger.warning("[plot] No plot tasks generated")
            return

        logger.info("[plot] Generated %d plot tasks", len(plot_tasks))

        # Execute plotting with multiprocessing
        try:
            # Try multiprocessing first
            if max_workers > 1:
                from concurrent.futures import ProcessPoolExecutor, as_completed

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    futures = []
                    for worker_func, args in plot_tasks:
                        future = executor.submit(worker_func, args)
                        futures.append(future)

                    # Collect results
                    completed = 0
                    failed = 0
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result.startswith("Success"):
                                completed += 1
                            else:
                                failed += 1
                                logger.warning("[plot] %s", result)
                        except Exception as e:
                            failed += 1
                            logger.warning("[plot] Task failed with exception: %s", str(e))

                        # Progress reporting
                        if (completed + failed) % 5 == 0 or (completed + failed) == len(plot_tasks):
                            logger.info("[plot] Progress: %d completed, %d failed, %d total",
                                        completed, failed, len(plot_tasks))

                logger.info("[plot] Multiprocess plotting completed: %d success, %d failed", completed, failed)

            else:
                # Single-threaded fallback
                logger.info("[plot] Using single-threaded mode (workers=1)")
                completed = 0
                failed = 0
                for worker_func, args in plot_tasks:
                    try:
                        result = worker_func(args)
                        if result.startswith("Success"):
                            completed += 1
                        else:
                            failed += 1
                            logger.warning("[plot] %s", result)
                    except Exception as e:
                        failed += 1
                        logger.warning("[plot] Task failed: %s", str(e))

                logger.info("[plot] Single-threaded plotting completed: %d success, %d failed", completed, failed)

        except Exception as e:
            # Fallback to single-threaded if multiprocessing fails
            logger.warning("[plot] Multiprocessing failed (%s), falling back to single-threaded", str(e))

            completed = 0
            failed = 0
            for worker_func, args in plot_tasks:
                try:
                    result = worker_func(args)
                    if result.startswith("Success"):
                        completed += 1
                    else:
                        failed += 1
                        logger.warning("[plot] %s", result)
                except Exception as e:
                    failed += 1
                    logger.warning("[plot] Task failed: %s", str(e))

            logger.info("[plot] Fallback plotting completed: %d success, %d failed", completed, failed)

        logger.info("[plot] All plotting completed. Output directory: %s", out_dir)

    except Exception as e:
        logger.warning("[plot] Overall plotting failed: %s", str(e))


# ------------------------- Main Run -------------------------

def run_fill_xy(cfg: FillXYConfig, logger: Optional[logging.Logger] = None) -> str:
    logger = logger or get_logger()

    try:
        # Validate inputs
        _ensure_file(cfg.sumo_csv, "sumo_csv")
        _ensure_file(cfg.trim_csv, "trim_csv")
        if cfg.use_sg_smooth and cfg.sg_csv:
            _ensure_file(cfg.sg_csv, "sg_csv")
        if cfg.use_method4 and cfg.m4_csv:
            _ensure_file(cfg.m4_csv, "m4_csv")

        _safe_mkdir(os.path.dirname(os.path.abspath(cfg.out_csv)))

        # Read and process SUMO data
        df_sumo_raw = pd.read_csv(cfg.sumo_csv, encoding=cfg.csv_encoding)
        df_sumo_raw = _normalize_columns(df_sumo_raw)

        sumo_req = [cfg.col_time, cfg.col_vid, cfg.col_speed, cfg.col_type]
        _require_cols(df_sumo_raw, sumo_req, "sumo_csv")

        if cfg.require_sumo_xy:
            _require_cols(df_sumo_raw, [cfg.col_x, cfg.col_y], "sumo_csv (xy)")

        # Determine dt
        if isinstance(cfg.dt, str) and cfg.dt.lower() == "auto":
            dt = _estimate_dt_from_sumo(df_sumo_raw, cfg.col_time, cfg.col_vid, logger)
        else:
            dt = float(cfg.dt)
            if dt <= 0 or not np.isfinite(dt):
                raise ValueError(f"Invalid dt: {cfg.dt}")
            logger.info("[dt] using dt=%.6f", dt)

        # Prepare SUMO dataset
        df_sumo = df_sumo_raw.copy()

        # Ensure all required columns exist
        for col in [cfg.col_trip, cfg.col_accel, cfg.col_jerk, cfg.col_odo]:
            if col not in df_sumo.columns:
                df_sumo[col] = np.nan

        df_sumo[cfg.col_vid] = df_sumo[cfg.col_vid].astype(str)
        df_sumo[cfg.col_trip] = df_sumo[cfg.col_trip].astype(str)
        _to_numeric_inplace(df_sumo, [cfg.col_time, cfg.col_speed, cfg.col_accel, cfg.col_jerk, cfg.col_odo, cfg.col_x,
                                      cfg.col_y])

        df_sumo = _compute_accel_jerk_odometer_per_vehicle(df_sumo, cfg, dt, logger)
        df_sumo[cfg.col_dtype] = "sumo"

        # Build vehicle type mapping
        type_map = {}
        if cfg.col_type in df_sumo.columns:
            type_map = (
                df_sumo[[cfg.col_vid, cfg.col_type]]
                .dropna()
                .drop_duplicates(subset=[cfg.col_vid], keep="first")
                .set_index(cfg.col_vid)[cfg.col_type]
                .astype(str)
                .to_dict()
            )

        logger.info("[fill_xy] SUMO vehicles with type: %d", len(type_map))

        # Process other datasets
        dfs = [df_sumo]

        # trim data
        df_trim = _read_standardize_one(cfg.trim_csv, "trim", cfg, dt, logger)
        if cfg.col_type in df_trim.columns:
            df_trim[cfg.col_type] = df_trim[cfg.col_vid].map(type_map).fillna(df_trim[cfg.col_type])
        else:
            df_trim[cfg.col_type] = df_trim[cfg.col_vid].map(type_map)
        dfs.append(df_trim)

        # Optional SG smooth data
        if cfg.use_sg_smooth and cfg.sg_csv:
            df_sg = _read_standardize_one(cfg.sg_csv, "sg", cfg, dt, logger)
            mapped = df_sg[cfg.col_vid].map(type_map)
            if cfg.col_type in df_sg.columns:
                mapped = mapped.fillna(df_sg[cfg.col_type])
            else:
                mapped = mapped.fillna("SG smooth")
            df_sg[cfg.col_type] = mapped
            dfs.append(df_sg)

        # Optional Method4 data
        if cfg.use_method4 and cfg.m4_csv:
            dtype = str(cfg.m4_type).strip() if cfg.m4_type else "METHOD4"
            df_m4 = _read_standardize_one(cfg.m4_csv, dtype, cfg, dt, logger)
            mapped = df_m4[cfg.col_vid].map(type_map)
            if cfg.col_type in df_m4.columns:
                mapped = mapped.fillna(df_m4[cfg.col_type])
            else:
                mapped = mapped.fillna(dtype)
            df_m4[cfg.col_type] = mapped
            dfs.append(df_m4)

        # Combine all datasets
        df_all = pd.concat(dfs, ignore_index=True)

        # Remove duplicates
        subset = [cfg.col_dtype, cfg.col_vid, cfg.col_time]
        if cfg.keep_trip_and_jerk and cfg.col_trip in df_all.columns:
            subset.insert(2, cfg.col_trip)
        before = len(df_all)
        df_all = df_all.drop_duplicates(subset=subset, keep="first").copy()
        after = len(df_all)
        if after < before:
            logger.info("[fill_xy] dropped duplicates: %d -> %d", before, after)

        # Fill missing XY coordinates
        df_all = _fill_xy_using_sumo_reference(df_all, df_sumo, cfg, logger)

        # Sort data
        sort_cols = [cfg.col_dtype, cfg.col_vid, cfg.col_time]
        if cfg.keep_trip_and_jerk and cfg.col_trip in df_all.columns:
            sort_cols = [cfg.col_dtype, cfg.col_trip, cfg.col_vid, cfg.col_time]
        df_all = df_all.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        # Filter output columns (preserves data_type for plotting)
        df_final = _filter_output_columns(df_all, cfg, logger)

        # Save output
        df_final.to_csv(cfg.out_csv, index=False, encoding=cfg.csv_encoding)
        logger.info("[OK] fill_xy output saved: %s", cfg.out_csv)

        # Generate plots (MULTIPROCESSING ENABLED)
        _plot_samples_multiprocess(df_final, cfg, logger)

        return cfg.out_csv

    except Exception as e:
        logger.error("[fill_xy] Failed: %s", str(e))
        raise


# ------------------------- GUI Integration -------------------------

def run_gui(*, in_path, out_dir, state=None, upstream=None, options=None):
    """GUI-friendly wrapper - MULTIPROCESSING ENABLED"""
    logger = get_logger()

    try:
        # Validate inputs
        if not in_path or not Path(in_path).exists():
            raise FileNotFoundError(f"Input TRIM CSV not found: {in_path}")

        if not state:
            raise ValueError("State object is required")

        if not upstream:
            raise ValueError("Upstream outputs are required")

        # Get SUMO baseline data
        sumo_csv_path = upstream.get("xml2csv_fcd")
        if not sumo_csv_path or not Path(sumo_csv_path).exists():
            raise FileNotFoundError(f"SUMO baseline CSV not found: {sumo_csv_path}")

        # Setup paths
        trim_csv_path = in_path
        sg_csv_path = upstream.get("sg_smooth")
        use_sg_smooth = sg_csv_path is not None and Path(sg_csv_path).exists()

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_prefix = getattr(state.inputs, 'safe_prefix', 'fcd')
        out_csv_path = out_dir / f"{safe_prefix}_filled_xy.csv"

        # Configure plotting based on state options
        enable_plots = getattr(state.options, 'enable_plots', False)
        plot_limit = getattr(state.options, 'plot_limit', 10)

        # Build configuration with multiprocessing support
        cfg = FillXYConfig(
            sumo_csv=str(sumo_csv_path),
            trim_csv=str(trim_csv_path),
            out_csv=str(out_csv_path),

            use_sg_smooth=use_sg_smooth,
            sg_csv=str(sg_csv_path) if use_sg_smooth else None,

            use_method4=False,  # Can be enabled if needed
            m4_csv=None,

            dt="auto",
            xy_extrapolate_mode="clamp",
            require_sumo_xy=True,
            require_sumo_odo=False,

            keep_trip_and_jerk=True,

            # Multiprocessing plotting configuration
            plot_xy=enable_plots,
            plot_saj=enable_plots,
            plot_n=plot_limit,
            plot_workers=0,  # Auto-detect optimal worker count
            random_seed=getattr(state.options, 'random_seed', 42),
            plot_dir=str(out_dir / "plots") if enable_plots else None,
        )

        logger.info("[fill_xy] Starting XY fill process (multiprocessing enabled)")
        logger.info(f"[fill_xy] SUMO baseline: {sumo_csv_path}")
        logger.info(f"[fill_xy] TRIM input: {trim_csv_path}")
        logger.info(f"[fill_xy] SG smooth: {'Used' if use_sg_smooth else 'Not used'}")
        logger.info(f"[fill_xy] Output: {out_csv_path}")
        logger.info(f"[fill_xy] Multiprocessing: {'Enabled' if enable_plots else 'Disabled (no plots)'}")

        # Run the main function
        result_path = run_fill_xy(cfg, logger=logger)
        logger.info(f"[fill_xy] Successfully completed. Output: {result_path}")
        return result_path

    except Exception as e:
        logger.error(f"[fill_xy] Failed: {e}")
        raise


# ------------------------- CLI Wrapper -------------------------

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
    logger = get_logger()

    p = argparse.ArgumentParser(description="Fill XY using SUMO reference mapping (Multiprocessing Enabled)")
    p.add_argument("--sumo_csv", required=True)
    p.add_argument("--trim_csv", required=True)
    p.add_argument("--out_csv", required=True)

    p.add_argument("--dt", default="auto", help="time step seconds, or 'auto'")

    p.add_argument("--use_sg_smooth", action="store_true")
    p.add_argument("--sg_csv", default=None)

    p.add_argument("--use_method4", action="store_true")
    p.add_argument("--m4_csv", default=None)
    p.add_argument("--m4_type", default="METHOD4")

    p.add_argument("--xy_extrapolate_mode", choices=["clamp", "extrapolate"], default="clamp")
    p.add_argument("--keep_trip_and_jerk", action="store_true")

    p.add_argument("--plot_xy", action="store_true")
    p.add_argument("--plot_saj", action="store_true")
    p.add_argument("--plot_n", type=int, default=10)
    p.add_argument("--plot_workers", type=int, default=0, help="Number of plot workers (0=auto)")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--plot_dir", default=None)

    args = p.parse_args()

    cfg = FillXYConfig(
        sumo_csv=args.sumo_csv,
        trim_csv=args.trim_csv,
        out_csv=args.out_csv,
        dt=args.dt,
        use_sg_smooth=bool(args.use_sg_smooth),
        sg_csv=args.sg_csv,
        use_method4=bool(args.use_method4),
        m4_csv=args.m4_csv,
        m4_type=args.m4_type,
        xy_extrapolate_mode=args.xy_extrapolate_mode,
        keep_trip_and_jerk=bool(args.keep_trip_and_jerk),
        plot_xy=bool(args.plot_xy),
        plot_saj=bool(args.plot_saj),
        plot_n=int(args.plot_n),
        plot_workers=int(args.plot_workers),
        random_seed=int(args.random_seed),
        plot_dir=args.plot_dir,
    )

    out = run_fill_xy(cfg, logger=logger)
    logger.info("Done. Output: %s", out)


if __name__ == "__main__":
    main()