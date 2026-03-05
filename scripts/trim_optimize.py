# -*- coding: utf-8 -*-
"""
trim_optimize.py

Key changes vs v4:
1) Stage2 only runs for BAD cases identified by jerk envelope check on Stage1 output,
   where "BAD" is determined per (trip_id, vehicle_id) and ONLY checks mid points:
   idx=1..n-2 (ignore first and last row), and n<3 => OK.

Public API:
- TRIMOptimizeConfig
- run_stage1(cfg, logger=None) -> stage1_path
- run_stage2(cfg, logger=None, case_keys_filter=None) -> stage2_path
- run_all(cfg) -> (stage1_path, stage2_path)   # stage2 auto-runs only bad (trip_id, vehicle_id)
"""

from __future__ import annotations

import os
import re
import sys
import time
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from gurobipy import Model, GRB, quicksum, Env


# ----------------------------
# Fully Adaptive max_workers Detection
# ----------------------------
import os
import functools
import platform
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

@functools.lru_cache(maxsize=1)
def _detect_max_workers() -> int:
    """
    Intelligent detection of system's maximum ProcessPoolExecutor workers.

    Optimization algorithm:
    1. Get CPU core count
    2. Use intelligent division strategy: cpu_count // [2,3,4,5], ensuring result >= 4
    3. Combine platform-specific limits and CPU multiplier candidates
    4. Test and select maximum available value

    Returns:
        Maximum safe number of workers (guaranteed to be at least 2)
    """
    cpu_count = os.cpu_count() or 4
    system = platform.system()

    # Step 1: Intelligent candidates based on CPU core count
    cpu_based_candidates = []

    # Division strategy candidates (more conservative, ensuring system stability)
    divisors = [2, 3, 4, 5]
    for divisor in divisors:
        candidate = cpu_count // divisor
        if candidate >= 4:  # Raise minimum value to ensure meaningful parallelism
            cpu_based_candidates.append(candidate)

    # Multiplier strategy candidates (for high core count systems)
    multipliers = [0.5, 0.6, 0.7, 0.8]
    for multiplier in multipliers:
        candidate = int(cpu_count * multiplier)
        if candidate >= 4:
            cpu_based_candidates.append(candidate)

    # Step 2: Platform-specific candidates
    platform_candidates = []

    if system == 'Windows':
        # Windows has ProcessPoolExecutor limit of 61
        platform_candidates = [61, 60, 50, 40, 30, 20, 15]
    elif system == 'Darwin':  # macOS
        # macOS typically supports medium concurrency
        platform_candidates = [
            min(128, cpu_count * 2),
            min(64, cpu_count),
            min(32, max(cpu_count // 2, 4))
        ]
    else:  # Linux and others
        # Linux supports high concurrency
        platform_candidates = [
            min(256, cpu_count * 2),
            min(128, cpu_count),
            min(64, max(cpu_count // 2, 4)),
            min(32, max(cpu_count // 3, 4))
        ]

    # Step 3: Add empirical candidates (based on common optimal configurations)
    experience_candidates = []
    if cpu_count >= 16:
        experience_candidates = [cpu_count // 2, cpu_count // 3]
    elif cpu_count >= 8:
        experience_candidates = [cpu_count // 2, max(4, cpu_count - 4)]
    elif cpu_count >= 4:
        experience_candidates = [max(4, cpu_count - 1), max(3, cpu_count - 2)]

    # Step 4: Merge all candidates
    all_candidates = (cpu_based_candidates + platform_candidates +
                      experience_candidates + [cpu_count, max(4, cpu_count - 1)])

    # Filter and sort (largest to smallest)
    meaningful_candidates = [c for c in all_candidates if c >= 3]
    candidates = sorted(set(meaningful_candidates), reverse=True)

    # Step 5: Test candidate values
    for workers in candidates:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future = executor.submit(lambda: True)
                if future.result(timeout=0.5):
                    return workers
        except (ValueError, OSError):
            # Exceeded limit, try smaller value
            continue
        except Exception:
            # Other errors, try smaller value
            continue

    # Extreme case: return minimum meaningful parallel number
    return max(2, min(4, cpu_count - 1))


def get_safe_n_jobs(requested: Optional[int] = None) -> int:
    """
    Get safe n_jobs value using optimized CPU core detection.

    Args:
        requested: User-requested worker count
                  - None: auto-detect (recommended)
                  - int: manual value, but limited by system maximum

    Returns:
        Safe n_jobs value (between 3 and max_workers)

    Examples:
        >>> n_jobs = get_safe_n_jobs()  # Auto: based on intelligent CPU detection
        >>> n_jobs = get_safe_n_jobs(16)  # Manual: not exceeding system limits
    """
    max_workers = _detect_max_workers()
    cpu_count = os.cpu_count() or 4

    if requested is None:
        # Auto mode: use detected maximum value, but consider system load
        # For CPU-intensive tasks like TRIM, can use close to maximum value
        auto_jobs = max_workers
        return max(3, auto_jobs)
    else:
        # Manual mode: limit within system maximum, but allow user to specify higher values
        return max(3, min(max_workers, requested))

# ----------------------------
# Logging
# ----------------------------
def setup_logger(
        log_path: Optional[str] = None,
        level: int = logging.INFO,
        to_console: bool = False,
        logger_name: str = "trim",
) -> logging.Logger:
    """
    Default: only file logging (no console).
    Set to_console=True if you really want console prints.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if to_console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ----------------------------
# Config
# ----------------------------
@dataclass
class TRIMOptimizeConfig:
    input_csv: str
    out_dir: str
    prefix: str = "fcd"

    n_jobs: int = 60

    # Gurobi
    time_limit_sec: int = 500
    gurobi_threads_per_worker: int = 1
    output_flag: int = 0
    gurobi_log_to_console: int = 0
    gurobi_silent_env: bool = True

    v_max: float = 20.0
    a_upper: float = 4.0
    a_lower: float = -4.0
    jerk_upper: float = 4.0
    jerk_lower: float = -4.0

    w_a: float = 5.0
    w_v: float = 1.0
    w_x: float = 0.0

    big_m: float = 1e6
    lock_when_base_v_zero: bool = True

    jerk_envelope_csv: Optional[str] = None
    accel_envelope_csv: Optional[str] = None
    use_accel_envelope: bool = False
    accel_round_decimals: int = 1

    out_format: str = "csv"
    csv_encoding: str = "utf-8-sig"

    # Logging control
    log_to_console: bool = False

    # Column names
    col_case: str = "trip_id"
    col_vid: str = "vehicle_id"
    col_t: str = "timestep_time"
    col_v: str = "vehicle_speed"
    col_s: str = "vehicle_odometer"
    col_pre_id: str = "preceding_vehicle_id"
    col_fol_id: str = "following_vehicle_id"
    col_pre_hd: str = "preceding_headway_distance"
    col_fol_hd: str = "following_headway_distance"
    col_fol_v: str = "following_vehicle_speed"


# ----------------------------
# Utilities
# ----------------------------
def sanitize_filename(s: str) -> str:
    return re.sub(r'[\\/*?:"<>|.]', "_", str(s))


def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{name}] missing required columns: {miss}")


def resolve_resource_path(user_path: Optional[str], filename: str) -> str:
    if user_path:
        p = os.path.abspath(user_path)
        if os.path.exists(p):
            return p

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    project_root = os.path.dirname(script_dir)
    candidates = [
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "resources", filename),
        os.path.join(project_root, "data", filename),
        os.path.join(cwd, filename),
        os.path.join(cwd, "resources", filename),
        os.path.join(cwd, "data", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Resource file not found: {filename}\n"
        f"Tried:\n- " + "\n- ".join(candidates) + "\n\n"
                                                  f"How to fix:\n"
                                                  f"1) Put {filename} next to this script or under resources/\n"
                                                  f"2) Or pass --jerk_env_csv \"<path>\" explicitly"
    )


def _force_key_dtypes(df: pd.DataFrame, cfg: TRIMOptimizeConfig) -> pd.DataFrame:
    df = df.copy()
    if cfg.col_case in df.columns:
        df[cfg.col_case] = df[cfg.col_case].astype(str)
    if cfg.col_vid in df.columns:
        df[cfg.col_vid] = df[cfg.col_vid].astype(str)
    if cfg.col_t in df.columns:
        df[cfg.col_t] = pd.to_numeric(df[cfg.col_t], errors="coerce")
    return df


# ----------------------------
# Gurobi: silent env/model
# ----------------------------
def _new_silent_env(cfg: TRIMOptimizeConfig) -> Optional[Env]:
    if not cfg.gurobi_silent_env:
        return None
    env = Env(empty=True)
    env.setParam("LogToConsole", int(cfg.gurobi_log_to_console))
    env.setParam("OutputFlag", int(cfg.output_flag))
    env.start()
    return env


def _new_model(cfg: TRIMOptimizeConfig) -> Tuple[Model, Optional[Env]]:
    env = _new_silent_env(cfg)
    if env is not None:
        m = Model(env=env)
    else:
        m = Model()

    m.setParam("OutputFlag", int(cfg.output_flag))
    try:
        m.setParam("LogToConsole", int(cfg.gurobi_log_to_console))
    except Exception:
        pass
    m.setParam("TimeLimit", int(cfg.time_limit_sec))
    m.setParam("Threads", int(cfg.gurobi_threads_per_worker))
    return m, env


# ----------------------------
# Prepare data
# ----------------------------
def load_and_prepare(df_all: pd.DataFrame, cfg: TRIMOptimizeConfig, logger: logging.Logger) -> Tuple[
    pd.DataFrame, float]:
    df_all = _force_key_dtypes(df_all, cfg)
    df_all = df_all.dropna(subset=[cfg.col_t]).copy()

    key_cols = [cfg.col_vid, cfg.col_t]
    df_all = df_all.drop_duplicates(subset=key_cols, keep="first", ignore_index=True)

    df_all = df_all.sort_values([cfg.col_vid, cfg.col_t]).reset_index(drop=True)

    safe_distance = np.nan
    if cfg.col_fol_hd in df_all.columns and cfg.col_pre_hd in df_all.columns:
        try:
            min_fhd = pd.to_numeric(df_all[cfg.col_fol_hd], errors="coerce").min()
            min_phd = pd.to_numeric(df_all[cfg.col_pre_hd], errors="coerce").min()
            safe_distance = float(np.nanmin([min_fhd, min_phd]))
        except Exception:
            safe_distance = np.nan

    if not np.isfinite(safe_distance):
        safe_distance = 2.0
        logger.warning(f"Failed to compute safe_distance; fallback to {safe_distance}")

    if cfg.col_s in df_all.columns:
        if cfg.col_pre_hd in df_all.columns:
            df_all["preceding_mileage_pos"] = (
                    pd.to_numeric(df_all[cfg.col_s], errors="coerce")
                    + pd.to_numeric(df_all[cfg.col_pre_hd], errors="coerce")
            )
        if cfg.col_fol_hd in df_all.columns:
            df_all["following_mileage_pos"] = (
                    pd.to_numeric(df_all[cfg.col_s], errors="coerce")
                    - pd.to_numeric(df_all[cfg.col_fol_hd], errors="coerce")
            )

    return df_all, safe_distance


def build_soft_locks(base_v: Dict[float, float], lock_when_base_v_zero: bool) -> Tuple[
    Dict[float, int], Dict[float, int]]:
    if not lock_when_base_v_zero:
        b_v = {k: 0 for k in base_v.keys()}
    else:
        b_v = {k: int(float(base_v[k]) == 0.0) for k in base_v.keys()}

    if b_v:
        keys = list(b_v.keys())
        b_v[keys[0]] = 1
        b_v[keys[-1]] = 1

    b_x = dict(b_v)
    return b_v, b_x


def prepare_params_for_case(df_case: pd.DataFrame, cfg: TRIMOptimizeConfig, safe_distance: float) -> Dict:
    df_case = df_case.sort_values(cfg.col_t)
    time_list = sorted(df_case[cfg.col_t].unique())

    base_v = df_case.set_index(cfg.col_t)[cfg.col_v].to_dict()
    base_x = df_case.set_index(cfg.col_t)[cfg.col_s].to_dict()

    preced_time = df_case.loc[
        df_case[cfg.col_pre_id].notna(), cfg.col_t].tolist() if cfg.col_pre_id in df_case.columns else []
    follow_time = df_case.loc[
        df_case[cfg.col_fol_id].notna(), cfg.col_t].tolist() if cfg.col_fol_id in df_case.columns else []

    preced_x = {}
    follow_x = {}
    follow_v = {}

    if "preceding_mileage_pos" in df_case.columns and preced_time:
        preced_x = df_case.set_index(cfg.col_t).loc[preced_time, "preceding_mileage_pos"].to_dict()
    if "following_mileage_pos" in df_case.columns and follow_time:
        follow_x = df_case.set_index(cfg.col_t).loc[follow_time, "following_mileage_pos"].to_dict()
    if cfg.col_fol_v in df_case.columns and follow_time:
        follow_v = df_case.set_index(cfg.col_t).loc[follow_time, cfg.col_fol_v].to_dict()

    b_v, b_x = build_soft_locks(base_v, cfg.lock_when_base_v_zero)

    vehicle_id = str(df_case[cfg.col_vid].iloc[0]) if cfg.col_vid in df_case.columns else "unknown"

    return {
        "vehicle_id": vehicle_id,
        "time": time_list,
        "base_v": base_v,
        "base_x": base_x,
        "b_v": b_v,
        "b_x": b_x,
        "preced_time": set(preced_time),
        "follow_time": set(follow_time),
        "preced_x": preced_x,
        "follow_x": follow_x,
        "follow_v": follow_v,
        "x_safe": float(safe_distance),
        "M": float(cfg.big_m),

        "v_limit": float(cfg.v_max),
        "v_max": float(cfg.v_max),
        "a_upper": float(cfg.a_upper),
        "a_lower": float(cfg.a_lower),
        "jerk_upper": float(cfg.jerk_upper),
        "jerk_lower": float(cfg.jerk_lower),

        "w_a": float(cfg.w_a),
        "w_v": float(cfg.w_v),
        "w_x": float(cfg.w_x),
    }


# ----------------------------
# Stage-1 solve
# ----------------------------
def solve_stage1_case(params: Dict, cfg: TRIMOptimizeConfig, case_id: str, logger: logging.Logger) -> pd.DataFrame:
    time_list = list(params["time"])
    if len(time_list) < 2:
        return pd.DataFrame()

    m = None
    env = None
    try:
        m, env = _new_model(cfg)

        a = m.addVars(time_list, lb=params["a_lower"], ub=params["a_upper"], name="a")
        v = m.addVars(time_list, lb=0.0, ub=params["v_limit"], name="v")
        x = m.addVars(time_list, lb=-GRB.INFINITY, name="x")

        for i in range(1, len(time_list)):
            t = time_list[i]
            tp = time_list[i - 1]
            m.addConstr(v[t] == v[tp] + a[t], name=f"update_v[{t}]")
            m.addConstr(x[t] == x[tp] + (v[tp] + 0.5 * a[t]), name=f"update_x[{t}]")
            m.addConstr(x[t] >= x[tp], name=f"mono_x[{t}]")

        m.addConstr(a[time_list[-1]] == 0.0, name="anchor_a_end")

        for i in range(1, len(time_list)):
            t = time_list[i]
            tp = time_list[i - 1]
            m.addConstr(a[t] - a[tp] <= params["jerk_upper"], name=f"jerk_up[{t}]")
            m.addConstr(a[t] - a[tp] >= params["jerk_lower"], name=f"jerk_lo[{t}]")

        for t in params["preced_time"]:
            if t in params["preced_x"]:
                m.addConstr(params["preced_x"][t] - x[t] >= params["x_safe"] + 0.1 * v[t], name=f"preced[{t}]")
        for t in params["follow_time"]:
            if t in params["follow_x"]:
                m.addConstr(x[t] - params["follow_x"][t] >= params["x_safe"] + 0.1 * params["v_max"],
                            name=f"follow[{t}]")

        M = params["M"]
        for t in time_list:
            bx = params["b_x"][t]
            bv = params["b_v"][t]
            m.addConstr(x[t] - params["base_x"][t] <= M * (1 - bx), name=f"x_soft_u[{t}]")
            m.addConstr(x[t] - params["base_x"][t] >= -M * (1 - bx), name=f"x_soft_l[{t}]")
            m.addConstr(v[t] - params["base_v"][t] <= M * (1 - bv), name=f"v_soft_u[{t}]")
            m.addConstr(v[t] - params["base_v"][t] >= -M * (1 - bv), name=f"v_soft_l[{t}]")

        obj = (
                params["w_a"] * quicksum(a[t] * a[t] for t in time_list)
                + params["w_v"] * quicksum(
            (v[t] - params["base_v"][t]) * (v[t] - params["base_v"][t]) for t in time_list)
        )
        if params["w_x"] > 0:
            obj += params["w_x"] * quicksum(
                (x[t] - params["base_x"][t]) * (x[t] - params["base_x"][t]) for t in time_list)

        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        if m.SolCount <= 0:
            return pd.DataFrame()

        df_out = pd.DataFrame({
            cfg.col_t: time_list,
            cfg.col_vid: params["vehicle_id"],
            cfg.col_case: str(case_id),
            "vehicle_accel": [a[t].X for t in time_list],
            "vehicle_speed": [v[t].X for t in time_list],
            cfg.col_s: [x[t].X for t in time_list],
        })
        return df_out

    except Exception as e:
        veh_id = params["vehicle_id"]
        logger.error(f"[stage1][vehicle={veh_id}, case={case_id}] solve failed: {e}")
        return pd.DataFrame()

    finally:
        try:
            if m is not None:
                m.dispose()
        except Exception:
            pass
        try:
            if env is not None:
                env.dispose()
        except Exception:
            pass


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


def select_bad_case_keys_from_stage1(
        df_stage1: pd.DataFrame,
        cfg: TRIMOptimizeConfig,
        jerk_env_path: str,
        logger: logging.Logger,
) -> Set[Tuple[str, str]]:
    """
    Implement your original rule on Stage1 output:

    - sort by (trip_id, vehicle_id, timestep_time)
    - jerk_t = a_{t+1} - a_t (assigned to current row)
    - accel_round = round(a_t, 0.1)
    - in_envelope requires jerk notna and bounds exist and within [min,max]
    - case-level: ONLY check mid points idx=1..n-2 (ignore first and last row)
      n<3 => OK (not bad)
    Return: set of bad keys (vehicle_id, trip_id)  (NOTE: stored as (vid, case) for filtering tasks)
    """
    req = [cfg.col_case, cfg.col_vid, cfg.col_t, "vehicle_accel", "vehicle_speed", cfg.col_s]
    require_cols(df_stage1, req, "stage1_df_for_envelope_check")

    df = df_stage1.copy()
    df = _force_key_dtypes(df, cfg)
    df = df.dropna(subset=[cfg.col_t]).copy()
    df = df[df["vehicle_speed"].notna()].copy()

    # stable sort
    df = df.sort_values([cfg.col_case, cfg.col_vid, cfg.col_t], kind="mergesort").reset_index(drop=True)

    # jerk forward diff assigned to current row
    df["_a_num"] = pd.to_numeric(df["vehicle_accel"], errors="coerce")
    df["_a_next"] = df.groupby([cfg.col_case, cfg.col_vid])["_a_num"].shift(-1)
    df["_jerk"] = df["_a_next"] - df["_a_num"]

    # accel round and merge envelope
    env = load_jerk_envelope_df(jerk_env_path, cfg.accel_round_decimals, cfg.csv_encoding)
    df["_accel_round"] = pd.to_numeric(df["_a_num"], errors="coerce").round(cfg.accel_round_decimals)
    df = df.merge(env, left_on="_accel_round", right_on="accel_round", how="left")

    df["_in_env"] = (
            df["_jerk"].notna()
            & df["jerk_min"].notna()
            & df["jerk_max"].notna()
            & (df["_jerk"] >= df["jerk_min"])
            & (df["_jerk"] <= df["jerk_max"])
    )

    # idx/n per case (use ALL rows as in your original script)
    df["n_in_case"] = df.groupby([cfg.col_case, cfg.col_vid])[cfg.col_case].transform("size")
    df["idx_in_case"] = df.groupby([cfg.col_case, cfg.col_vid]).cumcount()

    # mid points only: idx=1..n-2
    df_mid = df[(df["idx_in_case"] >= 1) & (df["idx_in_case"] <= df["n_in_case"] - 2)].copy()

    case_all = df_mid.groupby([cfg.col_case, cfg.col_vid])["_in_env"].all()

    # include all ids (default False)
    all_pairs = df[[cfg.col_case, cfg.col_vid]].drop_duplicates()
    all_index = pd.MultiIndex.from_frame(all_pairs[[cfg.col_case, cfg.col_vid]])
    case_all = case_all.reindex(all_index, fill_value=False)

    # n<3 => OK
    small_pairs = df.loc[df["n_in_case"] < 3, [cfg.col_case, cfg.col_vid]].drop_duplicates()
    if not small_pairs.empty:
        for _, r in small_pairs.iterrows():
            case_all.loc[(str(r[cfg.col_case]), str(r[cfg.col_vid]))] = True

    ok_mask = case_all.astype(bool)
    bad_index = case_all.index[~ok_mask]

    # return as (vid, case) set for filtering tasks
    bad_keys: Set[Tuple[str, str]] = set((str(vid), str(cid)) for (cid, vid) in bad_index.tolist())

    logger.info(
        f"[stage2-select] total_cases={len(case_all):,}, bad_cases={len(bad_keys):,}, ok_cases={(len(case_all) - len(bad_keys)):,}")
    return bad_keys


# ----------------------------
# Stage-2 solve (same as v4, but can filter cases)
# ----------------------------
def solve_stage2_case(
        params: Dict,
        accels: List[float],
        jmin_dict: Dict[float, float],
        jmax_dict: Dict[float, float],
        cfg: TRIMOptimizeConfig,
        case_id: str,
        logger: logging.Logger,
) -> pd.DataFrame:
    time_list = list(params["time"])
    if len(time_list) < 2:
        return pd.DataFrame()

    times1 = time_list[1:]
    prev_of = {time_list[i]: time_list[i - 1] for i in range(1, len(time_list))}

    jmin_arr = np.array([float(jmin_dict.get(float(a), np.nan)) for a in accels], dtype=float)
    jmax_arr = np.array([float(jmax_dict.get(float(a), np.nan)) for a in accels], dtype=float)
    if np.isnan(jmin_arr).any() or np.isnan(jmax_arr).any():
        logger.error(f"[stage2][case={case_id}] Jerk envelope alignment failed: missing accel_round entries")
        return pd.DataFrame()

    m = None
    env = None
    try:
        m, env = _new_model(cfg)

        jerk = m.addVars(time_list, lb=params["jerk_lower"], ub=params["jerk_upper"], name="jerk")
        a = m.addVars(time_list, lb=params["a_lower"], ub=params["a_upper"], name="a")
        v = m.addVars(time_list, lb=0.0, ub=params["v_limit"], name="v")
        x = m.addVars(time_list, lb=-GRB.INFINITY, name="x")

        lam = {(t, y): m.addVar(lb=0.0, name=f"lam[{t},{y}]") for t in times1 for y in range(len(accels))}

        for t in times1:
            tp = prev_of[t]
            idxs = list(range(len(accels)))
            lam_vars = [lam[(t, y)] for y in idxs]

            m.addConstr(quicksum(lam_vars) == 1.0, name=f"lam_sum[{t}]")
            m.addConstr(a[tp] == quicksum(accels[y] * lam[(t, y)] for y in idxs), name=f"a_interp[{t}]")
            m.addSOS(GRB.SOS_TYPE2, lam_vars, idxs)

            Jmax_at_a = quicksum(float(jmax_arr[y]) * lam[(t, y)] for y in idxs)
            Jmin_at_a = quicksum(float(jmin_arr[y]) * lam[(t, y)] for y in idxs)

            m.addConstr(jerk[tp] <= Jmax_at_a, name=f"j_env_u[{t}]")
            m.addConstr(jerk[tp] >= Jmin_at_a, name=f"j_env_l[{t}]")

        for t in times1:
            tp = prev_of[t]
            m.addConstr(v[t] == v[tp] + a[tp] + 0.5 * jerk[tp], name=f"update_v[{t}]")
            m.addConstr(a[t] == a[tp] + jerk[tp], name=f"update_a[{t}]")
            m.addConstr(x[t] == x[tp] + (v[tp] + 0.5 * a[tp] + jerk[tp] / 6.0), name=f"update_x[{t}]")

        for t in params["preced_time"]:
            if t in params["preced_x"]:
                m.addConstr(params["preced_x"][t] - x[t] >= params["x_safe"] + 0.1 * v[t], name=f"preced[{t}]")
        for t in params["follow_time"]:
            if t in params["follow_x"]:
                m.addConstr(x[t] - params["follow_x"][t] >= params["x_safe"] + 0.1 * params["v_max"],
                            name=f"follow[{t}]")

        M = params["M"]
        for t in time_list:
            bx = params["b_x"][t]
            bv = params["b_v"][t]
            m.addConstr(x[t] - params["base_x"][t] <= M * (1 - bx), name=f"x_soft_u[{t}]")
            m.addConstr(x[t] - params["base_x"][t] >= -M * (1 - bx), name=f"x_soft_l[{t}]")
            m.addConstr(v[t] - params["base_v"][t] <= M * (1 - bv), name=f"v_soft_u[{t}]")
            m.addConstr(v[t] - params["base_v"][t] >= -M * (1 - bv), name=f"v_soft_l[{t}]")
            m.addConstr(v[t] >= -M * bv, name=f"v_soft_l1[{t}]")

        # ---- 目标函数（与模型一致，仅最小化 a^2；需要可扩展 w_v/w_x 再加）----
        obj = params['w_a'] * quicksum(a[t]*a[t] for t in time_list)
        obj += params['w_v'] * quicksum((v[t]-params['base_v'][t])*(v[t]-params['base_v'][t]) for t in time_list)
        # obj += params['w_x'] * quicksum((x[t]-params['base_x'][t])*(x[t]-params['base_x'][t]) * (1-bx[t]) for t in time)

        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        if m.SolCount <= 0:
            return pd.DataFrame()

        df_out = pd.DataFrame({
            cfg.col_t: time_list,
            cfg.col_vid: params["vehicle_id"],
            cfg.col_case: str(case_id),
            "vehicle_jerk": [jerk[t].X for t in time_list],
            "vehicle_accel": [a[t].X for t in time_list],
            "vehicle_speed": [v[t].X for t in time_list],
            cfg.col_s: [x[t].X for t in time_list],
        })
        return df_out

    except Exception as e:
        logger.error(f"[stage2][case={case_id}] solve failed: {e}")
        return pd.DataFrame()

    finally:
        try:
            if m is not None:
                m.dispose()
        except Exception:
            pass
        try:
            if env is not None:
                env.dispose()
        except Exception:
            pass


# ----------------------------
# Task iterator and workers
# ----------------------------
def _iter_case_groups(df: pd.DataFrame, cfg: TRIMOptimizeConfig) -> List[Tuple[str, str, pd.DataFrame]]:
    tasks = []
    for (veh_id, cid), g in df.groupby([cfg.col_vid, cfg.col_case], sort=False):
        tasks.append((str(veh_id), str(cid), g))
    return tasks


def _stage1_worker(task, cfg: TRIMOptimizeConfig, safe_distance: float) -> pd.DataFrame:
    veh_id, cid, df_case = task
    params = prepare_params_for_case(df_case, cfg, safe_distance)
    if not (pd.to_numeric(df_case[cfg.col_v], errors="coerce").fillna(0.0) != 0.0).any():
        return pd.DataFrame()
    logger = logging.getLogger("trim")
    return solve_stage1_case(params, cfg, cid, logger)


def _stage2_worker(task, cfg: TRIMOptimizeConfig, safe_distance: float,
                   accels: List[float], jmin: Dict[float, float], jmax: Dict[float, float]) -> pd.DataFrame:
    veh_id, cid, df_case = task
    params = prepare_params_for_case(df_case, cfg, safe_distance)
    if not (pd.to_numeric(df_case[cfg.col_v], errors="coerce").fillna(0.0) != 0.0).any():
        return pd.DataFrame()
    logger = logging.getLogger("trim")
    return solve_stage2_case(params, accels, jmin, jmax, cfg, cid, logger)


# ----------------------------
# Load jerk envelope for stage2
# ----------------------------
def load_jerk_envelope(jerk_env_csv: str, accel_round_decimals: int) -> Tuple[
    List[float], Dict[float, float], Dict[float, float]]:
    env = pd.read_csv(jerk_env_csv, encoding="utf-8-sig")
    need = ["accel_round", "q1_smooth", "q99_smooth"]
    miss = [c for c in need if c not in env.columns]
    if miss:
        raise ValueError(f"Jerk envelope missing columns: {miss}")

    env["accel_round"] = pd.to_numeric(env["accel_round"], errors="coerce").round(accel_round_decimals)
    env["jerk_min"] = pd.to_numeric(env["q1_smooth"], errors="coerce")
    env["jerk_max"] = pd.to_numeric(env["q99_smooth"], errors="coerce")
    env = env.dropna(subset=["accel_round", "jerk_min", "jerk_max"]).copy()

    accels = sorted(env["accel_round"].astype(float).unique().tolist())
    jmin = dict(zip(env["accel_round"].astype(float), env["jerk_min"].astype(float)))
    jmax = dict(zip(env["accel_round"].astype(float), env["jerk_max"].astype(float)))
    return accels, jmin, jmax


# ----------------------------
# Run stage1/stage2/all
# ----------------------------
def run_stage1(cfg: TRIMOptimizeConfig, logger: Optional[logging.Logger] = None) -> str:
    ensure_dir(cfg.out_dir)
    logger = logger or setup_logger(
        log_path=os.path.join(cfg.out_dir, "trim_stage1.log"),
        to_console=cfg.log_to_console
    )

    df_all = pd.read_csv(cfg.input_csv, encoding=cfg.csv_encoding)
    need_cols = [cfg.col_case, cfg.col_vid, cfg.col_t, cfg.col_v, cfg.col_s]
    require_cols(df_all, need_cols, "input_csv")

    df_all, safe_distance = load_and_prepare(df_all, cfg, logger)

    df_clean = df_all.groupby(cfg.col_case).filter(
        lambda g: (pd.to_numeric(g[cfg.col_v], errors="coerce").fillna(0.0) != 0.0).any()
    )
    df_clean = df_clean.sort_values([cfg.col_case, cfg.col_vid], ascending=[False, False])

    tasks = _iter_case_groups(df_clean, cfg)
    logger.info(f"[stage1] tasks={len(tasks):,}, n_jobs={cfg.n_jobs}, safe_distance={safe_distance:.3f}")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=cfg.n_jobs) as ex:
        futs = [ex.submit(_stage1_worker, task, cfg, safe_distance) for task in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="stage1 optimize"):
            df_res = fut.result()
            if df_res is not None and not df_res.empty:
                results.append(df_res)

    if not results:
        raise RuntimeError("Stage1 produced no usable results (all trips infeasible or filtered)")

    df_results = pd.concat(results, ignore_index=True)
    df_results = _force_key_dtypes(df_results, cfg)
    df_results = df_results.dropna(subset=[cfg.col_t]).copy()
    df_results = df_results.sort_values([cfg.col_vid, cfg.col_case, cfg.col_t])

    key_cols = [cfg.col_vid, cfg.col_case, cfg.col_t]
    dfm = df_all.merge(df_results, on=key_cols, how="left", suffixes=("", "__trim"))

    for base_col, trim_col in [
        ("vehicle_speed", "vehicle_speed__trim"),
        ("vehicle_accel", "vehicle_accel__trim"),
        (cfg.col_s, f"{cfg.col_s}__trim"),
        ("vehicle_jerk", "vehicle_jerk__trim"),
    ]:
        if trim_col in dfm.columns:
            if base_col in dfm.columns:
                dfm[base_col] = dfm[trim_col].where(dfm[trim_col].notna(), dfm[base_col])
            else:
                dfm[base_col] = dfm[trim_col]
            dfm.drop(columns=[trim_col], inplace=True, errors="ignore")

    out_path = os.path.join(cfg.out_dir, f"{cfg.prefix}_trim_stage1.{cfg.out_format}")
    if cfg.out_format.lower() == "parquet":
        dfm.to_parquet(out_path, index=False)
    else:
        dfm.to_csv(out_path, index=False, encoding=cfg.csv_encoding)

    logger.info(f"[stage1] done -> {out_path} | time={(time.time() - t0) / 60:.2f} min")
    return out_path


def run_stage2(
        cfg: TRIMOptimizeConfig,
        logger: Optional[logging.Logger] = None,
        case_keys_filter: Optional[Set[Tuple[str, str]]] = None,  # (vehicle_id, trip_id)
) -> str:
    ensure_dir(cfg.out_dir)
    logger = logger or setup_logger(
        log_path=os.path.join(cfg.out_dir, "trim_stage2.log"),
        to_console=cfg.log_to_console
    )

    cfg.jerk_envelope_csv = resolve_resource_path(cfg.jerk_envelope_csv or "", "jerk_envelope.csv")

    df_all = pd.read_csv(cfg.input_csv, encoding=cfg.csv_encoding)
    need_cols = [cfg.col_case, cfg.col_vid, cfg.col_t, cfg.col_v, cfg.col_s]
    require_cols(df_all, need_cols, "input_csv_stage2")

    df_all, safe_distance = load_and_prepare(df_all, cfg, logger)

    # filter tasks by bad keys if provided
    if case_keys_filter is not None:
        df_all["_key"] = list(zip(df_all[cfg.col_vid].astype(str), df_all[cfg.col_case].astype(str)))
        df_all = df_all[df_all["_key"].isin(case_keys_filter)].drop(columns=["_key"], errors="ignore").copy()

    # if empty => stage2 nothing to do
    if df_all.empty:
        out_path = os.path.join(cfg.out_dir, f"{cfg.prefix}_trim_stage2.{cfg.out_format}")
        # write an empty file with headers similar to input
        df_all.to_csv(out_path, index=False, encoding=cfg.csv_encoding)
        logger.info("[stage2] no bad cases found -> stage2 skipped, output empty placeholder.")
        return out_path

    accels, jmin, jmax = load_jerk_envelope(cfg.jerk_envelope_csv, cfg.accel_round_decimals)

    df_clean = df_all.groupby(cfg.col_case).filter(
        lambda g: (pd.to_numeric(g[cfg.col_v], errors="coerce").fillna(0.0) != 0.0).any()
    )
    tasks = _iter_case_groups(df_clean, cfg)
    logger.info(f"[stage2] tasks={len(tasks):,}, n_jobs={cfg.n_jobs}, safe_distance={safe_distance:.3f}")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=cfg.n_jobs) as ex:
        futs = [ex.submit(_stage2_worker, task, cfg, safe_distance, accels, jmin, jmax) for task in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="stage2 optimize"):
            df_res = fut.result()
            if df_res is not None and not df_res.empty:
                results.append(df_res)

    if not results:
        raise RuntimeError("Stage2 produced no usable results (bad cases infeasible or filtered)")

    df_results = pd.concat(results, ignore_index=True)
    df_results = _force_key_dtypes(df_results, cfg)
    df_results = df_results.dropna(subset=[cfg.col_t]).copy()
    df_results = df_results.sort_values([cfg.col_vid, cfg.col_case, cfg.col_t])

    key_cols = [cfg.col_vid, cfg.col_case, cfg.col_t]
    dfm = df_all.merge(df_results, on=key_cols, how="left", suffixes=("", "__trim"))

    for base_col, trim_col in [
        ("vehicle_jerk", "vehicle_jerk__trim"),
        ("vehicle_accel", "vehicle_accel__trim"),
        ("vehicle_speed", "vehicle_speed__trim"),
        (cfg.col_s, f"{cfg.col_s}__trim"),
    ]:
        if trim_col in dfm.columns:
            if base_col in dfm.columns:
                dfm[base_col] = dfm[trim_col].where(dfm[trim_col].notna(), dfm[base_col])
            else:
                dfm[base_col] = dfm[trim_col]
            dfm.drop(columns=[trim_col], inplace=True, errors="ignore")

    out_path = os.path.join(cfg.out_dir, f"{cfg.prefix}_trim_stage2.{cfg.out_format}")
    if cfg.out_format.lower() == "parquet":
        dfm.to_parquet(out_path, index=False)
    else:
        dfm.to_csv(out_path, index=False, encoding=cfg.csv_encoding)

    logger.info(f"[stage2] done -> {out_path} | time={(time.time() - t0) / 60:.2f} min")
    return out_path
