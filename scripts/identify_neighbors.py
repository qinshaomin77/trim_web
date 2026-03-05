# -*- coding: utf-8 -*-
"""
identify_neighbors.py

Purpose:
- Input: FCD CSV (from xml2csv_fcd.py)
- Using lane topology (next/prev lanes) and lane lengths,
  identify preceding/following vehicles for each vehicle at each timestep.

Notes:
- This step depends on lane-based coordinates (vehicle_lane, vehicle_pos).
- This module uses logger "trim.identify_neighbors" and does NOT configure handlers.
  GUI/pipeline should configure logging handlers (File + Live Console).

Core logic rewrite (v2):
1) process_one_timestep now groups by lane internally, matching the
   reference process_timestep_lane approach.
2) For each lane, candidate pools for preceding/following vehicles are built
   by expanding into next/prev lanes up to `hop` layers (default=2).
3) Offset calculation follows physical SUMO lane semantics:
   - Forward (next lanes): offset = cumulative lane lengths leaving current lane.
   - Backward (prev lanes): offset = cumulative lane lengths of upstream lanes.
4) Deduplication keeps the minimum offset per candidate lane.
5) merge_asof matches nearest forward/backward vehicles using unified coordinates.

Outputs:
- {out_dir}/{prefix}_neighbors.csv
- (optional) parquet
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pathlib import Path

LOGGER_NAME = "trim.identify_neighbors"


def _get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)

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

# --------------------------- Config ---------------------------

@dataclass
class NeighborConfig:
    # inputs
    input_fcd_csv: str
    lane_next_csv: str
    lane_length_csv: str

    # outputs
    out_dir: str
    prefix: str = "fcd"

    # parallel
    n_jobs: int = 1
    hop: int = 2  # 0/1/2... how many lane-graph hops to consider (default=2 for 2-layer search)

    # column names (expect these in FCD CSV)
    col_time: str = "timestep_time"
    col_vid: str = "vehicle_id"
    col_lane: str = "vehicle_lane"
    col_pos: str = "vehicle_pos"
    col_speed: str = "vehicle_speed"
    col_edge: str = "vehicle_edge"  # created if missing
    col_lane_length: str = "lane_length"

    # output format
    export_parquet: bool = False
    csv_encoding: str = "utf-8-sig"

    # if True, attempt to standardize known alternative names (light-touch)
    standardize_schema: bool = True


# --------------------------- Globals for worker ---------------------------

G_NEXT_MAP: Dict[str, List[str]] = {}
G_PREV_MAP: Dict[str, List[str]] = {}
G_LEN_MAP: Dict[str, float] = {}


def _init_worker(next_map: Dict[str, List[str]], prev_map: Dict[str, List[str]], len_map: Dict[str, float]) -> None:
    global G_NEXT_MAP, G_PREV_MAP, G_LEN_MAP
    G_NEXT_MAP = next_map
    G_PREV_MAP = prev_map
    G_LEN_MAP = len_map


# --------------------------- Helpers ---------------------------

def _ensure_file(path: str, label: str) -> None:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _safe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _lane_to_edge(lane_id: str) -> str:
    s = str(lane_id)
    if s.startswith(":"):
        return s
    return s.rsplit("_", 1)[0] if "_" in s else s


def build_lane_maps(lane_next_csv: str, lane_length_csv: str, col_lane: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, float], pd.DataFrame]:
    """
    Build next_map, prev_map, len_map, and a lane_length table for merge.
    Expected lane_next_csv columns: lane_id, next_lane_id
    Expected lane_length_csv columns: lane_id, length
    """
    _ensure_file(lane_next_csv, "lane_next_csv")
    _ensure_file(lane_length_csv, "lane_length_csv")

    df_next = pd.read_csv(lane_next_csv)
    df_len = pd.read_csv(lane_length_csv)

    # normalize expected column names if needed
    if "lane_id" not in df_next.columns or "next_lane_id" not in df_next.columns:
        raise ValueError(f"lane_next_csv must contain columns lane_id,next_lane_id. Got: {list(df_next.columns)}")

    if "lane_id" not in df_len.columns:
        raise ValueError(f"lane_length_csv must contain column lane_id. Got: {list(df_len.columns)}")

    # length column might be length / lane_length
    len_col = "length" if "length" in df_len.columns else ("lane_length" if "lane_length" in df_len.columns else None)
    if len_col is None:
        raise ValueError(f"lane_length_csv must contain length (or lane_length). Got: {list(df_len.columns)}")

    df_len = df_len[["lane_id", len_col]].rename(columns={len_col: "lane_length"})

    next_map: Dict[str, List[str]] = defaultdict(list)
    prev_map: Dict[str, List[str]] = defaultdict(list)

    for a, b in df_next[["lane_id", "next_lane_id"]].itertuples(index=False, name=None):
        if pd.isna(a) or pd.isna(b):
            continue
        a = str(a)
        b = str(b)
        next_map[a].append(b)
        prev_map[b].append(a)

    # stable ordering
    for k in list(next_map.keys()):
        next_map[k] = sorted(set(next_map[k]))
    for k in list(prev_map.keys()):
        prev_map[k] = sorted(set(prev_map[k]))

    len_map = dict(zip(df_len["lane_id"].astype(str), df_len["lane_length"].astype(float)))

    # Prepare merge table: vehicle_lane -> lane_length
    df_lane_len_for_merge = df_len.rename(columns={"lane_id": col_lane})

    return dict(next_map), dict(prev_map), len_map, df_lane_len_for_merge


def _standardize_schema_if_needed(df: pd.DataFrame, cfg: NeighborConfig) -> pd.DataFrame:
    """
    Light-touch schema standardization:
    - If 'vehicle_acceleration' exists, rename to 'vehicle_accel' (project convention).
    - No heavy recalculation here.
    """
    if not cfg.standardize_schema:
        return df

    rename_map = {}
    if "vehicle_acceleration" in df.columns and "vehicle_accel" not in df.columns:
        rename_map["vehicle_acceleration"] = "vehicle_accel"

    # Common FCD alternatives (optional; safe only when target not present)
    if "time" in df.columns and cfg.col_time not in df.columns:
        rename_map["time"] = cfg.col_time
    if "id" in df.columns and cfg.col_vid not in df.columns:
        rename_map["id"] = cfg.col_vid
    if "lane" in df.columns and cfg.col_lane not in df.columns:
        rename_map["lane"] = cfg.col_lane
    if "pos" in df.columns and cfg.col_pos not in df.columns:
        rename_map["pos"] = cfg.col_pos
    if "speed" in df.columns and cfg.col_speed not in df.columns:
        rename_map["speed"] = cfg.col_speed

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}. Existing: {list(df.columns)}")


# --------------------------- Candidate lane expansion ---------------------------

def _build_forward_candidates(lane_id: str, hop: int) -> Dict[str, float]:
    """
    Build forward (next-direction) candidate lanes with offsets for preceding vehicle search.

    Physical meaning: a vehicle on a next-lane at position p is effectively at
    position (p + offset) in the current lane's coordinate system, where offset
    is the cumulative length of lanes between the current lane and the candidate lane.

    Returns dict: {candidate_lane_id: min_offset}
    Always includes the current lane with offset=0.
    """
    lane_id = str(lane_id)
    lane_length = G_LEN_MAP.get(lane_id, 0.0)

    # candidate_lane -> minimum offset (keep shortest path)
    candidates: Dict[str, float] = {lane_id: 0.0}

    if hop <= 0:
        return candidates

    # Layer 1: direct next lanes, offset = current lane length
    next_lane_list = G_NEXT_MAP.get(lane_id, [])
    for nl in next_lane_list:
        off = lane_length
        if nl not in candidates or off < candidates[nl]:
            candidates[nl] = off

    if hop <= 1:
        return candidates

    # Layer 2: next-of-next lanes, offset = current lane length + intermediate lane length
    for nl in next_lane_list:
        nl_len = G_LEN_MAP.get(nl, 0.0)
        for n2 in G_NEXT_MAP.get(nl, []):
            off = lane_length + nl_len
            if n2 not in candidates or off < candidates[n2]:
                candidates[n2] = off

    # Layer 3+ (if hop > 2, generalize with BFS)
    if hop > 2:
        # For hop > 2, do additional BFS layers beyond the 2 explicit layers above
        # frontier: list of (lane, accumulated_offset_up_to_that_lane)
        # We already have layers 0,1,2. Continue from layer-2 frontier.
        frontier: List[Tuple[str, float]] = []
        for nl in next_lane_list:
            nl_len = G_LEN_MAP.get(nl, 0.0)
            for n2 in G_NEXT_MAP.get(nl, []):
                frontier.append((n2, lane_length + nl_len))

        for _ in range(hop - 2):
            new_frontier: List[Tuple[str, float]] = []
            for fl, fl_off in frontier:
                fl_len = G_LEN_MAP.get(fl, 0.0)
                for fn in G_NEXT_MAP.get(fl, []):
                    off = fl_off + fl_len
                    if fn not in candidates or off < candidates[fn]:
                        candidates[fn] = off
                        new_frontier.append((fn, off))
            frontier = new_frontier
            if not frontier:
                break

    return candidates


def _build_backward_candidates(lane_id: str, hop: int) -> Dict[str, float]:
    """
    Build backward (prev-direction) candidate lanes with offsets for following vehicle search.

    Physical meaning: a vehicle on a prev-lane at position p is effectively at
    position (p - offset) in the current lane's coordinate system, where offset
    is the length of the prev-lane(s) traversed.

    Returns dict: {candidate_lane_id: min_offset}
    Always includes the current lane with offset=0.
    """
    lane_id = str(lane_id)

    candidates: Dict[str, float] = {lane_id: 0.0}

    if hop <= 0:
        return candidates

    # Layer 1: direct prev lanes, offset = prev lane's own length
    prev_lane_list = G_PREV_MAP.get(lane_id, [])
    for pl in prev_lane_list:
        pl_length = G_LEN_MAP.get(pl, 0.0)
        if pl not in candidates or pl_length < candidates[pl]:
            candidates[pl] = pl_length

    if hop <= 1:
        return candidates

    # Layer 2: prev-of-prev lanes, offset = prev lane length + prev-prev lane length
    for pl in prev_lane_list:
        pl_len = G_LEN_MAP.get(pl, 0.0)
        for p2 in G_PREV_MAP.get(pl, []):
            p2_len = G_LEN_MAP.get(p2, 0.0)
            off = pl_len + p2_len
            if p2 not in candidates or off < candidates[p2]:
                candidates[p2] = off

    # Layer 3+ (if hop > 2)
    if hop > 2:
        frontier: List[Tuple[str, float]] = []
        for pl in prev_lane_list:
            pl_len = G_LEN_MAP.get(pl, 0.0)
            for p2 in G_PREV_MAP.get(pl, []):
                p2_len = G_LEN_MAP.get(p2, 0.0)
                frontier.append((p2, pl_len + p2_len))

        for _ in range(hop - 2):
            new_frontier: List[Tuple[str, float]] = []
            for fl, fl_off in frontier:
                for fn in G_PREV_MAP.get(fl, []):
                    fn_len = G_LEN_MAP.get(fn, 0.0)
                    off = fl_off + fn_len
                    if fn not in candidates or off < candidates[fn]:
                        candidates[fn] = off
                        new_frontier.append((fn, off))
            frontier = new_frontier
            if not frontier:
                break

    return candidates


# --------------------------- Per-lane processing ---------------------------

def _process_lane(
    timestep_time: float,
    lane_id: str,
    df_lane: pd.DataFrame,
    df_timestep: pd.DataFrame,
    cfg: NeighborConfig,
) -> pd.DataFrame:
    """
    For a single (timestep, lane), find the nearest preceding and following
    vehicles using topology-aware unified coordinates and merge_asof.

    This follows the reference process_timestep_lane logic:
    1) Build forward candidate pool (same lane + next lanes with offsets)
       -> position_preceding = vehicle_pos + offset
    2) Build backward candidate pool (same lane + prev lanes with offsets)
       -> position_following = vehicle_pos - offset
    3) merge_asof forward for nearest ahead vehicle (following/preceding)
    4) merge_asof backward for nearest behind vehicle
    """
    col_vid = cfg.col_vid
    col_lane = cfg.col_lane
    col_pos = cfg.col_pos
    col_speed = cfg.col_speed

    lane_id = str(lane_id)

    # --- Build forward candidates (for finding the vehicle AHEAD = "following") ---
    fwd_cands = _build_forward_candidates(lane_id, cfg.hop)
    fwd_lane_list = list(fwd_cands.keys())
    fwd_offset_map = fwd_cands  # {lane_id: offset}

    next_df = df_timestep[df_timestep[col_lane].isin(fwd_lane_list)][[col_vid, col_lane, col_pos, col_speed]].copy()
    if not next_df.empty:
        next_df = next_df.reset_index(drop=True)
        next_df["position_preceding"] = (
            next_df[col_pos].astype(float) + next_df[col_lane].map(fwd_offset_map).fillna(0.0).astype(float)
        )
        next_df = next_df.sort_values("position_preceding").reset_index(drop=True)

    # --- Build backward candidates (for finding the vehicle BEHIND = "preceding") ---
    bwd_cands = _build_backward_candidates(lane_id, cfg.hop)
    bwd_lane_list = list(bwd_cands.keys())
    bwd_offset_map = bwd_cands  # {lane_id: offset}

    prev_df = df_timestep[df_timestep[col_lane].isin(bwd_lane_list)][[col_vid, col_lane, col_pos, col_speed]].copy()
    if not prev_df.empty:
        prev_df = prev_df.reset_index(drop=True)
        prev_df["position_following"] = (
            prev_df[col_pos].astype(float) - prev_df[col_lane].map(bwd_offset_map).fillna(0.0).astype(float)
        )
        prev_df = prev_df.sort_values("position_following").reset_index(drop=True)

    # --- Current lane vehicles ---
    cur = df_lane[[col_vid, col_lane, col_pos, col_speed]].copy()
    cur["position"] = cur[col_pos].astype(float)
    cur = cur.sort_values("position").reset_index(drop=True)

    # --- merge_asof: find nearest vehicle AHEAD (forward direction) ---
    if not next_df.empty:
        result_front = pd.merge_asof(
            cur,
            next_df,
            left_on="position",
            right_on="position_preceding",
            direction="forward",
            allow_exact_matches=False,
            suffixes=("", "_preceding"),
        )
    else:
        result_front = cur.copy()
        result_front[f"{col_vid}_preceding"] = np.nan
        result_front[f"{col_pos}_preceding"] = np.nan
        result_front[f"{col_speed}_preceding"] = np.nan
        result_front[f"{col_lane}_preceding"] = np.nan
        result_front["position_preceding"] = np.nan

    result_front["following_headway_distance"] = (
        result_front["position_preceding"] - result_front["position"]
    )

    # --- merge_asof: find nearest vehicle BEHIND (backward direction) ---
    if not prev_df.empty:
        result_both = pd.merge_asof(
            result_front.sort_values("position"),
            prev_df,
            left_on="position",
            right_on="position_following",
            direction="backward",
            allow_exact_matches=False,
            suffixes=("", "_following"),
        )
    else:
        result_both = result_front.copy()
        result_both[f"{col_vid}_following"] = np.nan
        result_both[f"{col_pos}_following"] = np.nan
        result_both[f"{col_speed}_following"] = np.nan
        result_both[f"{col_lane}_following"] = np.nan
        result_both["position_following"] = np.nan

    result_both["preceding_headway_distance"] = (
        result_both["position"] - result_both["position_following"]
    )

    # --- Assemble output columns ---
    # Naming convention (consistent with downstream):
    #   "following_*" = the vehicle AHEAD (in travel direction)
    #   "preceding_*" = the vehicle BEHIND
    rename_map = {
        f"{col_vid}_preceding": "following_vehicle_id",
        f"{col_pos}_preceding": "following_flow_pos",
        f"{col_speed}_preceding": "following_vehicle_speed",
        f"{col_lane}_preceding": "following_vehicle_lane",
        f"{col_vid}_following": "preceding_vehicle_id",
        f"{col_pos}_following": "preceding_flow_pos",
        f"{col_speed}_following": "preceding_vehicle_speed",
        f"{col_lane}_following": "preceding_vehicle_lane",
    }

    # Only rename columns that exist (defensive)
    existing_rename = {k: v for k, v in rename_map.items() if k in result_both.columns}
    result = result_both.rename(columns=existing_rename)

    out_cols = [
        col_vid, cfg.col_time,
        "following_vehicle_id", "following_flow_pos", "following_vehicle_speed",
        "following_vehicle_lane", "following_headway_distance",
        "preceding_vehicle_id", "preceding_flow_pos", "preceding_vehicle_speed",
        "preceding_vehicle_lane", "preceding_headway_distance",
    ]

    # Add timestep_time column
    result[cfg.col_time] = timestep_time

    # Select only output columns that exist
    final_cols = [c for c in out_cols if c in result.columns]
    return result[final_cols]


# --------------------------- Timestep processing ---------------------------

def process_one_timestep(task: Tuple[float, pd.DataFrame, NeighborConfig]) -> pd.DataFrame:
    """
    Process one timestep's records: group by lane, then for each lane find
    preceding/following vehicles using topology-aware coordinate unification.

    Input task: (timestep_time, df_timestep, cfg)
    Returns: DataFrame with neighbor columns keyed by (vehicle_id, timestep_time).
    """
    t, df_t, cfg = task

    # Ensure types
    df_t = df_t.copy()
    df_t[cfg.col_vid] = df_t[cfg.col_vid].astype(str)
    df_t[cfg.col_lane] = df_t[cfg.col_lane].astype(str)
    df_t[cfg.col_pos] = pd.to_numeric(df_t[cfg.col_pos], errors="coerce")
    df_t[cfg.col_speed] = pd.to_numeric(df_t[cfg.col_speed], errors="coerce")

    # Group by lane, process each lane independently
    out_parts: List[pd.DataFrame] = []

    for lane_id, df_lane in df_t.groupby(cfg.col_lane, sort=False):
        part = _process_lane(
            timestep_time=t,
            lane_id=lane_id,
            df_lane=df_lane,
            df_timestep=df_t,
            cfg=cfg,
        )
        if not part.empty:
            out_parts.append(part)

    if not out_parts:
        return pd.DataFrame(columns=[
            cfg.col_time, cfg.col_vid,
            "following_vehicle_id", "following_flow_pos", "following_vehicle_speed",
            "following_vehicle_lane", "following_headway_distance",
            "preceding_vehicle_id", "preceding_flow_pos", "preceding_vehicle_speed",
            "preceding_vehicle_lane", "preceding_headway_distance",
        ])

    out = pd.concat(out_parts, ignore_index=True)
    out[cfg.col_vid] = out[cfg.col_vid].astype(str)
    return out


# --------------------------- Main run ---------------------------

def run(cfg: NeighborConfig) -> str:
    """
    Execute neighbor identification. Return output CSV path.
    """
    logger = _get_logger()

    _ensure_file(cfg.input_fcd_csv, "input_fcd_csv")
    _ensure_file(cfg.lane_next_csv, "lane_next_csv")
    _ensure_file(cfg.lane_length_csv, "lane_length_csv")
    _safe_mkdir(cfg.out_dir)

    logger.info("=" * 80)
    logger.info("[identify_neighbors] START")
    logger.info("input_fcd_csv : %s", cfg.input_fcd_csv)
    logger.info("lane_next_csv : %s", cfg.lane_next_csv)
    logger.info("lane_length_csv: %s", cfg.lane_length_csv)
    logger.info("out_dir       : %s", cfg.out_dir)
    logger.info("prefix        : %s", cfg.prefix)
    logger.info("n_jobs        : %d", cfg.n_jobs)
    logger.info("hop           : %d", cfg.hop)
    logger.info("=" * 80)

    # Build topology maps
    next_map, prev_map, len_map, df_lane_len_for_merge = build_lane_maps(
        cfg.lane_next_csv, cfg.lane_length_csv, cfg.col_lane
    )

    # Read FCD
    df = pd.read_csv(cfg.input_fcd_csv)

    # Light-touch schema standardization (including vehicle_acceleration -> vehicle_accel)
    df = _standardize_schema_if_needed(df, cfg)

    # Require minimal columns for this step
    _require_cols(df, [cfg.col_time, cfg.col_vid, cfg.col_lane, cfg.col_speed, cfg.col_pos], "input_fcd_csv")

    # Merge lane length if available
    if "lane_length" not in df.columns:
        df = df.merge(df_lane_len_for_merge, on=cfg.col_lane, how="left")
        # rename to cfg col name
        if "lane_length" in df.columns and cfg.col_lane_length != "lane_length":
            df = df.rename(columns={"lane_length": cfg.col_lane_length})
    else:
        # standardize column name if user set col_lane_length
        if cfg.col_lane_length != "lane_length":
            df = df.rename(columns={"lane_length": cfg.col_lane_length})

    # Create edge column if missing
    if cfg.col_edge not in df.columns:
        df[cfg.col_edge] = df[cfg.col_lane].astype(str).map(_lane_to_edge)

    # Clean types
    df[cfg.col_vid] = df[cfg.col_vid].astype(str)
    df[cfg.col_lane] = df[cfg.col_lane].astype(str)
    df[cfg.col_time] = pd.to_numeric(df[cfg.col_time], errors="coerce")
    df[cfg.col_speed] = pd.to_numeric(df[cfg.col_speed], errors="coerce")
    df[cfg.col_pos] = pd.to_numeric(df[cfg.col_pos], errors="coerce")

    df = df.dropna(subset=[cfg.col_time, cfg.col_vid, cfg.col_lane, cfg.col_pos]).copy()

    # Sort by time for grouping
    df = df.sort_values([cfg.col_vid, cfg.col_time, cfg.col_lane, cfg.col_pos]).reset_index(drop=True)

    times = df[cfg.col_time].unique()
    logger.info("timestep count: %d", len(times))
    logger.info("vehicle count : %d", df[cfg.col_vid].nunique())

    # Build tasks: (time, df_time, cfg)
    tasks = [(t, g, cfg) for t, g in df.groupby(cfg.col_time, sort=False)]

    # IMPORTANT FIX: Initialize globals even for single-core mode
    _init_worker(next_map, prev_map, len_map)

    results = []

    if cfg.n_jobs and cfg.n_jobs > 1:
        logger.info("Running in multi-process mode (n_jobs=%d)", cfg.n_jobs)
        with ProcessPoolExecutor(
            max_workers=cfg.n_jobs,
            initializer=_init_worker,
            initargs=(next_map, prev_map, len_map),
        ) as ex:
            futures = [ex.submit(process_one_timestep, task) for task in tasks]
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        logger.info("Running in single-process mode")
        for task in tasks:
            results.append(process_one_timestep(task))

    if results:
        df_nb = pd.concat(results, ignore_index=True)
    else:
        df_nb = pd.DataFrame()

    # Merge neighbor columns back
    key = [cfg.col_vid, cfg.col_time]
    df_out = df.merge(df_nb, on=key, how="left")

    # Output
    out_csv = os.path.join(cfg.out_dir, f"{cfg.prefix}_neighbors.csv")
    df_out.to_csv(out_csv, index=False, encoding=cfg.csv_encoding)
    logger.info("[OK] Output CSV: %s", out_csv)

    if cfg.export_parquet:
        out_parquet = os.path.join(cfg.out_dir, f"{cfg.prefix}_neighbors.parquet")
        df_out.to_parquet(out_parquet, index=False)
        logger.info("[OK] Output Parquet: %s", out_parquet)

    logger.info("[identify_neighbors] DONE")
    return out_csv


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

    p = argparse.ArgumentParser(description="Identify preceding/following vehicles per timestep (lane-based).")
    p.add_argument("--input_fcd_csv", required=True)
    p.add_argument("--lane_next_csv", required=True)
    p.add_argument("--lane_length_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prefix", default="fcd")

    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--hop", type=int, default=2)
    p.add_argument("--export_parquet", action="store_true")

    args = p.parse_args()

    cfg = NeighborConfig(
        input_fcd_csv=args.input_fcd_csv,
        lane_next_csv=args.lane_next_csv,
        lane_length_csv=args.lane_length_csv,
        out_dir=args.out_dir,
        prefix=args.prefix,
        n_jobs=args.n_jobs,
        hop=args.hop,
        export_parquet=args.export_parquet,
    )

    out_csv = run(cfg)
    logger.info("Finished. Output: %s", out_csv)

def run_gui(*, in_path, out_dir, state=None, upstream=None, options=None):
    """
    GUI adapter entrypoint.
    Accepts the standard GUI calling convention and internally builds NeighborConfig.
    """
    logger = _get_logger()

    # 1) resolve input csv
    in_p = Path(in_path) if in_path is not None else None
    if in_p is None:
        raise RuntimeError("[identify_neighbors] in_path is None.")
    if in_p.is_dir():
        # pick first csv under directory
        csvs = sorted(in_p.glob("*.csv"))
        if not csvs:
            raise RuntimeError(f"[identify_neighbors] No CSV found in directory: {in_p}")
        input_csv = csvs[0]
    else:
        input_csv = in_p

    # 2) find topology tables from net_topology output dir
    upstream = upstream or {}
    net_out = upstream.get("net_topology", None)
    if net_out is None:
        raise RuntimeError("[identify_neighbors] missing upstream['net_topology'] output (needed for lane tables).")

    net_dir = Path(net_out)
    if net_dir.is_file():
        net_dir = net_dir.parent

    # ------------------------------------------------------------------
    # Prefer SUMO_NET_lane_attributes_edge.csv (one file contains both next+length)
    # Expected columns (from net_topology export):
    #   lane_id, next_lane_id, lane_length, next_lane_length
    # We will materialize two temporary CSVs that match build_lane_maps() expectations:
    #   (1) lane_next_csv  : lane_id, next_lane_id
    #   (2) lane_length_csv: lane_id, lane_length
    # ------------------------------------------------------------------

    lane_attr = net_dir / "SUMO_NET_lane_attributes_edge.csv"
    lane_next = None
    lane_len = None

    if lane_attr.exists() and lane_attr.is_file():
        df_attr = pd.read_csv(lane_attr)

        need_cols = {"lane_id", "next_lane_id", "lane_length"}
        missing = [c for c in need_cols if c not in df_attr.columns]
        if missing:
            raise RuntimeError(
                "[identify_neighbors] SUMO_NET_lane_attributes_edge.csv missing required columns.\n"
                f"missing={missing}\n"
                f"existing={list(df_attr.columns)}\n"
                f"file={lane_attr}"
            )

        # write temporary lane_next/lane_length csv under current step out_dir
        tmp_next = Path(out_dir) / "__tmp_lane_next.csv"
        tmp_len  = Path(out_dir) / "__tmp_lane_length.csv"

        df_attr[["lane_id", "next_lane_id"]].dropna().drop_duplicates().to_csv(tmp_next, index=False)
        df_attr[["lane_id", "lane_length"]].dropna().drop_duplicates().to_csv(tmp_len, index=False)

        lane_next = tmp_next
        lane_len  = tmp_len

        logger.info("[identify_neighbors] Using lane attributes file: %s", lane_attr)
        logger.info("[identify_neighbors] Materialized lane_next_csv : %s", lane_next)
        logger.info("[identify_neighbors] Materialized lane_length_csv: %s", lane_len)

    else:
        # fallback: try find separate next/length tables (older exports)
        def _find_one(patterns):
            for pat in patterns:
                hits = sorted(net_dir.glob(pat))
                if hits:
                    return hits[0]
            return None

        lane_next = _find_one(["*lane*next*.csv", "*next*lane*.csv", "*next_lane*.csv", "*conn*.csv", "*connection*.csv"])
        lane_len  = _find_one(["*lane*length*.csv", "*length*.csv", "*lane_len*.csv"])

    if lane_next is None or lane_len is None:
        raise RuntimeError(
            "[identify_neighbors] Cannot locate lane_next/lane_length CSV under net_topology output.\n"
            f"net_topology_dir={net_dir}\n"
            f"tried lane_attributes_edge={lane_attr} (exists={lane_attr.exists()})\n"
            f"found lane_next={lane_next}\n"
            f"found lane_length={lane_len}\n"
            "Please check net_topology exports."
        )

    # 3) prefix
    prefix = "fcd"
    try:
        if state is not None and getattr(state, "inputs", None) is not None:
            prefix = getattr(state.inputs, "safe_prefix", None) or getattr(state.inputs, "raw_prefix", None) or "fcd"
    except Exception:
        prefix = "fcd"

    # 4) build cfg and run (auto parallel fallback)
    logger.info("[identify_neighbors] GUI wrapper using lane_next=%s", lane_next)
    logger.info("[identify_neighbors] GUI wrapper using lane_length=%s", lane_len)

    n_jobs = detect_max_workers()

    cfg = NeighborConfig(
        input_fcd_csv=str(input_csv),
        lane_next_csv=str(lane_next),
        lane_length_csv=str(lane_len),
        out_dir=str(out_dir),
        prefix=str(prefix),
        n_jobs=n_jobs,
        hop=2,  # default 2-layer search matching reference logic
    )

    return run(cfg)


if __name__ == "__main__":
    main()