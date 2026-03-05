# scripts/emission_spatial_map.py
# -*- coding: utf-8 -*-
"""
STEP-05: Emission Spatial Map (Emission Distribution Along Axis)

GUI-compatible version with robust pollutant column matching.

Key features:
- GT data extracted from upstream CSV based on data_type column
- Robust pollutant column matching with aliases (PM2.5/PM25, NOx/NOX, etc.)
- Case-insensitive matching
- Automatic column name cleanup (strip whitespace)
- Detailed diagnostic logging
- ROI polygon filtering (point-in-polygon) applied BEFORE computation
- ROI-aligned bin coordinates: bin_start_m starts from polygon x_min (or y_min)
- GT error CSV includes bin coordinate columns
- Time range filtering for simulation and GT data
- Emission statistics: total emissions and emission intensity (g/km)

Processing flow:
  Raw data
    → [polygon ROI filter]   vehicle points outside polygon are removed
    → [time filter]          rows outside time range are removed
    → [compute_emission_spatial]
         · assign mass to meter bins based on vehicle trajectory
         · bin origin = polygon x_min (or y_min) so bins align to ROI edge
         · bin_start_m / bin_center_m / bin_end_m reflect real coordinates
    → [compute_emission_statistics]
         · total emissions (g) by pollutant and data_type
         · emission intensity (g/km) by pollutant and data_type
    → save CSVs:
         · spatial_emission_{x|y}.csv - spatial distribution
         · spatial_error_pct_{x|y}.csv - GT error (if GT available)
         · emission_totals.csv - total emissions
         · emission_intensity.csv - emission per km
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration and constants
# ============================================================================

GT_DATA_TYPES = {"gt", "ground_truth", "groundtruth"}
REQUIRED_COLUMNS = ("timestep_time", "vehicle_id")


@dataclass
class AnalysisConfig:
    """Analysis configuration parameters"""
    axis_bin_size: float
    analysis_axis: str  # "x", "y", or "both"
    import_gt: bool
    enable_roi: bool
    enabled_pollutants: List[str]

    # ROI polygon text (used for polygon filtering AND bin origin derivation)
    roi_sim_text: str = ""
    roi_gt_text: str = ""

    # ROI axis-aligned bounds — derived from polygon vertices in _parse_config.
    # Used ONLY as bin origin for coordinate alignment in compute_emission_spatial.
    # Data filtering is handled by point-in-polygon (ROIProcessor.apply_filter),
    # NOT by these numeric bounds.
    x_min_roi: Optional[float] = None
    x_max_roi: Optional[float] = None
    y_min_roi: Optional[float] = None
    y_max_roi: Optional[float] = None

    # Time range filtering options
    enable_sumo_time_filter: bool = False
    sumo_time_start: Optional[float] = None
    sumo_time_end: Optional[float] = None
    enable_gt_time_filter: bool = False
    gt_time_start: Optional[float] = None
    gt_time_end: Optional[float] = None


# ============================================================================
# Utility functions
# ============================================================================

def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove whitespace from column names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _pollutant_aliases(p: str) -> List[str]:
    """
    Return alternative names for a pollutant.

    Examples:
    - "NOx" → ["NOx", "NOX", "nox"]
    - "PM2.5" → ["PM2.5", "PM25", "pm2.5", "pm25"]
    - "CO" → ["CO", "co"]
    """
    p = str(p).strip()
    up = p.upper()

    if up == "NOX":
        return ["NOx", "NOX", "nox"]

    if up in {"PM2.5", "PM2_5", "PM2-5", "PM2 5", "PM2,5", "PM25"}:
        return ["PM2.5", "PM25", "pm2.5", "pm25"]

    if up in {"PM10", "PM_10"}:
        return ["PM10", "PM_10", "pm10"]

    return [p, p.upper(), p.lower()]


# ============================================================================
# Module 1: File discovery and data loading
# ============================================================================

class DataLoader:
    """Responsible for file discovery and data loading"""

    @staticmethod
    def find_csv_files(path: Path) -> List[Path]:
        """Recursively find CSV files"""
        path = Path(path)
        if path.is_file() and path.suffix.lower() == ".csv":
            return [path]
        if not path.is_dir():
            return []
        return sorted(f for f in path.rglob("*.csv")
                      if not f.name.startswith((".", "~")))

    @staticmethod
    def load_and_concat(csv_paths: List[Path], tag_if_missing: Optional[str] = None) -> pd.DataFrame:
        """Load and concatenate CSV files"""
        frames = []
        for p in csv_paths:
            try:
                df = pd.read_csv(p, low_memory=False)
                df = _strip_columns(df)
                if tag_if_missing and "data_type" not in df.columns:
                    df["data_type"] = tag_if_missing
                frames.append(df)
                logger.info(f"  Loaded {p.name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"  Failed to load {p}: {e}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================================
# Module 2: Column detection
# ============================================================================

class ColumnDetector:
    """Column detection utilities"""

    @staticmethod
    def detect_data_type_col(df: pd.DataFrame) -> Optional[str]:
        """Detect data_type column with various aliases"""
        candidates = ["data_type", "source", "type", "method", "category"]
        col_lower = {str(c).lower(): str(c) for c in df.columns}
        for name in candidates:
            if name in df.columns:
                return name
            if name.lower() in col_lower:
                return col_lower[name.lower()]
        return None


# ============================================================================
# Module 3: ROI processing
# ============================================================================

class ROIProcessor:
    """Responsible for ROI polygon filtering and bound extraction"""

    @staticmethod
    def parse_polygon(roi_text: str) -> Optional[np.ndarray]:
        """
        Parse ROI polygon from text: 'x1,y1;x2,y2;x3,y3;...'

        Returns numpy array of shape (N, 2) or None if parsing fails.
        """
        if not roi_text or not roi_text.strip():
            return None
        try:
            pts = []
            for pair in roi_text.strip().split(";"):
                if not pair.strip():
                    continue
                parts = pair.strip().split(",")
                if len(parts) == 2:
                    pts.append((float(parts[0]), float(parts[1])))
            return np.array(pts, dtype=float) if len(pts) >= 3 else None
        except Exception:
            return None

    @staticmethod
    def get_polygon_axis_bounds(roi_text: str, axis: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract the min and max coordinate value along the specified axis
        from the polygon vertices.

        This is used ONLY for bin coordinate alignment (bin origin), NOT for
        data filtering. Data filtering is done by point_in_polygon.

        Parameters
        ----------
        roi_text : polygon text string 'x1,y1;x2,y2;...'
        axis     : 'x' or 'y'

        Returns
        -------
        (axis_min, axis_max) or (None, None) if parsing fails

        Example
        -------
        roi_text = '29,-45;388,-38;388,-70;29,-73'
        get_polygon_axis_bounds(roi_text, 'x') → (29.0, 388.0)
        get_polygon_axis_bounds(roi_text, 'y') → (-73.0, -38.0)
        """
        poly = ROIProcessor.parse_polygon(roi_text)
        if poly is None:
            return None, None
        col_idx = 0 if axis == "x" else 1
        vals = poly[:, col_idx]
        return float(vals.min()), float(vals.max())

    @staticmethod
    def point_in_polygon(px: np.ndarray, py: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """
        Ray casting algorithm for point-in-polygon test.

        Returns boolean array, True = point is inside polygon.
        """
        n = len(polygon)
        inside = np.zeros(len(px), dtype=bool)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            cond = ((yi > py) != (yj > py)) & (
                    px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
            )
            inside ^= cond
            j = i
        return inside

    @staticmethod
    def apply_filter(df: pd.DataFrame, roi_text: str) -> pd.DataFrame:
        """
        Apply ROI polygon filter to DataFrame.

        Keeps only rows where (vehicle_x, vehicle_y) falls inside the polygon.
        This is the primary spatial filtering step — all subsequent computation
        operates only on data within the polygon boundary.
        """
        if df.empty:
            return df

        poly = ROIProcessor.parse_polygon(roi_text)
        if poly is None:
            logger.warning("ROI polygon parse failed, no filter applied")
            return df

        x_col = "vehicle_x" if "vehicle_x" in df.columns else None
        y_col = "vehicle_y" if "vehicle_y" in df.columns else None

        if not x_col or not y_col:
            logger.warning("vehicle_x/vehicle_y not found, ROI polygon filter skipped")
            return df

        mask = ROIProcessor.point_in_polygon(
            df[x_col].to_numpy(float),
            df[y_col].to_numpy(float),
            poly
        )
        n_before = len(df)
        df_filtered = df.loc[mask].copy()
        logger.info(f"ROI polygon filter: {n_before} → {len(df_filtered)} rows "
                    f"(removed {n_before - len(df_filtered)} outside polygon)")
        return df_filtered


# ============================================================================
# Module: Emission Statistics Calculator
# ============================================================================

class EmissionStatisticsCalculator:
    """Calculate emission totals and per-vehicle statistics."""

    @staticmethod
    def compute_vehicle_distance(
            df: pd.DataFrame,
            data_type_col: str = "data_type"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute total distance traveled for each data_type using kinematic formula.

        Distance calculation method:
        For each time interval: distance = speed * dt + 0.5 * acceleration * dt²

        Returns aggregated distance by data_type (not per-vehicle details).

        Parameters
        ----------
        df : DataFrame with columns:
             - timestep_time
             - vehicle_id
             - vehicle_speed (m/s)
             - vehicle_acceleration (m/s²) - optional, if not present uses 0
             - data_type
        data_type_col : Name of data_type column

        Returns
        -------
        Dict with structure:
        {
            "estimated": {"total_km": 1234.567, "n_vehicles": 100},
            "gt": {"total_km": 1230.456, "n_vehicles": 100}
        }
        """
        if df.empty:
            return {}

        # Check required columns
        required = ["timestep_time", "vehicle_id", "vehicle_speed"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns for distance calculation: {missing}")
            return {}

        # Check if acceleration column exists
        has_accel = "vehicle_acceleration" in df.columns
        if not has_accel:
            logger.info("vehicle_acceleration column not found, using acceleration = 0")

        # Sort by vehicle and time
        df = df.sort_values([data_type_col, "vehicle_id", "timestep_time"])

        # Compute distance for each vehicle (internal, not saved)
        def calc_distance_per_vehicle(group):
            """
            Calculate distance for one vehicle using kinematic formula.

            Formula: distance = v * dt + 0.5 * a * dt²
            where:
              v = speed at current timestep (m/s)
              a = acceleration at current timestep (m/s²)
              dt = time interval (s)
            """
            group = group.sort_values("timestep_time").copy()

            # Get time and speed arrays
            time = group["timestep_time"].to_numpy()
            speed = group["vehicle_speed"].to_numpy()

            # Get acceleration if available
            if has_accel:
                accel = group["vehicle_acceleration"].to_numpy()
            else:
                accel = np.zeros_like(speed)

            # Compute dt (time differences)
            dt = np.diff(time, prepend=np.nan)

            # Calculate distance for each interval using kinematic formula
            distance_intervals = np.zeros_like(speed)
            for i in range(1, len(speed)):
                if np.isfinite(dt[i]) and dt[i] > 0:
                    # Use current speed and acceleration
                    v = speed[i] if np.isfinite(speed[i]) else 0.0
                    a = accel[i] if np.isfinite(accel[i]) else 0.0

                    # Kinematic formula: distance = v * dt + 0.5 * a * dt²
                    distance_intervals[i] = v * dt[i] + 0.5 * a * (dt[i] ** 2)

                    # Ensure non-negative distance (physical constraint)
                    if distance_intervals[i] < 0:
                        distance_intervals[i] = 0.0

            # Total distance in meters
            total_distance_m = distance_intervals.sum()

            return pd.Series({
                "distance_km": total_distance_m / 1000.0
            })

        # Group by data_type and vehicle_id
        vehicle_distances = df.groupby([data_type_col, "vehicle_id"], as_index=False).apply(
            calc_distance_per_vehicle
        )

        # Aggregate by data_type
        result = {}
        for dtype in vehicle_distances[data_type_col].unique():
            dtype_data = vehicle_distances[vehicle_distances[data_type_col] == dtype]
            result[dtype] = {
                "total_km": float(dtype_data["distance_km"].sum()),
                "n_vehicles": int(dtype_data["vehicle_id"].nunique())
            }

        return result

    @staticmethod
    def compute_emission_totals(
            df: pd.DataFrame,
            pollutant_rate_cols: List[str],
            data_type_col: str = "data_type"
    ) -> pd.DataFrame:
        """
        Compute total emissions for each pollutant and data_type.

        Method:
        - For each timestep, mass = rate (g/s) * dt (s)
        - Total emission = sum of all masses

        Parameters
        ----------
        df : DataFrame with timestep_time, vehicle_id, {pollutant}_gs columns
        pollutant_rate_cols : List of rate column names (e.g., ['NOx_gs', 'PM25_gs'])
        data_type_col : Name of data_type column

        Returns
        -------
        DataFrame with columns:
            - data_type
            - {pollutant}_g_total : Total emission mass (grams)
            for each pollutant
        """
        if df.empty:
            return pd.DataFrame()

        # Sort by data_type, vehicle, time
        df = df.sort_values([data_type_col, "vehicle_id", "timestep_time"]).copy()

        # Compute dt for each row
        df["dt"] = df.groupby([data_type_col, "vehicle_id"])["timestep_time"].diff()

        # For each pollutant rate column, compute mass
        mass_cols = []
        for rate_col in pollutant_rate_cols:
            if rate_col not in df.columns:
                continue

            # Extract pollutant name (remove _gs, _g, etc.)
            pollutant_name = rate_col.rsplit("_", 1)[0]  # "NOx_gs" -> "NOx"
            mass_col = f"{pollutant_name}_g_total"

            # Mass = rate * dt (only where dt > 0)
            df[mass_col] = 0.0
            valid_mask = (df["dt"] > 0) & df[rate_col].notna()
            df.loc[valid_mask, mass_col] = df.loc[valid_mask, rate_col] * df.loc[valid_mask, "dt"]

            mass_cols.append(mass_col)

        # Sum by data_type
        if mass_cols:
            result = df.groupby(data_type_col, as_index=False)[mass_cols].sum()
            return result

        return pd.DataFrame()

    @staticmethod
    def compute_emission_intensity(
            emission_totals: pd.DataFrame,
            distance_summary: Dict[str, Dict[str, float]],
            data_type_col: str = "data_type"
    ) -> pd.DataFrame:
        """
        Compute emission intensity (g/km) for each pollutant.

        Formula:
        emission_intensity (g/km) = total_emission (g) / total_distance (km)

        Parameters
        ----------
        emission_totals : DataFrame with data_type and {pollutant}_g_total columns
        distance_summary : Dict with distance stats by data_type
            e.g., {"estimated": {"total_km": 1234.5, "n_vehicles": 100}}
        data_type_col : Name of data_type column

        Returns
        -------
        DataFrame with columns:
            - data_type
            - total_distance_km
            - n_vehicles
            - {pollutant}_g_total
            - {pollutant}_g_per_km (emission intensity)
            for each pollutant
        """
        if emission_totals.empty or not distance_summary:
            return pd.DataFrame()

        # Convert distance summary to DataFrame
        dist_data = []
        for dtype, stats in distance_summary.items():
            dist_data.append({
                data_type_col: dtype,
                "total_distance_km": stats["total_km"],
                "n_vehicles": stats["n_vehicles"]
            })
        dist_df = pd.DataFrame(dist_data)

        # Merge emissions and distances
        result = pd.merge(emission_totals, dist_df, on=data_type_col, how="left")

        # Compute intensity for each pollutant
        mass_cols = [c for c in emission_totals.columns if c.endswith("_g_total")]

        for mass_col in mass_cols:
            pollutant_name = mass_col.replace("_g_total", "")
            intensity_col = f"{pollutant_name}_g_per_km"

            # Avoid division by zero
            result[intensity_col] = 0.0
            valid_mask = result["total_distance_km"] > 0
            result.loc[valid_mask, intensity_col] = (
                    result.loc[valid_mask, mass_col] / result.loc[valid_mask, "total_distance_km"]
            )

        return result


def compute_and_save_statistics(
        df_all: pd.DataFrame,
        pollutant_cols: List[str],
        out_dir: Path,
        data_type_col: str = "data_type"
) -> None:
    """
    Compute and save emission statistics.

    Outputs (only 2 files):
    1. emission_totals.csv - Total emissions by data_type and pollutant
    2. emission_intensity.csv - Emission per km by data_type and pollutant

    Does NOT save individual vehicle distances.
    """
    logger = logging.getLogger(__name__)
    calc = EmissionStatisticsCalculator()

    logger.info("=" * 60)
    logger.info("Computing Emission Statistics")
    logger.info("=" * 60)

    # 1. Compute total emissions
    logger.info("Computing total emissions...")
    emission_totals = calc.compute_emission_totals(
        df_all,
        pollutant_rate_cols=pollutant_cols,
        data_type_col=data_type_col
    )

    if not emission_totals.empty:
        out_csv = out_dir / "emission_totals.csv"
        emission_totals.to_csv(out_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Saved: {out_csv.name}")

        # Log summary
        for _, row in emission_totals.iterrows():
            dtype = row[data_type_col]
            logger.info(f"  [{dtype}]")
            for col in emission_totals.columns:
                if col.endswith("_g_total"):
                    pollutant = col.replace("_g_total", "")
                    value = row[col]
                    logger.info(f"    {pollutant}: {value:.3f} g")

    # 2. Compute vehicle distances (aggregated, not saved)
    logger.info("Computing aggregated distances...")
    distance_summary = calc.compute_vehicle_distance(
        df_all,
        data_type_col=data_type_col
    )

    if distance_summary:
        logger.info("Distance summary:")
        for dtype, stats in distance_summary.items():
            logger.info(f"  [{dtype}]")
            logger.info(f"    Total distance: {stats['total_km']:.3f} km")
            logger.info(f"    N vehicles: {stats['n_vehicles']}")

    # 3. Compute emission intensity
    logger.info("Computing emission intensity (g/km)...")
    emission_intensity = calc.compute_emission_intensity(
        emission_totals,
        distance_summary,
        data_type_col=data_type_col
    )

    if not emission_intensity.empty:
        out_csv = out_dir / "emission_intensity.csv"
        emission_intensity.to_csv(out_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Saved: {out_csv.name}")

        # Log summary
        for _, row in emission_intensity.iterrows():
            dtype = row[data_type_col]
            logger.info(f"  [{dtype}]")
            logger.info(f"    Total distance: {row['total_distance_km']:.3f} km")
            logger.info(f"    N vehicles: {row['n_vehicles']}")
            for col in emission_intensity.columns:
                if col.endswith("_g_per_km"):
                    pollutant = col.replace("_g_per_km", "")
                    value = row[col]
                    logger.info(f"    {pollutant}: {value:.6f} g/km")

    logger.info("=" * 60)


# ============================================================================
# Module 4: Spatial emission calculation
# ============================================================================

class SpatialEmissionCalculator:
    """Responsible for spatial emission distribution computation"""

    @staticmethod
    def compute_emission_spatial(
            df: pd.DataFrame,
            axis: str,
            axis_bin_size: float,
            pollutant_rate_cols: List[str],
            data_type_col: str = "data_type",
            meter_step_m: float = 1.0,
            roi_origin: Optional[float] = None,
    ) -> pd.DataFrame:
        df = df.copy()
        coord_col = f"vehicle_{axis}"

        if coord_col not in df.columns:
            raise ValueError(f"Missing column: {coord_col}")

        # Sort for consistent shift operations
        df = df.sort_values([data_type_col, "vehicle_id", "timestep_time"], kind="mergesort")

        # Compute next-step coordinate and time
        df["coord_next"] = df.groupby([data_type_col, "vehicle_id"])[coord_col].shift(-1)
        df["time_next"] = df.groupby([data_type_col, "vehicle_id"])["timestep_time"].shift(-1)

        # Compute interval duration
        df["duration"] = df["time_next"] - df["timestep_time"]

        # Keep valid intervals only
        valid = (df["duration"] > 0) & df["coord_next"].notna()
        df = df[valid].copy()

        if df.empty:
            return pd.DataFrame(
                columns=["axis", "axis_bin", data_type_col] +
                        [f"{p}_g" for p in pollutant_rate_cols] +
                        ["bin_start_m", "bin_center_m", "bin_end_m"]
            )

        # Allocate mass to 1-meter bins
        records = []

        for _, row in df.iterrows():
            coord_start = float(row[coord_col])
            coord_end = float(row["coord_next"])
            duration = float(row["duration"])
            data_type = row[data_type_col]

            m_start = int(np.floor(min(coord_start, coord_end)))
            m_end = int(np.floor(max(coord_start, coord_end)))

            meter_bins = list(range(m_start, m_end + 1))
            if not meter_bins:
                continue

            n_bins = len(meter_bins)

            for m_bin in meter_bins:
                rec = {"meter_bin": m_bin, data_type_col: data_type}
                for pcol in pollutant_rate_cols:
                    rate = row.get(pcol, 0.0)
                    if pd.isna(rate):
                        rate = 0.0
                    mass = float(rate) * duration / n_bins
                    rec[f"{str(pcol)[:-3]}_g"] = mass
                records.append(rec)

        if not records:
            return pd.DataFrame(
                columns=["axis", "axis_bin", data_type_col] +
                        [f"{p[:-3]}_g" for p in pollutant_rate_cols] +
                        ["bin_start_m", "bin_center_m", "bin_end_m"]
            )

        df_meter = pd.DataFrame(records)

        if roi_origin is not None:
            origin = roi_origin
            logger.info(f"Bin origin for axis '{axis}': {origin} m (from ROI polygon min)")
        else:
            origin = float(df_meter["meter_bin"].min())
            logger.info(f"Bin origin for axis '{axis}': {origin} m (fallback to data minimum)")

        # Compute axis_bin relative to origin
        df_meter["axis_bin"] = np.floor(
            (df_meter["meter_bin"] - origin) / axis_bin_size
        ).astype(int)

        group_cols = ["axis_bin", data_type_col]
        mass_cols = [c for c in df_meter.columns if c.endswith("_g")]

        agg_dict = {c: "sum" for c in mass_cols}
        result = df_meter.groupby(group_cols, as_index=False).agg(agg_dict)

        # Bin coordinates in original coordinate system
        result.insert(0, "axis", axis)
        result["bin_start_m"] = origin + result["axis_bin"] * axis_bin_size
        result["bin_center_m"] = result["bin_start_m"] + axis_bin_size / 2
        result["bin_end_m"] = result["bin_start_m"] + axis_bin_size

        return result.sort_values(["axis_bin", data_type_col]).reset_index(drop=True)


# ============================================================================
# Module 5: GT error calculation
# ============================================================================

class GTErrorCalculator:
    """Responsible for GT error calculation"""

    @staticmethod
    def compute_error_pct(
            df: pd.DataFrame,
            axis: str,
            data_type_col: str,
            bin_col: str,
            pollutant_mass_cols: List[str],
    ) -> pd.DataFrame:
        """
        Compute GT error percentage.

        bin coordinate columns (bin_start_m, bin_center_m, bin_end_m) from
        the input df are included in the output so the error CSV has spatial
        reference information.
        """
        gt_types = [t for t in df[data_type_col].unique()
                    if str(t).strip().lower() in GT_DATA_TYPES]

        if not gt_types:
            logger.warning("No GT data_type found for error calculation")
            return pd.DataFrame()

        est_types = [t for t in df[data_type_col].unique() if t not in gt_types]

        if not est_types:
            logger.warning("No estimation data_type found")
            return pd.DataFrame()

        # Build bin → coordinate mapping (one unique row per axis_bin)
        coord_cols_available = [c for c in ("bin_start_m", "bin_center_m", "bin_end_m")
                                if c in df.columns]
        coord_map = None
        if coord_cols_available:
            coord_map = (
                df[[bin_col] + coord_cols_available]
                .drop_duplicates(subset=[bin_col])
                .reset_index(drop=True)
            )

        records = []

        for gt_type in gt_types:
            df_gt = df[df[data_type_col] == gt_type]

            for est_type in est_types:
                df_est = df[df[data_type_col] == est_type]

                merged = pd.merge(
                    df_est[[bin_col] + pollutant_mass_cols],
                    df_gt[[bin_col] + pollutant_mass_cols],
                    on=bin_col,
                    how="inner",
                    suffixes=("", "_gt")
                )

                for pcol in pollutant_mass_cols:
                    gt_col = f"{pcol}_gt"
                    if gt_col not in merged.columns:
                        continue

                    for _, row in merged.iterrows():
                        val = row[pcol]
                        gt_val = row[gt_col]

                        if pd.notna(gt_val) and abs(gt_val) > 1e-9:
                            error_pct = 100.0 * (val - gt_val) / gt_val
                        else:
                            error_pct = np.nan

                        records.append({
                            "axis": axis,
                            bin_col: row[bin_col],
                            "data_type": est_type,
                            "gt_type": gt_type,
                            "pollutant": str(pcol).replace("_g", ""),
                            "value": val,
                            "gt_value": gt_val,
                            "error_pct": error_pct,
                        })

        if not records:
            return pd.DataFrame()

        df_error = pd.DataFrame(records)

        # Attach bin coordinate columns so error CSV has spatial reference
        if coord_map is not None:
            df_error = pd.merge(df_error, coord_map, on=bin_col, how="left")

        return df_error


# ============================================================================
# Data preprocessing
# ============================================================================

class DataPreprocessor:
    """Data preprocessing utilities"""

    @staticmethod
    def merge_sim_and_gt(df_main: pd.DataFrame, df_gt: pd.DataFrame, data_type_col: str) -> pd.DataFrame:
        """Merge simulation and GT data"""
        if df_gt.empty:
            return df_main.copy()
        return pd.concat([df_main, df_gt], ignore_index=True)


# ============================================================================
# Configuration parsing
# ============================================================================

def _parse_config(state: Any, options: Any) -> AnalysisConfig:
    """
    Parse configuration from GUI state and options.

    ROI polygon text is stored as-is for:
      (a) point-in-polygon data filtering in _apply_roi_filters
      (b) bin origin derivation via get_polygon_axis_bounds

    x_min_roi / y_min_roi are derived from the Simulation ROI polygon vertices.
    They are used ONLY as bin alignment origins inside compute_emission_spatial,
    NOT for any numeric range filtering of data.
    """
    # 1. import_gt from state.inputs
    import_gt = False
    if state and hasattr(state, "inputs") and hasattr(state.inputs, "import_gt"):
        import_gt = state.inputs.import_gt

    # 2. enabled_pollutants
    enabled_pollutants = []
    if options and hasattr(options, "pollutants"):
        enabled_pollutants = [k for k, v in options.pollutants.items() if v]
    elif options and hasattr(options, "enabled_pollutants"):
        enabled_pollutants = getattr(options, "enabled_pollutants", [])

    # 3. ROI polygon texts
    roi_sim_text = ""
    roi_gt_text = ""
    if options:
        roi_sim_text = getattr(options, "spatial_roi_sim", "") or ""
        roi_gt_text = getattr(options, "spatial_roi_gt", "") or ""

    # 4. Derive axis-aligned bounds from Simulation ROI polygon vertices.
    #    Purpose: provide bin origin for compute_emission_spatial so that
    #    bin_start_m aligns with the actual polygon edge, not coordinate 0.
    #    GT ROI has its own polygon for filtering but does NOT affect bin alignment.
    x_min_roi, x_max_roi = None, None
    y_min_roi, y_max_roi = None, None

    if roi_sim_text:
        x_min_roi, x_max_roi = ROIProcessor.get_polygon_axis_bounds(roi_sim_text, "x")
        y_min_roi, y_max_roi = ROIProcessor.get_polygon_axis_bounds(roi_sim_text, "y")
        logger.info(
            f"Bin alignment bounds from Sim ROI polygon — "
            f"x: [{x_min_roi}, {x_max_roi}], y: [{y_min_roi}, {y_max_roi}]"
        )
    else:
        logger.info("No Sim ROI polygon; bin origin will fall back to data minimum.")

    # 5. Build config
    if options:
        return AnalysisConfig(
            axis_bin_size=getattr(options, "axis_bin_size", 10.0),
            analysis_axis=getattr(options, "spatial_axis", "both"),
            import_gt=import_gt,
            enable_roi=getattr(options, "spatial_enable_roi", False),
            enabled_pollutants=enabled_pollutants,
            roi_sim_text=roi_sim_text,
            roi_gt_text=roi_gt_text,
            x_min_roi=x_min_roi,
            x_max_roi=x_max_roi,
            y_min_roi=y_min_roi,
            y_max_roi=y_max_roi,
            enable_sumo_time_filter=getattr(options, "enable_sumo_time_filter", False),
            sumo_time_start=getattr(options, "sumo_time_start", None),
            sumo_time_end=getattr(options, "sumo_time_end", None),
            enable_gt_time_filter=getattr(options, "enable_gt_time_filter", False),
            gt_time_start=getattr(options, "gt_time_start", None),
            gt_time_end=getattr(options, "gt_time_end", None),
        )

    return AnalysisConfig(
        axis_bin_size=10.0,
        analysis_axis="both",
        import_gt=import_gt,
        enable_roi=False,
        enabled_pollutants=enabled_pollutants,
        enable_sumo_time_filter=False,
        sumo_time_start=None,
        sumo_time_end=None,
        enable_gt_time_filter=False,
        gt_time_start=None,
        gt_time_end=None,
    )


# ============================================================================
# Data loading with GT splitting
# ============================================================================

def _load_data(
        in_path: Path,
        state: Any,
        upstream: Optional[Dict[str, Path]],
        config: AnalysisConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load simulation and GT data.

    Splits combined CSV into simulation and GT based on data_type column.
    """
    loader = DataLoader()

    sim_csvs = loader.find_csv_files(in_path)
    if not sim_csvs:
        raise FileNotFoundError(f"No CSV files found in {in_path}")

    logger.info(f"Found {len(sim_csvs)} upstream CSV files")
    df_all = loader.load_and_concat(sim_csvs, tag_if_missing="estimated")

    if df_all.empty:
        raise ValueError("No data loaded from upstream")

    detector = ColumnDetector()
    data_type_col = detector.detect_data_type_col(df_all)

    if not data_type_col:
        logger.info("No data_type column found, treating all data as simulation")
        df_all["data_type"] = "estimated"
        data_type_col = "data_type"

    if data_type_col != "data_type":
        df_all = df_all.rename(columns={data_type_col: "data_type"})

    dt_counts = df_all["data_type"].value_counts(dropna=False)
    logger.info(f"data_type distribution: {dt_counts.to_dict()}")

    df_main = pd.DataFrame()
    df_gt = pd.DataFrame()

    if config.import_gt:
        gt_types = [
            t for t in df_all["data_type"].unique()
            if str(t).strip().lower() in GT_DATA_TYPES
        ]

        if gt_types:
            logger.info(f"Found GT data types: {gt_types}")
            df_gt = df_all[df_all["data_type"].isin(gt_types)].copy()
            df_main = df_all[~df_all["data_type"].isin(gt_types)].copy()
            logger.info(f"Split data: {len(df_main)} simulation rows, {len(df_gt)} GT rows")
        else:
            logger.warning(
                f"import_gt=True but no GT data found. "
                f"Available data_types: {df_all['data_type'].unique().tolist()}"
            )
            df_main = df_all.copy()
    else:
        df_main = df_all.copy()
        logger.info(f"import_gt=False, using all {len(df_main)} rows as simulation data")

    return df_main, df_gt


def _apply_roi_filters(df: pd.DataFrame, enable_roi: bool, roi_text: str) -> pd.DataFrame:
    """
    Apply ROI polygon filter using point-in-polygon on (vehicle_x, vehicle_y).

    This is the ONLY spatial data filtering step.
    Numeric range checks are NOT used.
    """
    if not enable_roi or not roi_text or df.empty:
        return df
    return ROIProcessor.apply_filter(df, roi_text)


def _apply_time_filter(
        df: pd.DataFrame,
        enable_filter: bool,
        time_start: Optional[float],
        time_end: Optional[float],
        label: str = "data"
) -> pd.DataFrame:
    """
    Apply time range filter based on timestep_time column.

    Parameters
    ----------
    df           : DataFrame to filter
    enable_filter: whether time filtering is enabled
    time_start   : start time (seconds, inclusive); None = no lower bound
    time_end     : end time   (seconds, inclusive); None = no upper bound
    label        : tag for log messages
    """
    if not enable_filter or df.empty:
        return df

    if "timestep_time" not in df.columns:
        logger.warning(f"[{label}] 'timestep_time' column not found, time filter skipped")
        return df

    n_before = len(df)
    mask = pd.Series([True] * n_before, index=df.index)

    if time_start is not None:
        mask &= df["timestep_time"] >= time_start
    if time_end is not None:
        mask &= df["timestep_time"] <= time_end

    df_filtered = df.loc[mask].copy()
    logger.info(
        f"[{label}] Time filter [{time_start}, {time_end}]s: "
        f"{n_before} → {len(df_filtered)} rows"
    )
    return df_filtered


# ============================================================================
# Robust pollutant column selection
# ============================================================================

def _select_pollutant_columns(
        df: pd.DataFrame,
        enabled_pollutants: List[str]
) -> List[str]:
    """
    Select pollutant rate columns with robust matching.

    Features:
    - Case-insensitive matching
    - Alias support (PM2.5/PM25, NOx/NOX, etc.)
    - Multiple suffix support (_gs, _g, _grams, _rate)
    """
    df = _strip_columns(df)

    SUFFIXES = ["_gs", "_g", "_grams", "_rate"]
    cols = list(df.columns)
    cols_lower_map = {str(c).lower(): str(c) for c in cols}

    available_candidates = []
    for sfx in SUFFIXES:
        available_candidates.extend([c for c in cols if str(c).lower().endswith(sfx)])

    selected = []
    missing = []

    if enabled_pollutants:
        logger.info(f"Searching for enabled pollutants: {enabled_pollutants}")

        for pollutant in enabled_pollutants:
            found = None

            for base in _pollutant_aliases(pollutant):
                for suffix in SUFFIXES:
                    expected = f"{base}{suffix}"
                    expected_lower = expected.lower()
                    if expected_lower in cols_lower_map:
                        found = cols_lower_map[expected_lower]
                        selected.append(found)
                        logger.info(f"  ✓ Found: {pollutant} → {found}")
                        break
                if found:
                    break

            if not found:
                missing.append(pollutant)
                logger.warning(
                    f"  ✗ Not found: {pollutant} "
                    f"(tried aliases: {_pollutant_aliases(pollutant)} with suffixes: {SUFFIXES})"
                )
    else:
        selected = [c for c in cols if str(c).lower().endswith("_gs")]
        logger.info(f"No specific pollutants enabled; using all *_gs columns: {selected}")

    if missing:
        logger.error(f"Missing pollutant columns: {missing}")
        logger.error(f"Available candidate columns: {sorted(set(available_candidates))}")
        raise ValueError(
            f"Missing pollutant columns for selections: {missing}. "
            f"Available candidate columns: {sorted(set(available_candidates))}"
        )

    if not selected:
        logger.error(f"No pollutant columns found")
        logger.error(f"Available candidate columns: {sorted(set(available_candidates))}")
        logger.error(f"Total columns in DataFrame: {len(cols)}")
        logger.error(f"First 30 columns: {cols[:30]}")
        raise ValueError(
            f"No pollutant columns found. "
            f"Available candidate columns: {sorted(set(available_candidates))}. "
            f"Total DataFrame columns: {len(cols)}"
        )

    return selected


def _prepare_analysis_data(
        df_main: pd.DataFrame,
        df_gt: pd.DataFrame,
        config: AnalysisConfig
) -> Tuple[pd.DataFrame, List[str]]:
    """Merge simulation and GT data, then detect pollutant columns."""
    preprocessor = DataPreprocessor()
    df_all = preprocessor.merge_sim_and_gt(df_main, df_gt, "data_type")

    for col in REQUIRED_COLUMNS:
        if col not in df_all.columns:
            raise ValueError(f"Missing required column: {col}")

    pollutant_cols = _select_pollutant_columns(df_all, config.enabled_pollutants)
    logger.info(f"Using {len(pollutant_cols)} pollutant column(s): {pollutant_cols}")

    return df_all, pollutant_cols


# ============================================================================
# Computation and save
# ============================================================================

def _compute_and_save(
        df_all: pd.DataFrame,
        pollutant_cols: List[str],
        config: AnalysisConfig,
        out_dir: Path
) -> None:
    """
    Compute and save spatial emission distributions.

    For each axis:
    1. Get roi_origin from config (x_min_roi or y_min_roi, derived from Sim polygon)
    2. Call compute_emission_spatial — input data is already ROI-filtered
    3. Save spatial_emission_{axis}.csv
    4. If GT present, compute error % and save spatial_error_pct_{axis}.csv
    """
    calculator = SpatialEmissionCalculator()
    error_calc = GTErrorCalculator()

    if config.analysis_axis == "both":
        axes = ["x", "y"]
    elif config.analysis_axis in ("x", "y"):
        axes = [config.analysis_axis]
    else:
        raise ValueError(f"Invalid analysis_axis: {config.analysis_axis}")

    for axis in axes:
        logger.info(f"Computing for axis '{axis}'...")

        # roi_origin: polygon min value along this axis → bin zero-point alignment
        roi_origin = config.x_min_roi if axis == "x" else config.y_min_roi
        logger.info(f"  roi_origin for axis '{axis}': {roi_origin}")

        df_result = calculator.compute_emission_spatial(
            df_all,
            axis=axis,
            axis_bin_size=config.axis_bin_size,
            pollutant_rate_cols=pollutant_cols,
            data_type_col="data_type",
            roi_origin=roi_origin,
        )

        logger.info(f"  Result columns: {df_result.columns.tolist()}")
        logger.info(f"  Result rows: {len(df_result)}")

        out_csv = out_dir / f"spatial_emission_{axis}.csv"
        df_result.to_csv(out_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Saved: {out_csv.name}")

        if config.import_gt:
            mass_cols = [c for c in df_result.columns if c.endswith("_g")]

            if mass_cols:
                df_error = error_calc.compute_error_pct(
                    df_result,
                    axis=axis,
                    data_type_col="data_type",
                    bin_col="axis_bin",
                    pollutant_mass_cols=mass_cols,
                )

                if not df_error.empty:
                    err_csv = out_dir / f"spatial_error_pct_{axis}.csv"
                    df_error.to_csv(err_csv, index=False, encoding="utf-8-sig")
                    logger.info(f"Saved: {err_csv.name} ({len(df_error)} rows)")
                else:
                    logger.warning("GT error calculation returned empty (no GT or no overlap)")


# ============================================================================
# Main execution
# ============================================================================

def run_spatial_map(
        *,
        in_path: Path,
        out_dir: Path,
        state: Any = None,
        upstream: Optional[Dict[str, Path]] = None,
        options: Any = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Emission Spatial Map (GUI version)")
    logger.info("=" * 60)

    # 1. Parse configuration
    logger.info("[1/6] Parsing configuration...")
    config = _parse_config(state, options)

    logger.info(f"  ├─ axis_bin_size           : {config.axis_bin_size}")
    logger.info(f"  ├─ analysis_axis           : {config.analysis_axis}")
    logger.info(f"  ├─ import_gt               : {config.import_gt}")
    logger.info(f"  ├─ enable_roi              : {config.enable_roi}")
    logger.info(f"  ├─ roi_sim_text            : {config.roi_sim_text}")
    logger.info(f"  ├─ roi_gt_text             : {config.roi_gt_text}")
    logger.info(f"  ├─ x bin origin (x_min_roi): {config.x_min_roi}")
    logger.info(f"  ├─ y bin origin (y_min_roi): {config.y_min_roi}")
    logger.info(f"  └─ enabled_pollutants      : {config.enabled_pollutants}")

    # 2. Load data
    logger.info("[2/6] Loading data...")
    df_main, df_gt = _load_data(in_path, state, upstream, config)

    # 3. Apply ROI polygon filters (point-in-polygon spatial filtering)
    logger.info("[3/6] Applying ROI polygon filters...")
    df_main = _apply_roi_filters(df_main, config.enable_roi, config.roi_sim_text)
    if config.import_gt:
        df_gt = _apply_roi_filters(df_gt, config.enable_roi, config.roi_gt_text)

    # 3.5. Apply time range filters
    logger.info("[3.5/6] Applying time range filters...")
    df_main = _apply_time_filter(
        df_main, config.enable_sumo_time_filter,
        config.sumo_time_start, config.sumo_time_end,
        label="simulation"
    )
    if config.import_gt:
        df_gt = _apply_time_filter(
            df_gt, config.enable_gt_time_filter,
            config.gt_time_start, config.gt_time_end,
            label="GT"
        )

    # 4. Prepare data for analysis
    logger.info("[4/6] Preparing analysis data...")
    df_all, pollutant_cols = _prepare_analysis_data(df_main, df_gt, config)

    # 5. Compute spatial distributions
    logger.info("[5/6] Computing spatial distributions...")
    _compute_and_save(df_all, pollutant_cols, config, out_dir)

    # 6. Compute emission statistics (totals and intensity)
    logger.info("[6/6] Computing emission statistics...")
    compute_and_save_statistics(df_all, pollutant_cols, out_dir, data_type_col="data_type")

    logger.info("Emission Spatial Map complete.")
    return out_dir


# ============================================================================
# GUI entry point
# ============================================================================

def run_gui(*, in_path, out_dir, state=None, upstream=None, options=None):
    """GUI adapter entry point"""
    run_spatial_map(
        in_path=Path(in_path),
        out_dir=Path(out_dir),
        state=state,
        upstream=upstream,
        options=options,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        run_spatial_map(in_path=Path(sys.argv[1]), out_dir=Path(sys.argv[2]))
    else:
        print("Usage: python emission_spatial_map.py <in_path> <out_dir>")