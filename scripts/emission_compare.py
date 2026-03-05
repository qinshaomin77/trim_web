# -*- coding: utf-8 -*-
"""
emission_compare.py - Emission visualization and comparison

This script reads output CSVs from emission_spatial_map.py and creates:
1. Line plots: emission vs axis_bin, grouped by data_type, one figure per pollutant.
2. Error-percentage bar charts: error_pct vs axis_bin, grouped bar by data_type,
   one figure per pollutant. Only created when GT data is available.

Upstream CSV formats (from emission_spatial_map.py):
  - spatial_emission_{x|y}.csv
      columns: axis, axis_bin, data_type, {Pollutant}_g, ...
  - spatial_error_pct_{x|y}.csv  (only when GT is present)
      columns: axis, axis_bin, data_type, gt_type, pollutant, value, gt_value, error_pct

The axis in the filename (x or y) depends on the user's GUI setting
(analysis_axis = x | y | both). When 'both' is selected, two sets of files
exist and both are processed.

Usage (CLI):
    python emission_compare.py --spatial_dir <dir> --out_dir <dir> \
        --pollutants NOx,CO2 --prefix fcd --dpi 300

Usage (GUI):
    Called via run_gui() adapter from app/bindings.py
"""

from __future__ import annotations

import os
import sys
import re
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

LOGGER_NAME = "trim.emission_compare"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def _ensure_cli_logging() -> None:
    """Set up basic logging when running from the command line."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


# ---------------------------------------------------------------------------
# String / filename helpers
# ---------------------------------------------------------------------------

def sanitize_filename(s: str) -> str:
    """Remove characters that are unsafe in file paths."""
    return re.sub(r'[\\/*?:"<>|]', "_", str(s))


def pollutant_to_col(pollutant: str, available_columns: Optional[List[str]] = None) -> str:
    """
    Convert a pollutant display name to the column-name prefix used in
    spatial_emission CSVs.  E.g. 'NOx' -> 'NOx', 'PM2.5' -> 'PM2.5' or 'PM25'.
    The actual column in the CSV is '{result}_g', e.g. 'NOx_g' or 'PM2.5_g'.

    Enhanced to handle multiple naming conventions by checking available columns.

    Parameters
    ----------
    pollutant : str
        Pollutant display name (e.g., 'NOx', 'PM2.5', 'CO2')
    available_columns : list of str, optional
        List of available column names in the CSV (to check which format exists)

    Returns
    -------
    str
        Column name prefix (without '_g' suffix)
    """
    p = str(pollutant).strip().strip('"').strip("'")

    # NOx special case
    if p.upper() == "NOX":
        return "NOx"

    # PM2.5 handling - try multiple formats
    if p.upper() in {"PM2.5", "PM2_5", "PM2-5", "PM2 5", "PM2,5", "PM25"}:
        # If we have available columns, check which format exists
        if available_columns:
            # Try common variants in order of preference
            candidates = ["PM2.5", "PM25", "PM2_5", "PM2-5"]
            for candidate in candidates:
                col_with_suffix = f"{candidate}_g"
                if col_with_suffix in available_columns:
                    return candidate
        # Fallback: return the original format with dot
        return "PM2.5"

    # For other pollutants, preserve dots if available_columns provided
    # This handles cases like CO2, THC, etc.
    if available_columns:
        # Try exact match first (e.g., "CO2_g")
        if f"{p}_g" in available_columns:
            return p
        # Try without dots (e.g., "CO2" -> "CO2")
        p_no_dot = p.replace(".", "")
        if f"{p_no_dot}_g" in available_columns:
            return p_no_dot

    # Default: remove dots and special characters
    p2 = p.replace(".", "")
    p2 = re.sub(r"[^A-Za-z0-9_]+", "", p2)
    return p2 if p2 else "POLLUTANT"


def _extract_axis_from_filename(csv_path: Path) -> str:
    """
    Extract the axis label ('x' or 'y') from filenames like
    spatial_emission_x.csv or spatial_error_pct_y.csv.
    Falls back to 'x' if unable to determine.
    """
    stem = csv_path.stem.lower()
    if stem.endswith("_y"):
        return "y"
    if stem.endswith("_x"):
        return "x"
    # Try to find x/y anywhere as last segment
    parts = stem.split("_")
    if parts and parts[-1] in ("x", "y"):
        return parts[-1]
    return "x"


# ---------------------------------------------------------------------------
# Plotting: emission line chart
# ---------------------------------------------------------------------------

def plot_emission_lines(
        df: pd.DataFrame,
        pollutants: List[str],
        axis_name: str,
        out_dir: Path,
        prefix: str = "fcd",
        dpi: int = 300,
) -> List[Path]:
    """
    Create line plots from a spatial_emission_{axis}.csv dataframe.

    For each pollutant a separate figure is saved.  Inside each figure all
    data_type groups are plotted on the same axes so they can be compared
    visually.

    Parameters
    ----------
    df : DataFrame with columns [axis, axis_bin, data_type, {Pollutant}_g, ...]
    pollutants : list of pollutant display names selected by the user
    axis_name : 'x' or 'y'
    out_dir : directory for saved PNG files
    prefix : filename prefix
    dpi : plot resolution

    Returns
    -------
    List of saved file paths.
    """
    logger = get_logger()
    saved: List[Path] = []

    if "axis_bin" not in df.columns or "data_type" not in df.columns:
        logger.error(
            f"Emission CSV missing required columns. "
            f"Expected 'axis_bin' and 'data_type', got: {list(df.columns)}"
        )
        return saved

    # Get available columns for smart matching
    available_columns = list(df.columns)

    for pollutant in pollutants:
        # Use enhanced column name detection
        col_prefix = pollutant_to_col(pollutant, available_columns)
        col = f"{col_prefix}_g"

        if col not in df.columns:
            logger.warning(
                f"Column '{col}' not found in emission CSV for pollutant '{pollutant}'. "
                f"Available pollutant columns: {[c for c in df.columns if c.endswith('_g')]}"
            )
            continue

        logger.info(f"Plotting {pollutant} using column '{col}'")

        fig, ax = plt.subplots(figsize=(12, 6))

        for dtype, grp in df.groupby("data_type", sort=True):
            grp_sorted = grp.sort_values("axis_bin")
            ax.plot(
                grp_sorted["axis_bin"],
                grp_sorted[col],
                label=str(dtype),
                marker="o",
                markersize=3,
                linewidth=1.5,
            )

        ax.set_xlabel(f"{axis_name.upper()}-axis bin (m)", fontsize=11)
        ax.set_ylabel(f"{pollutant} emission (g)", fontsize=11)
        ax.set_title(
            f"Spatial Emission - {pollutant} ({axis_name.upper()}-axis)",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(title="data_type", fontsize=9, title_fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = out_dir / f"{prefix}_line_{sanitize_filename(pollutant)}_{axis_name}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        saved.append(fname)
        logger.info(f"Saved emission line plot: {fname}")

    return saved


# ---------------------------------------------------------------------------
# Plotting: error-percentage grouped bar chart
# ---------------------------------------------------------------------------

def plot_error_pct_bars(
        df_err: pd.DataFrame,
        axis_name: str,
        out_dir: Path,
        prefix: str = "fcd",
        dpi: int = 300,
) -> List[Path]:
    """
    Create grouped bar charts from a spatial_error_pct_{axis}.csv dataframe.

    Each pollutant (column 'pollutant') gets its own figure.  Within each
    figure the x-axis is axis_bin and bars are grouped by data_type so that
    each bin shows one bar per data_type.

    Parameters
    ----------
    df_err : DataFrame with columns
             [axis, axis_bin, data_type, gt_type, pollutant, value, gt_value, error_pct]
    axis_name : 'x' or 'y'
    out_dir : directory for saved PNG files
    prefix : filename prefix
    dpi : plot resolution

    Returns
    -------
    List of saved file paths.
    """
    logger = get_logger()
    saved: List[Path] = []

    required = {"axis_bin", "data_type", "pollutant", "error_pct"}
    missing = required - set(df_err.columns)
    if missing:
        logger.error(
            f"Error-pct CSV missing required columns: {sorted(missing)}. "
            f"Available: {list(df_err.columns)}"
        )
        return saved

    for pollutant, grp in df_err.groupby("pollutant", sort=True):
        # Pivot so that index=axis_bin, columns=data_type, values=error_pct
        pivot = grp.pivot_table(
            index="axis_bin",
            columns="data_type",
            values="error_pct",
            aggfunc="first",
        )
        pivot = pivot.sort_index()

        if pivot.empty:
            logger.warning(f"No error data for pollutant '{pollutant}' on {axis_name}-axis")
            continue

        n_bins = len(pivot.index)
        n_types = max(len(pivot.columns), 1)

        fig, ax = plt.subplots(figsize=(max(10, n_bins * 0.6), 6))

        x = np.arange(n_bins)
        total_bar_width = 0.8
        single_width = total_bar_width / n_types

        for i, dtype in enumerate(pivot.columns):
            offset = (i - n_types / 2 + 0.5) * single_width
            bars = ax.bar(
                x + offset,
                pivot[dtype].values,
                single_width,
                label=str(dtype),
            )
            # Add value labels on top of bars when there are few bins
            if n_bins <= 20:
                for bar, val in zip(bars, pivot[dtype].values):
                    if pd.notna(val):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            f"{val:.1f}%",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            rotation=45 if n_bins > 10 else 0,
                        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{v:.0f}" for v in pivot.index],
            rotation=45 if n_bins > 8 else 0,
            fontsize=9,
        )
        ax.set_xlabel(f"{axis_name.upper()}-axis bin (m)", fontsize=11)
        ax.set_ylabel("Error (%)", fontsize=11)
        ax.set_title(
            f"Error Percentage - {pollutant} ({axis_name.upper()}-axis)",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(title="data_type", fontsize=9, title_fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        plt.tight_layout()

        safe_poll = sanitize_filename(str(pollutant))
        fname = out_dir / f"{prefix}_error_pct_{safe_poll}_{axis_name}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        saved.append(fname)
        logger.info(f"Saved error-pct bar chart: {fname}")

    return saved


# ---------------------------------------------------------------------------
# Core processing logic (used by both CLI and GUI)
# ---------------------------------------------------------------------------

def run_compare(
        spatial_dir: Path,
        out_dir: Path,
        pollutants: List[str],
        prefix: str = "fcd",
        dpi: int = 300,
) -> Dict[str, Any]:
    """
    Main comparison logic.

    Scans *spatial_dir* for upstream CSVs and creates all applicable plots.

    Parameters
    ----------
    spatial_dir : directory produced by emission_spatial_map step
    out_dir : output directory for plots and summary CSV
    pollutants : list of pollutant display names
    prefix : filename prefix
    dpi : plot resolution

    Returns
    -------
    Summary dict with counts and file lists.
    """
    logger = get_logger()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_line_plots: List[Path] = []
    all_bar_plots: List[Path] = []

    # ------------------------------------------------------------------
    # 1) Emission line plots from spatial_emission_{x|y}.csv
    # ------------------------------------------------------------------
    emission_csvs = sorted(spatial_dir.glob("spatial_emission_*.csv"))
    if not emission_csvs:
        logger.warning(f"No spatial_emission_*.csv files found in {spatial_dir}")
    else:
        logger.info(f"Found {len(emission_csvs)} emission CSV(s): "
                    f"{[c.name for c in emission_csvs]}")

    for csv_path in emission_csvs:
        axis_name = _extract_axis_from_filename(csv_path)
        logger.info(f"Processing emission line plots: {csv_path.name} (axis={axis_name})")

        try:
            df = pd.read_csv(csv_path)
            logger.info(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
            plots = plot_emission_lines(df, pollutants, axis_name, out_dir, prefix, dpi)
            all_line_plots.extend(plots)
        except Exception as e:
            logger.error(f"  Failed to process {csv_path.name}: {e}")

    # ------------------------------------------------------------------
    # 2) Error-percentage bar charts from spatial_error_pct_{x|y}.csv
    # ------------------------------------------------------------------
    error_csvs = sorted(spatial_dir.glob("spatial_error_pct_*.csv"))
    if not error_csvs:
        logger.info("No spatial_error_pct_*.csv files found (GT data not available)")
    else:
        logger.info(f"Found {len(error_csvs)} error-pct CSV(s): "
                    f"{[c.name for c in error_csvs]}")

    for csv_path in error_csvs:
        axis_name = _extract_axis_from_filename(csv_path)
        logger.info(f"Processing error-pct bar charts: {csv_path.name} (axis={axis_name})")

        try:
            df_err = pd.read_csv(csv_path)
            logger.info(f"  Loaded {len(df_err)} rows, columns: {list(df_err.columns)}")
            plots = plot_error_pct_bars(df_err, axis_name, out_dir, prefix, dpi)
            all_bar_plots.extend(plots)
        except Exception as e:
            logger.error(f"  Failed to process {csv_path.name}: {e}")

    # ------------------------------------------------------------------
    # 3) Summary
    # ------------------------------------------------------------------
    total_plots = len(all_line_plots) + len(all_bar_plots)
    summary = {
        "analysis_type": "emission_compare",
        "spatial_dir": str(spatial_dir),
        "output_dir": str(out_dir),
        "pollutants_requested": pollutants,
        "emission_csvs_processed": len(emission_csvs),
        "error_pct_csvs_processed": len(error_csvs),
        "line_plots_created": len(all_line_plots),
        "bar_plots_created": len(all_bar_plots),
        "total_plots_created": total_plots,
    }

    summary_csv = out_dir / f"{prefix}_emission_compare_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Summary saved: {summary_csv}")
    logger.info(f"Total plots created: {total_plots}")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize and compare spatial emission data"
    )
    parser.add_argument(
        "--spatial_dir",
        required=True,
        help="Directory containing spatial_emission_*.csv and spatial_error_pct_*.csv",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    parser.add_argument(
        "--pollutants",
        default="NOx",
        help="Comma-separated pollutant names (e.g. NOx,PM2.5,CO2)",
    )
    parser.add_argument("--prefix", default="fcd", help="Filename prefix")
    parser.add_argument("--dpi", type=int, default=300, help="Plot DPI")

    args = parser.parse_args()
    _ensure_cli_logging()
    logger = get_logger()

    spatial_dir = Path(args.spatial_dir)
    out_dir = Path(args.out_dir)
    pollutants = [p.strip() for p in args.pollutants.split(",") if p.strip()]

    if not spatial_dir.is_dir():
        raise FileNotFoundError(f"Spatial directory not found: {spatial_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== EMISSION COMPARE (CLI) ===")
    logger.info(f"Spatial dir : {spatial_dir}")
    logger.info(f"Output dir  : {out_dir}")
    logger.info(f"Pollutants  : {pollutants}")
    logger.info(f"Prefix      : {args.prefix}")
    logger.info(f"DPI         : {args.dpi}")

    run_compare(
        spatial_dir=spatial_dir,
        out_dir=out_dir,
        pollutants=pollutants,
        prefix=args.prefix,
        dpi=args.dpi,
    )

    logger.info("=== EMISSION COMPARE COMPLETED ===")


# ---------------------------------------------------------------------------
# GUI adapter entry point
# ---------------------------------------------------------------------------

def run_gui(*, in_path, out_dir, state=None, upstream=None, options=None):
    """
    GUI-friendly entry called by the adapter in app/bindings.py.

    Reads upstream directory from emission_spatial_map step, determines
    which pollutants are selected, and delegates to run_compare().
    """
    logger = get_logger()

    try:
        logger.info("[emission_compare] Starting")

        # --- Validate upstream ---
        if not upstream:
            raise ValueError("Upstream data is required for emission_compare")

        if "emission_spatial_map" not in upstream:
            raise ValueError(
                "emission_spatial_map output not found in upstream data. "
                "Make sure that step ran successfully before this one."
            )

        spatial_dir = Path(upstream["emission_spatial_map"])
        if not spatial_dir.is_dir():
            raise FileNotFoundError(
                f"Upstream spatial_map path is not a directory: {spatial_dir}"
            )

        # --- Output directory ---
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Pollutants from GUI state ---
        pollutants = ["NOx"]  # default
        if state and hasattr(state, "options") and hasattr(state.options, "pollutants"):
            selected = [k for k, v in state.options.pollutants.items() if v]
            if selected:
                pollutants = selected

        # --- Prefix ---
        prefix = "fcd"
        if state and hasattr(state, "inputs") and hasattr(state.inputs, "safe_prefix"):
            prefix = state.inputs.safe_prefix or "fcd"

        # --- DPI (use default; could be added to Options later) ---
        dpi = 300

        logger.info(f"[emission_compare] spatial_dir : {spatial_dir}")
        logger.info(f"[emission_compare] out_dir     : {out_dir}")
        logger.info(f"[emission_compare] pollutants  : {pollutants}")
        logger.info(f"[emission_compare] prefix      : {prefix}")

        # --- Run ---
        summary = run_compare(
            spatial_dir=spatial_dir,
            out_dir=out_dir,
            pollutants=pollutants,
            prefix=prefix,
            dpi=dpi,
        )

        logger.info(
            f"[emission_compare] Completed: "
            f"{summary.get('total_plots_created', 0)} plots created"
        )
        return str(out_dir)

    except Exception as e:
        logger.error(f"[emission_compare] Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()