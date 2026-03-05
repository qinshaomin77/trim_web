# app/bindings.py
# -*- coding: utf-8 -*-
"""
Step bindings for TRIM GUI - UPDATED VERSION

修改记录:
1. 将 CANDIDATES 列表中的 "run_gui" 提前到最前面
2. 这样适配器会优先尝试调用 run_gui() 函数而不是 main()

Provides:
    build_entrypoints() -> dict[step_id, callable]

Each callable signature:
    run_fn(*, state, out_root, step_dir, upstream) -> Path
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from app.adapters import make_function_step
from app.adapters import make_class_step, import_module

# -----------------------------
# Input selection rules
# -----------------------------

def _select_input(step_id: str, state: Any, upstream: Dict[str, Path]) -> Any:
    """
    Decide the input path for each step.
    This is intentionally simple: prefer upstream output of dependency step.
    If a step consumes GUI inputs, read from state.inputs.
    """
    inp = getattr(state, "inputs", None)
    if inp is None:
        raise RuntimeError("State has no 'inputs' attribute. Please check AppState definition.")

    # GUI direct inputs
    if step_id == "net_topology":
        return inp.net_xml
    if step_id == "xml2csv_fcd":
        return inp.fcd_input
    if step_id == "gt_standardize":
        return inp.gt_csv

    # Upstream chain (mainline)
    chain_map = {
        "identify_neighbors": "xml2csv_fcd",
        "trip_split": "identify_neighbors",
        "trim_postprocess": "trip_split",
        "fill_xy": "trim_postprocess",
        "ef_match": "fill_xy",
        "emission_spatial_map": "ef_match",
        "emission_compare": "emission_spatial_map",
        # optional
        "sg_smooth": "trip_split",
    }

    dep = chain_map.get(step_id)
    if dep:
        if dep not in upstream:
            raise RuntimeError(f"[{step_id}] Upstream output not found for dependency '{dep}'.")
        return upstream[dep]

    # default fallback: no input
    return None


def _select_output(step_id: str, step_dir: Path) -> Path:
    """
    Decide the output path returned by runner.
    We return step_dir for 'Open' button to open the directory.
    If your script returns a CSV path, adapters will return it and override this.
    """
    return Path(step_dir)


# Common candidate names for function entrypoints inside scripts/*.py
CANDIDATES = ["run_gui", "run", "main", "execute", "process", "cli_main", "extract"]

# -----------------------------
# Build entrypoints
# -----------------------------

def build_entrypoints() -> Dict[str, Any]:
    """
    Map step_id -> runner callable.
    The module_name must match scripts/<module_name>.py (without .py).
    """
    mod = import_module("scripts.net_topology")
    NetTopologyConfig = getattr(mod, "NetTopologyConfig")
    NetTopologyExtractor = getattr(mod, "NetTopologyExtractor")

    def cfg_builder(state, step_dir):
        return NetTopologyConfig(
            net_xml=state.inputs.net_xml,
            out_dir=Path(step_dir),
            rou_xml=None,
            include_internal=True,
            max_steps=100_000,
            debug=False,
            export_paths=True,
            export_lane_tables=True,
        )

    eps = {
        "net_topology": make_class_step(
            step_id="net_topology",
            cfg_cls=NetTopologyConfig,
            extractor_cls=NetTopologyExtractor,
            cfg_builder=cfg_builder,
            return_dir=True,
        ),
        "xml2csv_fcd": make_function_step(
            step_id="xml2csv_fcd",
            module_name="xml2csv_fcd",
            candidates=CANDIDATES,
            input_selector=_select_input,
            output_selector=_select_output,
        ),
        "identify_neighbors": make_function_step(
            step_id="identify_neighbors",
            module_name="identify_neighbors",
            candidates=CANDIDATES,
            input_selector=_select_input,
            output_selector=_select_output,
        ),
        "trip_split": make_function_step(
            step_id="trip_split",
            module_name="trip_split",
            candidates=CANDIDATES,
            input_selector=_select_input,
            output_selector=_select_output,
        ),
        # TRIM postprocess - 现在会优先尝试 run_gui()
        "trim_postprocess": make_function_step(
            step_id="trim_postprocess",
            module_name="trim_postprocess",
            candidates=CANDIDATES,  # run_gui 在最前面
            input_selector=_select_input,
            output_selector=_select_output,
        ),
        "fill_xy": make_function_step(
            step_id="fill_xy",
            module_name="fill_xy",
            candidates=CANDIDATES,
            input_selector=_select_input,
            output_selector=_select_output,
        ),
        "ef_match": make_function_step(
            step_id="ef_match",
            module_name="ef_match",
            candidates=CANDIDATES,
            input_selector=_select_input,
            output_selector=_select_output,
        ),
        "emission_spatial_map": make_function_step(
            step_id="emission_spatial_map",
            module_name="emission_spatial_map",
            candidates=CANDIDATES,
            input_selector=_select_input,
            output_selector=_select_output,
        ),
        "emission_compare": make_function_step(
            step_id="emission_compare",
            module_name="emission_compare",
            candidates=CANDIDATES,
            input_selector=_select_input,
            output_selector=_select_output,
        ),
        # optional
        "sg_smooth": make_function_step(
            step_id="sg_smooth",
            module_name="sg_smooth",
            candidates=CANDIDATES,
            input_selector=_select_input,
            output_selector=_select_output,
        ),
        "gt_standardize": make_function_step(
            step_id="gt_standardize",
            module_name="gt_standardize",
            candidates=CANDIDATES,
            input_selector=_select_input,
            output_selector=_select_output,
        ),
    }
    return eps