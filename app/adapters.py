# app/adapters.py
# -*- coding: utf-8 -*-
"""
Adapters for TRIM GUI step bindings.

Goal:
- Keep trim_gui.py small.
- Support heterogeneous script styles:
  1) function-based entrypoints (run/main/...)
  2) class-based API (Config + Extractor(cfg).run())

All adapters return a callable with signature:
    run_fn(*, state, out_root, step_dir, upstream) -> Path

Notes:
- Do NOT configure logging here. Let GUI configure logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Any, Dict, List


def ensure_path(x) -> Optional[Path]:
    if x is None:
        return None
    return x if isinstance(x, Path) else Path(x)


def import_module(mod_fullname: str):
    """Import module by fullname, raise clear error."""
    try:
        return __import__(mod_fullname, fromlist=["*"])
    except Exception as e:
        raise RuntimeError(
            f"Failed to import module '{mod_fullname}'.\n"
            f"Check sys.path includes project root, and package __init__.py exists.\n\n{e}"
        ) from e


def pick_callable(mod, candidates: List[str]) -> tuple[Optional[Callable[..., Any]], str]:
    """Pick first callable attribute in module by candidate names."""
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn, name
    return None, ""


# -------------------------------
# Adapter: function-style entry
# -------------------------------

def make_function_step(
    *,
    step_id: str,
    module_name: str,
    candidates: List[str],
    input_selector: Callable[[str, Any, Dict[str, Path]], Any],
    output_selector: Callable[[str, Path], Path],
) -> Callable[..., Path]:
    """
    Create a step runner for function-based scripts.

    Parameters
    ----------
    step_id : step id string
    module_name : module under scripts/, without "scripts." prefix
    candidates : list of possible function names inside module
    input_selector : function returning 'in_path' for the step
    output_selector : function returning output path/dir

    Calling strategy (flexible):
    - Try fn(in_path=..., out_dir=..., state=..., upstream=..., options=...)
    - Try fn(input_csv=..., out_dir=...)
    - Try fn(in_csv=..., out_dir=...)
    - Try fn(in_path, out_dir)
    """
    mod = import_module(f"scripts.{module_name}")
    fn, picked = pick_callable(mod, candidates)
    if fn is None:
        raise RuntimeError(
            f"[{step_id}] No callable entry function found in scripts.{module_name}.\n"
            f"Tried: {candidates}\n"
            f"Please expose one of these names OR provide a custom binding in app/bindings.py."
        )

    def _runner(*, state, out_root, step_dir, upstream) -> Path:
        in_path = input_selector(step_id, state, upstream)
        out_path = output_selector(step_id, Path(step_dir))

        # normalize
        in_path_p = ensure_path(in_path)
        out_dir = Path(step_dir)

        # options object (if present)
        opts = getattr(state, "options", None)

        # Try a few common signatures
        try:
            r = fn(in_path=in_path_p, out_dir=out_dir, state=state, upstream=upstream, options=opts)
            return ensure_path(r) or out_path
        except TypeError:
            pass

        try:
            r = fn(input_csv=in_path_p, out_dir=out_dir)
            return ensure_path(r) or out_path
        except TypeError:
            pass

        try:
            r = fn(in_csv=in_path_p, out_dir=out_dir)
            return ensure_path(r) or out_path
        except TypeError:
            pass

        try:
            r = fn(in_path_p, out_dir)
            return ensure_path(r) or out_path
        except TypeError:
            pass

        # If still failed, raise a clear error
        raise RuntimeError(
            f"[{step_id}] Failed to call function '{picked}' in scripts.{module_name}.\n"
            f"Please adjust binding in app/bindings.py to match the true signature.\n"
            f"Debug: in_path={in_path_p}, out_dir={out_dir}"
        )

    return _runner


# -------------------------------
# Adapter: class-style entry
# -------------------------------

def make_class_step(
    *,
    step_id: str,
    cfg_cls,
    extractor_cls,
    cfg_builder: Callable[[Any, Path], Any],
    return_dir: bool = True
) -> Callable[..., Path]:
    """
    Create a step runner for class-based scripts:
        cfg = cfg_builder(state, step_dir)
        extractor = extractor_cls(cfg)
        extractor.run()

    Parameters
    ----------
    cfg_cls / extractor_cls : class objects
    cfg_builder : function to construct cfg instance from (state, step_dir)
    return_dir : return step_dir as output (recommended for directory output)
    """

    def _runner(*, state, out_root, step_dir, upstream) -> Path:
        step_dir = Path(step_dir)
        cfg = cfg_builder(state, step_dir)
        extractor = extractor_cls(cfg)
        # run may return dict/df; we only care that exports happen
        extractor.run()
        return step_dir if return_dir else Path(step_dir)

    return _runner
