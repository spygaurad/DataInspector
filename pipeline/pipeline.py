import inspect
from typing import Callable, Optional, Any, Dict, Tuple, Sequence, List

import inspect
from typing import Optional, Callable, Tuple, Dict, Any
import pandas as pd

def make_step(func: Callable, *, name: Optional[str] = None, use_original_df: bool = False):
    """
    Wrap `func` into a pipeline step builder.
    Assumes: func(...) -> (output_df, stats: dict)

    If the function has a `df` parameter, the step will:
      - by default use the previous step's output as df
      - if `use_original_df=True`, use the original df passed to `run_pipeline`
      - if you bind `df=` at build time, that takes precedence over both
    """
    sig = inspect.signature(func)
    step_name = name or func.__name__

    def builder(**params):
        sig.bind_partial(**params)  # validate early

        def _run(prev_df: pd.DataFrame, orig_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], str]:
            call = dict(params)

            if "df" in sig.parameters:
                # explicit df bound at build time wins
                if "df" not in call:
                    call["df"] = orig_df if use_original_df and orig_df is not None else prev_df

            out_df, stats = func(**call)
            if not isinstance(stats, dict):
                raise TypeError(f"{step_name}: expected stats to be dict, got {type(stats)}")

            # for logging: the input df actually used by the function (if any), else prev_df
            input_df = call.get("df", prev_df)
            return input_df, out_df, stats, step_name

        _run.__name__ = step_name
        _run.__doc__  = func.__doc__
        _run.__signature__ = sig
        return _run

    builder.__name__ = f"{step_name}_step"
    builder.__doc__  = f"Step builder for `{func.__name__}`.\n\n" + (func.__doc__ or "")
    builder.__signature__ = sig
    return builder

def run_pipeline(steps: Sequence[Callable], df: pd.DataFrame):
    """
    Calls each step as step(prev_df, orig_df) and chains outputs.
    Returns final_df, logs.
    """
    orig_df = df
    prev_df = df
    logs: List[Dict[str, Any]] = []
    for step in steps:
        in_df, out_df, stats, name = step(prev_df, orig_df)
        logs.append({"step": name, **stats})
        prev_df = out_df if out_df is not None else prev_df
    return prev_df, logs

