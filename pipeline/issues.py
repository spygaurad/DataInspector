from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression, SGDRegressor
from tqdm.auto import tqdm

from .utils_cool import PhaseProgress, _ensure_dense32, _infer_task


def find_issues(
    df: pd.DataFrame,
    *,
    label: str,
    task: Optional[str] = None,                 # "classification" | "regression"; if None we infer
    model: Optional[Any] = None,                # sklearn estimator; default chosen by task
    progress: Optional[Any] = None,             # tqdm bar; None -> auto-create
    verbose: bool = False,
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Detect label issues using Cleanlab's CleanLearning.

    Parameters
    ----------
    df : DataFrame
        Input table containing features and a label column.
    label : str
        Name of the label column. Rows with NaN in label are dropped.
    task : {"classification","regression"}, optional
        If not provided, inferred from y.
    model : sklearn estimator, optional
        If not provided, defaults to LogisticRegression or SGDRegressor.
    progress : tqdm, optional
        Phase-aware progress bar. If None, a local bar is created and closed.
    verbose : bool, default False
        Print warnings/timings in addition to returning them in stats.

    Returns
    -------
    (df_out, stats)
        df_out : DataFrame with label-issues optionally removed (or original df if `remove_issues=False`).
        stats  : dict with minimal metadata and counts.
    """
    # ---- progress setup ----
    local_bar = None
    pp = None
    if progress is None:
        local_bar = tqdm(total=100, leave=True)
        pp = PhaseProgress(local_bar, weights={"clean": .15, "cleanlab": .75, "finalize": .10})
    elif hasattr(progress, "set_description") and hasattr(progress, "update"):
        if not hasattr(progress, "_last_val"):
            progress._last_val = 0
        pp = PhaseProgress(progress, weights={"clean": .15, "cleanlab": .75, "finalize": .10})

    warnings: List[str] = []
    n_before = len(df)

    # ---- clean: ensure label exists, drop NaNs in label ----
    pp and pp.start("clean", extra={"N": n_before})
    if label not in df.columns:
        if local_bar is not None:
            pp.close()
        raise KeyError(f"Label column '{label}' not found in DataFrame.")
    df_in = df.dropna(subset=[label]).reset_index(drop=True)
    X = df_in
    y_raw = X[label].to_numpy()
    pp and pp.tick_abs("clean", 1.0, extra={"N": len(X)})
    pp and pp.end("clean")

    # ---- task/model selection ----
    task_applied = _infer_task(y_raw, task)
    if model is None:
        if task_applied == "classification":
            model = LogisticRegression(solver="saga", n_jobs=-1)
        else:
            model = SGDRegressor()

    # y encoding for classification (Cleanlab expects numeric labels)
    if task_applied == "classification":
        classes, y = np.unique(y_raw, return_inverse=True)
    else:
        classes = None
        y = y_raw.astype(np.float64, copy=False)

    # ---- cleanlab: find label issues (with auto-dense fallback for tiny/sparse edge cases) ----
    pp and pp.start("cleanlab", extra={"task": task_applied})
    used_dense_fallback = False
    try:
        if task_applied == "classification":
            from cleanlab.classification import CleanLearning as _CL
        else:
            from cleanlab.regression.learn import CleanLearning as _CL
        cl = _CL(model)
        issues = cl.find_label_issues(X, y)
    except Exception as e:
        if issparse(X):
            # retry dense (cleanlab/small-N often prefers dense)
            Xd = _ensure_dense32(X)
            try:
                cl = _CL(model)
                issues = cl.find_label_issues(Xd, y)
                used_dense_fallback = True
            except Exception as e2:
                if local_bar is not None:
                    pp.close()
                raise RuntimeError(f"Cleanlab failed on sparse and dense features: {e2}") from e
        else:
            if local_bar is not None:
                pp.close()
            raise

    # Parse outputs robustly
    if isinstance(issues, pd.DataFrame):
        is_issue = issues.get("is_label_issue", None)
        label_quality = issues.get("label_quality", None)
    else:
        is_issue = None
        label_quality = None

    n = len(issues) if hasattr(issues, "__len__") else len(y)
    n_issues = int(is_issue.sum()) if isinstance(is_issue, (pd.Series, np.ndarray)) else 0
    pct = round((n_issues / n) * 100.0, 3) if n else 0.0
    avg_quality = float(np.nanmean(label_quality.values)) if isinstance(label_quality, pd.Series) else float("nan")

    pp and pp.tick_abs("cleanlab", 1.0, extra={"issues": n_issues})
    pp and pp.end("cleanlab", extra={"issues": n_issues})

    # ---- finalize: optionally drop issue rows ----
    pp and pp.start("finalize")
    df_out = df_in
    if isinstance(is_issue, (pd.Series, np.ndarray)):
        mask_keep = ~(is_issue.astype(bool).values)
        df_out = df_in.loc[mask_keep].copy()

    stats: Dict[str, Any] = {
        "n_rows_before_cleanlab": int(len(df_in)),
        "n_label_issues": int(n_issues),
        "pct_label_issues": float(pct),
        "avg_label_quality": float(avg_quality),
        "n_rows_after_cleanlab": int(len(df_out)),
        "task_applied": task_applied,
        "model_name": type(model).__name__,
        "used_dense_fallback": bool(used_dense_fallback),
        "warnings": warnings,
    }
    pp and pp.tick_abs("finalize", 1.0)
    pp and pp.end("finalize")

    if verbose and warnings:
        print("\n".join(warnings))
    if local_bar is not None:
        pp.close()

    return df_out, stats
