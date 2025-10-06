from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from cleanlab import Datalab
from cleanlab.datalab.internal.issue_manager.duplicate import NearDuplicateIssueManager
from scipy.sparse import issparse
from scipy.special import comb
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from .utils_cool import PhaseProgress, _ensure_dense32, choose_k


# =========================
# Core near-duplicate finder
# =========================
def find_near_duplicates(
    df: pd.DataFrame,
    *,
    # Cleanlab params
    metric: str = "cosine",
    threshold: float = 0.13,
    k: Optional[int] = None,
    # Vectorizer / features
    vectorizer: Optional[TransformerMixin] = None,
    force_dense: bool = False,   # set True if your cleanlab version needs dense
    # Behavior
    verbose: bool = False,
    # Progress: either a tqdm object or a callable phase,p in [0,1]
    progress: Optional[Any] = None,
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Detect near-duplicates using Cleanlab's NearDuplicateIssueManager.

    Parameters
    ----------
    df : DataFrame
        Rows to analyze. If no vectorizer is passed, all columns are joined as strings for TF-IDF.
    metric : {"cosine", "euclidean", "manhattan"}
        Distance metric for kNN graph.
    threshold : float
        Near-duplicate radius is based on threshold × median NN distance (internal to Cleanlab).
    k : int or None
        Neighborhood size. If None, uses sqrt(N) clipped to [5, 50] and ≤ N-1.
    vectorizer : sklearn Transformer
        Any transformer with fit_transform/transform. If None, uses TF-IDF (float32).
    force_dense : bool
        If True, densify features before passing to Cleanlab.
    verbose : bool
        Print timing breakdown.
    progress : tqdm or Callable[[str, float], None]
        Phase-aware progress reporting.

    Returns
    -------
    (output_df, stats)
        output_df : DataFrame after deduplication
        stats     : dict of counts/timings/params
    """
    # Progress setup (one bar per call unless user provided one)
    local_bar = None
    pp = None
    if progress is None:
        local_bar = tqdm(total=100, leave=True)
        pp = PhaseProgress(local_bar, weights={"vectorize": .2, "find_issues": .7, "grouping": .1})
    else:
        if hasattr(progress, "set_description") and hasattr(progress, "update"):
            # treat given tqdm as the bar
            if not hasattr(progress, "_last_val"):
                progress._last_val = 0
            pp = PhaseProgress(progress, weights={"vectorize": .2, "find_issues": .7, "grouping": .1})

    timings: Dict[str, float] = {}
    t0 = perf_counter()
    N = int(len(df))

    # --- Vectorize ---
    pp and pp.start("vectorize", extra={"N": N})
    t_vec0 = perf_counter()
    text_series = df.astype(str).agg(" ".join, axis=1)

    if vectorizer is None:
        vectorizer = TfidfVectorizer(dtype=np.float32)

    if hasattr(vectorizer, "fit_transform"):
        X = vectorizer.fit_transform(text_series.tolist())
    elif hasattr(vectorizer, "transform"):
        X = vectorizer.transform(text_series.tolist())
    else:
        raise TypeError("`vectorizer` must implement fit_transform or transform.")

    if force_dense:
        X = _ensure_dense32(X)

    timings["vectorize"] = perf_counter() - t_vec0
    pp and pp.tick_abs("vectorize", 1.0)
    pp and pp.end("vectorize")

    # --- Cleanlab duplicate finder ---
    pp and pp.start("find_issues", extra={"metric": metric})
    t_cl0 = perf_counter()

    if k is None:
        k = choose_k(N)

    lab = Datalab(data={"__row__": list(range(N))})
    ndm = NearDuplicateIssueManager(datalab=lab, metric=metric, threshold=threshold, k=k)

    pp and pp.tick_abs("find_issues", 0.0, extra={"k": k})

    try:
        ndm.find_issues(features=X)
    except Exception:
        if issparse(X):
            X_dense = _ensure_dense32(X)
            ndm.find_issues(features=X_dense)  # retry dense
        else:
            raise

    pp and pp.tick_abs("find_issues", 1.0)
    timings["cleanlab_find_issues"] = perf_counter() - t_cl0
    pp and pp.end("find_issues", extra={"k": k})

    near_dup_sets: List[List[int]] = getattr(ndm, "near_duplicate_sets", []) or []

    # --- Representatives & output ---
    pp and pp.start("grouping")
    t_out0 = perf_counter()

    # Keep smallest index as representative; skip any empty groups defensively
    reps = [int(np.min(g)) for g in near_dup_sets if np.size(g) > 0]
    in_any = {int(i) for g in near_dup_sets for i in np.asarray(g).ravel()}
    keep_set = set(reps) | (set(range(N)) - in_any)

    out_df = None
    if N > 0:
        keep_mask = np.zeros(N, dtype=bool)
        if keep_set:
            keep_mask[list(keep_set)] = True
        out_df = df.iloc[np.where(keep_mask)[0]].copy()
    else:
        out_df = df.copy()

    # Stats
    n_groups = sum(1 for g in near_dup_sets if np.size(g) > 0)
    group_sizes = [int(np.size(g)) for g in near_dup_sets if np.size(g) > 0]
    n_flagged = sum(max(0, s - 1) for s in group_sizes)  # rows we'd drop if keeping 1 per group
    n_pairs = int(sum(comb(s, 2, exact=True) for s in group_sizes))

    timings["groups_and_output"] = perf_counter() - t_out0
    total_time = perf_counter() - t0

    stats = {
        "n_rows_before_dedup": N,
        "n_near_dupe_pairs": n_pairs,
        "n_groups": n_groups,
        "avg_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "max_group_size": max(group_sizes) if group_sizes else 0,
        "n_rows_flagged_duplicates": n_flagged,
        "n_rows_after_dedup": int(len(out_df)) if out_df is not None else N,
        "metric": metric,
        "threshold": threshold,
        "k": int(k),
        "timings": timings,
        "total_time_sec": total_time,
    }

    if verbose:
        print(f"[timing] TOTAL: {total_time*1000:.1f} ms")
        for k_, v in timings.items():
            print(f"  - {k_}: {v*1000:.1f} ms")

    pp and pp.tick_abs("grouping", 1.0, extra={"groups": n_groups, "pairs": n_pairs})
    pp and pp.end("grouping")

    if local_bar is not None:
        pp.close()

    # Return the standardized pair expected by your pipeline
    return out_df, stats