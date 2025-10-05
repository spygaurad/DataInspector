from __future__ import annotations

from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from utils import PhaseProgress

def _safe_apply_pipeline(
    group_name: str,
    df_in: pd.DataFrame,
    cols: List[str],
    pipeline: Pipeline,
    *,
    drop_on_error: bool,
    warnings: List[str],
    max_new_cols: int = 2000,
) -> pd.DataFrame:
    """
    Apply a user pipeline to df_in[cols] safely.
    - If success and returns DataFrame with same index: replace group columns with returned columns (prefixed).
    - If success and returns ndarray/sparse: if width <= max_new_cols, expand with auto names; else drop group.
    - If failure and drop_on_error: drop group columns and record a warning.
    """
    Xsub = df_in[cols]
    try:
        out = pipeline.fit_transform(Xsub)
    except Exception as e:
        if drop_on_error:
            warnings.append(
                f"[custom_featurizer] dropped '{group_name}' columns ({len(cols)}): {cols[:6]}{'...' if len(cols)>6 else ''} "
                f"reason={type(e).__name__}: {str(e)[:180]}"
            )
            return df_in.drop(columns=cols)
        raise

    # If pipeline returns a DataFrame -> use its columns (prefixed)
    if isinstance(out, pd.DataFrame):
        out_df = out.copy()
        # ensure index alignment
        out_df.index = df_in.index
        # prefix to avoid collisions
        out_df.columns = [f"{group_name}__{c}" for c in out_df.columns]
        df_out = df_in.drop(columns=cols).join(out_df)
        return df_out

    # If ndarray / sparse matrix -> expand to columns cautiously
    try:
        import numpy as _np
        import scipy.sparse as _sp
        if _sp.issparse(out):
            out = out.toarray()
        out = _np.asarray(out)
        n_rows, n_cols = out.shape[0], (out.shape[1] if out.ndim == 2 else 1)
        if n_rows != len(df_in):
            raise ValueError(f"Pipeline for '{group_name}' returned {n_rows} rows; expected {len(df_in)}.")
        if n_cols > max_new_cols:
            warnings.append(
                f"[custom_featurizer] '{group_name}' produced {n_cols} columns (> {max_new_cols}); dropping this group to avoid explosion."
            )
            return df_in.drop(columns=cols)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
            n_cols = 1
        new_cols = [f"{group_name}__f{i}" for i in range(n_cols)]
        out_df = pd.DataFrame(out, index=df_in.index, columns=new_cols)
        df_out = df_in.drop(columns=cols).join(out_df)
        return df_out
    except Exception as e:
        if drop_on_error:
            warnings.append(
                f"[custom_featurizer] failed to materialize output for '{group_name}'; dropping group. "
                f"reason={type(e).__name__}: {str(e)[:180]}"
            )
            return df_in.drop(columns=cols)
        raise


def custom_featurizer(
    df: pd.DataFrame,
    *,
    # optional label just for cleaning rows with NaN in label; not returned as y
    label: Optional[str] = None,

    # user-provided pipelines (override defaults for that group)
    numeric_pipeline: Optional[Pipeline] = None,
    low_card_pipeline: Optional[Pipeline] = None,
    text_pipeline: Optional[Pipeline] = None,

    # defaults (used ONLY if the corresponding pipeline is NOT provided)
    numeric_scale: bool = True,                  # scale numeric after impute
    text_lowercase: bool = True,                 # lower+strip text
    max_ohe_cardinality: int = 50,               # threshold to split low-card vs text

    # NaN handling for DEFAULTS ONLY (ignored if a custom pipeline is provided for that group)
    nan_strategy: str = "impute",                # "impute" or "drop"

    # failure policy for user pipelines
    on_pipeline_error: str = "drop",             # "drop" -> drop group; "raise" -> bubble error

    # control expansion when user pipelines return big matrices
    max_new_cols_per_group: int = 2000,

    # progress / logging
    progress: Optional[Any] = None,              # pass a tqdm here; None -> auto-create

    # logging
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Featurize a mixed DataFrame while keeping the output as a DataFrame for downstream steps.

    Overview
    --------
    Columns are split into three groups:
      • numeric (including boolean)  
      • low-cardinality categoricals (nunique ≤ `max_ohe_cardinality`)  
      • text/high-cardinality (everything else)

    For each group you may provide your own sklearn `Pipeline`. If a user-provided
    pipeline **raises** during `fit/transform` and `on_pipeline_error="drop"`, that
    entire group of columns is **dropped** and a short message is recorded in
    `stats["warnings"]` (the step never crashes). If `"raise"`, the error is propagated.

    If no pipeline is provided for a group, a sensible **default** is used:
      • numeric: `SimpleImputer(median)` then optional `StandardScaler(with_mean=False)`
      • low-card: `SimpleImputer(most_frequent)` (no one-hot here; avoids column blow-up)
      • text: fill NaN with empty string, optional lowercase/strip

    `nan_strategy` only applies to groups that use the **default** pipeline. For groups
    with custom pipelines, NaN handling is assumed to be part of the user pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input table. If `label` is provided, rows with NaN in `label` are dropped
        before featurization. The returned DataFrame preserves row order after cleaning.
    label : Optional[str], default=None
        Name of the target column (used **only** to drop NaN targets). Not returned as `y`.
        If provided but missing in `df`, a `KeyError` is raised.
    numeric_pipeline : Optional[sklearn.pipeline.Pipeline], default=None
        Custom pipeline applied to numeric/boolean columns. If provided, `nan_strategy`
        is ignored for this group and NaN handling is your pipeline’s responsibility.
    low_card_pipeline : Optional[sklearn.pipeline.Pipeline], default=None
        Custom pipeline for low-cardinality categorical columns. Same NaN rule as above.
    text_pipeline : Optional[sklearn.pipeline.Pipeline], default=None
        Custom pipeline for text/high-cardinality columns. Same NaN rule as above.
    numeric_scale : bool, default=True
        Only used if `numeric_pipeline` is None. When True, applies
        `StandardScaler(with_mean=False)` after imputation.
    text_lowercase : bool, default=True
        Only used if `text_pipeline` is None. When True, lowercases and strips text
        after filling NaNs with "".
    max_ohe_cardinality : int, default=50
        Threshold to decide whether a non-numeric column is treated as low-cardinality
        categorical (≤ threshold) or text/high-cardinality (> threshold).
    nan_strategy : {"impute","drop"}, default="impute"
        For **default** group pipelines only:
          - "impute": impute NaNs (median for numeric, most_frequent for low-card; text fills "")
          - "drop": drop rows containing NaNs in any default-handled feature column
        Ignored for groups with a custom pipeline.
    on_pipeline_error : {"drop","raise"}, default="drop"
        What to do if a **user-provided** pipeline raises during fit/transform:
          - "drop": drop the entire group’s columns and record a warning
          - "raise": propagate the exception
    max_new_cols_per_group : int, default=2000
        If a user pipeline returns an array/sparse matrix with more than this many
        columns, the group is dropped (to avoid exploding the DataFrame) and a warning
        is recorded. When the width is acceptable, new columns are added with a
        `"{group}__f{i}"` prefix.
    progress : Optional[tqdm], default=None
        If provided, progress is reported across phases:
        clean → split → numeric → low_card → text → finalize.
        If None, a local tqdm(total=100) is created and closed.
    verbose : bool, default=False
        When True, prints collected warnings (also available in `stats["warnings"]`).

    Returns
    -------
    output_df : pd.DataFrame
        Transformed DataFrame to feed the next pipeline step. May have:
          - imputed/scaled numeric values (if default numeric pipeline used)
          - cleaned low-card categorical values (if default low-card pipeline used)
          - normalized text (if default text pipeline used)
          - or columns replaced/expanded by user pipelines (subject to `max_new_cols_per_group`)
    stats : Dict[str, Any]
        Minimal metadata:
          - "warnings": List[str] — messages for any dropped/failed groups
          - "cols": Dict[str, List[str]] — column names per group:
                {"numeric": [...], "low_card": [...], "text": [...]}
          - "n_rows_before": int — input row count before cleaning
          - "n_rows_after" : int — final row count

    Notes
    -----
    • If a user pipeline returns a DataFrame, its columns are joined back with a
      `"{group}__{original}"` prefix to avoid collisions. If it returns an array/sparse
      matrix, columns are auto-named `"{group}__f{i}"`.
    • Group failure handling only applies to **user** pipelines (via `on_pipeline_error`).
      Default pipelines are designed to be robust.

    Raises
    ------
    KeyError
        If `label` is provided but not found in `df`.
    ValueError
        If `nan_strategy` not in {"impute","drop"} or `on_pipeline_error` not in {"drop","raise"}.

    Examples
    --------
    >>> # Defaults for all groups
    >>> out_df, stats = custom_featurizer(df, label="target", nan_strategy="impute")

    >>> # Custom text pipeline; drop group on failure
    >>> txt_pipe = Pipeline([...])
    >>> out_df, stats = custom_featurizer(df, text_pipeline=txt_pipe, on_pipeline_error="drop")

    >>> # Strict mode: raise if any user pipeline fails
    >>> out_df, stats = custom_featurizer(df, numeric_pipeline=my_num, on_pipeline_error="raise")
    """
    # progress setup
    local_bar = None
    pp = None
    if progress is None:
        from tqdm.auto import tqdm  # local import to keep module lightweight
        local_bar = tqdm(total=100, leave=True)
        pp = PhaseProgress(local_bar, weights={
            "clean": .10, "split": .10, "numeric": .30, "low_card": .25, "text": .20, "finalize": .05
        })
    elif hasattr(progress, "set_description") and hasattr(progress, "update"):
        if not hasattr(progress, "_last_val"):
            progress._last_val = 0
        pp = PhaseProgress(progress, weights={
            "clean": .10, "split": .10, "numeric": .30, "low_card": .25, "text": .20, "finalize": .05
        })

    warnings: List[str] = []

    # 1) drop rows with NaN label if present
    n_before = len(df)
    pp and pp.start("clean", extra={"N": n_before})
    if label is not None and label not in df.columns:
        # ensure the bar is closed even on error
        if local_bar is not None:
            pp.close()
        raise KeyError(f"Target column '{label}' not in DataFrame.")
    if label is not None:
        df = df.dropna(subset=[label]).reset_index(drop=True)
    pp and pp.tick_abs("clean", 1.0, extra={"N": len(df)})
    pp and pp.end("clean")

    # 2) split columns
    pp and pp.start("split")
    Xf = df if label is None else df.drop(columns=[label])
    numeric_cols: List[str] = Xf.select_dtypes(include=[np.number, "boolean"]).columns.tolist()
    non_numeric = [c for c in Xf.columns if c not in numeric_cols]

    low_card_cols: List[str] = []
    text_cols: List[str] = []
    for c in non_numeric:
        nunq = Xf[c].nunique(dropna=True)
        (low_card_cols if nunq <= max_ohe_cardinality else text_cols).append(c)
    pp and pp.tick_abs("split", 1.0, extra={
        "num": len(numeric_cols), "low": len(low_card_cols), "txt": len(text_cols)
    })
    pp and pp.end("split")

    # 3) NaN strategy for default groups (custom pipelines assumed to handle NaNs)
    if nan_strategy not in {"impute", "drop"}:
        if local_bar is not None:
            pp.close()
        raise ValueError("nan_strategy must be 'impute' or 'drop'")
    if nan_strategy == "drop":
        cols_to_check = []
        if numeric_cols and numeric_pipeline is None:
            cols_to_check += numeric_cols
        if low_card_cols and low_card_pipeline is None:
            cols_to_check += low_card_cols
        if text_cols and text_pipeline is None:
            cols_to_check += text_cols
        if cols_to_check:
            mask = ~Xf[cols_to_check].isna().any(axis=1)
            df = df.loc[mask].reset_index(drop=True)
            Xf = df if label is None else df.drop(columns=[label])

    # 4) numeric
    pp and pp.start("numeric", extra={"cols": len(numeric_cols)})
    if numeric_cols:
        if numeric_pipeline is not None:
            df = _safe_apply_pipeline(
                "num", df, numeric_cols, numeric_pipeline,
                drop_on_error=(on_pipeline_error == "drop"),
                warnings=warnings,
                max_new_cols=max_new_cols_per_group,
            )
        else:
            if nan_strategy == "impute":
                imputed = SimpleImputer(strategy="median").fit_transform(df[numeric_cols])
                df.loc[:, numeric_cols] = imputed
            if numeric_scale:
                scaler = StandardScaler(with_mean=False)
                scaled = scaler.fit_transform(df[numeric_cols].astype(float, copy=False))
                df.loc[:, numeric_cols] = np.asarray(scaled)
    pp and pp.tick_abs("numeric", 1.0)
    pp and pp.end("numeric", extra={"cols": len(numeric_cols)})

    # 5) low-card categoricals
    pp and pp.start("low_card", extra={"cols": len(low_card_cols)})
    if low_card_cols:
        if low_card_pipeline is not None:
            df = _safe_apply_pipeline(
                "cat", df, low_card_cols, low_card_pipeline,
                drop_on_error=(on_pipeline_error == "drop"),
                warnings=warnings,
                max_new_cols=max_new_cols_per_group,
            )
        else:
            if nan_strategy == "impute":
                df.loc[:, low_card_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[low_card_cols])
    pp and pp.tick_abs("low_card", 1.0)
    pp and pp.end("low_card", extra={"cols": len(low_card_cols)})

    # 6) text
    pp and pp.start("text", extra={"cols": len(text_cols)})
    if text_cols:
        if text_pipeline is not None:
            df = _safe_apply_pipeline(
                "txt", df, text_cols, text_pipeline,
                drop_on_error=(on_pipeline_error == "drop"),
                warnings=warnings,
                max_new_cols=max_new_cols_per_group,
            )
        else:
            df.loc[:, text_cols] = df[text_cols].fillna("").astype(str)
            if text_lowercase:
                df.loc[:, text_cols] = df[text_cols].applymap(lambda s: s.strip().lower())
    pp and pp.tick_abs("text", 1.0)
    pp and pp.end("text", extra={"cols": len(text_cols)})

    # 7) finalize
    pp and pp.start("finalize")
    stats: Dict[str, Any] = {
        "warnings": warnings,
        "cols": {"numeric": numeric_cols, "low_card": low_card_cols, "text": text_cols},
        "n_rows_before": n_before,
        "n_rows_after": len(df),
    }
    pp and pp.tick_abs("finalize", 1.0, extra={"warn": len(warnings)})
    pp and pp.end("finalize", extra={"warn": len(warnings)})

    if verbose and warnings:
        print("\n".join(warnings))
    if local_bar is not None:
        pp.close()

    return df, stats