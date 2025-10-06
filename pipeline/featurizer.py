from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .utils_cool import PhaseProgress


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
      • text/high-cardinality (nunique > `max_ohe_cardinality`)
 
    You may pass custom sklearn `Pipeline`s per group. If a user pipeline raises and
    `on_pipeline_error="drop"`, that entire group of columns is dropped and a short
    warning is recorded in `stats["warnings"]` (the step does not crash). If `"raise"`,
    the error is propagated.
 
    Defaults when pipelines are NOT provided
    ---------------------------------------
    • Numeric (when `numeric_pipeline is None`)
        - NaNs: if `nan_strategy="impute"`, apply `SimpleImputer(strategy="median")`;
                if `"drop"`, rows with NaNs in numeric columns are dropped BEFORE this step.
        - Scaling: if `numeric_scale=True`, apply `StandardScaler(with_mean=False)`.
        - Columns: replaced in place (same column names remain; values become numeric/float).
    • Low-cardinality categoricals (when `low_card_pipeline is None`)
        - NaNs: if `nan_strategy="impute"`, apply `SimpleImputer(strategy="most_frequent")`;
                if `"drop"`, rows with NaNs in these columns are dropped BEFORE this step.
        - Encoding: **no one-hot by default** (values stay as cleaned strings/categories).
          If you want encodings, pass your own `low_card_pipeline` (e.g., OneHotEncoder/CatBoostEncoder).
        - Columns: preserved (same names).
    • Text / high-cardinality (when `text_pipeline is None`)
        - Build a tiny pipeline:
              concat selected text cols → `TfidfVectorizer(dtype=float32, lowercase=text_lowercase)`
        - Output: numeric TF-IDF features. Original text columns are **replaced** by new
          columns named `txt__f0`, `txt__f1`, … (or `txt__<col>` if your pipeline returns a DataFrame).
        - Feature explosion guard: if the produced matrix has more than `max_new_cols_per_group` columns,
          the entire text group is dropped and a warning is recorded.
        - If TF-IDF fails (e.g., empty vocabulary on tiny data), the text group is dropped with a warning.
 
    NaN handling
    ------------
    `nan_strategy` applies only to groups using the **default** pipeline:
      - "impute": impute NaNs (as above)
      - "drop"  : drop rows containing NaNs in any default-handled feature column **before**
                  transforming. For groups with a **custom** pipeline, NaN handling is your pipeline’s responsibility.
 
    Parameters
    ----------
    df : pd.DataFrame
        Input table. If `label` is provided, rows with NaN in `label` are dropped first.
    label : Optional[str], default=None
        Name of the target column (only used to drop NaN labels). Not returned as `y`.
    numeric_pipeline : Optional[sklearn.pipeline.Pipeline], default=None
        Custom pipeline for numeric/boolean columns. If provided, `nan_strategy` is ignored for this group.
        By default: median imputation (+ optional scaling) and columns are preserved.
    low_card_pipeline : Optional[sklearn.pipeline.Pipeline], default=None
        Custom pipeline for low-card categorical columns. If provided, `nan_strategy` is ignored for this group.
        By default: most-frequent imputation only; **no encoding**; columns are preserved.
    text_pipeline : Optional[sklearn.pipeline.Pipeline], default=None
        Custom pipeline for text/high-card columns. If provided, it replaces the text columns with whatever
        it outputs (DataFrame → prefixed columns; array/sparse → `txt__f*` columns). If not provided, the
        built-in concat+TF-IDF is used (see defaults above).
    numeric_scale : bool, default=True
        Applies `StandardScaler(with_mean=False)` to numeric columns in the default numeric pipeline.
    text_lowercase : bool, default=True
        Forwarded to the default `TfidfVectorizer(lowercase=...)`.
    max_ohe_cardinality : int, default=50
        Threshold to classify non-numeric columns as low-card (≤ threshold) vs text/high-card (> threshold).
    nan_strategy : {"impute","drop"}, default="impute"
        Strategy for **default** pipelines only (custom pipelines manage their own NaNs).
    on_pipeline_error : {"drop","raise"}, default="drop"
        If a **user** pipeline raises: "drop" → drop that group and record a warning; "raise" → propagate.
    max_new_cols_per_group : int, default=2000
        Upper bound on the number of columns a group is allowed to add (applies to array/sparse outputs).
        If exceeded, the group is dropped and a warning is recorded.
    progress : Optional[tqdm], default=None
        Phase-aware progress bar (clean → split → numeric → low_card → text → finalize).
        If None, a local bar is created and closed automatically.
    verbose : bool, default=False
        When True, prints recorded warnings (in addition to returning them in `stats["warnings"]`).
 
    Returns
    -------
    output_df : pd.DataFrame
        Transformed DataFrame ready for the next pipeline step. Numeric/low-card columns are
        updated in place by defaults; text columns are replaced by TF-IDF features when using
        the default text path.
    stats : Dict[str, Any]
        Minimal metadata:
          - "warnings": List[str]
          - "cols": {"numeric": [...], "low_card": [...], "text": [...]}
          - "n_rows_before": int
          - "n_rows_after": int
 
    Notes
    -----
    • If a user pipeline returns a DataFrame, columns are prefixed with the group name to avoid collisions.
      If it returns an array/sparse matrix, columns are auto-named `"{group}__f{i}"`.
    • The feature explosion guard applies after transformation; if you often hit it with text, consider passing
      your own `TfidfVectorizer(max_features=...)` in `text_pipeline` to cap the width proactively.
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
        if text_pipeline is None:
            def _concat_cols(Xframe: pd.DataFrame):
                return Xframe.fillna("").astype(str).agg(" ".join, axis=1).values
 
            text_pipeline = Pipeline([
            ("concat", FunctionTransformer(_concat_cols, validate=False)),
            ("tfidf",  TfidfVectorizer(
                dtype=np.float32,
                lowercase=bool(text_lowercase),
            )),
        ])
 
        df = _safe_apply_pipeline(
            "txt", df, text_cols, text_pipeline,
            drop_on_error=(on_pipeline_error == "drop"),
            warnings=warnings,
            max_new_cols=max_new_cols_per_group,
        )
 
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