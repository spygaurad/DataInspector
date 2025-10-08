from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from pipeline.deduplication import find_near_duplicates
from pipeline.featurizer import custom_featurizer
from pipeline.issues import find_issues
from pipeline.utils_cool import (
    get_signature_dict,
    guess_task_and_label,
)

from .runtime_ctx import (
    get_df_payload,  # now supports version spec (None|'current'|'prev'|'base'|'@-1'|int)
    )
from .runtime_ctx import (
    list_versions as _list_versions_state,
)
from .runtime_ctx import (
    reset_current_to as _reset_current_to,
)

# Registry of runnable steps (names used by the agent/UI)
STEP_FUNCS = {
    "dedup": find_near_duplicates,
    "featurize": custom_featurizer,
    "find_label_issues": find_issues,
}


@tool("inspect_dataset", return_direct=True)
def tool_inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize the CURRENT dataset (no arguments required).

    Behavior:
      • Reads the dataset from the runtime context (set by the graph).
      • Returns a compact summary of columns, dtypes, shape, and a guessed label/task.

    Returns:
      {
        "type": "dataset_summary",
        "columns": [...],
        "dtypes": {col: dtype, ...},
        "shape": (rows, cols),
        "label_guess": "<name or None>",
        "task_guess": "classification|regression|unsupervised",
        "issues": [ ... ]  # e.g., missing labels, single-class, etc.
      }
    """
    if df is None:
        raise RuntimeError("inspect_dataset: no dataset available in runtime context.")
    summary = guess_task_and_label(df)
    # keep context fresh for downstream tools
    return {"type": "dataset_summary", **summary}


@tool("sota_preprocessing", return_direct=True)
def tool_sota_preprocessing(
    df_summary: Dict[str, Any],
    task: Optional[str] = None,
    modality: Optional[str] = None,
    domain: Optional[str] = None,
    target: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search state-of-the-art preprocessing best practices (modality-aware).

    Args:
      task: e.g., "classification", "regression", "segmentation", "NER", "ASR", "forecasting".
            If omitted, inferred from the dataset summary if available.
      modality: one of {"tabular","text","image","audio","video","time_series","graph","multimodal"}.
      domain: optional domain context (e.g., "clinical", "finance").
      target: optional target structure (e.g., "segmentation masks", "bounding boxes").

    Returns:
      {
        "type": "sota",
        "task": ...,
        "modality": ...,
        "domain": ...,
        "target": ...,
        "queries": [...],
        "bundled_results": [{ "query": q, "results": <tavily-results> }, ...],
        "results": <first tavily-results batch>
      }
    """
    if not task:
        task = df_summary.get("task_guess") or "classification"

    yr = "2024 2025"
    m = (modality or "").lower().strip()

    modality_terms = {
        "tabular": ["imputation", "encoding", "scaling", "outliers", "leakage prevention"],
        "text": ["tokenization", "normalization", "subword", "BPE", "SentencePiece", "stopwords", "lemmatization", "augmentation"],
        "image": ["normalization", "resizing", "color space", "augmentation", "RandAugment", "MixUp", "CutMix"],
        "audio": ["resampling", "log-mel spectrogram", "MFCC", "pre-emphasis", "SpecAugment", "denoising"],
        "time_series": ["resampling", "windowing", "detrending", "imputation", "outlier detection", "scaling"],
        "video": ["frame sampling", "temporal augmentation", "clip normalization", "optical flow"],
        "graph": ["feature normalization", "self-loops", "adjacency normalization", "sparsification"],
        "multimodal": ["alignment", "synchronization", "fusion", "tokenization"],
    }
    m_terms = modality_terms.get(m, [])

    # Build candidate queries
    queries: List[str] = []
    queries.append(f"state of the art preprocessing {task} {yr}")
    queries.append(f"best practices data preprocessing {task} {yr}")
    if m:
        queries.append(f"{m} {task} preprocessing best practices {yr}")
    if domain:
        queries.append(f"{domain} {m or ''} {task} preprocessing best practices {yr}".strip())
    if target:
        queries.append(f"{m or ''} {task} {target} preprocessing pipeline {yr}".strip())
    if m_terms:
        queries.append(f"{m} {task} preprocessing {' '.join(m_terms)} {yr}")

    # Deduplicate, preserve order
    seen = set()
    queries = [q for q in (q.strip() for q in queries) if q and (q not in seen and not seen.add(q))]

    tavily = TavilySearchResults(k=6)
    bundled: List[Dict[str, Any]] = [{"query": q, "results": tavily.invoke({"query": q})} for q in queries]
    flat_first = bundled[0]["results"] if (bundled and "results" in bundled[0]) else []

    # persist for planning

    return {
        "type": "sota",
        "task": task,
        "modality": m or "unknown",
        "domain": domain,
        "target": target,
        "queries": queries,
        "bundled_results": bundled,
        "results": flat_first,
    }

@tool("describe_step", return_direct=True)
def tool_describe_step(name: str) -> Dict[str, Any]:
    """
    Return the exact docstring + parameter schema for a single step by name.
    This prevents the model from inventing params.
    """
    if name not in STEP_FUNCS:
        raise ValueError(f"Unknown step '{name}'. Available: {list(STEP_FUNCS)}")
    fn = STEP_FUNCS[name]
    sig = get_signature_dict(fn)  # your util that introspects defaults/annotations
    return {"type": "step_description", "name": name, **sig}

@tool("list_steps", return_direct=True)
def tool_list_steps() -> Dict[str, Any]:
    """
    List available pipeline steps (name, docstring, and signature).

    Returns:
      {
        "type": "steps",
        "steps": [
          {
            "name": "dedup" | "featurize" | "find_label_issues",
            "doc": "<docstring>",
            "params": [{"name": "...", "default": ..., "annotation": "...", "kind": "..."}]
          }, ...
        ]
      }
    """
    return {
        "type": "steps",
        "steps": [{"name": n, **get_signature_dict(fn)} for n, fn in STEP_FUNCS.items()],
    }


@tool("propose_plan", return_direct=True)
def tool_propose_plan(
    df_summary: Dict[str, Any],
    bundled: List[Dict[str, Any]],
    task: Optional[str] = None,
    modality: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Propose an ordered preprocessing plan grounded in SOTA + dataset summary.
    (Planning only — does not execute steps.)
    """
    if not task:
        task = df_summary.get("task_guess") or "classification"
    label_guess = df_summary.get("label_guess")

    KEYWORDS = {
        "dedup": {"duplicate", "near-duplicate", "near duplicate", "dupe", "dedup", "similarity", "knn", "kNN"},
        "featurize": {
            "impute", "imputation", "encoding", "one-hot", "scale", "scaling",
            "normalize", "normalization", "standardize", "tfidf", "tokenization",
            "lemmatization", "augmentation"
        },
        "find_label_issues": {"label noise", "noisy labels", "cleanlab", "confident learning", "label issues", "weak labels"},
    }

    def _score(text: str, keys: set[str]) -> int:
        t = (text or "").lower()
        return sum(1 for k in keys if k in t)

    hits = {"dedup": 0, "featurize": 0, "find_label_issues": 0}
    evidence: Dict[str, List[Dict[str, str]]] = {"dedup": [], "featurize": [], "find_label_issues": []}

    for pack in bundled:
        q = pack.get("query", "")
        for item in (pack.get("results") or []):
            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")
            for step, keys in KEYWORDS.items():
                s = _score(f"{q} {title} {content}", keys)
                if s > 0:
                    hits[step] += s
                    if len(evidence[step]) < 5:
                        evidence[step].append({"query": q, "title": title, "url": url})

    options: List[Dict[str, Any]] = []

    if hits["dedup"] > 0 or modality in {None, "tabular", "text", "image", "time_series"}:
        options.append(
            {
                "reason": "SOTA emphasizes handling near-duplicates early" if hits["dedup"] else "Practical first step to prevent leakage/skew",
                "step": "dedup",
                "params": {"threshold": 0.95, "metric": "cosine"},
                "evidence": evidence["dedup"][:3],
            }
        )

    options.append(
        {
            "reason": "SOTA emphasizes robust imputation/encoding/scaling" if hits["featurize"] else "Prepare features based on modality",
            "step": "featurize",
            "params": {"nan_strategy": "impute"},
            "evidence": evidence["featurize"][:3],
        }
    )

    if (task == "classification" and label_guess) or hits["find_label_issues"] > 0:
        options.append(
            {
                "reason": "SOTA recommends checking noisy labels" if hits["find_label_issues"] else "Check label quality before training",
                "step": "find_label_issues",
                "params": {"label": label_guess or "<CONFIRM>"},
                "evidence": evidence["find_label_issues"][:3],
            }
        )

    if not options:
        options = [{"reason": "Generic best practice", "step": "featurize", "params": {"nan_strategy": "impute"}, "evidence": []}]

    return {
        "type": "plan",
        "task": task,
        "modality": modality,
        "label_guess": label_guess,
        "options": options,
        "keyword_hits": hits,
    }


@tool("run_step", return_direct=True)
def tool_run_step(df: pd.DataFrame, name: str, params_json: str = "") -> Dict[str, Any]:
    """
    Execute a single pipeline step on the CURRENT dataset (no df argument).
    Returns ONLY a compact summary; the updated df is stored in runtime context.
    """
    if df is None:
        raise RuntimeError("run_step: no dataset available in runtime context.")
    if name not in STEP_FUNCS:
        raise ValueError(f"Unknown step '{name}'. Available: {list(STEP_FUNCS)}")

    params = json.loads(params_json) if params_json else {}
    if not isinstance(params, dict):
        raise ValueError("params_json must decode to a JSON object")

    df_out, stats = STEP_FUNCS[name](df=df, **params)
    df_next = df_out if df_out is not None else df
    
    # Build a tiny, safe summary for the model
    shape_before = (len(df), len(df.columns))
    shape_after  = (len(df_next), len(df_next.columns))
    compact_stats = {k: stats.get(k) for k in [
        "n_rows_before_dedup", "n_near_dupe_pairs", "n_groups",
        "n_rows_flagged_duplicates", "n_rows_after_dedup",
        "metric", "threshold", "k", "total_time_sec"
    ] if k in stats}

    return {
        "type": "step_result",
        "name": name,
        "params_used": params,
        "shape_before": shape_before,
        "shape_after": shape_after,
        "stats": compact_stats,   # small dict only
        "note": "Dataset updated in runtime context; use list_versions/reset_to_version if needed."
    }

# ---------------------------
# Optional helpers for version control from chat/agent
# ---------------------------
@tool("list_versions", return_direct=True)
def tool_list_versions() -> Dict[str, Any]:
    """
    Return a lightweight view of the version stack:
      { count, current_index, has_base, meta: [{...}, ...] }
    """
    return {"type": "versions", **_list_versions_state()}


@tool("reset_to_version", return_direct=True)
def tool_reset_to_version(spec: str) -> Dict[str, Any]:
    """
    Move CURRENT pointer to a prior version without deleting history.
    spec can be: "base" | "prev" | "@-1" | "@-2" | "3"
    """
    # accept int or @-k in string form
    try:
        if spec.isdigit():
            _reset_current_to(int(spec))
        else:
            _reset_current_to(spec)
    except Exception as e:
        raise RuntimeError(f"reset_to_version: {e}")

    df_payload = get_df_payload()
    return {
        "type": "reset",
        "current": _list_versions_state(),
        "df": df_payload,
    }
