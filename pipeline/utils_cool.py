import inspect
import math
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from tqdm import tqdm


@dataclass
class PhaseProgress:
    bar: "tqdm"
    weights: Dict[str, float]
    total: int = 100

    def __post_init__(self):
        self._norm = sum(self.weights.values()) or 1.0
        self._done = 0.0
        self._phase = None
        self._phase_t0 = None
        # for smooth updates
        if not hasattr(self.bar, "_last_val"):
            self.bar._last_val = 0

    def start(self, phase: str, extra: Optional[Dict] = None):
        self._phase = phase
        self._phase_t0 = perf_counter()
        self.bar.set_description_str(phase)
        if extra:
            self.bar.set_postfix(extra, refresh=False)

    def tick_abs(self, phase: str, p01: float, extra: Optional[Dict] = None):
        """Update absolute progress based on within-phase progress p01 ∈ [0,1]."""
        p01 = max(0.0, min(1.0, float(p01)))
        w = self.weights.get(phase, 0.0) / self._norm
        target = int(round(self.total * (self._done + w * p01)))
        delta = target - self.bar._last_val
        if delta > 0:
            self.bar.update(delta)
            self.bar._last_val = target
        self.bar.set_description_str(f"{phase} {int(100*p01)}%")
        if extra:
            self.bar.set_postfix(extra, refresh=False)

    def end(self, phase: str, extra: Optional[Dict] = None):
        w = self.weights.get(phase, 0.0) / self._norm
        self._done += w
        elapsed_ms = (perf_counter() - (self._phase_t0 or perf_counter())) * 1000
        post = dict(extra or {})
        post["t"] = f"{elapsed_ms:.0f}ms"
        self.bar.set_postfix(post, refresh=False)

    def close(self):
        try:
            if self.bar._last_val < self.total:
                self.bar.update(self.total - self.bar._last_val)
        finally:
            self.bar.close()

def choose_k(N: int, k_min: int = 5, k_max: int = 50) -> int:
    """sqrt(N) clipped to [k_min, k_max] and ≤ N-1."""
    if N <= 1:
        return 1
    k = int(math.sqrt(N))
    k = max(k_min, min(k, k_max))
    return min(k, N - 1)

def _ensure_dense32(X) -> np.ndarray:
    """Convert to contiguous float32 ndarray (densify only if needed)."""
    if issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32, order="C")

def decide_task_and_model(
    y: np.ndarray,
    series: pd.Series,
    *,
    is_categorical: bool = False,
    few_class_floor: int = 20,
    few_class_frac: float = 0.05,
):
    N = len(y)

    # dtype checks
    is_bool = pd.api.types.is_bool_dtype(series)
    is_numeric = pd.api.types.is_numeric_dtype(series)

    # unique values (ignore NaNs)
    y_nonnull = y[~pd.isnull(y)]
    n_unique = len(pd.unique(y_nonnull))

    # numeric-but-few-classes heuristic
    few_classes_threshold = max(few_class_floor, int(np.ceil(few_class_frac * max(N, 1))))
    numeric_few_classes = is_numeric and (n_unique <= few_classes_threshold)

    use_classification = (
        is_categorical
        or is_bool
        or (not is_numeric)
        or numeric_few_classes
    )

    if use_classification:
        return "classification"
    else:
        return "regression"

def _infer_task(y: np.ndarray, task: Optional[str]) -> str:
    """Decide task if not provided: numeric with many uniques -> regression, else classification."""
    if task in {"classification", "regression"}:
        return task

    if np.issubdtype(y.dtype, np.number):
        nunq = len(np.unique(y[~pd.isna(y)]))
        is_categorical = nunq <= max(2, int(0.02 * max(1, len(y))))
    else:
        is_categorical = True
    return "classification" if is_categorical else "regression"


# --------- DataFrame payload helpers (for tool IO) ---------

def df_to_payload(df: pd.DataFrame) -> Dict[str, Any]:
    return {"orient": "split", "data": df.to_dict(orient="split")}


def df_from_payload(p: Dict[str, Any]) -> pd.DataFrame:
    d = p["data"]
    return pd.DataFrame(d["data"], columns=d["columns"])

# --------- Light heuristics for task/label guess ---------

def guess_task_and_label(df: pd.DataFrame) -> Dict[str, Any]:
    cols = list(df.columns)
    label_candidates = [c for c in cols if c.lower() in {"label","target","y","class","outcome"}]
    label = label_candidates[0] if label_candidates else None


    task = None
    if label and (pd.api.types.is_integer_dtype(df[label]) or pd.api.types.is_bool_dtype(df[label])):
        nuniq = df[label].nunique(dropna=True)
        task = "classification" if nuniq <= max(20, int(0.05*len(df))) else "regression"
    elif label and pd.api.types.is_float_dtype(df[label]):
        task = "regression"
    else:
        task = "unsupervised"


    issues = []
    if label and df[label].isna().any():
        issues.append(f"Missing values in label `{label}`")
    if label and df[label].nunique() == 1:
        issues.append(f"Label `{label}` has a single class")


    return {
        "columns": cols,
        "dtypes": {c: str(df[c].dtype) for c in cols},
        "label_guess": label,
        "task_guess": task,
        "issues": issues,
        "shape": df.shape,
    }

# --------- Signature extraction for asking params ---------


def get_signature_dict(fn) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    doc = (fn.__doc__ or "").strip()
    params = []
    for p in sig.parameters.values():
        if p.name == "df":
            continue
        default = None if (p.default is inspect._empty) else p.default
        annotation = None if (p.annotation is inspect._empty) else str(p.annotation)
        params.append({"name": p.name, "default": default, "annotation": annotation, "kind": str(p.kind)})
    return {"params": params, "doc": doc}

# --------- Parse free-text confirmation like "Run dedup threshold=0.93 metric=cosine" ---------
STEP_ALIASES = {
"dedup": {"dedup","de-dup","duplicates","near-dup"},
"featurize": {"featurize","features","featureize","engineering"},
"find_label_issues": {"find_label_issues","label issues","cleanlab","label noise"},
}


def parse_user_choice(text: str) -> Tuple[Optional[str], Dict[str, Any]]:
    t = text.lower()
    chosen = None
    for step, aliases in STEP_ALIASES.items():
        if any(a in t for a in aliases):
            chosen = step
            break

    params: Dict[str, Any] = {}
    for m in re.finditer(r"(\w+)\s*=\s*([\-\w\.]+)", text):
        k, v = m.group(1), m.group(2)
        if v.replace('.', '', 1).isdigit():
            v = float(v) if '.' in v else int(v)
        elif v.lower() in {"true","false"}:
            v = (v.lower() == "true")
        params[k] = v
    return chosen, params