# agent/runtime_ctx.py
from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Union

# ---------------- Versioned dataset store ----------------
# Each new mutating step can create a new "version".
# You can address versions as: "current", "base", "prev", "@-1", "@-2", "@3" (0-based).
_VERSIONS_CV: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar("VERSIONS", default=None)
_CUR_INDEX_CV: ContextVar[int] = ContextVar("CUR_INDEX", default=-1)
_VERS_META_CV: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar("VERS_META", default=None)

# Legacy singletons (fallback across tasks)
_STORE: Dict[str, Any] = {
    "versions": [],            # list of df_payloads
    "version_meta": [],        # parallel list of metadata dicts
    "cur_index": -1,
    # kept for backward compat with old getters:
    "df_payload": None,        # alias of current
    "base_df_payload": None,   # alias of versions[0]
    "sota_bundled": None,
    "df_summary": None,
}

_SOTA_BUNDLED_CV: ContextVar[Optional[list]] = ContextVar("SOTA_BUNDLED", default=None)
_DF_SUMMARY_CV: ContextVar[Optional[Dict[str, Any]]] = ContextVar("DF_SUMMARY", default=None)


# -------- internal helpers --------
def _get_versions() -> List[Dict[str, Any]]:
    return _VERSIONS_CV.get() or _STORE["versions"]

def _get_meta() -> List[Dict[str, Any]]:
    return _VERS_META_CV.get() or _STORE["version_meta"]

def _set_versions(vers: List[Dict[str, Any]], meta: List[Dict[str, Any]], cur: int) -> None:
    _VERSIONS_CV.set(vers)
    _VERS_META_CV.set(meta)
    _CUR_INDEX_CV.set(cur)
    _STORE["versions"] = vers
    _STORE["version_meta"] = meta
    _STORE["cur_index"] = cur
    # keep legacy aliases in sync
    _STORE["df_payload"] = vers[cur] if (0 <= cur < len(vers)) else None
    _STORE["base_df_payload"] = vers[0] if vers else None


# =========================
# Init / Set / Annotate
# =========================
def init_dataset(p: Optional[Dict[str, Any]]) -> None:
    """Initialize version stack with a single BASE version."""
    vers = [] if p is None else [p]
    meta = [] if p is None else [dict(tag="base")]
    cur = -1 if p is None else 0
    _set_versions(vers, meta, cur)

def set_df_payload(p: Optional[Dict[str, Any]], *, new_version: bool = True) -> None:
    """
    Set CURRENT dataset.
    - new_version=True: truncate any forward history and append p (like a new commit).
    - new_version=False: replace the current version in place (no new snapshot).
    """
    vers = list(_get_versions())
    meta = list(_get_meta())
    cur = _CUR_INDEX_CV.get() if _CUR_INDEX_CV.get() is not None else _STORE["cur_index"]

    if cur < 0 or not vers:
        # not initialized yet
        init_dataset(p)
        return

    if new_version:
        # drop any versions after current (no branching for simplicity)
        vers = vers[:cur + 1]
        meta = meta[:cur + 1]
        vers.append(p)
        meta.append({})
        cur = len(vers) - 1
    else:
        vers[cur] = p

    _set_versions(vers, meta, cur)

def annotate_current(**kv) -> None:
    """Attach metadata to the current version (e.g., step/params/stats)."""
    vers = list(_get_versions())
    meta = list(_get_meta())
    cur = _CUR_INDEX_CV.get() if _CUR_INDEX_CV.get() is not None else _STORE["cur_index"]
    if 0 <= cur < len(meta):
        meta[cur] = {**meta[cur], **kv}
        _set_versions(vers, meta, cur)


# =========================
# Getters / Navigation
# =========================
def _resolve_index(spec: Union[str, int, None]) -> int:
    vers = _get_versions()
    cur = _CUR_INDEX_CV.get() if _CUR_INDEX_CV.get() is not None else _STORE["cur_index"]
    if spec is None or spec == "current":
        return cur
    if spec == "base":
        return 0 if vers else -1
    if spec == "prev":
        return max(-1, cur - 1)
    if isinstance(spec, int):
        idx = spec if spec >= 0 else len(vers) + spec
        return idx
    # strings like "@-1", "@3"
    if isinstance(spec, str) and spec.startswith("@"):
        try:
            n = int(spec[1:])
        except Exception:
            return cur
        idx = n if n >= 0 else len(vers) + n
        return idx
    return cur

def get_df_payload(version: Union[str, int, None] = None) -> Optional[Dict[str, Any]]:
    """Return dataset payload for the requested version (default: current)."""
    vers = _get_versions()
    idx = _resolve_index(version)
    if 0 <= idx < len(vers):
        return vers[idx]
    # legacy fallback
    return _STORE["df_payload"]

def get_base_df_payload() -> Optional[Dict[str, Any]]:
    return get_df_payload("base")

def get_prev_df_payload() -> Optional[Dict[str, Any]]:
    return get_df_payload("prev")

def list_versions() -> Dict[str, Any]:
    """Lightweight overview for debugging/UI."""
    vers = _get_versions()
    meta = _get_meta()
    cur = _CUR_INDEX_CV.get() if _CUR_INDEX_CV.get() is not None else _STORE["cur_index"]
    return {
        "count": len(vers),
        "current_index": cur,
        "has_base": bool(vers),
        "meta": meta,  # [{tag:..., step:..., params:...}, ...]
    }

def reset_current_to(version: Union[str, int]) -> None:
    """Move the current pointer to a prior version (no deletion)."""
    vers = _get_versions()
    meta = _get_meta()
    idx = _resolve_index(version)
    if 0 <= idx < len(vers):
        _set_versions(vers, meta, idx)

def reset_current_to_base() -> None:
    reset_current_to("base")


# =========================
# SOTA / Summary passthrough
# =========================
def set_sota_bundled(b: Optional[list]) -> None:
    _SOTA_BUNDLED_CV.set(b)
    _STORE["sota_bundled"] = b

def get_sota_bundled() -> Optional[list]:
    return _SOTA_BUNDLED_CV.get() or _STORE["sota_bundled"]

def set_df_summary(s: Optional[Dict[str, Any]]) -> None:
    _DF_SUMMARY_CV.set(s)
    _STORE["df_summary"] = s

def get_df_summary() -> Optional[Dict[str, Any]]:
    return _DF_SUMMARY_CV.get() or _STORE["df_summary"]
