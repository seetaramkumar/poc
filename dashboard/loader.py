from __future__ import annotations

import fnmatch
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.config import resolve_output_root


@st.cache_data(show_spinner=False, ttl=60)
def discover_output_root() -> Path:
    return resolve_output_root()


@st.cache_data(show_spinner=False, ttl=60)
def discover_run_dates(output_root: str | Path | None = None) -> list[str]:
    root = Path(output_root or discover_output_root())
    if not root.exists():
        return []

    dates: set[str] = set()
    for path in root.rglob("*.parquet"):
        try:
            parts = path.parts
            for part in parts:
                if len(part) == 10 and part[4] == "-" and part[7] == "-":
                    dates.add(part)
        except Exception:
            continue
    return sorted(dates, reverse=True)


@st.cache_data(show_spinner=False, ttl=60)
def load_parquet_file(path: str | Path) -> pd.DataFrame:
    candidate = Path(path)
    if not candidate.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(candidate)
    except Exception:
        return pd.DataFrame()


def resolve_parquet_path(
    output_root: str | Path,
    run_date: str,
    category_candidates: Iterable[str],
    filename_patterns: Iterable[str],
) -> Path | None:
    root = Path(output_root)
    category_names = {c.lower() for c in category_candidates}
    filename_globs = [pattern.lower() for pattern in filename_patterns]

    candidate_roots = []
    for category in category_names:
        candidate_roots.append(root / category / run_date)
        candidate_roots.append(root / run_date / category)

    candidates: list[Path] = []
    for base in candidate_roots:
        if not base.exists():
            continue
        for path in base.rglob("*.parquet"):
            if any(fnmatch.fnmatch(path.name.lower(), pattern) for pattern in filename_globs):
                candidates.append(path)

    if not candidates:
        for path in root.rglob("*.parquet"):
            if run_date not in path.parts:
                continue
            parts = [p.lower() for p in path.parts]
            if not any(part in category_names for part in parts):
                continue
            if any(fnmatch.fnmatch(path.name.lower(), pattern) for pattern in filename_globs):
                candidates.append(path)

    if not candidates:
        return None
    candidates.sort(key=lambda item: (len(item.parts), str(item)))
    return candidates[0]


@st.cache_data(show_spinner=False, ttl=60)
def load_dataset(
    output_root: str | Path,
    run_date: str,
    category_candidates: Iterable[str],
    filename_patterns: Iterable[str],
) -> pd.DataFrame:
    path = resolve_parquet_path(output_root, run_date, category_candidates, filename_patterns)
    if path is None:
        return pd.DataFrame()
    return load_parquet_file(path)
