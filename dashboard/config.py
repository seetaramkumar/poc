from __future__ import annotations

from pathlib import Path
from typing import Final

ROOT_CANDIDATES: Final[tuple[Path, ...]] = (
    Path("output"),
    Path("runner/output"),
    Path(__file__).resolve().parents[1] / "output",
    Path(__file__).resolve().parents[1] / "runner" / "output",
)


def resolve_output_root() -> Path:
    for candidate in ROOT_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    return ROOT_CANDIDATES[0].resolve()
