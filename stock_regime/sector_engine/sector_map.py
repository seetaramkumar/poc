"""
stock_regime/sector_engine/sector_map.py
==========================================
Symbol → sector resolution.

Resolution priority
-------------------
1. Loaded from a CSV file at:
       data/sectors/<universe>_sectors.csv
   Expected columns: symbol, sector [, industry, market_cap_category]

2. Inline override dict passed at construction time
   (useful for tests and one-off overrides without touching CSV).

3. UNKNOWN (graceful degradation — the engine never fails on missing data)

Design notes
------------
- This class has NO dependency on any engine or indicator module.
- It is a pure lookup table with a well-defined loading contract.
- The CSV schema is intentionally minimal so it works with any data source.
- Sectors are normalised to UPPER_CASE with spaces → underscores so
  "Information Technology" and "INFORMATION_TECHNOLOGY" map to the same key.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_UNKNOWN = "UNKNOWN"


class SectorMap:
    """
    Immutable symbol → sector lookup table.

    Parameters
    ----------
    csv_path :
        Path to a CSV with at minimum columns: symbol, sector.
        Optional columns: industry, market_cap_category.
    inline_overrides :
        Dict of {symbol: sector} applied on top of the CSV.
        Useful for single-stock overrides without editing CSV files.
    normalise_sectors :
        When True, convert sector names to UPPER_SNAKE_CASE for consistency.
    """

    def __init__(
        self,
        csv_path:          Optional[str | Path] = None,
        inline_overrides:  Optional[dict[str, str]] = None,
        normalise_sectors: bool = True,
    ) -> None:
        self._normalise = normalise_sectors
        self._map: dict[str, str] = {}            # symbol → sector
        self._industry_map: dict[str, str] = {}   # symbol → industry (optional)

        if csv_path is not None:
            self._load_csv(Path(csv_path))

        if inline_overrides:
            for sym, sec in inline_overrides.items():
                self._map[sym.strip().upper()] = self._norm(sec)
            logger.debug("Applied %d inline sector overrides.", len(inline_overrides))

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def get_sector(self, symbol: str) -> str:
        """Return the sector for symbol, or UNKNOWN."""
        return self._map.get(symbol.strip().upper(), _UNKNOWN)

    def get_industry(self, symbol: str) -> str:
        """Return the industry for symbol, or UNKNOWN."""
        return self._industry_map.get(symbol.strip().upper(), _UNKNOWN)

    def get_all_sectors(self) -> list[str]:
        """Return sorted list of all known sectors (excluding UNKNOWN)."""
        return sorted(set(self._map.values()) - {_UNKNOWN})

    def symbols_in_sector(self, sector: str) -> list[str]:
        """Return all symbols mapped to a given sector."""
        target = self._norm(sector)
        return [sym for sym, sec in self._map.items() if sec == target]

    def coverage(self) -> dict:
        """Return mapping statistics for diagnostics."""
        return {
            "total_mapped":    len(self._map),
            "unique_sectors":  len(self.get_all_sectors()),
            "has_industry":    len(self._industry_map),
        }

    def __len__(self) -> int:
        return len(self._map)

    # ──────────────────────────────────────────────────────────────
    #  Factory methods
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def from_project_root(
        cls,
        project_root: str | Path,
        universe:     str,
        **kwargs,
    ) -> "SectorMap":
        """
        Convenience factory: looks for data/sectors/<universe>_sectors.csv
        relative to project_root.

        Always succeeds — returns an empty map if the file is absent.
        """
        path = Path(project_root) / "data" / "sectors" / f"{universe.lower()}_sectors.csv"
        if not path.exists():
            logger.info(
                "No sector map found at '%s'. "
                "All symbols will be mapped to UNKNOWN. "
                "Create the CSV to enable sector intelligence.",
                path,
            )
            return cls(**kwargs)
        return cls(csv_path=path, **kwargs)

    # ──────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────

    def _load_csv(self, path: Path) -> None:
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]

            if "symbol" not in df.columns:
                logger.error("Sector CSV '%s' missing 'symbol' column — skipped.", path)
                return
            if "sector" not in df.columns:
                logger.error("Sector CSV '%s' missing 'sector' column — skipped.", path)
                return

            # Drop rows with null symbol or sector
            df = df.dropna(subset=["symbol", "sector"])

            for _, row in df.iterrows():
                sym = str(row["symbol"]).strip().upper()
                sec = self._norm(str(row["sector"]))
                self._map[sym] = sec
                if "industry" in df.columns and pd.notna(row.get("industry")):
                    self._industry_map[sym] = self._norm(str(row["industry"]))

            logger.info(
                "SectorMap: loaded %d symbols across %d sectors from '%s'.",
                len(self._map),
                len(self.get_all_sectors()),
                path,
            )
        except Exception as exc:
            logger.warning("Could not load sector CSV '%s': %s", path, exc)

    def _norm(self, name: str) -> str:
        """Normalise sector/industry name to UPPER_SNAKE_CASE."""
        if not self._normalise:
            return name.strip()
        return (
            name.strip()
                .upper()
                .replace(" & ", "_AND_")
                .replace(" / ", "_")
                .replace("/", "_")
                .replace(" ", "_")
                .replace("-", "_")
                .replace("__", "_")
        )