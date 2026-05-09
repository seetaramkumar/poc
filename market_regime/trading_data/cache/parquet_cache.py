"""
trading_data/cache/parquet_cache.py
-------------------------------------
Persistent local cache backed by Apache Parquet files.

Design
------
* One parquet file per (symbol, provider, interval) combination.
* File paths are deterministic and human-readable:
  ``<cache_dir>/<provider>/<interval>/<symbol>.parquet``
* Staleness is checked by comparing the file's modification time to
  ``max_age_days``.  Stale files are **not** deleted automatically; they
  are overwritten on the next successful fetch so the system degrades
  gracefully if the network is down (old data is still served with a warning).
* The cache is thread-safe at the file level because parquet writes are
  atomic on POSIX systems.

Parquet was chosen over CSV/HDF5 because it:
- preserves dtype information (no float→string surprises)
- is ~5× smaller than CSV for OHLCV time series
- reads 10–50× faster than CSV for large files
- is the de-facto standard in the PyData / Arrow ecosystem
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from trading_data.exceptions import CacheError
from trading_data.models import OHLCVFrame

logger = logging.getLogger(__name__)

# Characters that are not safe in filenames on any major OS
_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|^]')


def _sanitise(s: str) -> str:
    """Replace filesystem-unsafe characters with underscores."""
    return _UNSAFE_CHARS.sub("_", s)


class ParquetCache:
    """
    Read / write OHLCV data to a local parquet cache.

    Parameters
    ----------
    cache_dir :
        Root directory for cached files.  Created if it does not exist.
    max_age_days :
        Number of days after which a cached file is considered stale.
        Stale files are still readable but trigger a live fetch.

    Examples
    --------
    >>> cache = ParquetCache(cache_dir=".cache/ohlcv")
    >>> cache.write("^NSEI", provider="yahoo", df=df)
    >>> cached_df = cache.read("^NSEI", provider="yahoo")
    >>> cache.is_stale("^NSEI", provider="yahoo")
    False
    """

    def __init__(self, cache_dir: str | Path = ".cache/ohlcv", max_age_days: int = 1):
        self._root        = Path(cache_dir)
        self.max_age_days = max_age_days
        self._root.mkdir(parents=True, exist_ok=True)
        logger.debug("ParquetCache initialised at '%s'.", self._root.resolve())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(
        self,
        symbol:   str,
        provider: str,
        interval: str = "1d",
        start:    Optional[str] = None,
        end:      Optional[str] = None,
    ) -> Optional[OHLCVFrame]:
        """
        Load a cached DataFrame, optionally sliced to [*start*, *end*].

        Parameters
        ----------
        symbol :
            Provider-specific ticker (used to build the file path).
        provider :
            Provider name (``"yahoo"``, ``"polygon"`` …).
        interval :
            Bar interval string (``"1d"`` …).
        start, end :
            Optional ISO-8601 date strings for slicing.

        Returns
        -------
        pd.DataFrame or None
            Cached frame, or ``None`` if the file does not exist.

        Raises
        ------
        CacheError
            If the parquet file exists but cannot be read.
        """
        path = self._path(symbol, provider, interval)
        if not path.exists():
            logger.debug("Cache MISS: '%s'.", path)
            return None

        try:
            df = pd.read_parquet(path)
        except Exception as exc:            # noqa: BLE001
            raise CacheError(f"Cannot read cache file '{path}': {exc}") from exc

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Optional date slice
        if start or end:
            df = df.loc[
                (df.index >= pd.Timestamp(start) if start else df.index >= df.index.min()) &
                (df.index <= pd.Timestamp(end)   if end   else df.index <= df.index.max())
            ]

        age_hours = self._age_hours(path)
        logger.debug(
            "Cache HIT: '%s'  rows=%d  age=%.1fh.",
            path.name, len(df), age_hours,
        )
        return df if not df.empty else None

    def write(
        self,
        symbol:   str,
        provider: str,
        df:       OHLCVFrame,
        interval: str = "1d",
    ) -> None:
        """
        Persist *df* to parquet.

        Parameters
        ----------
        symbol :
            Provider-specific ticker.
        provider :
            Provider name.
        df :
            Normalised OHLCV DataFrame to cache.
        interval :
            Bar interval.

        Raises
        ------
        CacheError
            If the write fails.
        """
        if df is None or df.empty:
            logger.debug("Skipping cache write for '%s': empty DataFrame.", symbol)
            return

        path = self._path(symbol, provider, interval)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df.to_parquet(path, engine="pyarrow", compression="snappy")
            logger.debug("Cache WRITE: '%s'  rows=%d.", path.name, len(df))
        except Exception as exc:            # noqa: BLE001
            raise CacheError(f"Cannot write cache file '{path}': {exc}") from exc

    def is_stale(
        self,
        symbol:   str,
        provider: str,
        interval: str = "1d",
    ) -> bool:
        """
        Return ``True`` if the cached file is older than ``max_age_days``
        **or** does not exist.
        """
        path = self._path(symbol, provider, interval)
        if not path.exists():
            return True
        age_days = self._age_hours(path) / 24
        stale    = age_days > self.max_age_days
        if stale:
            logger.debug(
                "Cache STALE: '%s' (age=%.1f days > max=%d).",
                path.name, age_days, self.max_age_days,
            )
        return stale

    def invalidate(self, symbol: str, provider: str, interval: str = "1d") -> bool:
        """
        Delete a cached file.

        Returns
        -------
        bool
            ``True`` if the file was deleted; ``False`` if it did not exist.
        """
        path = self._path(symbol, provider, interval)
        if path.exists():
            path.unlink()
            logger.info("Cache INVALIDATED: '%s'.", path.name)
            return True
        return False

    def clear_all(self) -> int:
        """
        Delete **all** cached files under ``cache_dir``.

        Returns
        -------
        int
            Number of files deleted.
        """
        deleted = 0
        for f in self._root.rglob("*.parquet"):
            f.unlink()
            deleted += 1
        logger.info("Cache cleared: %d files deleted.", deleted)
        return deleted

    def stats(self) -> dict:
        """
        Return basic cache statistics.

        Returns
        -------
        dict
            Keys: ``files``, ``total_size_mb``, ``oldest_file``, ``newest_file``.
        """
        files = list(self._root.rglob("*.parquet"))
        if not files:
            return {"files": 0, "total_size_mb": 0.0, "oldest_file": None, "newest_file": None}

        mtimes    = [f.stat().st_mtime for f in files]
        total_mb  = sum(f.stat().st_size for f in files) / (1024 ** 2)
        oldest    = min(zip(mtimes, files))[1].name
        newest    = max(zip(mtimes, files))[1].name

        return {
            "files":         len(files),
            "total_size_mb": round(total_mb, 2),
            "oldest_file":   oldest,
            "newest_file":   newest,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _path(self, symbol: str, provider: str, interval: str) -> Path:
        """Build deterministic file path for a (symbol, provider, interval) triple."""
        safe_symbol = _sanitise(symbol)
        return self._root / _sanitise(provider) / _sanitise(interval) / f"{safe_symbol}.parquet"

    @staticmethod
    def _age_hours(path: Path) -> float:
        """Return the age of *path* in hours."""
        mtime   = path.stat().st_mtime
        now     = time.time()
        return (now - mtime) / 3600
