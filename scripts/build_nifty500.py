"""
scripts/build_nifty500.py
==========================
Fetches the current NIFTY 500 constituent list from NSE India and writes
one Yahoo Finance ticker per line to:

    data/universes/nifty500.txt

Run from the project root:
    python scripts/build_nifty500.py

Re-run whenever the index reconstitution happens (typically quarterly).
The pipeline reads this file at runtime — it does NOT call this script
automatically.  Build the file once, commit it to version control, and
refresh it on a schedule (cron / CI) as needed.

Data source
-----------
NSE India publishes a machine-readable CSV of all NIFTY 500 constituents at:
  https://archives.nseindia.com/content/indices/ind_nifty500list.csv

Columns used:  "Symbol"  (NSE ticker, e.g. "RELIANCE")
Yahoo Finance format: append ".NS" suffix  → "RELIANCE.NS"

Fallback strategy
-----------------
If NSE blocks the request (common for automated scrapers), the script
tries two backup sources in order:
  1. NSE India Indices page (different endpoint, same CSV schema)
  2. nsepython library (pip install nsepython) if installed
  3. Manual CSV path override via --csv argument

Usage
-----
    # Standard run
    python scripts/build_nifty500.py

    # Use a locally downloaded CSV (if NSE blocks automated requests)
    python scripts/build_nifty500.py --csv /path/to/ind_nifty500list.csv

    # Preview without writing (dry-run)
    python scripts/build_nifty500.py --dry-run

    # Custom output path
    python scripts/build_nifty500.py --output data/universes/nifty500_test.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# ── Project root on path (run from anywhere) ─────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# NSE India direct CSV download.
# This URL is stable but NSE occasionally requires a cookie obtained by first
# visiting the main website.  We replicate that below.
_NSE_HOME      = "https://www.nseindia.com"
_NSE_CSV_URL   = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
_NSE_BACKUP    = "https://nseindia.com/market-data/index-constituents"

# Realistic browser headers — NSE returns 403 on barebones requests
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer":         "https://www.nseindia.com",
    "DNT":             "1",
}

_DEFAULT_OUTPUT = ROOT / "data" / "universes" / "nifty500.txt"


# ─────────────────────────────────────────────────────────────────────────────
#  Fetchers  (tried in order; first success wins)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_via_nse_session(timeout: int = 15) -> pd.DataFrame:
    """
    Obtain a session cookie by visiting the NSE home page, then download
    the CSV.  NSE requires an active session to serve the download.
    """
    logger.info("Attempting NSE India (session cookie method) …")
    session = requests.Session()
    session.headers.update(_HEADERS)

    # Step 1 — visit home page to set session cookies
    try:
        home = session.get(_NSE_HOME, timeout=timeout)
        home.raise_for_status()
        logger.debug("NSE home page: %d  cookies=%s", home.status_code, dict(session.cookies))
        time.sleep(1)   # brief pause to appear human
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach NSE home page: {exc}") from exc

    # Step 2 — download CSV with active session
    resp = session.get(_NSE_CSV_URL, timeout=timeout)
    resp.raise_for_status()

    if "Symbol" not in resp.text[:200]:
        raise ValueError(f"Unexpected response — first 200 chars: {resp.text[:200]}")

    df = pd.read_csv(StringIO(resp.text))
    logger.info("NSE session method: %d rows fetched.", len(df))
    return df


def _fetch_via_nsepython() -> pd.DataFrame:
    """
    Use the nsepython library if installed.
    Install with:  pip install nsepython
    """
    try:
        from nsepython import nse_eq_symbols   # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("nsepython not installed (pip install nsepython)") from exc

    logger.info("Attempting via nsepython …")
    symbols = nse_eq_symbols()   # returns list of NSE tickers
    df = pd.DataFrame({"Symbol": symbols})
    logger.info("nsepython: %d symbols.", len(df))
    return df


def _fetch_from_local_csv(csv_path: Path) -> pd.DataFrame:
    """Load from a locally downloaded CSV (manual download fallback)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Local CSV not found: {csv_path}")
    logger.info("Loading from local CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Local CSV: %d rows.", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Normalisation
# ─────────────────────────────────────────────────────────────────────────────

# NSE tickers that have non-standard Yahoo Finance suffixes.
# Most NSE stocks use .NS; these are the exceptions.
_NSE_YAHOO_OVERRIDES: dict[str, str] = {
    # symbol → full Yahoo ticker
    # Add any known exceptions here as you discover them.
    # e.g. "M&M": "M&M.NS" is usually fine, but some tools mangle the &
    "M&M":   "M-M.NS",
    "M&MFIN":"M-MFIN.NS",
    "L&TFH": "L-TFH.NS",
}

def _to_yahoo_ticker(nse_symbol: str) -> str:
    """
    Convert a raw NSE ticker to Yahoo Finance format.

    Rules
    -----
    1. Apply known overrides first.
    2. Otherwise append ".NS".
    3. Strip leading/trailing whitespace.
    """
    s = nse_symbol.strip().upper()
    if s in _NSE_YAHOO_OVERRIDES:
        return _NSE_YAHOO_OVERRIDES[s]
    return f"{s}.NS"


def _normalise(df: pd.DataFrame) -> list[str]:
    """
    Extract and normalise the Symbol column.

    Handles common column name variants returned by different NSE sources.

    Returns
    -------
    list[str]
        Sorted list of Yahoo Finance tickers, deduplicated.
    """
    # Find the symbol column (case-insensitive, strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    col_map    = {c.lower(): c for c in df.columns}

    symbol_col = None
    for candidate in ["symbol", "symbols", "ticker", "nse_symbol", "scrip_code"]:
        if candidate in col_map:
            symbol_col = col_map[candidate]
            break

    if symbol_col is None:
        raise ValueError(
            f"Cannot find a symbol column.  Columns present: {list(df.columns)}"
        )

    raw_symbols = df[symbol_col].dropna().astype(str).str.strip().unique()
    tickers     = sorted({_to_yahoo_ticker(s) for s in raw_symbols if s})

    logger.info(
        "Normalised %d raw symbols → %d Yahoo tickers.",
        len(raw_symbols), len(tickers),
    )
    return tickers


# ─────────────────────────────────────────────────────────────────────────────
#  Writer
# ─────────────────────────────────────────────────────────────────────────────

def _write(tickers: list[str], output_path: Path, dry_run: bool) -> None:
    """Write tickers to a one-per-line text file."""
    if dry_run:
        print(f"\n[dry-run] Would write {len(tickers)} tickers to '{output_path}'")
        print(f"  First 10: {tickers[:10]}")
        print(f"  Last  10: {tickers[-10:]}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(tickers) + "\n", encoding="utf-8")
    logger.info("Wrote %d tickers → '%s'.", len(tickers), output_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def build(
    output_path: Path = _DEFAULT_OUTPUT,
    csv_path:    Path | None = None,
    dry_run:     bool = False,
    timeout:     int  = 15,
) -> list[str]:
    """
    Fetch NIFTY 500 constituents and write to *output_path*.

    Parameters
    ----------
    output_path :
        Destination file.
    csv_path :
        If set, load from this local CSV instead of fetching from NSE.
    dry_run :
        Print output without writing to disk.
    timeout :
        HTTP request timeout in seconds.

    Returns
    -------
    list[str]
        The Yahoo Finance tickers that were written.

    Raises
    ------
    RuntimeError
        When all fetch strategies are exhausted.
    """
    df: pd.DataFrame | None = None

    # Strategy 1: local CSV override (highest priority, always offline)
    if csv_path:
        df = _fetch_from_local_csv(csv_path)

    # Strategy 2: NSE India website (session cookie)
    if df is None:
        try:
            df = _fetch_via_nse_session(timeout=timeout)
        except Exception as exc:
            logger.warning("NSE session method failed: %s", exc)

    # Strategy 3: nsepython library
    if df is None:
        try:
            df = _fetch_via_nsepython()
        except Exception as exc:
            logger.warning("nsepython method failed: %s", exc)

    if df is None:
        raise RuntimeError(
            "All fetch strategies exhausted.\n"
            "Options:\n"
            "  1. Download ind_nifty500list.csv from https://www.nseindia.com "
            "and pass --csv /path/to/file.csv\n"
            "  2. pip install nsepython and re-run\n"
            "  3. Check your network / VPN settings"
        )

    tickers = _normalise(df)

    # Sanity check: NIFTY 500 should have ~500 stocks
    if len(tickers) < 400:
        logger.warning(
            "Only %d tickers found — expected ~500.  "
            "Check the source data quality.",
            len(tickers),
        )

    _write(tickers, output_path, dry_run)
    return tickers


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build NIFTY 500 universe file for the Algo Trading Platform.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help=f"Output file path (default: {_DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--csv", type=Path, default=None,
        metavar="PATH",
        help="Use a locally downloaded ind_nifty500list.csv instead of fetching",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print tickers without writing to disk",
    )
    p.add_argument(
        "--timeout", type=int, default=15,
        help="HTTP request timeout in seconds (default: 15)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        tickers = build(
            output_path = args.output,
            csv_path    = args.csv,
            dry_run     = args.dry_run,
            timeout     = args.timeout,
        )
        print(f"\n✓  {len(tickers)} NIFTY 500 tickers written to '{args.output}'")
        print(f"   Sample: {tickers[:5]} … {tickers[-3:]}")

    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)