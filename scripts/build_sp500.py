"""
scripts/build_sp500.py
=======================
Fetches the current S&P 500 constituent list and writes one Yahoo Finance
ticker per line to:

    data/universes/sp500.txt

Run from the project root:
    python scripts/build_sp500.py

Re-run whenever the index rebalancing happens (typically quarterly).

Data sources (tried in order)
------------------------------
1. Wikipedia  — "List of S&P 500 companies" HTML table
   URL: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
   Column: "Symbol" (e.g. "AAPL", "BRK.B")

2. DataHub.io — machine-readable CSV maintained by the Open Data community
   URL: https://datahub.io/core/s-and-p-500-companies/r/constituents.csv

3. Local CSV override via --csv argument

Ticker normalisation
--------------------
S&P 500 tickers on Wikipedia use "." as a class separator (BRK.B, BF.B).
Yahoo Finance uses "-" for the same (BRK-B, BF-B).  This is handled
automatically.

Usage
-----
    # Standard run
    python scripts/build_sp500.py

    # Use a locally downloaded CSV
    python scripts/build_sp500.py --csv /path/to/sp500.csv

    # Preview without writing
    python scripts/build_sp500.py --dry-run

    # Custom output path
    python scripts/build_sp500.py --output data/universes/sp500_test.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

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

_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# DataHub.io hosts a regularly maintained CSV of S&P 500 constituents.
# This is the most machine-friendly source.
_DATAHUB_URL = (
    "https://pkgstore.datahub.io/core/s-and-p-500-companies/"
    "constituents_csv/data/f2da8a43f7e7d4a3a59f1640e0c09c39/constituents_csv.csv"
)

# Realistic browser headers to avoid bot-detection 403s
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_DEFAULT_OUTPUT = ROOT / "data" / "universes" / "sp500.txt"


# ─────────────────────────────────────────────────────────────────────────────
#  Fetchers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_via_wikipedia(timeout: int = 15) -> pd.DataFrame:
    """
    Parse the S&P 500 constituents table from Wikipedia.

    Wikipedia's "List of S&P 500 companies" page has a reliably structured
    HTML table with a "Symbol" column.  pandas.read_html() handles the
    parsing automatically.
    """
    logger.info("Attempting Wikipedia …")
    resp = requests.get(_WIKIPEDIA_URL, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()

    # read_html returns a list; the first table is the constituents.
    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise ValueError("No tables found on Wikipedia page.")

    # The constituents table has a "Symbol" column — find it.
    df = None
    for table in tables:
        if "Symbol" in table.columns:
            df = table
            break

    if df is None:
        raise ValueError(
            f"No table with a 'Symbol' column found.  "
            f"Columns in first table: {list(tables[0].columns)}"
        )

    logger.info("Wikipedia: %d rows.", len(df))
    return df


def _fetch_via_datahub(timeout: int = 15) -> pd.DataFrame:
    """Fetch from DataHub.io CSV (maintained by the Open Data community)."""
    logger.info("Attempting DataHub.io …")
    resp = requests.get(_DATAHUB_URL, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    logger.info("DataHub: %d rows.", len(df))
    return df


def _fetch_from_local_csv(csv_path: Path) -> pd.DataFrame:
    """Load from a locally downloaded CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Local CSV not found: {csv_path}")
    logger.info("Loading from local CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Local CSV: %d rows.", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Normalisation
# ─────────────────────────────────────────────────────────────────────────────

# Wikipedia uses "." as separator (BRK.B), Yahoo Finance uses "-" (BRK-B).
# These are the known cases as of 2024.
_YAHOO_OVERRIDES: dict[str, str] = {
    "BRK.B": "BRK-B",
    "BF.B":  "BF-B",
    "BRK.A": "BRK-A",
    "BF.A":  "BF-A",
}

# Symbols to exclude entirely (e.g. SPACs, rights, warrants that appear briefly)
_EXCLUDE: set[str] = set()


def _to_yahoo_ticker(raw: str) -> str:
    """
    Normalise a raw S&P 500 ticker to Yahoo Finance format.

    1. Apply known overrides.
    2. Replace any remaining "." with "-" for Yahoo compatibility.
    3. Strip whitespace.
    """
    s = raw.strip().upper()
    if s in _YAHOO_OVERRIDES:
        return _YAHOO_OVERRIDES[s]
    # Generic rule: "." in the middle of a ticker means a share class
    return s.replace(".", "-")


def _normalise(df: pd.DataFrame) -> list[str]:
    """
    Extract and normalise the Symbol column from the raw DataFrame.

    Handles column name variants from different sources.

    Returns
    -------
    list[str]
        Sorted list of Yahoo Finance tickers, deduplicated.
    """
    df.columns = [c.strip() for c in df.columns]
    col_map    = {c.lower(): c for c in df.columns}

    symbol_col = None
    for candidate in ["symbol", "ticker", "symbols", "ticker symbol"]:
        if candidate in col_map:
            symbol_col = col_map[candidate]
            break

    if symbol_col is None:
        raise ValueError(
            f"Cannot find symbol column. Columns: {list(df.columns)}"
        )

    raw_symbols = df[symbol_col].dropna().astype(str).str.strip().unique()
    tickers     = sorted({
        _to_yahoo_ticker(s)
        for s in raw_symbols
        if s and s.upper() not in _EXCLUDE
    })

    logger.info(
        "Normalised %d raw symbols → %d Yahoo tickers.",
        len(raw_symbols), len(tickers),
    )
    return tickers


# ─────────────────────────────────────────────────────────────────────────────
#  Writer
# ─────────────────────────────────────────────────────────────────────────────

def _write(tickers: list[str], output_path: Path, dry_run: bool) -> None:
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
    Fetch S&P 500 constituents and write to *output_path*.

    Parameters
    ----------
    output_path :
        Destination file.
    csv_path :
        If set, load from this local CSV instead of fetching.
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

    if csv_path:
        df = _fetch_from_local_csv(csv_path)

    if df is None:
        try:
            df = _fetch_via_wikipedia(timeout=timeout)
        except Exception as exc:
            logger.warning("Wikipedia method failed: %s", exc)

    if df is None:
        try:
            df = _fetch_via_datahub(timeout=timeout)
        except Exception as exc:
            logger.warning("DataHub method failed: %s", exc)

    if df is None:
        raise RuntimeError(
            "All fetch strategies exhausted.\n"
            "Options:\n"
            "  1. Download constituents.csv from Wikipedia or DataHub and pass "
            "--csv /path/to/file.csv\n"
            "  2. Check your network / VPN settings"
        )

    tickers = _normalise(df)

    if len(tickers) < 480:
        logger.warning(
            "Only %d tickers found — expected ~503.  "
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
        description="Build S&P 500 universe file for the Algo Trading Platform.",
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
        help="Use a locally downloaded CSV instead of fetching from the web",
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
        print(f"\n✓  {len(tickers)} S&P 500 tickers written to '{args.output}'")
        print(f"   Sample: {tickers[:5]} … {tickers[-3:]}")

    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)