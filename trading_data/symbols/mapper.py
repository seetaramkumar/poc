"""
trading_data/symbols/mapper.py
------------------------------
Translates canonical / user-supplied symbol names to provider-specific tickers
and enriches them with metadata (exchange, currency, asset class).

Design
------
* A static registry covers well-known indices and popular tickers.
* Unknown symbols fall back to a passthrough strategy – the raw symbol is
  forwarded to the provider unchanged (works well for Yahoo Finance tickers).
* A custom registry file (JSON) can be mounted at runtime for team-specific
  overrides without touching source code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from trading_data.models import AssetClass, Provider, SymbolInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static registry
# ---------------------------------------------------------------------------
# Structure:  canonical_key -> {provider -> ticker, meta fields}
# The canonical key is always UPPER-CASE and provider-agnostic.
_STATIC_REGISTRY: dict[str, dict] = {
    # ── Indices ──────────────────────────────────────────────────────────────
    "NIFTY50": {
        "description":  "Nifty 50 Index",
        "asset_class":  AssetClass.INDEX,
        "exchange":     "NSE",
        "currency":     "INR",
        Provider.YAHOO:   "^NSEI",
        Provider.ZERODHA: "NIFTY 50",
        Provider.POLYGON: None,          # not available
        Provider.IBKR:    "NIFTY50-NSE",
    },
    "BANKNIFTY": {
        "description":  "Nifty Bank Index",
        "asset_class":  AssetClass.INDEX,
        "exchange":     "NSE",
        "currency":     "INR",
        Provider.YAHOO:   "^NSEBANK",
        Provider.ZERODHA: "NIFTY BANK",
        Provider.POLYGON: None,
        Provider.IBKR:    "BANKNIFTY-NSE",
    },
    "SP500": {
        "description":  "S&P 500 Index",
        "asset_class":  AssetClass.INDEX,
        "exchange":     "NYSE",
        "currency":     "USD",
        Provider.YAHOO:   "^GSPC",
        Provider.ZERODHA: None,
        Provider.POLYGON: "SPX",
        Provider.IBKR:    "SPX",
    },
    "NASDAQ": {
        "description":  "NASDAQ Composite",
        "asset_class":  AssetClass.INDEX,
        "exchange":     "NASDAQ",
        "currency":     "USD",
        Provider.YAHOO:   "^IXIC",
        Provider.ZERODHA: None,
        Provider.POLYGON: "COMP",
        Provider.IBKR:    "COMP",
    },
    "DOW": {
        "description":  "Dow Jones Industrial Average",
        "asset_class":  AssetClass.INDEX,
        "exchange":     "NYSE",
        "currency":     "USD",
        Provider.YAHOO:   "^DJI",
        Provider.POLYGON: "INDU",
        Provider.IBKR:    "INDU",
    },
    "NIFTYIT": {
        "description":  "Nifty IT Index",
        "asset_class":  AssetClass.INDEX,
        "exchange":     "NSE",
        "currency":     "INR",
        Provider.YAHOO:   "^CNXIT",
        Provider.ZERODHA: "NIFTY IT",
    },

    # ── Indian equities (NSE) ────────────────────────────────────────────────
    "RELIANCE": {
        "description":  "Reliance Industries",
        "asset_class":  AssetClass.EQUITY,
        "exchange":     "NSE",
        "currency":     "INR",
        Provider.YAHOO:   "RELIANCE.NS",
        Provider.ZERODHA: "RELIANCE",
    },
    "TCS": {
        "description":  "Tata Consultancy Services",
        "asset_class":  AssetClass.EQUITY,
        "exchange":     "NSE",
        "currency":     "INR",
        Provider.YAHOO:   "TCS.NS",
        Provider.ZERODHA: "TCS",
    },
    "INFY": {
        "description":  "Infosys",
        "asset_class":  AssetClass.EQUITY,
        "exchange":     "NSE",
        "currency":     "INR",
        Provider.YAHOO:   "INFY.NS",
        Provider.ZERODHA: "INFY",
        Provider.POLYGON: "INFY",
        Provider.IBKR:    "INFY",
    },
    "HDFCBANK": {
        "description":  "HDFC Bank",
        "asset_class":  AssetClass.EQUITY,
        "exchange":     "NSE",
        "currency":     "INR",
        Provider.YAHOO:   "HDFCBANK.NS",
        Provider.ZERODHA: "HDFCBANK",
    },

    # ── US equities ──────────────────────────────────────────────────────────
    "AAPL": {
        "description":  "Apple Inc.",
        "asset_class":  AssetClass.EQUITY,
        "exchange":     "NASDAQ",
        "currency":     "USD",
        Provider.YAHOO:   "AAPL",
        Provider.POLYGON: "AAPL",
        Provider.IBKR:    "AAPL",
    },
    "MSFT": {
        "description":  "Microsoft Corporation",
        "asset_class":  AssetClass.EQUITY,
        "exchange":     "NASDAQ",
        "currency":     "USD",
        Provider.YAHOO:   "MSFT",
        Provider.POLYGON: "MSFT",
        Provider.IBKR:    "MSFT",
    },
    "GOOGL": {
        "description":  "Alphabet Inc.",
        "asset_class":  AssetClass.EQUITY,
        "exchange":     "NASDAQ",
        "currency":     "USD",
        Provider.YAHOO:   "GOOGL",
        Provider.POLYGON: "GOOGL",
        Provider.IBKR:    "GOOGL",
    },
}


class SymbolMapper:
    """
    Resolves user-facing symbols to provider-specific tickers.

    Parameters
    ----------
    custom_registry_path :
        Optional path to a JSON file with additional / override entries.
        The file must follow the same structure as ``_STATIC_REGISTRY``.

    Examples
    --------
    >>> mapper = SymbolMapper()
    >>> info = mapper.resolve("NIFTY50", Provider.YAHOO)
    >>> info.provider_symbol
    '^NSEI'
    >>> info.asset_class
    <AssetClass.INDEX: 'index'>
    """

    def __init__(self, custom_registry_path: Optional[str | Path] = None):
        self._registry: dict[str, dict] = dict(_STATIC_REGISTRY)
        if custom_registry_path:
            self._load_custom(Path(custom_registry_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, symbol: str, provider: Provider = Provider.YAHOO) -> SymbolInfo:
        """
        Resolve *symbol* to a :class:`~trading_data.models.SymbolInfo` for
        the requested *provider*.

        Unknown symbols are passed through unchanged (passthrough strategy).

        Parameters
        ----------
        symbol :
            User-supplied ticker.  Looked up case-insensitively.
        provider :
            Target data provider.

        Returns
        -------
        SymbolInfo
        """
        key = symbol.upper()
        entry = self._registry.get(key)

        if entry is None:
            # Passthrough: assume the raw symbol works for this provider
            logger.debug("Symbol '%s' not in registry – using passthrough.", symbol)
            return SymbolInfo(
                canonical=key,
                provider_symbol=symbol,
                asset_class=AssetClass.EQUITY,
                description=symbol,
            )

        provider_symbol = entry.get(provider, symbol)
        if provider_symbol is None:
            raise ValueError(
                f"Symbol '{symbol}' is not available for provider '{provider}'."
            )

        return SymbolInfo(
            canonical=key,
            provider_symbol=provider_symbol,
            asset_class=entry.get("asset_class", AssetClass.EQUITY),
            exchange=entry.get("exchange", ""),
            currency=entry.get("currency", "USD"),
            description=entry.get("description", symbol),
        )

    def list_symbols(self) -> list[str]:
        """Return all canonical symbols in the registry."""
        return sorted(self._registry.keys())

    def register(self, canonical: str, entry: dict) -> None:
        """
        Register or override a symbol at runtime.

        Parameters
        ----------
        canonical :
            Canonical key (will be upper-cased).
        entry :
            Dict matching the ``_STATIC_REGISTRY`` schema.
        """
        self._registry[canonical.upper()] = entry
        logger.debug("Registered symbol '%s'.", canonical.upper())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_custom(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Custom registry path '%s' not found – skipped.", path)
            return
        try:
            with path.open() as fh:
                custom: dict = json.load(fh)
            self._registry.update(
                {k.upper(): v for k, v in custom.items()}
            )
            logger.info("Loaded %d entries from custom registry '%s'.", len(custom), path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load custom registry '%s': %s", path, exc)
