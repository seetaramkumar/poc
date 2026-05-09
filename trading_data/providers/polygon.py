"""
trading_data/providers/polygon.py
----------------------------------
Polygon.io data provider (stub / skeleton).

Status
------
**Not yet implemented.**

Implementation notes (for future maintainers)
---------------------------------------------
1. Install: ``pip install polygon-api-client``
2. Authenticate with an API key from https://polygon.io
3. Fetch via ``RESTClient(api_key).get_aggs(ticker, multiplier, timespan, from_, to)``
4. Polygon returns paginated results; the client handles pagination automatically.
5. Free-tier data is delayed 15 min; Starter+ tiers give real-time access.
6. Polygon covers US equities, options, forex, crypto (not Indian markets).
7. Use ``"day"`` as ``timespan`` for daily OHLCV.
8. Each bar dict keys: ``open, high, low, close, volume, timestamp`` (ms epoch).
   Convert ``timestamp`` (ms) → DatetimeIndex before normalising.
"""

from __future__ import annotations

import logging

from trading_data.exceptions import ConfigurationError
from trading_data.models import DateLike, OHLCVFrame
from trading_data.providers.base import BaseDataProvider

logger = logging.getLogger(__name__)


class PolygonProvider(BaseDataProvider):
    """
    Polygon.io data provider — **stub implementation**.

    Parameters
    ----------
    api_key :
        Polygon.io REST API key.
    timeout :
        Socket timeout (seconds).

    Raises
    ------
    ConfigurationError
        If ``api_key`` is not provided.
    NotImplementedError
        Raised by all public methods until fully implemented.
    """

    def __init__(self, api_key: str = "", timeout: int = 30):
        super().__init__(timeout=timeout)
        if not api_key:
            raise ConfigurationError(self.name, "'api_key' must be provided.")
        self._api_key = api_key
        # Uncomment when polygon-api-client is installed:
        # from polygon import RESTClient
        # self._client = RESTClient(api_key=api_key)

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "polygon"

    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol:   str,
        start:    DateLike,
        end:      DateLike,
        interval: str = "1d",
    ) -> OHLCVFrame:
        """
        Fetch OHLCV data from Polygon.io.

        .. todo::
            Implement using the Polygon REST client:

            .. code-block:: python

                aggs = self._client.get_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan="day",
                    from_=self._to_date_str(start),
                    to=self._to_date_str(end),
                )
                import pandas as pd
                df = pd.DataFrame([a.__dict__ for a in aggs])
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("date", inplace=True)
                df.rename(columns={"vw": "vwap", "n": "trades"}, inplace=True)
                return self.normalise(df, symbol=symbol, provider=self.name)
        """
        raise NotImplementedError(
            "PolygonProvider.fetch_ohlcv() is not yet implemented. "
            "See module docstring for implementation notes."
        )

    def validate_connection(self) -> bool:
        """
        .. todo:: Call ``client.get_ticker_details("AAPL")`` and return ``True`` on success.
        """
        raise NotImplementedError("PolygonProvider.validate_connection() is not yet implemented.")
