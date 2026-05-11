"""
trading_data/providers/zerodha.py
----------------------------------
Zerodha Kite Connect provider (stub / skeleton).

Status
------
**Not yet implemented.**  This file defines the full interface contract so
that swapping in the real implementation requires zero changes to the
DataManager or any calling code.

Implementation notes (for future maintainers)
---------------------------------------------
1. Install: ``pip install kiteconnect``
2. Authenticate with ``KiteConnect(api_key=...).generate_session(request_token, api_secret)``
3. Fetch historical data via ``kite.historical_data(instrument_token, from_date, to_date, interval)``
4. Zerodha requires ``instrument_token`` (an integer), not a ticker string.
   The SymbolMapper provides the mapping via ``Provider.ZERODHA``.
5. Rate limits: 3 requests/second for historical data.
6. The Kite API returns a list of dicts; convert to DataFrame and call
   ``self.normalise()`` before returning.
"""

from __future__ import annotations

import logging

from trading_data.exceptions import ConfigurationError
from trading_data.models import DateLike, OHLCVFrame
from trading_data.providers.base import BaseDataProvider

logger = logging.getLogger(__name__)


class ZerodhaProvider(BaseDataProvider):
    """
    Zerodha Kite Connect data provider — **stub implementation**.

    Parameters
    ----------
    api_key :
        Kite Connect API key.
    access_token :
        Session access token obtained after OAuth login.
    timeout :
        Socket timeout (seconds).

    Raises
    ------
    ConfigurationError
        If ``api_key`` or ``access_token`` is not provided.
    NotImplementedError
        Raised by all public methods until fully implemented.
    """

    def __init__(
        self,
        api_key:      str = "",
        access_token: str = "",
        timeout:      int = 30,
    ):
        super().__init__(timeout=timeout)
        if not api_key or not access_token:
            raise ConfigurationError(
                self.name,
                "Both 'api_key' and 'access_token' must be provided.",
            )
        self._api_key      = api_key
        self._access_token = access_token
        # Uncomment when kiteconnect is installed:
        # from kiteconnect import KiteConnect
        # self._kite = KiteConnect(api_key=api_key)
        # self._kite.set_access_token(access_token)

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "zerodha"

    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol:   str,
        start:    DateLike,
        end:      DateLike,
        interval: str = "1d",
    ) -> OHLCVFrame:
        """
        Fetch OHLCV data from Zerodha Kite.

        .. todo::
            Implement using ``kite.historical_data()``:

            .. code-block:: python

                from datetime import datetime
                data = self._kite.historical_data(
                    instrument_token=int(symbol),   # symbol is an int token
                    from_date=self._to_date_str(start),
                    to_date=self._to_date_str(end),
                    interval="day",
                )
                df = pd.DataFrame(data)
                df.rename(columns={"date": "date", "open": "open", ...}, inplace=True)
                df.set_index("date", inplace=True)
                return self.normalise(df, symbol=symbol, provider=self.name)
        """
        raise NotImplementedError(
            "ZerodhaProvider.fetch_ohlcv() is not yet implemented. "
            "See module docstring for implementation notes."
        )

    def validate_connection(self) -> bool:
        """
        .. todo:: Call ``kite.profile()`` and return ``True`` on success.
        """
        raise NotImplementedError("ZerodhaProvider.validate_connection() is not yet implemented.")
