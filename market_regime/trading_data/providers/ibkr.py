"""
trading_data/providers/ibkr.py
--------------------------------
Interactive Brokers TWS / IB Gateway provider (stub / skeleton).

Status
------
**Not yet implemented.**

Implementation notes (for future maintainers)
---------------------------------------------
1. Options:
   a. ``ib_insync`` (async-friendly, simpler API):
      ``pip install ib_insync``
   b. ``ibapi`` (official, more verbose):
      ``pip install ibapi``
2. TWS or IB Gateway must be running locally with the API port enabled.
3. Connect: ``ib = IB(); ib.connect("127.0.0.1", 7497, clientId=1)``
4. Request historical bars:
   ``ib.reqHistoricalData(contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH)``
5. IB returns a list of ``BarData`` objects; convert to DataFrame.
6. ``whatToShow = "TRADES"`` for OHLCV; ``"MIDPOINT"`` for forex.
7. ``barSizeSetting = "1 day"`` for daily data.
8. IB pacing rules: ≤ 60 requests / 10 min; ≤ 2000 bars per request.
9. Requires a funded or paper account; market-data subscriptions may be needed.
"""

from __future__ import annotations

import logging

from trading_data.exceptions import ConfigurationError
from trading_data.models import DateLike, OHLCVFrame
from trading_data.providers.base import BaseDataProvider

logger = logging.getLogger(__name__)


class IBKRProvider(BaseDataProvider):
    """
    Interactive Brokers TWS data provider — **stub implementation**.

    Parameters
    ----------
    host :
        TWS / IB Gateway host (default ``"127.0.0.1"``).
    port :
        API port (paper: ``7497``, live: ``7496``).
    client_id :
        Unique client ID for this connection.
    timeout :
        Socket timeout (seconds).

    Raises
    ------
    ConfigurationError
        If connection parameters are invalid.
    NotImplementedError
        Raised by all public methods until fully implemented.
    """

    def __init__(
        self,
        host:      str = "127.0.0.1",
        port:      int = 7497,
        client_id: int = 1,
        timeout:   int = 30,
    ):
        super().__init__(timeout=timeout)
        self._host      = host
        self._port      = port
        self._client_id = client_id
        # Uncomment when ib_insync is installed:
        # from ib_insync import IB
        # self._ib = IB()

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "ibkr"

    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol:   str,
        start:    DateLike,
        end:      DateLike,
        interval: str = "1d",
    ) -> OHLCVFrame:
        """
        Fetch OHLCV data from Interactive Brokers TWS.

        .. todo::
            Implement using ``ib_insync``:

            .. code-block:: python

                from ib_insync import Stock, Index
                contract = Stock(symbol, "SMART", "USD")
                self._ib.connect(self._host, self._port, self._client_id)
                bars = self._ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr="5 Y",
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                )
                import pandas as pd
                df = pd.DataFrame([b.__dict__ for b in bars])
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                return self.normalise(df, symbol=symbol, provider=self.name)
        """
        raise NotImplementedError(
            "IBKRProvider.fetch_ohlcv() is not yet implemented. "
            "See module docstring for implementation notes."
        )

    def validate_connection(self) -> bool:
        """
        .. todo:: Call ``ib.connect()`` and return ``ib.isConnected()``.
        """
        raise NotImplementedError("IBKRProvider.validate_connection() is not yet implemented.")
