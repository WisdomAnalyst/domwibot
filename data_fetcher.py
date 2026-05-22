"""
Market data fetcher -- OHLCV, order book, funding rates via ccxt.

Two exchange instances:
  _pub_exchange  -- no auth, always mainnet, used for historical OHLCV and prices
  _exchange      -- with API keys and testnet setting, used for orders and balance
This lets backtesting and paper trading pull years of real historical data
even when TESTNET=true.
"""

import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import ccxt
import pandas as pd
import numpy as np

from config import cfg
from models import Candle

logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self):
        self._pub_exchange = self._init_public_exchange()   # historical data
        self._exchange = self._init_trading_exchange()      # orders / balance
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_ts: Dict[str, float] = {}
        self._cache_ttl = 25

    # ─── Exchange init ────────────────────────────────────────────────────────

    def _init_public_exchange(self) -> ccxt.Exchange:
        """
        Unauthenticated mainnet exchange for public OHLCV.
        Always uses mainnet so backtests get full multi-year history.
        """
        ec = cfg.exchange
        exchange_class = getattr(ccxt, ec.name)
        exchange = exchange_class({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        try:
            exchange.load_markets()
            logger.info("Public data feed connected to %s (mainnet)", ec.name)
        except Exception as exc:
            logger.warning("Public exchange load_markets failed: %s", exc)
        return exchange

    def _init_trading_exchange(self) -> ccxt.Exchange:
        """
        Authenticated exchange for live order placement and balance checks.
        Respects testnet setting.
        """
        ec = cfg.exchange
        exchange_class = getattr(ccxt, ec.name)
        params: dict = {
            "apiKey": ec.api_key,
            "secret": ec.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future" if ec.use_futures else "spot"},
        }
        exchange = exchange_class(params)
        if ec.testnet:
            if hasattr(exchange, "set_sandbox_mode"):
                exchange.set_sandbox_mode(True)
            elif "test" in exchange.urls:
                exchange.urls["api"] = exchange.urls["test"]
        try:
            exchange.load_markets()
            logger.info("Trading exchange connected to %s (testnet=%s)", ec.name, ec.testnet)
        except Exception as exc:
            logger.warning("Trading exchange load_markets failed: %s", exc)
        return exchange

    # ─── OHLCV (uses public mainnet exchange) ────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: Optional[str] = None,
        timeframe: str = "1h",
        limit: int = 300,
        since: Optional[int] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        symbol = symbol or cfg.exchange.symbol
        cache_key = f"{symbol}_{timeframe}"
        now = time.time()

        if use_cache and cache_key in self._cache:
            if now - self._cache_ts.get(cache_key, 0) < self._cache_ttl:
                return self._cache[cache_key]

        try:
            raw = self._pub_exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=limit
            )
        except ccxt.NetworkError as exc:
            logger.error("Network error fetching %s %s: %s", symbol, timeframe, exc)
            return self._cache.get(cache_key, pd.DataFrame())
        except ccxt.ExchangeError as exc:
            logger.error("Exchange error: %s", exc)
            return self._cache.get(cache_key, pd.DataFrame())

        if not raw:
            return self._cache.get(cache_key, pd.DataFrame())

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.astype({c: float for c in ["open", "high", "low", "close", "volume"]})
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        self._cache[cache_key] = df
        self._cache_ts[cache_key] = now
        return df

    def fetch_ohlcv_range(
        self,
        symbol: Optional[str] = None,
        timeframe: str = "1h",
        start_date: str = "2023-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """
        Fetch full date range by paginating Binance in 500-bar chunks.
        Used by the backtester to get multi-year history.
        """
        symbol = symbol or cfg.exchange.symbol
        start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)

        all_bars = []
        since = start_ms
        limit = 500

        logger.info("Downloading %s %s from %s to %s...", symbol, timeframe, start_date, end_date)

        while since < end_ms:
            try:
                bars = self._pub_exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit
                )
            except Exception as exc:
                logger.error("Paginated fetch error: %s", exc)
                break

            if not bars:
                break

            all_bars.extend(bars)
            since = bars[-1][0] + 1

            # Respect Binance rate limit
            time.sleep(self._pub_exchange.rateLimit / 1000)

            if bars[-1][0] >= end_ms:
                break

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.astype({c: float for c in ["open", "high", "low", "close", "volume"]})
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]

        end_ts = pd.Timestamp(end_date, tz="UTC")
        df = df[df.index <= end_ts]

        logger.info("Downloaded %d bars for %s %s", len(df), symbol, timeframe)
        return df

    def fetch_multi_tf(self, timeframes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        tfs = timeframes or [
            cfg.timeframes.trend_tf,
            cfg.timeframes.structure_tf,
            cfg.timeframes.entry_tf,
        ]
        return {tf: self.fetch_ohlcv(timeframe=tf, limit=cfg.timeframes.candles_required) for tf in tfs}

    def fetch_multi_tf_historical(self) -> Dict[str, pd.DataFrame]:
        """Full historical range fetch for backtesting."""
        bc = cfg.backtest
        tfs = [cfg.timeframes.trend_tf, cfg.timeframes.structure_tf, cfg.timeframes.entry_tf]
        return {
            tf: self.fetch_ohlcv_range(
                timeframe=tf,
                start_date=bc.start_date,
                end_date=bc.end_date,
            )
            for tf in tfs
        }

    # ─── Price and market data ────────────────────────────────────────────────

    def get_current_price(self, symbol: Optional[str] = None) -> float:
        symbol = symbol or cfg.exchange.symbol
        try:
            ticker = self._pub_exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as exc:
            logger.warning("Price fetch failed: %s", exc)
            return 0.0

    def fetch_orderbook(self, symbol: Optional[str] = None, depth: int = 20) -> dict:
        symbol = symbol or cfg.exchange.symbol
        try:
            return self._pub_exchange.fetch_order_book(symbol, limit=depth)
        except Exception as exc:
            logger.warning("Orderbook fetch failed: %s", exc)
            return {"bids": [], "asks": []}

    def get_bid_ask_spread(self, symbol: Optional[str] = None) -> Tuple[float, float, float]:
        ob = self.fetch_orderbook(symbol)
        best_bid = ob["bids"][0][0] if ob["bids"] else 0.0
        best_ask = ob["asks"][0][0] if ob["asks"] else 0.0
        return best_bid, best_ask, best_ask - best_bid

    def fetch_funding_rate(self, symbol: Optional[str] = None) -> float:
        if not cfg.exchange.use_futures:
            return 0.0
        symbol = symbol or cfg.exchange.futures_symbol
        try:
            info = self._exchange.fetch_funding_rate(symbol)
            return float(info.get("fundingRate", 0.0))
        except Exception as exc:
            logger.debug("Funding rate unavailable: %s", exc)
            return 0.0

    # ─── Account / orders (authenticated, testnet-aware) ─────────────────────

    def get_balance(self) -> float:
        try:
            balance = self._exchange.fetch_balance()
            return float(balance["USDT"]["free"])
        except Exception as exc:
            logger.warning("Balance fetch failed: %s", exc)
            return cfg.risk.account_balance_usdt

    def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[dict]:
        try:
            order = self._exchange.create_limit_order(symbol, side, amount, price)
            logger.info("Limit %s order: %.4f @ %.4f", side, amount, price)
            return order
        except Exception as exc:
            logger.error("Limit order failed: %s", exc)
            return None

    def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[dict]:
        try:
            order = self._exchange.create_market_order(symbol, side, amount)
            logger.info("Market %s order: %.4f", side, amount)
            return order
        except Exception as exc:
            logger.error("Market order failed: %s", exc)
            return None

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        symbol = symbol or cfg.exchange.symbol
        try:
            self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as exc:
            logger.error("Cancel order failed: %s", exc)
            return False

    @staticmethod
    def df_to_candles(df: pd.DataFrame, timeframe: str = "") -> List[Candle]:
        candles = []
        for ts, row in df.iterrows():
            candles.append(Candle(
                timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=float(row["volume"]), timeframe=timeframe,
            ))
        return candles
