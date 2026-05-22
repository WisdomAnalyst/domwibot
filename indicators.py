"""
Pure-function technical indicators that operate on pandas DataFrames.
All functions expect a DataFrame with columns: open, high, low, close, volume.
All return new columns or Series — nothing mutates in place.
"""

import numpy as np
import pandas as pd
from typing import Tuple

from config import cfg


# ─── Trend indicators ────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sc = cfg.strategy
    df["ema_fast"] = ema(df["close"], sc.ema_fast)
    df["ema_slow"] = ema(df["close"], sc.ema_slow)
    df["ema_trend"] = ema(df["close"], sc.ema_trend)
    return df


# ─── Momentum ────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stoch_rsi(
    series: pd.Series, rsi_period: int = 14, stoch_period: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Returns (%K, %D) — both in [0, 100]."""
    rsi_vals = rsi(series, rsi_period)
    min_rsi = rsi_vals.rolling(stoch_period).min()
    max_rsi = rsi_vals.rolling(stoch_period).max()
    stoch = 100 * (rsi_vals - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sc = cfg.strategy
    df["rsi"] = rsi(df["close"], sc.rsi_period)
    ml, sl, hist = macd(df["close"], sc.macd_fast, sc.macd_slow, sc.macd_signal)
    df["macd"] = ml
    df["macd_signal"] = sl
    df["macd_hist"] = hist
    df["stoch_k"], df["stoch_d"] = stoch_rsi(df["close"])
    return df


# ─── Volatility ──────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper_band, middle_band, lower_band)."""
    middle = sma(series, period)
    std = series.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sc = cfg.strategy
    df["atr"] = atr(df, sc.atr_period)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(df["close"])
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    return df


# ─── Volume ──────────────────────────────────────────────────────────────────

def add_volume_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sc = cfg.strategy
    df["vol_ma"] = sma(df["volume"], sc.volume_ma_period)
    df["vol_ratio"] = df["volume"] / df["vol_ma"]
    df["vol_spike"] = df["vol_ratio"] > sc.volume_spike_mult
    df["obv"] = (
        np.sign(df["close"].diff()) * df["volume"]
    ).fillna(0).cumsum()
    return df


# ─── ADX (trend strength) ────────────────────────────────────────────────────

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    mask2 = minus_dm < plus_dm
    minus_dm[mask2] = 0

    tr_series = atr(df, period)
    plus_di = 100 * ema(plus_dm, period) / tr_series
    minus_di = 100 * ema(minus_dm, period) / tr_series
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return ema(dx, period)


def add_adx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["adx"] = adx(df)
    return df


# ─── Composite: apply all indicators ────────────────────────────────────────

def apply_all(df: pd.DataFrame) -> pd.DataFrame:
    df = add_emas(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_volume_analysis(df)
    df = add_adx(df)
    return df


# ─── Swing pivot detection ───────────────────────────────────────────────────

def find_swing_highs(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """Returns boolean series where True = swing high (local max with n bars each side)."""
    highs = df["high"]
    is_swing = pd.Series(False, index=df.index)
    for i in range(n, len(df) - n):
        window = highs.iloc[i - n : i + n + 1]
        if highs.iloc[i] == window.max():
            is_swing.iloc[i] = True
    return is_swing


def find_swing_lows(df: pd.DataFrame, n: int = 5) -> pd.Series:
    lows = df["low"]
    is_swing = pd.Series(False, index=df.index)
    for i in range(n, len(df) - n):
        window = lows.iloc[i - n : i + n + 1]
        if lows.iloc[i] == window.min():
            is_swing.iloc[i] = True
    return is_swing


# ─── Divergence ──────────────────────────────────────────────────────────────

def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Returns Series: +1 (bullish divergence), -1 (bearish), 0 (none).
    Bullish: price makes lower low but RSI makes higher low.
    Bearish: price makes higher high but RSI makes lower high.
    """
    result = pd.Series(0, index=df.index)
    price = df["close"]
    rsi_vals = df["rsi"] if "rsi" in df.columns else rsi(price, cfg.strategy.rsi_period)

    for i in range(lookback, len(df)):
        window_price = price.iloc[i - lookback : i + 1]
        window_rsi = rsi_vals.iloc[i - lookback : i + 1]

        if price.iloc[i] == window_price.min() and rsi_vals.iloc[i] > window_rsi.idxmin():
            prev_low_idx = window_price.idxmin()
            if rsi_vals.iloc[i] > rsi_vals.loc[prev_low_idx]:
                result.iloc[i] = 1   # bullish divergence

        if price.iloc[i] == window_price.max() and rsi_vals.iloc[i] < window_rsi.max():
            prev_high_idx = window_price.idxmax()
            if rsi_vals.iloc[i] < rsi_vals.loc[prev_high_idx]:
                result.iloc[i] = -1  # bearish divergence

    return result
