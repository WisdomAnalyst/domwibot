"""
Central configuration — all tuneable parameters in one place.
Load secrets from .env; strategy constants are hardcoded defaults here.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ─── Exchange ────────────────────────────────────────────────────────────────

@dataclass
class ExchangeConfig:
    name: str = "binance"
    api_key: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_SECRET", ""))
    testnet: bool = field(default_factory=lambda: os.getenv("TESTNET", "true").lower() == "true")
    symbol: str = "SOL/USDT"
    futures_symbol: str = "SOL/USDT:USDT"    # USDT-margined perpetual
    use_futures: bool = field(default_factory=lambda: os.getenv("USE_FUTURES", "false").lower() == "true")


# ─── Timeframes ───────────────────────────────────────────────────────────────

@dataclass
class TimeframeConfig:
    trend_tf: str = "1d"       # macro bias
    structure_tf: str = "4h"   # SMC structure
    entry_tf: str = "15m"      # precise entry (15m is better for SOL volatility)
    candles_required: int = 300


# ─── Risk management ─────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    account_balance_usdt: float = float(os.getenv("ACCOUNT_BALANCE", "1000"))
    risk_per_trade_pct: float = 1.0       # 1% risk — strict, professional standard
    max_open_trades: int = 2              # SOL is volatile — keep max 2
    daily_loss_limit_pct: float = 4.0    # hard stop for the day at -4%
    min_reward_risk: float = 3.0         # minimum 3:1 R:R — only A+ setups
    max_leverage: float = 3.0            # conservative leverage
    trailing_activate_r: float = 2.0     # trail activates at 2R
    trailing_atr_mult: float = 2.0       # SOL needs wider trail due to wicks
    partial_close_pct: float = 0.5       # close 50% at TP1


# ─── Strategy (SOL/USDT tuned) ───────────────────────────────────────────────

@dataclass
class StrategyConfig:
    # Trend EMAs
    ema_trend: int = 200
    ema_fast: int = 21
    ema_slow: int = 55

    # RSI — wider band for SOL's volatility
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0

    # ATR — wider stops for SOL wicks
    atr_period: int = 14
    atr_sl_mult: float = 2.0

    # Order Block / FVG
    ob_lookback: int = 60
    ob_min_body_pct: float = 0.35
    ob_max_age_bars: int = 100
    fvg_min_size_atr: float = 0.25

    # Swing pivots
    swing_lookback: int = 5

    # Volume
    volume_ma_period: int = 20
    volume_spike_mult: float = 1.8

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ── ICT Kill Zones (UTC) ──────────────────────────────────────────────────
    # London open: 02-05 UTC  |  NY open: 12-15 UTC  |  London close: 15-17 UTC
    kill_zones: list = field(default_factory=lambda: [
        (2, 5),    # London open — institutional manipulation
        (12, 15),  # NY open    — highest volume, best moves
        (15, 17),  # London close — second liquidity window
    ])
    filter_by_session: bool = True

    # ── Liquidity sweep detection ─────────────────────────────────────────────
    sweep_lookback: int = 20          # bars to look back for equal highs/lows
    sweep_equal_tolerance: float = 0.003  # 0.3% price tolerance for "equal" levels

    # ── Funding rate filter ───────────────────────────────────────────────────
    funding_extreme_long: float = 0.001    # >0.1% funding → crowded longs → avoid longs
    funding_extreme_short: float = -0.0005 # <-0.05% funding → crowded shorts → avoid shorts

    # ── Confluence scoring ────────────────────────────────────────────────────
    min_confluence_score: int = 6     # out of 10 — only take high-probability setups


# ─── Backtesting ─────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    commission_pct: float = 0.04       # Binance taker fee
    slippage_pct: float = 0.03         # SOL has decent liquidity but wider than BTC
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"


# ─── Logging ─────────────────────────────────────────────────────────────────

@dataclass
class LogConfig:
    level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = "logs"
    trade_log_file: str = "logs/trades.csv"


# ─── Master config ────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    timeframes: TimeframeConfig = field(default_factory=TimeframeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    log: LogConfig = field(default_factory=LogConfig)


cfg = BotConfig()
