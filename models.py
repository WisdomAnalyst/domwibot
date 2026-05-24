"""
Shared data models.  Using plain dataclasses for zero-overhead, no extra deps.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


class SignalStrength(str, Enum):
    STRONG = "strong"   # all confluence factors align
    MODERATE = "moderate"
    WEAK = "weak"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class StructureType(str, Enum):
    BOS = "bos"         # Break of Structure — trend continuation
    CHOCH = "choch"     # Change of Character — potential reversal
    NONE = "none"


# ─── Market data ─────────────────────────────────────────────────────────────

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = ""

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body_pct(self) -> float:
        return self.body / self.range if self.range > 0 else 0.0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low


# ─── Smart Money Concepts ─────────────────────────────────────────────────────

@dataclass
class OrderBlock:
    """Last opposing candle before a strong impulsive move."""
    direction: Direction           # LONG OB (bullish) or SHORT OB (bearish)
    top: float
    bottom: float
    origin_time: datetime
    origin_index: int
    mitigated: bool = False
    mitigation_time: Optional[datetime] = None
    strength: float = 1.0         # based on move size after OB

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def size(self) -> float:
        return self.top - self.bottom

    def contains_price(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass
class FairValueGap:
    """Three-candle imbalance; price tends to return and fill."""
    direction: Direction           # bullish FVG or bearish FVG
    top: float
    bottom: float
    origin_time: datetime
    origin_index: int
    filled: bool = False
    fill_time: Optional[datetime] = None

    @property
    def size(self) -> float:
        return self.top - self.bottom

    def contains_price(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass
class SwingPoint:
    direction: Direction           # HIGH or LOW (stored as LONG/SHORT)
    price: float
    timestamp: datetime
    index: int
    broken: bool = False


@dataclass
class MarketStructure:
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    last_bos: Optional[StructureType] = None
    last_bos_price: Optional[float] = None
    last_bos_time: Optional[datetime] = None
    trend: Optional[Direction] = None      # current inferred trend


# ─── Signals & Trades ────────────────────────────────────────────────────────

@dataclass
class TradeSignal:
    symbol: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit_1: float           # TP1 — partial close at 1R
    take_profit_2: float           # TP2 — full close, 2.5R+
    timestamp: datetime
    timeframe: str
    strength: SignalStrength = SignalStrength.MODERATE
    confluence_notes: List[str] = field(default_factory=list)
    atr: float = 0.0
    risk_reward: float = 0.0

    @property
    def risk_distance(self) -> float:
        if self.direction == Direction.LONG:
            return self.entry_price - self.stop_loss
        return self.stop_loss - self.entry_price

    @property
    def reward_distance(self) -> float:
        if self.direction == Direction.LONG:
            return self.take_profit_2 - self.entry_price
        return self.entry_price - self.take_profit_2


@dataclass
class Position:
    signal: TradeSignal
    size: float                    # in base asset (BTC)
    usdt_risked: float
    opened_at: datetime
    current_price: float = 0.0
    trailing_stop: Optional[float] = None
    partial_closed: bool = False   # TP1 hit
    bars_held: int = 0
    status: TradeStatus = TradeStatus.OPEN

    @property
    def unrealized_pnl(self) -> float:
        diff = self.current_price - self.signal.entry_price
        if self.signal.direction == Direction.SHORT:
            diff = -diff
        return diff * self.size

    @property
    def unrealized_pnl_pct(self) -> float:
        cost = self.signal.entry_price * self.size
        return (self.unrealized_pnl / cost) * 100 if cost > 0 else 0.0


@dataclass
class ClosedTrade:
    signal: TradeSignal
    size: float
    usdt_risked: float
    opened_at: datetime
    closed_at: datetime
    exit_price: float
    exit_reason: str               # tp1, tp2, sl, trailing_sl, manual
    commission: float
    pnl: float
    pnl_r: float                   # profit in R multiples

    @property
    def win(self) -> bool:
        return self.pnl > 0


@dataclass
class PortfolioStats:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    expectancy_r: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def net_profit(self) -> float:
        return self.gross_profit + self.gross_loss   # gross_loss is negative
