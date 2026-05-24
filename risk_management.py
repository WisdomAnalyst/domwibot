"""
Risk management: position sizing, stop adjustment, portfolio heat checks.

Core rules (what separates the 1%):
  - Never risk more than cfg.risk.risk_per_trade_pct of account per trade
  - Max cfg.risk.max_open_trades concurrent positions
  - Halt if daily PnL drops below -cfg.risk.daily_loss_limit_pct
  - Trailing stop activated at cfg.risk.trailing_activate_r × risk distance
  - Partial close (50%) at TP1 (1R) to lock in profit and reduce heat
"""

import logging
from datetime import datetime, timezone, date
from typing import List, Optional

from config import cfg
from models import Direction, Position, TradeSignal, ClosedTrade, TradeStatus

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self):
        self.rc = cfg.risk
        self.open_positions: List[Position] = []
        self.closed_trades: List[ClosedTrade] = []
        self._daily_pnl: float = 0.0
        self._daily_reset_date: date = datetime.now(timezone.utc).date()

    # ─── Position sizing ─────────────────────────────────────────────────────

    def calculate_position_size(
        self, signal: TradeSignal, account_balance: float
    ) -> float:
        """
        Fixed-fractional sizing: risk exactly risk_per_trade_pct% of balance.
        Returns the position size in BTC (base currency).
        """
        risk_amount_usdt = account_balance * (self.rc.risk_per_trade_pct / 100)
        risk_distance = signal.risk_distance

        if risk_distance <= 0:
            logger.warning("Invalid risk distance (%s) — skipping", risk_distance)
            return 0.0

        raw_size = risk_amount_usdt / risk_distance

        # Cap by leverage
        max_size = (account_balance * self.rc.max_leverage) / signal.entry_price
        size = min(raw_size, max_size)

        # Round down to 4 decimal places (BTC lot precision)
        size = int(size * 10_000) / 10_000
        logger.info(
            "Size calc: balance=%.2f USDT, risk=%.2f USDT, dist=%.2f → %.4f BTC",
            account_balance, risk_amount_usdt, risk_distance, size,
        )
        return size

    # ─── Pre-trade checks ────────────────────────────────────────────────────

    def can_open_trade(self, signal: TradeSignal, account_balance: float) -> bool:
        self._reset_daily_if_needed()

        if len(self.open_positions) >= self.rc.max_open_trades:
            logger.info("Max open trades (%d) reached", self.rc.max_open_trades)
            return False

        if self._daily_pnl <= -(account_balance * self.rc.daily_loss_limit_pct / 100):
            logger.warning(
                "Daily loss limit hit (%.2f USDT) — no new trades today", self._daily_pnl
            )
            return False

        # Prevent correlated positions in same direction (already at risk)
        same_dir = [p for p in self.open_positions if p.signal.direction == signal.direction]
        if len(same_dir) >= 2:
            logger.info("Already have 2 positions in %s direction", signal.direction)
            return False

        return True

    # ─── Position management ─────────────────────────────────────────────────

    def open_position(self, signal: TradeSignal, account_balance: float) -> Optional[Position]:
        if not self.can_open_trade(signal, account_balance):
            return None

        size = self.calculate_position_size(signal, account_balance)
        if size == 0:
            return None

        usdt_risked = size * signal.risk_distance
        pos = Position(
            signal=signal,
            size=size,
            usdt_risked=usdt_risked,
            opened_at=datetime.now(timezone.utc),
            current_price=signal.entry_price,
        )
        self.open_positions.append(pos)
        logger.info(
            "Opened %s: %.4f BTC @ %.2f | SL %.2f | TP1 %.2f | TP2 %.2f | R:R %.2f",
            signal.direction.value.upper(),
            size, signal.entry_price,
            signal.stop_loss, signal.take_profit_1, signal.take_profit_2,
            signal.risk_reward,
        )
        return pos

    def update_positions(self, current_price: float, commission_pct: float = 0.04) -> List[ClosedTrade]:
        """
        Update all open positions against current_price.
        Returns list of any trades that were closed this tick.
        """
        newly_closed: List[ClosedTrade] = []

        for pos in list(self.open_positions):
            pos.current_price = current_price
            pos.bars_held += 1
            s = pos.signal
            dir_ = s.direction

            # ── Time-based exit: close if stalling before TP1 ────────────
            if not pos.partial_closed and pos.bars_held > 12:
                trade = self._close_position(pos, current_price, "time_exit", commission_pct)
                newly_closed.append(trade)
                self.open_positions.remove(pos)
                continue

            # ── Partial close at TP1 ──────────────────────────────────────
            if not pos.partial_closed:
                tp1_hit = (
                    (dir_ == Direction.LONG and current_price >= s.take_profit_1) or
                    (dir_ == Direction.SHORT and current_price <= s.take_profit_1)
                )
                if tp1_hit:
                    partial_size = pos.size * self.rc.partial_close_pct
                    pnl_per_unit = (current_price - s.entry_price) * (1 if dir_ == Direction.LONG else -1)
                    commission = partial_size * current_price * (commission_pct / 100)
                    pnl = pnl_per_unit * partial_size - commission

                    self._daily_pnl += pnl
                    pos.size -= partial_size
                    pos.partial_closed = True

                    # Move SL to breakeven after TP1
                    if dir_ == Direction.LONG:
                        s = TradeSignal(
                            **{k: getattr(s, k) for k in s.__dataclass_fields__}
                        )
                        object.__setattr__(s, 'stop_loss', max(s.entry_price, s.stop_loss))
                        pos.signal = s

                    logger.info(
                        "TP1 hit — partial close %.4f BTC @ %.2f, PnL=%.2f USDT, SL→BE",
                        partial_size, current_price, pnl
                    )

            # ── Trailing stop ─────────────────────────────────────────────
            profit_r = pos.unrealized_pnl / pos.usdt_risked if pos.usdt_risked > 0 else 0
            if profit_r >= self.rc.trailing_activate_r:
                trail_distance = self.rc.trailing_atr_mult * s.atr
                if dir_ == Direction.LONG:
                    new_trail = current_price - trail_distance
                    if pos.trailing_stop is None or new_trail > pos.trailing_stop:
                        pos.trailing_stop = new_trail
                else:
                    new_trail = current_price + trail_distance
                    if pos.trailing_stop is None or new_trail < pos.trailing_stop:
                        pos.trailing_stop = new_trail

            # ── Check exit conditions ─────────────────────────────────────
            exit_reason: Optional[str] = None
            exit_price = current_price

            if pos.trailing_stop is not None:
                ts_hit = (
                    (dir_ == Direction.LONG and current_price <= pos.trailing_stop) or
                    (dir_ == Direction.SHORT and current_price >= pos.trailing_stop)
                )
                if ts_hit:
                    exit_reason = "trailing_sl"
                    exit_price = pos.trailing_stop

            if exit_reason is None:
                sl_hit = (
                    (dir_ == Direction.LONG and current_price <= s.stop_loss) or
                    (dir_ == Direction.SHORT and current_price >= s.stop_loss)
                )
                if sl_hit:
                    exit_reason = "sl"
                    exit_price = s.stop_loss

            if exit_reason is None:
                tp2_hit = (
                    (dir_ == Direction.LONG and current_price >= s.take_profit_2) or
                    (dir_ == Direction.SHORT and current_price <= s.take_profit_2)
                )
                if tp2_hit:
                    exit_reason = "tp2"

            if exit_reason:
                trade = self._close_position(pos, exit_price, exit_reason, commission_pct)
                newly_closed.append(trade)
                self.open_positions.remove(pos)

        return newly_closed

    def _close_position(
        self, pos: Position, exit_price: float, reason: str, commission_pct: float
    ) -> ClosedTrade:
        s = pos.signal
        dir_ = s.direction
        pnl_per_unit = (exit_price - s.entry_price) * (1 if dir_ == Direction.LONG else -1)
        commission = pos.size * exit_price * (commission_pct / 100)
        pnl = pnl_per_unit * pos.size - commission
        pnl_r = pnl / pos.usdt_risked if pos.usdt_risked > 0 else 0

        self._daily_pnl += pnl
        pos.status = TradeStatus.CLOSED

        trade = ClosedTrade(
            signal=s,
            size=pos.size,
            usdt_risked=pos.usdt_risked,
            opened_at=pos.opened_at,
            closed_at=datetime.now(timezone.utc),
            exit_price=exit_price,
            exit_reason=reason,
            commission=commission,
            pnl=pnl,
            pnl_r=pnl_r,
        )
        self.closed_trades.append(trade)

        outcome = "WIN" if pnl > 0 else "LOSS"
        logger.info(
            "%s closed [%s] @ %.2f | PnL=%.2f USDT (%.2fR) | Reason: %s",
            dir_.value.upper(), outcome, exit_price, pnl, pnl_r, reason
        )
        return trade

    def close_all(self, current_price: float, commission_pct: float = 0.04) -> List[ClosedTrade]:
        """Emergency close all positions."""
        closed = []
        for pos in list(self.open_positions):
            trade = self._close_position(pos, current_price, "manual", commission_pct)
            closed.append(trade)
        self.open_positions.clear()
        return closed

    # ─── Daily limit management ───────────────────────────────────────────────

    def _reset_daily_if_needed(self):
        today = datetime.now(timezone.utc).date()
        if today != self._daily_reset_date:
            self._daily_pnl = 0.0
            self._daily_reset_date = today

    @property
    def daily_pnl(self) -> float:
        self._reset_daily_if_needed()
        return self._daily_pnl

    @property
    def portfolio_heat(self) -> float:
        """Total USDT currently at risk across all open positions."""
        return sum(p.usdt_risked for p in self.open_positions)

    @property
    def open_count(self) -> int:
        return len(self.open_positions)
