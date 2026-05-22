"""
Event-driven backtester.

Simulates the bot bar-by-bar on historical 1H data, applying the same strategy,
risk manager, and commission model as live trading.  Outputs full performance
metrics at the end.
"""

import logging
import os
import csv
from datetime import datetime, timezone
from typing import List, Optional, Dict

import pandas as pd
import numpy as np

from config import cfg
from models import Direction, ClosedTrade, PortfolioStats, TradeSignal
from strategy import SMCStrategy
from risk_management import RiskManager
from indicators import apply_all

logger = logging.getLogger(__name__)


class Backtester:
    def __init__(self, tf_data: Dict[str, pd.DataFrame]):
        """
        tf_data: dict with keys matching cfg.timeframes (trend_tf, structure_tf, entry_tf).
        All DFs must cover the same date range; they'll be aligned by the entry_tf index.
        """
        self.tf_data = {k: apply_all(v) for k, v in tf_data.items()}
        self.bc = cfg.backtest
        self.commission = self.bc.commission_pct / 100
        self.slippage = self.bc.slippage_pct / 100
        self.strategy = SMCStrategy()
        self.rm = RiskManager()
        self.rm.rc.account_balance_usdt = self.bc.initial_capital

        self.equity_curve: List[float] = []
        self.trade_log: List[ClosedTrade] = []

    # ─── Main run ─────────────────────────────────────────────────────────────

    def run(self) -> PortfolioStats:
        entry_tf = cfg.timeframes.entry_tf
        struct_tf = cfg.timeframes.structure_tf
        trend_tf = cfg.timeframes.trend_tf

        entry_df = self.tf_data[entry_tf]
        struct_df = self.tf_data[struct_tf]
        trend_df = self.tf_data[trend_tf]

        # Filter by backtest date range
        start = pd.Timestamp(self.bc.start_date, tz="UTC")
        end = pd.Timestamp(self.bc.end_date, tz="UTC")
        entry_df = entry_df.loc[start:end]

        balance = self.bc.initial_capital
        peak_balance = balance
        max_dd = 0.0

        logger.info(
            "Backtesting %s bars from %s to %s | capital=%.2f USDT",
            len(entry_df), self.bc.start_date, self.bc.end_date, balance
        )

        for i in range(200, len(entry_df)):
            bar_ts = entry_df.index[i]
            current_price = float(entry_df.iloc[i]["close"])

            # Simulate slippage on fills
            effective_price = current_price * (1 + self.slippage)

            # Update open positions
            closed_this_bar = self.rm.update_positions(current_price, cfg.backtest.commission_pct)
            for t in closed_this_bar:
                balance += t.pnl
                self.trade_log.append(t)

            # Slice history up to current bar for each timeframe
            entry_slice = entry_df.iloc[:i + 1]
            struct_slice = struct_df.loc[struct_df.index <= bar_ts].tail(cfg.timeframes.candles_required)
            trend_slice = trend_df.loc[trend_df.index <= bar_ts].tail(cfg.timeframes.candles_required)

            if len(struct_slice) < 50 or len(trend_slice) < 50:
                self.equity_curve.append(balance)
                continue

            tf_slices = {
                entry_tf: entry_slice,
                struct_tf: struct_slice,
                trend_tf: trend_slice,
            }

            # Evaluate strategy
            signal: Optional[TradeSignal] = self.strategy.evaluate(tf_slices, effective_price)

            if signal is not None and self.rm.can_open_trade(signal, balance):
                pos = self.rm.open_position(signal, balance)
                if pos:
                    logger.debug("BT signal @ %s: %s R:R=%.2f", bar_ts, signal.direction.value, signal.risk_reward)

            # Track equity
            unrealised = sum(p.unrealized_pnl for p in self.rm.open_positions)
            total_equity = balance + unrealised
            self.equity_curve.append(total_equity)

            peak_balance = max(peak_balance, total_equity)
            dd = peak_balance - total_equity
            max_dd = max(max_dd, dd)

        # Force-close any remaining positions at last bar price
        last_price = float(entry_df.iloc[-1]["close"])
        for t in self.rm.close_all(last_price, cfg.backtest.commission_pct):
            balance += t.pnl
            self.trade_log.append(t)

        stats = self._compute_stats(balance, max_dd, peak_balance)
        self._print_report(stats, balance)
        self._save_trade_log()

        # Send results to Telegram if configured
        try:
            from telegram_notifier import notify_backtest_results
            notify_backtest_results(stats, balance, self.bc.initial_capital)
        except Exception:
            pass

        return stats

    # ─── Performance metrics ─────────────────────────────────────────────────

    def _compute_stats(self, final_balance: float, max_dd: float, peak: float) -> PortfolioStats:
        trades = self.trade_log
        if not trades:
            return PortfolioStats()

        wins = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = sum(t.pnl for t in losses)
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")

        avg_win_r = np.mean([t.pnl_r for t in wins]) if wins else 0.0
        avg_loss_r = np.mean([t.pnl_r for t in losses]) if losses else 0.0
        win_rate = len(wins) / len(trades)
        expectancy_r = win_rate * avg_win_r + (1 - win_rate) * avg_loss_r

        # Sharpe / Sortino (using daily equity returns)
        eq = pd.Series(self.equity_curve)
        daily_returns = eq.resample("D", closed="right").last().pct_change().dropna()
        sharpe = 0.0
        sortino = 0.0
        if len(daily_returns) > 1:
            mean_ret = daily_returns.mean()
            std_ret = daily_returns.std()
            downside_std = daily_returns[daily_returns < 0].std()
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
            sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        max_dd_pct = (max_dd / peak * 100) if peak > 0 else 0.0
        net_profit = final_balance - self.bc.initial_capital
        calmar = (net_profit / self.bc.initial_capital * 100 / max_dd_pct) if max_dd_pct > 0 else 0.0

        return PortfolioStats(
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3),
            profit_factor=round(profit_factor, 3),
            avg_win_r=round(avg_win_r, 3),
            avg_loss_r=round(avg_loss_r, 3),
            expectancy_r=round(expectancy_r, 3),
        )

    # ─── Reporting ────────────────────────────────────────────────────────────

    def _print_report(self, stats: PortfolioStats, final_balance: float):
        net = final_balance - self.bc.initial_capital
        net_pct = net / self.bc.initial_capital * 100
        sep = "─" * 50
        print(f"\n{sep}")
        print(f"  BACKTEST RESULTS  ({self.bc.start_date} → {self.bc.end_date})")
        print(sep)
        print(f"  Initial capital   : ${self.bc.initial_capital:>12,.2f}")
        print(f"  Final balance     : ${final_balance:>12,.2f}")
        print(f"  Net profit        : ${net:>12,.2f}  ({net_pct:+.2f}%)")
        print(f"  Total trades      : {stats.total_trades}")
        print(f"  Win rate          : {stats.win_rate * 100:.1f}%  ({stats.wins}W / {stats.losses}L)")
        print(f"  Profit factor     : {stats.profit_factor:.2f}")
        print(f"  Avg win (R)       : {stats.avg_win_r:.2f}R")
        print(f"  Avg loss (R)      : {stats.avg_loss_r:.2f}R")
        print(f"  Expectancy        : {stats.expectancy_r:.2f}R/trade")
        print(f"  Max drawdown      : ${stats.max_drawdown:,.2f}  ({stats.max_drawdown_pct:.1f}%)")
        print(f"  Sharpe ratio      : {stats.sharpe_ratio:.3f}")
        print(f"  Sortino ratio     : {stats.sortino_ratio:.3f}")
        print(f"  Calmar ratio      : {stats.calmar_ratio:.3f}")
        print(sep + "\n")

    def _save_trade_log(self):
        os.makedirs("logs", exist_ok=True)
        path = cfg.log.trade_log_file
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "opened_at", "closed_at", "symbol", "direction",
                "entry_price", "exit_price", "size", "pnl_usdt", "pnl_r",
                "exit_reason", "rr_ratio", "strength",
            ])
            for t in self.trade_log:
                writer.writerow([
                    t.opened_at, t.closed_at, t.signal.symbol,
                    t.signal.direction.value, t.signal.entry_price,
                    t.exit_price, t.size, round(t.pnl, 4), round(t.pnl_r, 4),
                    t.exit_reason, round(t.signal.risk_reward, 2),
                    t.signal.strength.value,
                ])
        logger.info("Trade log saved to %s", path)

    # ─── Equity curve chart ───────────────────────────────────────────────────

    def plot_equity_curve(self, save_path: str = "logs/equity_curve.png"):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_theme(style="darkgrid")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

            eq = pd.Series(self.equity_curve)
            ax1.plot(eq.values, color="#00d4aa", linewidth=1.5, label="Equity")
            ax1.axhline(self.bc.initial_capital, color="#ff6b6b", linestyle="--", alpha=0.5, label="Initial capital")
            ax1.set_title("Equity Curve — BTC/USDT SMC Strategy", fontsize=14)
            ax1.set_ylabel("Portfolio Value (USDT)")
            ax1.legend()

            # Drawdown panel
            running_max = eq.cummax()
            drawdown = (eq - running_max) / running_max * 100
            ax2.fill_between(range(len(drawdown)), drawdown.values, 0, color="#ff4757", alpha=0.5)
            ax2.set_ylabel("Drawdown (%)")
            ax2.set_xlabel("Bar #")

            plt.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info("Equity curve saved to %s", save_path)
            plt.close()
        except Exception as exc:
            logger.warning("Plot failed: %s", exc)
