"""
Run modes: live (real money), paper (simulated), backtest.

Live mode places real orders on Binance via ccxt.
TESTNET=true in .env routes to Binance testnet -- always start there.
"""

import logging
import time
import os
from datetime import datetime, timezone, date
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

from config import cfg
from data_fetcher import DataFetcher
from strategy import SMCStrategy
from risk_management import RiskManager
from models import Direction, Position, TradeSignal
import telegram_notifier as tg

logger = logging.getLogger(__name__)
console = Console(highlight=False)


def _render_dashboard(rm: RiskManager, price: float, balance: float, daily_pnl: float):
    unrealised = sum(p.unrealized_pnl for p in rm.open_positions)
    equity = balance + unrealised
    pnl_color = "green" if daily_pnl >= 0 else "red"

    console.print(
        f"[dim]{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}[/dim]  "
        f"SOL/USDT: [bold yellow]{price:,.4f}[/bold yellow]  |  "
        f"Equity: [bold]{equity:,.2f} USDT[/bold]  |  "
        f"Daily P&L: [{pnl_color}]{daily_pnl:+.2f}[/{pnl_color}]  |  "
        f"Open: {rm.open_count}/{cfg.risk.max_open_trades}"
    )

    if not rm.open_positions:
        return

    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", padding=(0, 1))
    tbl.add_column("Dir")
    tbl.add_column("Entry", justify="right")
    tbl.add_column("SL", justify="right")
    tbl.add_column("TP2", justify="right")
    tbl.add_column("Size SOL", justify="right")
    tbl.add_column("PnL USDT", justify="right")
    tbl.add_column("R:R")
    tbl.add_column("Trail SL", justify="right")

    for pos in rm.open_positions:
        s = pos.signal
        pnl = pos.unrealized_pnl
        c = "green" if pnl >= 0 else "red"
        tbl.add_row(
            f"[bold {c}]{s.direction.value.upper()}[/bold {c}]",
            f"{s.entry_price:,.4f}",
            f"{s.stop_loss:,.4f}",
            f"{s.take_profit_2:,.4f}",
            f"{pos.size:.4f}",
            f"[{c}]{pnl:+.2f}[/{c}]",
            f"{s.risk_reward:.2f}",
            f"{pos.trailing_stop:,.4f}" if pos.trailing_stop else "--",
        )
    console.print(tbl)


# ─── Live mode ────────────────────────────────────────────────────────────────

class LiveMode:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.strategy = SMCStrategy()
        self.rm = RiskManager()
        self._order_ids: dict = {}
        self._last_daily_summary: date = None

    def run(self):
        net_label = "MAINNET -- REAL MONEY" if not cfg.exchange.testnet else "TESTNET"
        console.print(Panel(
            f"[bold cyan]DomwiBot LIVE[/bold cyan]  [{('red' if not cfg.exchange.testnet else 'green')}]{net_label}[/]"
            f"\nSymbol: [yellow]{cfg.exchange.symbol}[/yellow]  |  "
            f"Risk/trade: [yellow]{cfg.risk.risk_per_trade_pct}%[/yellow]  |  "
            f"Max trades: [yellow]{cfg.risk.max_open_trades}[/yellow]  |  "
            f"Min R:R: [yellow]{cfg.risk.min_reward_risk}[/yellow]",
            title="SOL/USDT SMC Trading Bot",
            border_style="cyan",
        ))

        balance = self.fetcher.get_balance()
        tg.notify_startup("live", balance)

        while True:
            try:
                self._tick()
                self._maybe_send_daily_summary()
            except KeyboardInterrupt:
                console.print("\n[yellow]Shutting down -- closing all positions...[/yellow]")
                price = self.fetcher.get_current_price()
                closed = self.rm.close_all(price)
                for t in closed:
                    self._cancel_exchange_orders(t.signal.symbol)
                    side = "sell" if t.signal.direction == Direction.LONG else "buy"
                    self.fetcher.place_market_order(t.signal.symbol, side, t.size)
                tg.notify_shutdown("live", self.rm.daily_pnl, len(self.rm.closed_trades))
                console.print("[yellow]All positions closed. Goodbye.[/yellow]")
                break
            except Exception as exc:
                logger.exception("Tick error: %s", exc)
                tg.notify_error("LiveMode._tick", str(exc))
            time.sleep(30)

    def _tick(self):
        price = self.fetcher.get_current_price()
        balance = self.fetcher.get_balance()
        funding = self.fetcher.fetch_funding_rate()
        tf_data = self.fetcher.fetch_multi_tf()

        closed = self.rm.update_positions(price, cfg.backtest.commission_pct)
        for t in closed:
            self._cancel_exchange_orders(t.signal.symbol)
            outcome_c = "green" if t.win else "red"
            console.print(
                f"[bold {outcome_c}]{'WIN' if t.win else 'LOSS'}[/bold {outcome_c}] "
                f"{t.signal.direction.value.upper()} closed @ {t.exit_price:,.4f} | "
                f"PnL: {t.pnl:+.2f} USDT ({t.pnl_r:+.2f}R) | {t.exit_reason}"
            )
            tg.notify_trade_closed(t, balance)

        # Check daily loss limit
        if self.rm.daily_pnl <= -(balance * cfg.risk.daily_loss_limit_pct / 100):
            tg.notify_daily_limit_hit(self.rm.daily_pnl, cfg.risk.daily_loss_limit_pct)

        signal = self.strategy.evaluate(tf_data, price, funding)
        if signal and self.rm.can_open_trade(signal, balance):
            self._execute_live(signal, balance)

        _render_dashboard(self.rm, price, balance, self.rm.daily_pnl)

    def _execute_live(self, signal: TradeSignal, balance: float):
        pos = self.rm.open_position(signal, balance)
        if not pos:
            return

        symbol = signal.symbol
        side = "buy" if signal.direction == Direction.LONG else "sell"
        close_side = "sell" if signal.direction == Direction.LONG else "buy"

        entry_order = self.fetcher.place_market_order(symbol, side, pos.size)
        if not entry_order:
            self.rm.open_positions.remove(pos)
            return

        sl_order = self.fetcher.place_limit_order(symbol, close_side, pos.size, signal.stop_loss)
        tp_order = self.fetcher.place_limit_order(symbol, close_side, pos.size, signal.take_profit_2)

        self._order_ids[id(pos)] = {
            "sl": sl_order["id"] if sl_order else None,
            "tp": tp_order["id"] if tp_order else None,
        }

        console.print(Panel(
            f"[bold green]LIVE ENTRY EXECUTED[/bold green]\n"
            f"  Direction : [bold]{signal.direction.value.upper()}[/bold]\n"
            f"  Entry     : {signal.entry_price:,.4f}\n"
            f"  Stop Loss : {signal.stop_loss:,.4f}\n"
            f"  TP1       : {signal.take_profit_1:,.4f}  (partial close at 1R)\n"
            f"  TP2       : {signal.take_profit_2:,.4f}\n"
            f"  Size      : {pos.size:.4f} SOL\n"
            f"  Risk      : {pos.usdt_risked:.2f} USDT  ({cfg.risk.risk_per_trade_pct}%)\n"
            f"  R:R       : {signal.risk_reward:.2f}\n"
            f"  Edge      : {' | '.join(signal.confluence_notes)}",
            border_style="green",
        ))

        # Telegram alert
        tg.notify_signal(signal, pos.size, pos.usdt_risked)

    def _cancel_exchange_orders(self, symbol: str):
        for pos_id, orders in list(self._order_ids.items()):
            for order_id in orders.values():
                if order_id:
                    self.fetcher.cancel_order(order_id, symbol)
            del self._order_ids[pos_id]

    def _maybe_send_daily_summary(self):
        today = datetime.now(timezone.utc).date()
        now_hour = datetime.now(timezone.utc).hour
        if now_hour == 23 and self._last_daily_summary != today:
            self._last_daily_summary = today
            balance = self.fetcher.get_balance()
            wins = len([t for t in self.rm.closed_trades if t.win])
            losses = len([t for t in self.rm.closed_trades if not t.win])
            tg.notify_daily_summary(
                balance=balance,
                start_balance=cfg.risk.account_balance_usdt,
                daily_pnl=self.rm.daily_pnl,
                wins=wins,
                losses=losses,
                open_positions=self.rm.open_count,
            )


# ─── Paper trading mode ───────────────────────────────────────────────────────

class PaperMode:
    def __init__(self, initial_balance: Optional[float] = None):
        self.fetcher = DataFetcher()
        self.strategy = SMCStrategy()
        self.rm = RiskManager()
        self.balance = initial_balance or cfg.risk.account_balance_usdt
        self.start_balance = self.balance
        self._last_daily_summary: date = None

    def run(self):
        console.print(Panel(
            f"[bold cyan]PAPER MODE[/bold cyan]\n"
            f"Symbol: [yellow]{cfg.exchange.symbol}[/yellow]  |  "
            f"Starting: [yellow]{self.balance:,.2f} USDT[/yellow]  |  "
            f"Risk/trade: [yellow]{cfg.risk.risk_per_trade_pct}%[/yellow]",
            title="SOL/USDT -- Paper Trading",
            border_style="blue",
        ))

        tg.notify_startup("paper", self.balance)

        while True:
            try:
                self._tick()
                self._maybe_send_daily_summary()
            except KeyboardInterrupt:
                self._final_report()
                tg.notify_shutdown("paper", self.rm.daily_pnl, len(self.rm.closed_trades))
                break
            except Exception as exc:
                logger.exception("Tick error: %s", exc)
                tg.notify_error("PaperMode._tick", str(exc))
            time.sleep(30)

    def _tick(self):
        price = self.fetcher.get_current_price()
        if not price:
            return

        funding = self.fetcher.fetch_funding_rate()
        tf_data = self.fetcher.fetch_multi_tf()

        closed = self.rm.update_positions(price, cfg.backtest.commission_pct)
        for t in closed:
            self.balance += t.pnl
            c = "green" if t.win else "red"
            console.print(
                f"[{c}]{'WIN' if t.win else 'LOSS'}[/{c}] "
                f"{t.signal.direction.value.upper()} @ {t.exit_price:,.4f} | "
                f"PnL: {t.pnl:+.2f} USDT ({t.pnl_r:+.2f}R) | {t.exit_reason}"
            )
            tg.notify_trade_closed(t, self.balance)

        signal = self.strategy.evaluate(tf_data, price, funding)
        if signal and self.rm.can_open_trade(signal, self.balance):
            pos = self.rm.open_position(signal, self.balance)
            if pos:
                console.print(
                    f"[bold magenta]PAPER ENTRY[/bold magenta] "
                    f"{signal.direction.value.upper()} {pos.size:.4f} SOL @ {price:,.4f} | "
                    f"SL: {signal.stop_loss:,.4f} | TP: {signal.take_profit_2:,.4f} | "
                    f"R:R {signal.risk_reward:.2f}"
                )
                tg.notify_signal(signal, pos.size, pos.usdt_risked)

        _render_dashboard(self.rm, price, self.balance, self.rm.daily_pnl)

    def _final_report(self):
        trades = self.rm.closed_trades
        wins = [t for t in trades if t.win]
        net = self.balance - self.start_balance
        if trades:
            console.print(Panel(
                f"Trades: {len(trades)}  |  Wins: {len(wins)}  |  "
                f"Win rate: {len(wins)/len(trades)*100:.1f}%\n"
                f"Net PnL: {net:+.2f} USDT  ({net/self.start_balance*100:+.2f}%)",
                title="Paper Trading Summary",
            ))

    def _maybe_send_daily_summary(self):
        today = datetime.now(timezone.utc).date()
        now_hour = datetime.now(timezone.utc).hour
        if now_hour == 23 and self._last_daily_summary != today:
            self._last_daily_summary = today
            wins = len([t for t in self.rm.closed_trades if t.win])
            losses = len([t for t in self.rm.closed_trades if not t.win])
            tg.notify_daily_summary(
                balance=self.balance,
                start_balance=self.start_balance,
                daily_pnl=self.rm.daily_pnl,
                wins=wins,
                losses=losses,
                open_positions=self.rm.open_count,
            )
