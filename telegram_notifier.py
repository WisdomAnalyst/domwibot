"""
Telegram notification system for DomwiBot.

Sends alerts to your Telegram for:
  - Bot startup / shutdown
  - New trade signal (entry details, confluence, R:R)
  - Partial close at TP1
  - Trade closed (WIN or LOSS with full stats)
  - Daily P&L summary
  - Daily loss limit hit (emergency alert)
  - Backtest results
  - Any critical errors

Setup (do this once):
  1. Open Telegram -> search @BotFather -> send /newbot
  2. Follow prompts, copy the token it gives you
  3. Send any message to your new bot (to activate the chat)
  4. Visit: https://api.telegram.org/bot{YOUR_TOKEN}/getUpdates
     -> find "chat" -> "id" -> that is your TELEGRAM_CHAT_ID
  5. Add to .env:
       TELEGRAM_BOT_TOKEN=123456:ABCdef...
       TELEGRAM_CHAT_ID=987654321
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import requests

from config import cfg
from models import ClosedTrade, TradeSignal, Direction, PortfolioStats

logger = logging.getLogger(__name__)

_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
_BASE_URL = f"https://api.telegram.org/bot{_TOKEN}"


def _enabled() -> bool:
    return bool(_TOKEN and _CHAT_ID)


def send(message: str, silent: bool = False) -> bool:
    """Send a plain or markdown message. Returns True on success."""
    if not _enabled():
        logger.debug("Telegram not configured -- skipping notification")
        return False
    try:
        resp = requests.post(
            f"{_BASE_URL}/sendMessage",
            json={
                "chat_id": _CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_notification": silent,
            },
            timeout=10,
        )
        if not resp.ok:
            logger.warning("Telegram send failed: %s", resp.text)
            return False
        return True
    except Exception as exc:
        logger.warning("Telegram error: %s", exc)
        return False


# ─── Notification templates ───────────────────────────────────────────────────

def notify_startup(mode: str, balance: float):
    emoji = "🟢" if mode == "live" else "📋" if mode == "paper" else "📊"
    send(
        f"{emoji} <b>DomwiBot Started</b>\n"
        f"Mode     : <b>{mode.upper()}</b>\n"
        f"Symbol   : SOL/USDT\n"
        f"Balance  : <b>${balance:,.2f} USDT</b>\n"
        f"Risk/trade: {cfg.risk.risk_per_trade_pct}%  |  Max R:R min: {cfg.risk.min_reward_risk}\n"
        f"Max trades: {cfg.risk.max_open_trades}  |  Daily limit: -{cfg.risk.daily_loss_limit_pct}%\n"
        f"Time     : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


def notify_shutdown(mode: str, daily_pnl: float, total_trades: int):
    emoji = "🔴"
    pnl_str = f"+${daily_pnl:.2f}" if daily_pnl >= 0 else f"-${abs(daily_pnl):.2f}"
    send(
        f"{emoji} <b>DomwiBot Stopped</b>\n"
        f"Mode       : {mode.upper()}\n"
        f"Daily P&L  : <b>{pnl_str} USDT</b>\n"
        f"Trades today: {total_trades}\n"
        f"Time       : {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )


def notify_signal(signal: TradeSignal, size: float, usdt_risked: float):
    direction_emoji = "📈" if signal.direction == Direction.LONG else "📉"
    direction_word = "LONG" if signal.direction == Direction.LONG else "SHORT"
    strength_emoji = "🔥" if signal.strength.value == "strong" else "✅" if signal.strength.value == "moderate" else "⚠️"

    confluence_lines = "\n".join(
        f"  • {note}" for note in signal.confluence_notes
    )

    send(
        f"{direction_emoji} <b>NEW TRADE -- SOL/USDT {direction_word}</b> {strength_emoji}\n\n"
        f"Entry     : <b>${signal.entry_price:,.4f}</b>\n"
        f"Stop Loss : <code>${signal.stop_loss:,.4f}</code>  (-{abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100:.2f}%)\n"
        f"TP1       : <code>${signal.take_profit_1:,.4f}</code>  (1R partial close)\n"
        f"TP2       : <code>${signal.take_profit_2:,.4f}</code>\n"
        f"R:R Ratio : <b>{signal.risk_reward:.2f}</b>\n"
        f"Size      : {size:.4f} SOL\n"
        f"Risked    : <b>${usdt_risked:.2f} USDT</b>  ({cfg.risk.risk_per_trade_pct}%)\n"
        f"ATR       : {signal.atr:.4f}\n\n"
        f"<b>Confluence ({signal.confluence_notes[-1] if signal.confluence_notes else ''}):</b>\n"
        f"{confluence_lines}\n\n"
        f"<i>{datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"
    )


def notify_partial_close(signal: TradeSignal, close_price: float, pnl: float, remaining_size: float):
    send(
        f"🎯 <b>TP1 HIT -- Partial Close</b>\n\n"
        f"Symbol    : SOL/USDT {signal.direction.value.upper()}\n"
        f"Close price: <b>${close_price:,.4f}</b>\n"
        f"PnL        : <b>+${pnl:.2f} USDT</b>\n"
        f"Stop -> Breakeven at <code>${signal.entry_price:,.4f}</code>\n"
        f"Remaining : {remaining_size:.4f} SOL still open\n"
        f"<i>{datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"
    )


def notify_trade_closed(trade: ClosedTrade, balance_after: float):
    if trade.win:
        emoji = "✅"
        result = "WIN"
        pnl_str = f"+${trade.pnl:.2f}"
    else:
        emoji = "❌"
        result = "LOSS"
        pnl_str = f"-${abs(trade.pnl):.2f}"

    duration_secs = (trade.closed_at - trade.opened_at).total_seconds()
    duration_str = _format_duration(duration_secs)

    send(
        f"{emoji} <b>TRADE CLOSED -- {result}</b>\n\n"
        f"Symbol    : SOL/USDT {trade.signal.direction.value.upper()}\n"
        f"Entry     : ${trade.signal.entry_price:,.4f}\n"
        f"Exit      : <b>${trade.exit_price:,.4f}</b>  ({trade.exit_reason})\n"
        f"P&L       : <b>{pnl_str} USDT  ({trade.pnl_r:+.2f}R)</b>\n"
        f"Duration  : {duration_str}\n"
        f"Commission: ${trade.commission:.3f}\n"
        f"Balance   : <b>${balance_after:,.2f} USDT</b>\n"
        f"<i>{trade.closed_at.strftime('%H:%M UTC')}</i>"
    )


def notify_daily_summary(
    balance: float,
    start_balance: float,
    daily_pnl: float,
    wins: int,
    losses: int,
    open_positions: int,
):
    pnl_pct = daily_pnl / start_balance * 100 if start_balance > 0 else 0
    total = wins + losses
    win_rate = wins / total * 100 if total > 0 else 0
    pnl_emoji = "📈" if daily_pnl >= 0 else "📉"

    send(
        f"{pnl_emoji} <b>Daily Summary -- SOL/USDT</b>\n\n"
        f"Balance   : <b>${balance:,.2f} USDT</b>\n"
        f"Daily P&L : <b>${daily_pnl:+.2f}  ({pnl_pct:+.2f}%)</b>\n"
        f"Trades    : {total}  (W:{wins} / L:{losses}  --  {win_rate:.0f}% win rate)\n"
        f"Open now  : {open_positions}\n"
        f"Date      : {datetime.now(timezone.utc).strftime('%Y-%m-%d UTC')}"
    )


def notify_daily_limit_hit(daily_pnl: float, limit_pct: float):
    send(
        f"🚨 <b>DAILY LOSS LIMIT HIT -- TRADING HALTED</b>\n\n"
        f"Daily P&L : <b>-${abs(daily_pnl):.2f} USDT</b>\n"
        f"Limit     : -{limit_pct}%\n"
        f"Status    : No new trades until tomorrow\n"
        f"<i>{datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"
    )


def notify_backtest_results(stats: PortfolioStats, final_balance: float, initial_capital: float):
    net = final_balance - initial_capital
    net_pct = net / initial_capital * 100
    result_emoji = "✅" if net > 0 else "❌"

    send(
        f"📊 <b>Backtest Complete -- SOL/USDT</b> {result_emoji}\n\n"
        f"Period    : {cfg.backtest.start_date} to {cfg.backtest.end_date}\n"
        f"Capital   : $10,000 USDT\n"
        f"Final     : <b>${final_balance:,.2f} USDT</b>\n"
        f"Net P&L   : <b>${net:+,.2f}  ({net_pct:+.2f}%)</b>\n\n"
        f"<b>Performance:</b>\n"
        f"  Trades      : {stats.total_trades}  (W:{stats.wins} / L:{stats.losses})\n"
        f"  Win rate    : <b>{stats.win_rate * 100:.1f}%</b>\n"
        f"  Profit factor: <b>{stats.profit_factor:.2f}</b>\n"
        f"  Expectancy  : {stats.expectancy_r:.2f}R per trade\n"
        f"  Avg Win     : +{stats.avg_win_r:.2f}R\n"
        f"  Avg Loss    : {stats.avg_loss_r:.2f}R\n\n"
        f"<b>Risk:</b>\n"
        f"  Max drawdown: <b>${stats.max_drawdown:,.2f}  ({stats.max_drawdown_pct:.1f}%)</b>\n"
        f"  Sharpe      : {stats.sharpe_ratio:.3f}\n"
        f"  Sortino     : {stats.sortino_ratio:.3f}\n"
        f"  Calmar      : {stats.calmar_ratio:.3f}"
    )


def notify_error(context: str, error: str):
    send(
        f"⚠️ <b>DomwiBot Error</b>\n\n"
        f"Context : {context}\n"
        f"Error   : <code>{error[:300]}</code>\n"
        f"Time    : {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )


# ─── Helper ───────────────────────────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


def test_connection() -> bool:
    """Send a test message to confirm the bot is working."""
    ok = send(
        "✅ <b>DomwiBot Telegram connected!</b>\n"
        "You will receive all trade notifications here.\n"
        f"Symbol: SOL/USDT | Risk: {cfg.risk.risk_per_trade_pct}% per trade"
    )
    if ok:
        logger.info("Telegram test message sent successfully")
    else:
        logger.warning("Telegram test failed -- check token and chat ID in .env")
    return ok
