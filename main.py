"""
Entry point.  Usage:

    python main.py backtest           # run historical simulation
    python main.py paper              # paper trading (no real orders)
    python main.py live               # LIVE trading -- testnet by default
    python main.py live --mainnet     # real money (ensure .env is set!)
    python main.py signal             # print current signal once and exit
"""

import argparse
import logging
import os
import sys

# Force UTF-8 output on Windows so the terminal doesn't crash on special chars
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.logging import RichHandler

from config import cfg

console = Console(highlight=False)


def _setup_logging():
    os.makedirs(cfg.log.log_dir, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, cfg.log.level, logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True, markup=True),
            logging.FileHandler(f"{cfg.log.log_dir}/bot.log", encoding="utf-8"),
        ],
    )


def _banner():
    console.print("\n[bold cyan]=============================[/bold cyan]")
    console.print("[bold cyan]  DomwiBot -- SOL/USDT Bot  [/bold cyan]")
    console.print("[bold cyan]  Smart Money + MTF Trend   [/bold cyan]")
    console.print("[bold cyan]=============================[/bold cyan]\n")
    console.print(f"  Exchange  : [yellow]{cfg.exchange.name}[/yellow]")
    console.print(f"  Symbol    : [yellow]{cfg.exchange.symbol}[/yellow]")
    console.print(f"  Testnet   : [yellow]{cfg.exchange.testnet}[/yellow]")
    console.print(f"  Risk/trade: [yellow]{cfg.risk.risk_per_trade_pct}%[/yellow]")
    console.print(f"  Max trades: [yellow]{cfg.risk.max_open_trades}[/yellow]")
    console.print(f"  Min R:R   : [yellow]{cfg.risk.min_reward_risk}[/yellow]\n")


def run_backtest():
    from data_fetcher import DataFetcher
    from backtest import Backtester

    console.print("[bold]Downloading full SOL/USDT history from Binance...[/bold]")
    console.print(f"  Range : {cfg.backtest.start_date} to {cfg.backtest.end_date}")
    console.print("  Note  : First run takes 1-2 min to download years of data\n")
    fetcher = DataFetcher()
    tf_data = fetcher.fetch_multi_tf_historical()

    for tf, df in tf_data.items():
        if df.empty:
            console.print(f"[red]ERROR: No data for {tf}. Check connection.[/red]")
            sys.exit(1)
        console.print(
            f"  {tf}: {len(df)} bars "
            f"({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})"
        )

    bt = Backtester(tf_data)
    stats = bt.run()
    bt.plot_equity_curve()
    return stats


def run_paper():
    from run_modes import PaperMode
    PaperMode().run()


def run_live(mainnet: bool = False):
    if mainnet:
        console.print(
            "[bold red]WARNING: LIVE MAINNET -- real orders will be placed.[/bold red]\n"
            "Press Ctrl+C within 5s to abort."
        )
        import time
        try:
            for i in range(5, 0, -1):
                console.print(f"  Starting in {i}...")
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("[yellow]Aborted.[/yellow]")
            return
        cfg.exchange.testnet = False

    from run_modes import LiveMode
    LiveMode().run()


def run_signal():
    from data_fetcher import DataFetcher
    from strategy import SMCStrategy

    fetcher = DataFetcher()
    strategy = SMCStrategy()
    price = fetcher.get_current_price()
    funding = fetcher.fetch_funding_rate()
    tf_data = fetcher.fetch_multi_tf()

    console.print(f"SOL/USDT price : [yellow]{price:,.4f}[/yellow]")
    console.print(f"Funding rate   : [yellow]{funding*100:.4f}%[/yellow]")

    signal = strategy.evaluate(tf_data, price, funding)
    if signal is None:
        console.print("[dim]No signal right now -- waiting for A+ setup.[/dim]")
        return

    color = "green" if signal.direction.value == "long" else "red"
    console.print(f"\n[bold {color}]SIGNAL: {signal.direction.value.upper()}[/bold {color}]")
    console.print(f"  Entry     : {signal.entry_price:,.4f}")
    console.print(f"  Stop Loss : {signal.stop_loss:,.4f}")
    console.print(f"  TP1       : {signal.take_profit_1:,.4f}  (partial close 1R)")
    console.print(f"  TP2       : {signal.take_profit_2:,.4f}")
    console.print(f"  R:R       : {signal.risk_reward:.2f}")
    console.print(f"  Strength  : {signal.strength.value}")
    console.print(f"  Confluence:")
    for note in signal.confluence_notes:
        console.print(f"    - {note}")


def main():
    parser = argparse.ArgumentParser(description="DomwiBot SOL/USDT Trading Bot")
    parser.add_argument(
        "mode",
        choices=["backtest", "paper", "live", "signal", "telegram-test"],
        help="Run mode",
    )
    parser.add_argument(
        "--mainnet",
        action="store_true",
        help="Use real money (live mode only)",
    )
    args = parser.parse_args()

    _setup_logging()
    _banner()

    if args.mode == "backtest":
        run_backtest()
    elif args.mode == "paper":
        run_paper()
    elif args.mode == "live":
        run_live(mainnet=args.mainnet)
    elif args.mode == "signal":
        run_signal()
    elif args.mode == "telegram-test":
        from telegram_notifier import test_connection
        ok = test_connection()
        console.print(
            "[green]Telegram connected! Check your bot.[/green]" if ok
            else "[red]Telegram failed. Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env[/red]"
        )


if __name__ == "__main__":
    main()
