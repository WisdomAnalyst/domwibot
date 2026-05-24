# DomwiBot — Sol/USDT SMC Trading Bot

A professional-grade Solana swing trading bot built around the same edge
the top 1% of crypto traders use: **Smart Money Concepts (SMC)** combined
with **multi-timeframe trend filtering** and strict risk management.

---

## Strategy Overview

| Layer | Timeframe | Purpose |
|---|---|---|
| Trend bias | Daily (1D) | EMA-200 + EMA-50/100 crossover — only trade with the trend |
| Structure | 4-Hour | BOS/CHOCH, Order Blocks, Fair Value Gaps, discount/premium zones |
| Entry | 1-Hour | Price retest of OB/FVG + RSI confluence + volume confirmation |

### Entry checklist (LONG)
1. Daily close **above EMA-200** and EMA-50 > EMA-100
2. 4H Break of Structure (BOS) to the upside — confirmed uptrend
3. Price pulls back into **discount zone** (below 0.5 Fibonacci of last swing)
4. Unmitigated **bullish Order Block** or unfilled **bullish FVG** in that zone
5. 1H RSI < 40 (oversold), declining pullback volume
6. Minimum **2.5 R:R** to the next liquidity target

### Risk rules
- Risk **1.5% of account** per trade (configurable)
- Maximum **3 simultaneous** open positions
- Daily loss limit: **−5%** — bot halts for the day
- **Partial close at TP1** (1R profit), stop moved to breakeven
- **Trailing stop** activates at 2R profit (1.5× ATR trail)
- Stop placed below OB + 1.5× ATR buffer

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env: add API key/secret, set TESTNET=true

# 3. Check current signal
python main.py signal

# 4. Run paper trading (no real money)
python main.py paper

# 5. Backtest on historical data
python main.py backtest

# 6. Go live (testnet)
python main.py live

# 7. Go live (real money — be careful!)
python main.py live --mainnet
```

---

## File Structure

```
domwibot/
├── main.py            CLI entry point
├── config.py          All tuneable parameters
├── models.py          Data classes (Candle, OrderBlock, FVG, TradeSignal…)
├── data_fetcher.py    OHLCV, orderbook, funding rate via ccxt
├── indicators.py      EMA, RSI, MACD, ATR, BB, ADX, volume, pivots
├── strategy.py        SMC signal engine (OB + FVG + structure + MTF)
├── risk_management.py Position sizing, trailing stops, daily limits
├── backtest.py        Event-driven backtester + performance metrics
├── run_modes.py       Live, paper, and backtest orchestration
└── .env.example       Environment variable template
```

---

## Key Metrics the Bot Tracks

- Win rate, profit factor, expectancy (R/trade)
- Sharpe, Sortino, Calmar ratios
- Maximum drawdown (USDT and %)
- Per-trade: entry/exit, PnL in USDT and R-multiples, exit reason

---

## Exchange Support

Uses [ccxt](https://github.com/ccxt/ccxt) — supports 100+ exchanges.
Default: **Binance** (spot or USDT-margined perpetual futures).
Switch by changing `exchange.name` in `config.py`.

---

## Disclaimer

This bot is for **educational and research purposes**.  
Crypto trading carries significant risk. Past backtest results do not
guarantee future performance. Always start on **testnet** and use only
capital you can afford to lose.
