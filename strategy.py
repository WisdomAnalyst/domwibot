"""
SOL/USDT strategy — tactics of the top 1% most profitable traders.

Techniques implemented:
  1. ICT Kill Zones — only trade during institutional manipulation windows
  2. Liquidity Sweeps — enter AFTER smart money hunts retail stops
  3. Smart Money Concepts — unmitigated Order Blocks + Fair Value Gaps
  4. Wyckoff Phase detection — Spring/UTAD as the highest-probability reversals
  5. CVD Divergence — Cumulative Volume Delta reveals hidden buying/selling
  6. Funding Rate Extremes — crowded positioning as a contrarian filter
  7. Multi-timeframe BOS/CHOCH — only trade in direction of institutional flow
  8. Confluence Scoring (0–10) — entry only if score ≥ min_confluence_score

Why this works:
  Retail traders enter at obvious levels and get stopped out.
  Smart money CREATES those stop-hunts to fill their own positions.
  This bot waits for the sweep, watches for the reversal, then enters
  with the institutions — targeting the move that was set up by the sweep.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from config import cfg
from models import (
    Direction, SignalStrength, OrderBlock, FairValueGap,
    SwingPoint, MarketStructure, StructureType, TradeSignal,
)
from indicators import apply_all, find_swing_highs, find_swing_lows

logger = logging.getLogger(__name__)


class SMCStrategy:
    def __init__(self):
        self.sc = cfg.strategy

    # ─── Public entry point ──────────────────────────────────────────────────

    def evaluate(
        self,
        tf_data: Dict[str, pd.DataFrame],
        current_price: float,
        funding_rate: float = 0.0,
    ) -> Optional[TradeSignal]:
        trend_tf = cfg.timeframes.trend_tf
        struct_tf = cfg.timeframes.structure_tf
        entry_tf = cfg.timeframes.entry_tf

        if any(tf not in tf_data or tf_data[tf].empty for tf in [trend_tf, struct_tf, entry_tf]):
            logger.warning("Missing timeframe data — skipping")
            return None

        trend_df = apply_all(tf_data[trend_tf])
        struct_df = apply_all(tf_data[struct_tf])
        entry_df = apply_all(tf_data[entry_tf])

        if len(entry_df) < 50:
            return None

        # ── Gate 1: ICT Kill Zone ────────────────────────────────────────────
        if self.sc.filter_by_session and not self._in_kill_zone():
            logger.debug("Outside kill zone — skip")
            return None

        # ── Gate 2: Macro trend bias (Daily) ─────────────────────────────────
        bias = self._get_trend_bias(trend_df)
        if bias is None:
            logger.debug("No clear daily bias")
            return None

        # ── Gate 3: Funding rate extreme filter ──────────────────────────────
        if not self._funding_allows(bias, funding_rate):
            logger.debug("Funding rate blocks %s entry (%.4f)", bias.value, funding_rate)
            return None

        # ── Gate 4: Gather all 4H structure ──────────────────────────────────
        struct_ms = self._analyse_market_structure(struct_df)
        obs = self._find_order_blocks(struct_df)
        fvgs = self._find_fvgs(struct_df)

        # ── Gate 5: Detect liquidity sweep on entry timeframe ─────────────────
        sweep = self._detect_liquidity_sweep(entry_df, bias)

        # ── Score & build signal ──────────────────────────────────────────────
        return self._build_signal(
            bias, entry_df, struct_df, obs, fvgs, struct_ms,
            current_price, sweep, funding_rate
        )

    # ─── ICT Kill Zone ───────────────────────────────────────────────────────

    def _in_kill_zone(self) -> bool:
        hour = datetime.now(timezone.utc).hour
        for start, end in self.sc.kill_zones:
            if start <= hour < end:
                return True
        return False

    # ─── Trend bias (Daily EMA stack) ───────────────────────────────────────

    def _get_trend_bias(self, df: pd.DataFrame) -> Optional[Direction]:
        if len(df) < self.sc.ema_trend:
            return None
        last = df.iloc[-1]
        price = last["close"]
        ema200 = last.get("ema_trend", 0)
        ema21 = last.get("ema_fast", 0)
        ema55 = last.get("ema_slow", 0)

        if price > ema200 and ema21 > ema55:
            return Direction.LONG
        if price < ema200 and ema21 < ema55:
            return Direction.SHORT
        return None

    # ─── Funding rate filter ─────────────────────────────────────────────────

    def _funding_allows(self, bias: Direction, rate: float) -> bool:
        sc = self.sc
        if bias == Direction.LONG and rate > sc.funding_extreme_long:
            return False   # everyone is long → don't follow the crowd
        if bias == Direction.SHORT and rate < sc.funding_extreme_short:
            return False   # everyone is short → don't follow the crowd
        return True

    # ─── Market Structure ────────────────────────────────────────────────────

    def _analyse_market_structure(self, df: pd.DataFrame) -> MarketStructure:
        n = self.sc.swing_lookback
        sh_mask = find_swing_highs(df, n)
        sl_mask = find_swing_lows(df, n)

        swing_highs, swing_lows = [], []
        for i, (ts, row) in enumerate(df.iterrows()):
            if sh_mask.iloc[i]:
                swing_highs.append(SwingPoint(Direction.SHORT, float(row["high"]), ts, i))
            if sl_mask.iloc[i]:
                swing_lows.append(SwingPoint(Direction.LONG, float(row["low"]), ts, i))

        ms = MarketStructure(swing_highs=swing_highs, swing_lows=swing_lows)
        ms.last_bos, ms.last_bos_price, ms.last_bos_time, ms.trend = (
            self._detect_bos_choch(df, swing_highs, swing_lows)
        )
        return ms

    def _detect_bos_choch(
        self, df, swing_highs, swing_lows
    ) -> Tuple[Optional[StructureType], Optional[float], Optional[datetime], Optional[Direction]]:
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None, None, None, None

        prev_sh = swing_highs[-2]
        prev_sl = swing_lows[-2]
        last_sh = swing_highs[-1]
        last_sl = swing_lows[-1]
        close = df.iloc[-1]["close"]
        ts = df.index[-1]

        if close > prev_sh.price:
            return StructureType.BOS, prev_sh.price, ts, Direction.LONG
        if close < prev_sl.price:
            return StructureType.BOS, prev_sl.price, ts, Direction.SHORT
        if last_sh.price > prev_sh.price and close > last_sh.price:
            return StructureType.CHOCH, last_sh.price, ts, Direction.LONG
        if last_sl.price < prev_sl.price and close < last_sl.price:
            return StructureType.CHOCH, last_sl.price, ts, Direction.SHORT

        trend = None
        if last_sh.price > prev_sh.price and last_sl.price > prev_sl.price:
            trend = Direction.LONG
        elif last_sh.price < prev_sh.price and last_sl.price < prev_sl.price:
            trend = Direction.SHORT
        return None, None, None, trend

    # ─── Liquidity Sweep detection ───────────────────────────────────────────

    def _detect_liquidity_sweep(
        self, df: pd.DataFrame, bias: Direction
    ) -> Optional[dict]:
        """
        A liquidity sweep is the #1 setup of institutional traders:
        - Price spikes above equal highs (grabbing sell-side stops) then closes back below → SELL
        - Price spikes below equal lows (grabbing buy-side stops) then closes back above → BUY

        Returns dict with sweep details or None.
        """
        sc = self.sc
        lookback = min(sc.sweep_lookback, len(df) - 5)
        tol = sc.sweep_equal_tolerance
        recent = df.iloc[-(lookback + 5):]
        last = df.iloc[-1]
        prev = df.iloc[-2]

        if bias == Direction.LONG:
            # Look for equal lows in recent bars — these are the buy-side liquidity
            lows = recent["low"].iloc[:-2]
            level_candidates = []
            for i in range(len(lows) - 1):
                for j in range(i + 1, len(lows)):
                    if abs(lows.iloc[i] - lows.iloc[j]) / lows.iloc[i] < tol:
                        level_candidates.append((lows.iloc[i] + lows.iloc[j]) / 2)

            for level in level_candidates:
                # Sweep: prev candle wick pierced below level, current candle closes above
                swept = prev["low"] < level and prev["close"] > level
                if swept:
                    logger.debug("BUY-SIDE liquidity sweep detected at %.4f", level)
                    return {
                        "type": "buy_side_sweep",
                        "level": level,
                        "sweep_low": prev["low"],
                        "close_above": prev["close"],
                        "direction": Direction.LONG,
                    }

        elif bias == Direction.SHORT:
            highs = recent["high"].iloc[:-2]
            level_candidates = []
            for i in range(len(highs) - 1):
                for j in range(i + 1, len(highs)):
                    if abs(highs.iloc[i] - highs.iloc[j]) / highs.iloc[i] < tol:
                        level_candidates.append((highs.iloc[i] + highs.iloc[j]) / 2)

            for level in level_candidates:
                swept = prev["high"] > level and prev["close"] < level
                if swept:
                    logger.debug("SELL-SIDE liquidity sweep detected at %.4f", level)
                    return {
                        "type": "sell_side_sweep",
                        "level": level,
                        "sweep_high": prev["high"],
                        "close_below": prev["close"],
                        "direction": Direction.SHORT,
                    }

        return None

    # ─── Wyckoff Spring / UTAD ───────────────────────────────────────────────

    def _detect_wyckoff(
        self, df: pd.DataFrame, bias: Direction
    ) -> Optional[str]:
        """
        Simplified Wyckoff detection:
        Spring (LONG): sharp drop on high volume, immediate recovery, narrow range
        UTAD (SHORT): sharp spike above range on high volume, immediate fail
        """
        if len(df) < 30:
            return None

        last5 = df.iloc[-5:]
        atr_val = float(df["atr"].iloc[-1])
        vol_ratio = float(df["vol_ratio"].iloc[-1]) if "vol_ratio" in df.columns else 1.0

        if bias == Direction.LONG:
            # Spring: previous bar made a new low but closed mid-range on high volume
            prev = df.iloc[-2]
            prev_prev = df.iloc[-3]
            is_spring = (
                prev["low"] < prev_prev["low"]  # lower low
                and prev["close"] > prev["low"] + (prev["high"] - prev["low"]) * 0.5  # close in upper half
                and vol_ratio > 1.5              # above-average volume (absorption)
                and (prev["high"] - prev["low"]) < atr_val * 1.5  # compressed range
            )
            if is_spring:
                return "wyckoff_spring"

        elif bias == Direction.SHORT:
            prev = df.iloc[-2]
            prev_prev = df.iloc[-3]
            is_utad = (
                prev["high"] > prev_prev["high"]
                and prev["close"] < prev["high"] - (prev["high"] - prev["low"]) * 0.5
                and vol_ratio > 1.5
                and (prev["high"] - prev["low"]) < atr_val * 1.5
            )
            if is_utad:
                return "wyckoff_utad"

        return None

    # ─── CVD divergence ──────────────────────────────────────────────────────

    def _cvd_divergence(self, df: pd.DataFrame, bias: Direction) -> bool:
        """
        Cumulative Volume Delta: estimate buyer vs seller volume using candle body.
        Bullish divergence: price lower but CVD higher → hidden buying.
        Bearish divergence: price higher but CVD lower → hidden selling.
        """
        if len(df) < 20:
            return False

        # Estimate delta: positive when close > open, negative otherwise
        delta = (df["close"] - df["open"]).abs() * np.sign(df["close"] - df["open"]) * df["volume"]
        cvd = delta.cumsum()

        lookback = 15
        recent_cvd = cvd.iloc[-lookback:]
        recent_price = df["close"].iloc[-lookback:]

        if bias == Direction.LONG:
            price_trend = np.polyfit(range(lookback), recent_price.values, 1)[0]
            cvd_trend = np.polyfit(range(lookback), recent_cvd.values, 1)[0]
            return price_trend < 0 and cvd_trend > 0  # price down, buying pressure up

        elif bias == Direction.SHORT:
            price_trend = np.polyfit(range(lookback), recent_price.values, 1)[0]
            cvd_trend = np.polyfit(range(lookback), recent_cvd.values, 1)[0]
            return price_trend > 0 and cvd_trend < 0  # price up, selling pressure up

        return False

    # ─── Order Blocks ────────────────────────────────────────────────────────

    def _find_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        sc = self.sc
        obs = []
        current_price = df.iloc[-1]["close"]
        atr_vals = df["atr"]

        for i in range(2, len(df) - 3):
            candle = df.iloc[i]
            atr_val = float(atr_vals.iloc[i])
            if atr_val == 0:
                continue
            body_pct = abs(candle["close"] - candle["open"]) / (candle["high"] - candle["low"] + 1e-9)
            if body_pct < sc.ob_min_body_pct:
                continue

            next3 = df.iloc[i + 1: i + 4]
            impulse_up = (next3["close"].max() - candle["high"]) > 2 * atr_val
            impulse_down = (candle["low"] - next3["close"].min()) > 2 * atr_val

            if candle["close"] < candle["open"] and impulse_up:
                ob = OrderBlock(
                    direction=Direction.LONG,
                    top=float(candle["open"]),
                    bottom=float(candle["close"]),
                    origin_time=df.index[i],
                    origin_index=i,
                    strength=min((next3["close"].max() - candle["high"]) / atr_val, 5.0),
                )
                ob.mitigated = current_price < ob.bottom or (len(df) - i > sc.ob_max_age_bars)
                if not ob.mitigated:
                    obs.append(ob)

            elif candle["close"] > candle["open"] and impulse_down:
                ob = OrderBlock(
                    direction=Direction.SHORT,
                    top=float(candle["close"]),
                    bottom=float(candle["open"]),
                    origin_time=df.index[i],
                    origin_index=i,
                    strength=min((candle["low"] - next3["close"].min()) / atr_val, 5.0),
                )
                ob.mitigated = current_price > ob.top or (len(df) - i > sc.ob_max_age_bars)
                if not ob.mitigated:
                    obs.append(ob)

        return obs

    # ─── Fair Value Gaps ─────────────────────────────────────────────────────

    def _find_fvgs(self, df: pd.DataFrame) -> List[FairValueGap]:
        fvgs = []
        current_price = df.iloc[-1]["close"]
        atr_vals = df["atr"]

        for i in range(1, len(df) - 1):
            prev = df.iloc[i - 1]
            nxt = df.iloc[i + 1]
            atr_val = float(atr_vals.iloc[i])
            min_size = self.sc.fvg_min_size_atr * atr_val

            gap_top = float(nxt["low"])
            gap_bot = float(prev["high"])
            if gap_top > gap_bot and (gap_top - gap_bot) >= min_size:
                fvg = FairValueGap(
                    direction=Direction.LONG, top=gap_top, bottom=gap_bot,
                    origin_time=df.index[i], origin_index=i,
                )
                fvg.filled = current_price < gap_bot
                if not fvg.filled:
                    fvgs.append(fvg)

            gap_top2 = float(prev["low"])
            gap_bot2 = float(nxt["high"])
            if gap_top2 > gap_bot2 and (gap_top2 - gap_bot2) >= min_size:
                fvg = FairValueGap(
                    direction=Direction.SHORT, top=gap_top2, bottom=gap_bot2,
                    origin_time=df.index[i], origin_index=i,
                )
                fvg.filled = current_price > gap_top2
                if not fvg.filled:
                    fvgs.append(fvg)

        return fvgs

    # ─── Confluence scoring & signal assembly ────────────────────────────────

    def _build_signal(
        self, bias, entry_df, struct_df, obs, fvgs,
        struct_ms, current_price, sweep, funding_rate
    ) -> Optional[TradeSignal]:

        atr_val = float(entry_df["atr"].iloc[-1])
        if atr_val == 0:
            return None

        last = entry_df.iloc[-1]
        rsi_val = float(last.get("rsi", 50))
        vol_ratio = float(last.get("vol_ratio", 1.0))
        macd_hist = float(last.get("macd_hist", 0))
        adx_val = float(last.get("adx", 0))
        sc = self.sc

        score = 0
        notes = []

        # ── Score 1: Structure trend alignment (2pts) ─────────────────────────
        if struct_ms.trend == bias:
            score += 2
            notes.append(f"4H structure {bias.value}")
        elif struct_ms.last_bos == StructureType.BOS and struct_ms.trend == bias:
            score += 2
            notes.append("4H BOS confirmed")
        elif struct_ms.last_bos == StructureType.CHOCH:
            score += 1
            notes.append("4H CHOCH — potential reversal")

        # ── Score 2: Liquidity sweep (2pts — highest weight) ──────────────────
        if sweep and sweep["direction"] == bias:
            score += 2
            notes.append(f"Liquidity sweep at {sweep['level']:.4f}")

        # ── Score 3: Order Block or FVG confluence (2pts) ─────────────────────
        zone_found = False
        zone_top = zone_bot = 0.0

        for ob in sorted(obs, key=lambda x: -x.origin_index):
            if ob.direction == bias and ob.contains_price(current_price):
                score += 2
                zone_top, zone_bot = ob.top, ob.bottom
                notes.append(f"{'Bullish' if bias == Direction.LONG else 'Bearish'} OB {ob.bottom:.4f}–{ob.top:.4f}")
                zone_found = True
                break

        if not zone_found:
            for fvg in sorted(fvgs, key=lambda x: -x.origin_index):
                if fvg.direction == bias and fvg.contains_price(current_price):
                    score += 1
                    zone_top, zone_bot = fvg.top, fvg.bottom
                    notes.append(f"FVG {fvg.bottom:.4f}–{fvg.top:.4f}")
                    zone_found = True
                    break

        # ── Score 4: Wyckoff Spring / UTAD (1pt) ──────────────────────────────
        wyckoff = self._detect_wyckoff(entry_df, bias)
        if wyckoff:
            score += 1
            notes.append(wyckoff.replace("_", " ").title())

        # ── Score 5: CVD divergence (1pt) ─────────────────────────────────────
        if self._cvd_divergence(entry_df, bias):
            score += 1
            notes.append("CVD divergence")

        # ── Score 6: RSI extreme (1pt) ────────────────────────────────────────
        if bias == Direction.LONG and rsi_val < sc.rsi_oversold:
            score += 1
            notes.append(f"RSI oversold {rsi_val:.1f}")
        elif bias == Direction.SHORT and rsi_val > sc.rsi_overbought:
            score += 1
            notes.append(f"RSI overbought {rsi_val:.1f}")

        # ── Score 7: MACD + ADX momentum (1pt) ───────────────────────────────
        macd_ok = (bias == Direction.LONG and macd_hist > 0) or (bias == Direction.SHORT and macd_hist < 0)
        if macd_ok and adx_val > 20:
            score += 1
            notes.append(f"MACD + ADX {adx_val:.0f}")

        logger.debug("Confluence score: %d/10 | %s", score, " | ".join(notes))

        if score < sc.min_confluence_score:
            return None

        # ── Build stops and targets ───────────────────────────────────────────
        if not zone_found:
            # No structural zone — use ATR-based stops only if score is very high
            if score < 8:
                return None
            if bias == Direction.LONG:
                zone_bot = current_price - atr_val
            else:
                zone_top = current_price + atr_val

        if bias == Direction.LONG:
            sl = zone_bot - sc.atr_sl_mult * atr_val
            risk_dist = current_price - sl
        else:
            sl = zone_top + sc.atr_sl_mult * atr_val
            risk_dist = sl - current_price

        if risk_dist <= 0:
            return None

        tp2 = self._next_liquidity_target(struct_ms, current_price, bias)
        if tp2 == 0:
            tp2 = (current_price + cfg.risk.min_reward_risk * risk_dist
                   if bias == Direction.LONG
                   else current_price - cfg.risk.min_reward_risk * risk_dist)

        tp1 = (current_price + risk_dist if bias == Direction.LONG
               else current_price - risk_dist)

        actual_rr = (abs(tp2 - current_price) / risk_dist) if risk_dist > 0 else 0
        if actual_rr < cfg.risk.min_reward_risk:
            return None

        strength = (
            SignalStrength.STRONG if score >= 8
            else SignalStrength.MODERATE if score >= 6
            else SignalStrength.WEAK
        )

        return TradeSignal(
            symbol=cfg.exchange.symbol,
            direction=bias,
            entry_price=current_price,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            timestamp=datetime.now(timezone.utc),
            timeframe=cfg.timeframes.entry_tf,
            strength=strength,
            confluence_notes=notes + [f"Score: {score}/10"],
            atr=atr_val,
            risk_reward=round(actual_rr, 2),
        )

    # ─── Next liquidity target ────────────────────────────────────────────────

    def _next_liquidity_target(
        self, ms: MarketStructure, price: float, direction: Direction
    ) -> float:
        if direction == Direction.LONG:
            targets = [sp.price for sp in ms.swing_highs if sp.price > price]
            return min(targets) if targets else 0.0
        targets = [sp.price for sp in ms.swing_lows if sp.price < price]
        return max(targets) if targets else 0.0
