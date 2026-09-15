"""
Turns raw ticks + candles + SMMA state into the feature vector the ML
model scores at every crossover, and into a live feature snapshot for
the deterioration monitor on open positions.

This is the heart of the "LTQ + Bid/Ask + SMMA relationship" requirement:
every feature here is designed to capture whether a crossover has real
participation behind it, or is a thin/false move likely to fail.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from statistics import mean, pstdev
from typing import List, Optional

from brokers.base import Tick
from config import CONFIG
from data.resample import Candle
from indicators.smma import SMMAPair


@dataclass
class FeatureSnapshot:
    symbol: str
    ts: float
    ltp: float

    # --- SMMA context ---
    smma_fast: float
    smma_slow: float
    smma_spread_pct: float
    smma_slope_fast: float          # fast SMMA change over lookback (trend strength)

    # --- LTQ / volume dynamics ---
    ltq_last: int
    ltq_avg_recent: float
    ltq_change_pct: float           # last ltq vs recent average
    ltq_acceleration: float         # 2nd derivative proxy: is LTQ growth speeding up?
    volume_last_candle: int

    # --- Bid/Ask depth ---
    bid_price: float
    bid_qty: int
    ask_price: float
    ask_qty: int
    bid_ask_imbalance: float        # (bidQty - askQty) / (bidQty + askQty), range [-1, 1]
    imbalance_trend: float          # change in imbalance over recent ticks
    spread_pct: float               # (ask - bid) / ltp

    # --- Price action ---
    price_change_pct_1m: float      # close vs close 1 candle ago
    price_change_pct_5m: float

    def to_feature_vector(self) -> list[float]:
        """Ordered numeric vector consumed by the ML model. Keep FEATURE_ORDER in sync."""
        return [
            self.smma_spread_pct, self.smma_slope_fast,
            self.ltq_change_pct, self.ltq_acceleration,
            self.bid_ask_imbalance, self.imbalance_trend, self.spread_pct,
            self.price_change_pct_1m, self.price_change_pct_5m,
            self.ltq_avg_recent, float(self.volume_last_candle),
        ]

    def as_dict(self) -> dict:
        return asdict(self)


FEATURE_ORDER = [
    "smma_spread_pct", "smma_slope_fast",
    "ltq_change_pct", "ltq_acceleration",
    "bid_ask_imbalance", "imbalance_trend", "spread_pct",
    "price_change_pct_1m", "price_change_pct_5m",
    "ltq_avg_recent", "volume_last_candle",
]


def passes_screen(ltp: float, bid_qty: int, ask_qty: int) -> bool:
    """
    NSE LTP band + minimum depth-quantity filter from the assignment spec.

    CONFIG.min_bid_qty / min_ask_qty stay at the assignment's literal
    "10 lakh" value by default -- do not change those defaults. This
    function reads an optional environment override,
    SCREENER_MIN_DEPTH_QTY, so a real live-data run can be calibrated
    without altering the spec-matching default in source.

    Why an override exists at all: verified against two full live NSE
    sessions (2026-08-17, 2026-08-18; see data/store/ and
    reports/depth_threshold_analysis.json) that aggregate 5-level bid+ask
    quantity, summed per side, essentially never exceeds 10 lakh on BOTH
    sides simultaneously for stocks in the ₹30-500 band -- only 1 of 43
    watchlist symbols ever cleared it, and only ~0-2% of the time. At the
    literal threshold, essentially zero crossovers are trainable per day.
    Full writeup: README.md "On the 10 lakh depth threshold".
    """
    if ltp < CONFIG.price_min or ltp > CONFIG.price_max:
        return False

    min_qty = _effective_min_depth_qty()
    return bid_qty > min_qty and ask_qty > min_qty


def _effective_min_depth_qty() -> int:
    override = os.environ.get("SCREENER_MIN_DEPTH_QTY")
    if override is not None:
        try:
            return int(override)
        except ValueError:
            pass  # fall through to spec default on a malformed override
    return CONFIG.min_bid_qty


def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b


def build_feature_snapshot(
    symbol: str,
    recent_ticks: List[Tick],
    recent_candles: List[Candle],
    smma_pair: SMMAPair,
) -> Optional[FeatureSnapshot]:
    """
    recent_ticks: most recent ticks for this symbol, oldest-first, length
                  >= CONFIG.ltq_accel_lookback for full accuracy (degrades
                  gracefully with fewer).
    recent_candles: most recent CLOSED 1-min candles, oldest-first.
    """
    if not recent_ticks or smma_pair.fast.value is None or smma_pair.slow.value is None:
        return None

    last_tick = recent_ticks[-1]

    # --- LTQ dynamics ---
    window = recent_ticks[-CONFIG.ltq_accel_lookback:]
    ltqs = [t.ltq for t in window]
    ltq_avg = mean(ltqs) if ltqs else 0.0
    ltq_change_pct = _pct(last_tick.ltq, ltq_avg)

    if len(ltqs) >= 3:
        first_half = mean(ltqs[: len(ltqs) // 2])
        second_half = mean(ltqs[len(ltqs) // 2:])
        ltq_acceleration = _pct(second_half, first_half)
    else:
        ltq_acceleration = 0.0

    # --- Bid/Ask imbalance ---
    total_depth = last_tick.bid_qty + last_tick.ask_qty
    imbalance = (last_tick.bid_qty - last_tick.ask_qty) / total_depth if total_depth else 0.0

    imb_window = recent_ticks[-CONFIG.imbalance_lookback:]
    if len(imb_window) >= 2:
        earlier = imb_window[0]
        earlier_total = earlier.bid_qty + earlier.ask_qty
        earlier_imb = (earlier.bid_qty - earlier.ask_qty) / earlier_total if earlier_total else 0.0
        imbalance_trend = imbalance - earlier_imb
    else:
        imbalance_trend = 0.0

    spread_pct = _pct(last_tick.ask_price, last_tick.bid_price) if last_tick.bid_price else 0.0

    # --- SMMA context ---
    smma_spread_pct = smma_pair.spread_pct or 0.0
    if len(recent_candles) >= 3:
        # crude slope proxy: fast SMMA distance moved relative to price, over last few candles
        smma_slope_fast = _pct(smma_pair.fast.value, recent_candles[-3].close)
    else:
        smma_slope_fast = 0.0

    # --- Price action ---
    price_change_1m = _pct(recent_candles[-1].close, recent_candles[-2].close) if len(recent_candles) >= 2 else 0.0
    price_change_5m = _pct(recent_candles[-1].close, recent_candles[-6].close) if len(recent_candles) >= 6 else 0.0

    return FeatureSnapshot(
        symbol=symbol, ts=last_tick.ts, ltp=last_tick.ltp,
        smma_fast=smma_pair.fast.value, smma_slow=smma_pair.slow.value,
        smma_spread_pct=smma_spread_pct, smma_slope_fast=smma_slope_fast,
        ltq_last=last_tick.ltq, ltq_avg_recent=ltq_avg,
        ltq_change_pct=ltq_change_pct, ltq_acceleration=ltq_acceleration,
        volume_last_candle=recent_candles[-1].volume if recent_candles else 0,
        bid_price=last_tick.bid_price, bid_qty=last_tick.bid_qty,
        ask_price=last_tick.ask_price, ask_qty=last_tick.ask_qty,
        bid_ask_imbalance=imbalance, imbalance_trend=imbalance_trend, spread_pct=spread_pct,
        price_change_pct_1m=price_change_1m, price_change_pct_5m=price_change_5m,
    )


def explain_signal(signal: str, feat: FeatureSnapshot) -> list[str]:
    """
    Human-readable reasons behind ACCEPT/AVOID, independent of the ML
    model's own probability -- used in the dashboard's "Reason" column
    and the trade log, per the assignment's requirement for reasons.
    """
    reasons = []
    bullish_imbalance = feat.bid_ask_imbalance > 0.1
    bearish_imbalance = feat.bid_ask_imbalance < -0.1

    if signal == "BUY":
        reasons.append("LTQ rising" if feat.ltq_change_pct > 0.1 else "LTQ flat/falling")
        reasons.append("Bid support strong" if bullish_imbalance else
                        "Ask pressure building" if bearish_imbalance else "Depth balanced")
        reasons.append("SMMA spread widening" if feat.smma_slope_fast > 0 else "SMMA spread flat/narrowing")
    else:  # SELL
        reasons.append("LTQ rising into weakness" if feat.ltq_change_pct > 0.1 else "LTQ fading")
        reasons.append("Ask pressure strong" if bearish_imbalance else
                        "Bid support building" if bullish_imbalance else "Depth balanced")
        reasons.append("SMMA spread widening down" if feat.smma_slope_fast < 0 else "SMMA spread flat/narrowing")

    if feat.ltq_acceleration > 0.2:
        reasons.append("LTQ accelerating")
    elif feat.ltq_acceleration < -0.2:
        reasons.append("LTQ decelerating")

    return reasons
