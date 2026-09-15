"""
Wires together candle building + SMMA + universe screening.

Produces a `CrossoverEvent` every time a screened symbol's SMMA20/120
cross, which the ML layer (ml/predict.py) then scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from brokers.base import Tick
from config import CONFIG
from data.resample import CandleBuilder
from data.tick_store import TickStore
from features.engineering import build_feature_snapshot, passes_screen, FeatureSnapshot
from indicators.smma import SMMATracker


@dataclass
class CrossoverEvent:
    symbol: str
    signal: str  # "BUY" or "SELL"
    features: FeatureSnapshot


class CrossoverEngine:
    def __init__(self):
        self.tick_store = TickStore()
        self.candles = CandleBuilder(CONFIG.timeframe_minutes)
        self.smma = SMMATracker(CONFIG.smma_fast, CONFIG.smma_slow)

    def on_tick(self, tick: Tick) -> Optional[CrossoverEvent]:
        """Feed one live tick through the whole pipeline. Returns a
        CrossoverEvent if this tick's candle close triggered a fresh
        SMMA20/120 crossover on a symbol that currently passes the
        price/depth screen -- else None."""
        self.tick_store.add(tick)
        closed_candle = self.candles.on_tick(tick)

        if closed_candle is None:
            return None  # candle still forming; SMMA only updates on close

        signal = self.smma.update(tick.symbol, closed_candle.close)
        if signal is None:
            return None

        if not passes_screen(tick.ltp, tick.bid_qty, tick.ask_qty):
            return None  # crossover happened but symbol fails universe filters right now

        feat = build_feature_snapshot(
            symbol=tick.symbol,
            recent_ticks=self.tick_store.recent(tick.symbol, 50),
            recent_candles=self.candles.closed_candles(tick.symbol, 30),
            smma_pair=self.smma.get(tick.symbol),
        )
        if feat is None:
            return None

        return CrossoverEvent(symbol=tick.symbol, signal=signal, features=feat)

    def warm_start(self, symbol: str, historical_candles: list) -> int:
        """
        Prime SMMA + candle history from a broker's historical-candle
        REST response (rows of [epoch, open, high, low, close, volume],
        oldest first) instead of waiting for ~2 hours of live 1-minute
        candles to accumulate after every startup or reconnect.

        Any crossover signal detected during this replay is intentionally
        discarded -- warm-start is for priming indicator state only, not
        for generating trade signals from historical data. Returns the
        number of candles successfully applied.
        """
        from data.resample import Candle

        applied = 0
        for row in historical_candles:
            try:
                epoch, o, h, l, c, v = row[0], row[1], row[2], row[3], row[4], row[5]
            except (IndexError, TypeError):
                continue
            candle = Candle(
                symbol=symbol, minute_epoch=int(epoch),
                open=float(o), high=float(h), low=float(l), close=float(c),
                volume=int(v),
            )
            self.candles.seed_closed(candle)
            self.smma.update(symbol, candle.close)  # discard crossover signal
            applied += 1
        return applied
