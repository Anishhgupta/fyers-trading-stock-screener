"""
Smoothed Moving Average (SMMA), a.k.a. RMA / Wilder's moving average.

SMMA_t = SMMA_{t-1} + (price_t - SMMA_{t-1}) / period   for t > period
SMMA_period = simple average of the first `period` closes (seed value)

Kept as an incremental, per-symbol running calculator so it's O(1) per new
candle instead of recomputing over the whole history each time -- matters
for a live, ticking system.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional


class SMMA:
    def __init__(self, period: int):
        self.period = period
        self._seed_buffer = deque(maxlen=period)
        self.value: Optional[float] = None

    def update(self, price: float) -> Optional[float]:
        if self.value is None:
            self._seed_buffer.append(price)
            if len(self._seed_buffer) == self.period:
                self.value = sum(self._seed_buffer) / self.period
            return self.value
        self.value = self.value + (price - self.value) / self.period
        return self.value

    @property
    def ready(self) -> bool:
        return self.value is not None


class SMMAPair:
    """Tracks fast + slow SMMA for one symbol and exposes crossover state."""

    def __init__(self, fast_period: int, slow_period: int):
        self.fast = SMMA(fast_period)
        self.slow = SMMA(slow_period)
        self._prev_fast: Optional[float] = None
        self._prev_slow: Optional[float] = None

    def update(self, close_price: float) -> Optional[str]:
        """
        Feed one closed candle's close price. Returns "BUY", "SELL", or
        None (no crossover this bar / not enough history yet).
        """
        f = self.fast.update(close_price)
        s = self.slow.update(close_price)

        signal = None
        if f is not None and s is not None and self._prev_fast is not None and self._prev_slow is not None:
            crossed_up = self._prev_fast <= self._prev_slow and f > s
            crossed_down = self._prev_fast >= self._prev_slow and f < s
            if crossed_up:
                signal = "BUY"
            elif crossed_down:
                signal = "SELL"

        if f is not None:
            self._prev_fast = f
        if s is not None:
            self._prev_slow = s
        return signal

    @property
    def spread(self) -> Optional[float]:
        if self.fast.value is None or self.slow.value is None:
            return None
        return self.fast.value - self.slow.value

    @property
    def spread_pct(self) -> Optional[float]:
        if self.fast.value is None or self.slow.value is None or self.slow.value == 0:
            return None
        return (self.fast.value - self.slow.value) / self.slow.value


class SMMATracker:
    """Owns one SMMAPair per symbol."""

    def __init__(self, fast_period: int, slow_period: int):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._pairs: Dict[str, SMMAPair] = {}

    def _pair(self, symbol: str) -> SMMAPair:
        if symbol not in self._pairs:
            self._pairs[symbol] = SMMAPair(self.fast_period, self.slow_period)
        return self._pairs[symbol]

    def update(self, symbol: str, close_price: float) -> Optional[str]:
        return self._pair(symbol).update(close_price)

    def get(self, symbol: str) -> SMMAPair:
        return self._pair(symbol)
