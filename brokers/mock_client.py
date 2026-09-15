"""
Synthetic tick generator standing in for a real broker feed.

This is what lets every other module (SMMA, feature engineering, ML,
paper trading, dashboard) be developed and demoed WITHOUT a Fyers/Angel
One account or network access. Swap `broker: "mock"` -> `"fyers"` or
`"angelone"` in config.py once real credentials are available; nothing
else in the codebase changes.

Each symbol follows a mean-reverting random walk with an injected drift
that occasionally trends, so SMMA20/120 crossovers actually occur.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Callable

from brokers.base import BrokerClient, Tick


class MockClient(BrokerClient):
    name = "mock"

    def __init__(self, symbols, tick_interval: float = 0.25, speed: float = 1.0):
        super().__init__(symbols)
        self.tick_interval = tick_interval
        self.speed = speed
        self._running = False
        self._state = {}
        for s in self.symbols:
            base = random.uniform(40, 480)
            self._state[s] = {
                "price": base,
                "drift": 0.0,
                "drift_timer": 0,
                "cum_vol": random.randint(50_000, 500_000),
            }

    def connect(self) -> None:
        self._running = True

    def disconnect(self) -> None:
        self._running = False

    def subscribe(self, on_tick: Callable[[Tick], None]) -> None:
        self._running = True
        thread = threading.Thread(target=self._run, args=(on_tick,), daemon=True)
        thread.start()
        self._thread = thread

    def _run(self, on_tick: Callable[[Tick], None]):
        while self._running:
            for symbol in self.symbols:
                st = self._state[symbol]

                # occasionally start/refresh a directional drift so SMMA20/120
                # actually cross instead of just chopping sideways
                if st["drift_timer"] <= 0:
                    st["drift"] = random.choice([-1, -1, 0, 1, 1]) * random.uniform(0.01, 0.08)
                    st["drift_timer"] = random.randint(20, 120)
                st["drift_timer"] -= 1

                noise = random.gauss(0, 0.15)
                st["price"] = max(1.0, st["price"] + st["drift"] + noise)

                ltq = random.randint(1, 5000)
                st["cum_vol"] += ltq

                spread = max(0.05, st["price"] * 0.0008)
                bid_price = round(st["price"] - spread, 2)
                ask_price = round(st["price"] + spread, 2)

                # bias depth quantities in the direction of the current drift,
                # so bid/ask imbalance correlates loosely with trend -- this
                # is what the ML model should learn to exploit
                skew = 1.0 + max(-0.6, min(0.6, st["drift"] * 4))
                bid_qty = int(max(50_000, random.gauss(9_00_000, 3_00_000) * skew))
                ask_qty = int(max(50_000, random.gauss(9_00_000, 3_00_000) * (2 - skew)))

                tick = Tick(
                    symbol=symbol,
                    ts=time.time(),
                    ltp=round(st["price"], 2),
                    ltq=ltq,
                    total_traded_qty=st["cum_vol"],
                    bid_price=bid_price,
                    bid_qty=bid_qty,
                    ask_price=ask_price,
                    ask_qty=ask_qty,
                )
                on_tick(tick)
            time.sleep(self.tick_interval / self.speed)
