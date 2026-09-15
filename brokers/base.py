"""
Broker abstraction layer.

Every broker adapter (Fyers, Angel One, Mock) implements `BrokerClient` so
the rest of the system (data pipeline, screener, ML, paper trading) never
needs to know which broker is behind it.

Credentials are pulled from environment variables ONLY. Never hardcode
API keys / client IDs / TOTP secrets / access tokens in source files.
Required env vars per broker are documented in each adapter module and
in README.md.
"""
from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass
class Tick:
    """Normalized market-depth tick, identical shape regardless of broker."""
    symbol: str
    ts: float                # unix epoch seconds
    ltp: float                # last traded price
    ltq: int                  # last traded quantity
    total_traded_qty: int     # cumulative volume for the day (ETQ proxy)
    bid_price: float
    bid_qty: int
    ask_price: float
    ask_qty: int


class BrokerClient(abc.ABC):
    """Common interface every broker adapter must implement."""

    name: str = "base"

    def __init__(self, symbols: Iterable[str]):
        self.symbols = list(symbols)

    @abc.abstractmethod
    def connect(self) -> None:
        """Authenticate and open the streaming session."""

    @abc.abstractmethod
    def subscribe(self, on_tick: Callable[[Tick], None]) -> None:
        """
        Start streaming market-depth ticks for self.symbols.
        `on_tick` is called once per normalized Tick.
        This call is expected to block (run its own loop) or spawn a
        background thread — adapters document which.
        """

    @abc.abstractmethod
    def disconnect(self) -> None:
        ...

    # Optional: historical 1-min candles for warm-starting SMMA on startup
    def historical_candles(self, symbol: str, minutes_back: int = 300):
        raise NotImplementedError(f"{self.name} adapter has no historical fetch implemented")


def get_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in your shell or a local .env file (never commit it)."
        )
    return val


def build_broker(broker_name: str, symbols: Iterable[str]) -> BrokerClient:
    """Factory so main.py / dashboard don't import every adapter directly."""
    broker_name = broker_name.lower()
    if broker_name == "fyers":
        from brokers.fyers_client import FyersClient
        return FyersClient(symbols)
    if broker_name == "angelone":
        from brokers.angelone_client import AngelOneClient
        return AngelOneClient(symbols)
    if broker_name == "mock":
        from brokers.mock_client import MockClient
        return MockClient(symbols)
    raise ValueError(f"Unknown broker '{broker_name}'")
