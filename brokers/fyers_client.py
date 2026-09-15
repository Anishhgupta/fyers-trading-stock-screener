"""
Fyers adapter using the official `fyers-apiv3` SDK.

Install:
    pip install fyers-apiv3

Required environment variables:
    FYERS_CLIENT_ID
    FYERS_SECRET_ID
    FYERS_REDIRECT_URI
    FYERS_ACCESS_TOKEN    generated DAILY via fyers_login.py (Fyers access
                          tokens expire at end of each trading day, unlike
                          Angel One's self-regenerating TOTP)

IMPORTANT -- message schema note (corrected 2026-08-31 against real
live data, not assumed from docs):

Fyers' v3 websocket sends TWO SEPARATE message types on the same
socket, and neither one alone has everything this system needs:

  "SymbolUpdate" (msg['type'] == 'sf'):
      Has ltp, last_traded_qty, vol_traded_today, and top-of-book
      bid_price/ask_price/bid_size/ask_size -- but only ONE level of
      depth, not five.

  "DepthUpdate" (msg['type'] == 'dp'):
      Has FLAT (not nested) keys bid_price1..5, ask_price1..5,
      bid_size1..5, ask_size1..5, bid_order1..5, ask_order1..5 -- full
      5-level depth, but NO ltp/ltq/volume fields at all.

So this adapter subscribes to BOTH data types and merges the latest
known state of each per symbol into one normalized Tick, emitting a new
Tick whenever either side updates (using the most recent cached value
for whichever side didn't just update). This matches the assignment's
"5-level Bid/Ask depth" requirement while still getting LTQ/ETQ.
"""
from __future__ import annotations

import time
from typing import Callable

from brokers.base import BrokerClient, Tick, get_env


class FyersClient(BrokerClient):
    name = "fyers"

    def __init__(self, symbols):
        super().__init__(symbols)
        self.client_id = get_env("FYERS_CLIENT_ID")
        self.access_token = get_env("FYERS_ACCESS_TOKEN")
        self._fyers = None
        self._ws = None
        # Per-symbol caches of the most recent update from each message
        # type -- a Tick is only ever emitted once both have been seen
        # at least once for that symbol.
        self._last_symbol_data: dict[str, dict] = {}
        self._last_depth_data: dict[str, dict] = {}

    def connect(self) -> None:
        from fyers_apiv3 import fyersModel
        print("  [Fyers] Initializing FyersModel client...")
        self._fyers = fyersModel.FyersModel(
            client_id=self.client_id,
            token=self.access_token,
            is_async=False,
            log_path="",
        )
        print("  [Fyers] Verifying token via get_profile()...")
        profile = self._fyers.get_profile()
        if profile.get("s") != "ok":
            raise RuntimeError(f"Fyers auth failed: {profile}")
        print("  [Fyers] connect() complete.")

    def subscribe(self, on_tick: Callable[[Tick], None]) -> None:
        from fyers_apiv3.FyersWebsocket import data_ws

        def _on_message(msg):
            tick = self._handle_message(msg)
            if tick is not None:
                on_tick(tick)

        def _on_open():
            print(f"  [Fyers] Websocket opened, subscribing to {len(self.symbols)} symbol(s) "
                  f"on BOTH SymbolUpdate (LTQ/volume) and DepthUpdate (5-level depth)...")
            self._ws.subscribe(symbols=self.symbols, data_type="SymbolUpdate")
            self._ws.subscribe(symbols=self.symbols, data_type="DepthUpdate")
            print("  [Fyers] Subscribe requests sent. Waiting for ticks...")
            self._ws.keep_running()

        def _on_error(message):
            print(f"  [Fyers] Websocket ERROR: {message}")

        def _on_close(message):
            print(f"  [Fyers] Websocket closed: {message}")

        print("  [Fyers] Creating websocket connection...")
        self._ws = data_ws.FyersDataSocket(
            access_token=f"{self.client_id}:{self.access_token}",
            log_path="",
            litemode=False,
            write_to_file=False,
            reconnect=True,
            on_connect=_on_open,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
        )
        print("  [Fyers] Calling ws.connect() (this blocks until the socket closes)...")
        self._ws.connect()

    def _handle_message(self, msg: dict) -> Tick | None:
        """
        Merge-on-update logic: cache whichever message type just arrived,
        then -- if we've now seen at least one of EACH type for this
        symbol -- emit a merged Tick using the freshest known value of
        both sides.
        """
        msg_type = msg.get("type")
        symbol = msg.get("symbol")
        if not symbol or msg_type not in ("sf", "dp"):
            return None  # control/ack messages ('cn', 'ful', 'sub') have no symbol

        try:
            if msg_type == "sf":
                self._last_symbol_data[symbol] = {
                    "ltp": float(msg["ltp"]),
                    "ltq": int(msg.get("last_traded_qty", 0)),
                    "total_traded_qty": int(msg.get("vol_traded_today", 0)),
                }
            else:  # "dp"
                bid_qty_total = sum(int(msg.get(f"bid_size{i}", 0)) for i in range(1, 6))
                ask_qty_total = sum(int(msg.get(f"ask_size{i}", 0)) for i in range(1, 6))
                self._last_depth_data[symbol] = {
                    "bid_price": float(msg.get("bid_price1", 0.0)),
                    "ask_price": float(msg.get("ask_price1", 0.0)),
                    "bid_qty": bid_qty_total,
                    "ask_qty": ask_qty_total,
                }
        except (KeyError, ValueError, TypeError):
            return None  # malformed/partial packet, skip it

        sd = self._last_symbol_data.get(symbol)
        dd = self._last_depth_data.get(symbol)
        if sd is None or dd is None:
            return None  # still waiting on the other side's first update for this symbol

        return Tick(
            symbol=symbol,
            ts=time.time(),
            ltp=sd["ltp"], ltq=sd["ltq"], total_traded_qty=sd["total_traded_qty"],
            bid_price=dd["bid_price"], bid_qty=dd["bid_qty"],
            ask_price=dd["ask_price"], ask_qty=dd["ask_qty"],
        )

    def disconnect(self) -> None:
        if self._ws is not None:
            self._ws.close_connection()

    def historical_candles(self, symbol: str, minutes_back: int = 300, end_ts: float | None = None):
        """
        end_ts: optional unix timestamp to fetch UP TO, instead of the
        current time. Used when warm-starting an OFFLINE replay for a
        past day -- pass the timestamp of that day's first live tick so
        the historical window ends exactly where the tick CSV begins,
        with no overlap between the two (overlap would double-count
        those minutes' prices in SMMA and risk fabricating duplicate
        crossover detections).
        """
        end = int(end_ts) if end_ts is not None else int(time.time())
        start = end - minutes_back * 60
        data = self._fyers.history({
            "symbol": symbol,
            "resolution": "1",
            "date_format": "0",
            "range_from": str(start),
            "range_to": str(end),
            "cont_flag": "1",
        })
        if data.get("s") != "ok":
            raise RuntimeError(f"Fyers history fetch failed: {data}")
        return data["candles"]
