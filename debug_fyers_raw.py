"""
Raw Fyers websocket diagnostic -- prints EVERY message exactly as Fyers
sends it, with no processing. Run this live, during market hours, to
see ground truth before assuming anything about message format.

Usage:
    python debug_fyers_raw.py

Tries "DepthUpdate" first for a few seconds; if nothing arrives, retries
with "SymbolUpdate" so we can tell whether the issue is depth-specific
(often a separate paid data entitlement) or affects all live data.
"""
import time
from dotenv import load_dotenv
load_dotenv()

from brokers.base import get_env

client_id = get_env("FYERS_CLIENT_ID")
access_token = get_env("FYERS_ACCESS_TOKEN")

TEST_SYMBOLS = ["NSE:SBIN-EQ", "NSE:AXISBANK-EQ"]

message_count = 0


def on_message(msg):
    global message_count
    message_count += 1
    print(f"\n--- MESSAGE #{message_count} ---")
    print(msg)


def on_error(msg):
    print(f"ERROR: {msg}")


def on_close(msg):
    print(f"CLOSED: {msg}")


def try_data_type(data_type: str, seconds: int = 15):
    global message_count
    message_count = 0
    from fyers_apiv3.FyersWebsocket import data_ws

    def on_open():
        print(f"Websocket open. Subscribing with data_type='{data_type}'...")
        ws.subscribe(symbols=TEST_SYMBOLS, data_type=data_type)
        ws.keep_running()

    ws = data_ws.FyersDataSocket(
        access_token=f"{client_id}:{access_token}",
        log_path="",
        litemode=False,
        write_to_file=False,
        reconnect=False,
        on_connect=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    print(f"\n{'=' * 60}\nTesting data_type='{data_type}' for {seconds}s...\n{'=' * 60}")
    import threading
    t = threading.Thread(target=ws.connect, daemon=True)
    t.start()
    time.sleep(seconds)
    try:
        ws.close_connection()
    except Exception:
        pass
    print(f"\n>>> Result for '{data_type}': {message_count} message(s) received in {seconds}s\n")
    return message_count


if __name__ == "__main__":
    depth_count = try_data_type("DepthUpdate", seconds=15)
    time.sleep(2)
    symbol_count = try_data_type("SymbolUpdate", seconds=15)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"DepthUpdate:  {depth_count} messages")
    print(f"SymbolUpdate: {symbol_count} messages")
    if depth_count == 0 and symbol_count > 0:
        print(
            "\n-> DepthUpdate specifically is not delivering data, while "
            "SymbolUpdate works. This strongly suggests your Fyers account "
            "does not have Level-2 market depth data enabled (often a "
            "separate paid data-vendor entitlement, not just an API "
            "permission). Contact Fyers support to confirm/enable it, or "
            "adapt the system to work from SymbolUpdate's LTP + top-of-book "
            "fields if full 5-level depth isn't available on your plan."
        )
    elif depth_count == 0 and symbol_count == 0:
        print(
            "\n-> NEITHER data type produced any messages. This points to "
            "something more basic: market may genuinely be closed, the "
            "access token may have an issue despite passing get_profile(), "
            "or symbol format may be wrong. Re-verify market hours and "
            "token freshness before investigating further."
        )
    else:
        print("\n-> Both worked, or DepthUpdate worked -- re-check the main pipeline's _normalize() logic instead.")