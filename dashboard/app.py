"""
Live dashboard. Reads dashboard/state.json, which main.py refreshes
roughly once a second while it runs.

Run alongside main.py:
    streamlit run dashboard/app.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASE_DIR  # noqa: E402

STATE_PATH = BASE_DIR / "dashboard" / "state.json"

st.set_page_config(page_title="Stock Screener - Live", layout="wide")
st.title("Real-Time SMMA + LTQ + Bid/Ask AI Screener")

placeholder = st.empty()


def load_state():
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def render(state: dict):
    with placeholder.container():
        if state is None:
            st.warning("Waiting for main.py to start writing state... run `python main.py --broker mock`")
            return

        age = time.time() - state["updated_at"]
        st.caption(f"Last update: {age:.1f}s ago")

        col1, col2 = st.columns(2)
        for col, key, label in ((col1, "basic_summary", "Basic Strategy"), (col2, "filtered_summary", "AI/ML Filtered Strategy")):
            s = state.get(key)
            if s:
                with col:
                    st.subheader(label)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Trades", s["total_trades"])
                    m2.metric("Win rate", f"{s['win_rate']:.0%}")
                    m3.metric("Open", s["open_trades"])
                    m4.metric("P/L (Rs)", f"{s['total_pnl']:.0f}")

        st.subheader("Watchlist — Live Screen")
        rows = list(state["symbols"].values())
        if rows:
            df = pd.DataFrame(rows)
            df = df[[
                "symbol", "ltp", "smma_fast", "smma_slow", "signal",
                "ltq", "etq", "bid_price", "bid_qty", "ask_price", "ask_qty",
                "imbalance", "probability", "decision", "reasons",
            ]]
            df.columns = [
                "Symbol", "LTP", "SMMA20", "SMMA120", "Signal",
                "LTQ", "ETQ", "Bid Px", "Bid Qty", "Ask Px", "Ask Qty",
                "Bid/Ask Imbalance", "AI Probability", "Decision", "Reasons",
            ]

            def highlight(row):
                if row["Decision"] == "ACCEPT":
                    return ["background-color: #d4f7d4"] * len(row)
                if row["Decision"] == "AVOID":
                    return ["background-color: #f7d4d4"] * len(row)
                return [""] * len(row)

            st.dataframe(df.style.apply(highlight, axis=1), use_container_width=True, height=400)
        else:
            st.info("No ticks received yet.")

        st.subheader("Open Paper Positions")
        positions = state.get("open_positions", [])
        if positions:
            pdf = pd.DataFrame(positions)
            st.dataframe(pdf, use_container_width=True)
        else:
            st.caption("No open positions.")


state = load_state()
render(state)

st_autorefresh_seconds = 2
st.caption(f"Auto-refreshing every {st_autorefresh_seconds}s")
time.sleep(st_autorefresh_seconds)
st.rerun()
