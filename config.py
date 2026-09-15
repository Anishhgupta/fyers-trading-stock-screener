"""
Central configuration for the screening/paper-trading system.

Nothing secret lives here. Real credentials are read from environment
variables (or a local, git-ignored `.env`) at runtime — see brokers/base.py.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

if getattr(sys, "frozen", False):
    # Running as a PyInstaller --onefile .exe: __file__ would resolve
    # inside the temporary extraction folder (sys._MEIPASS), which is
    # deleted when the exe exits. Anchor persistent data/model/log
    # folders next to the actual .exe on disk instead, so collected
    # ticks and trained models survive between runs.
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "store"
MODEL_DIR = BASE_DIR / "ml" / "model_store"
LOG_DIR = BASE_DIR / "logs"

for d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class ScreenerConfig:
    # --- Universe filters ---
    timeframe_minutes: int = 1
    smma_fast: int = 20
    smma_slow: int = 120
    price_min: float = 30.0
    price_max: float = 500.0
    min_bid_qty: int = 10_00_000   # 10 lakh, per assignment spec. Verified
                                    # against a real live session (Aug 17
                                    # 2026): top-of-book quantity alone
                                    # almost never reaches this for NSE
                                    # names. brokers/*.py now sum quantity
                                    # across all 5 published depth levels
                                    # before this filter is applied -- see
                                    # README "On the 10 lakh depth threshold".
    min_ask_qty: int = 10_00_000   # 10 lakh -- same aggregate-depth basis
    exchange: str = "NSE"

    # --- ML decision thresholds ---
    accept_probability: float = 0.62      # >= this -> ACCEPT
    avoid_probability: float = 0.45       # <= this -> AVOID outright
    deteriorate_prob_drop: float = 0.15   # drop in live prob vs entry prob -> DETERIORATING
    ltq_accel_lookback: int = 5           # ticks used to compute LTQ acceleration
    imbalance_lookback: int = 10

    # --- Paper trading ---
    capital_per_trade: float = 50_000.0
    stop_loss_pct: float = 0.006          # 0.6%
    target_pct: float = 0.012             # 1.2%
    max_hold_minutes: int = 45
    max_concurrent_positions: int = 10

    # --- Broker ---
    broker: str = "mock"                  # "fyers" | "angelone" | "mock"
    watchlist: list = field(default_factory=lambda: [
        # Wider NSE watchlist for a real training day -- SMMA20/120 is a
        # slow, infrequent indicator, so ml/train.py needs enough symbols
        # in play to accumulate 30+ crossovers in a single session.
        # Live screening (price ₹30-500, min depth) still applies to
        # every symbol below at runtime -- anything outside the band or
        # too thin is auto-excluded, so it's fine to cast a wide net here.
        # Before market open, sanity-check symbol->token resolution with:
        #   python -m data.instrument_master SBIN PNB IOC ...
        "NSE:SBIN-EQ", "NSE:AXISBANK-EQ", "NSE:PNB-EQ", "NSE:BANKBARODA-EQ",
        "NSE:CANBK-EQ", "NSE:UNIONBANK-EQ", "NSE:IDFCFIRSTB-EQ", "NSE:INDIANB-EQ",
        "NSE:IOB-EQ", "NSE:UCOBANK-EQ", "NSE:MAHABANK-EQ",

        "NSE:TATASTEEL-EQ", "NSE:VEDL-EQ", "NSE:SAIL-EQ", "NSE:NATIONALUM-EQ",
        "NSE:HINDCOPPER-EQ", "NSE:JINDALSTEL-EQ", "NSE:NMDC-EQ",

        "NSE:IOC-EQ", "NSE:BPCL-EQ", "NSE:GAIL-EQ", "NSE:ONGC-EQ",
        "NSE:OIL-EQ", "NSE:PETRONET-EQ",

        "NSE:IDEA-EQ", "NSE:TATAPOWER-EQ", "NSE:NHPC-EQ", "NSE:NTPC-EQ",
        "NSE:PFC-EQ", "NSE:RECLTD-EQ", "NSE:SJVN-EQ", "NSE:JPPOWER-EQ",
        "NSE:RPOWER-EQ", "NSE:SUZLON-EQ",

        "NSE:IRFC-EQ", "NSE:HUDCO-EQ", "NSE:NBCC-EQ", "NSE:RVNL-EQ",
        "NSE:RAILTEL-EQ", "NSE:IRCON-EQ",

        "NSE:YESBANK-EQ", "NSE:IDBI-EQ", "NSE:ETERNAL-EQ",
    ])


CONFIG = ScreenerConfig()
