# AI/ML Real-Time Stock Screening & Paper Trading — Project Scaffold

Full pipeline: broker feed → 1-min candles → SMMA20/120 crossover detection
→ LTQ/Bid-Ask/SMMA feature engineering → ML ACCEPT/AVOID scoring → paper
trading (Basic vs ML-Filtered) → live dashboard → performance report.

**Status: this is a working, tested scaffold**, not a finished production
system. Every module runs end-to-end against a built-in mock broker (no
credentials or network needed), and every piece has been exercised:
tick → candle → SMMA → feature vector → model training → model scoring →
paper trade → report. The real broker adapters (Fyers, Angel One) are
written to each SDK's documented interface but haven't been run against
live accounts in this environment — see "What still needs a real broker"
below.

## Architecture

```
brokers/       Broker abstraction. fyers_client.py, angelone_client.py,
                mock_client.py (synthetic data, no creds needed) all
                implement the same BrokerClient interface + normalized Tick.
data/          tick_store.py (buffers + persists ticks to CSV),
                resample.py (builds 1-min OHLCV candles from ticks)
indicators/    smma.py — incremental SMMA20/SMMA120 + crossover detection
features/      engineering.py — LTQ acceleration, bid/ask imbalance,
                SMMA spread/slope, price action -> the feature vector
                the ML model scores. Also the price/depth universe screen.
signals/       crossover.py — wires candles+SMMA+screen+features together,
                emits a CrossoverEvent whenever a screened symbol crosses.
ml/            labeling.py  — replays a day's tick CSV to reconstruct
                               crossovers + forward-looking profit/loss
                               labels (target/stop/max-hold, mirroring
                               the paper trading exit rules)
                train.py    — trains a GradientBoostingClassifier, saves
                               to ml/model_store/
                predict.py  — loads the model, scores live crossovers
                               into ACCEPT/AVOID + reasons, and monitors
                               open positions for DETERIORATING conditions
trading/       paper_engine.py — simulated order book, two parallel
                books ("basic" = every crossover, "filtered" = ML ACCEPT
                only + early deterioration exit)
reports/       performance.py — Basic vs ML-Filtered comparison report
                (win rate, P/L, losing trades avoided, avoid reasons)
dashboard/     app.py — Streamlit live dashboard reading dashboard/state.json
main.py        Orchestrator: broker -> pipeline -> paper trading -> state file
```

## Quick start (no broker account needed)

```bash
pip install -r requirements.txt

# 1. Run the live pipeline against the mock broker
python main.py --broker mock --duration 300

# 2. In a second terminal, watch the live dashboard
streamlit run dashboard/app.py
```

This proves out the whole system, but SMMA20/120 crossovers on a 1-minute
timeframe are inherently rare per symbol per day (SMMA120 alone needs 2
hours of candles just to warm up) — see "On crossover frequency" below.

## Training the model

```bash
# after collecting a full trading day of ticks (data/store/ticks_YYYY-MM-DD.csv)
python -m ml.train --tick-csv data/store/ticks_2026-08-13.csv
```

This reconstructs every crossover from that day using the *same*
`CrossoverEngine` used live (so training features exactly match serving
features), labels each one by walking forward with the identical
stop-loss/target/max-hold rules the paper trading engine uses, and trains
a `GradientBoostingClassifier` on the result. Metrics are written to
`ml/model_store/train_metrics.json`.

**Next-day validation**: run `main.py` again on the following trading day
with the model already trained — it will score live crossovers using
*only* the frozen model from day 1, with zero access to day-2 outcomes
during scoring. `reports/latest_report.json` after that session is your
validation report.

## On the 10 lakh depth threshold

The assignment specifies "Bid Quantity > 10 lakh" / "Ask Quantity > 10
lakh". **`CONFIG.min_bid_qty` / `min_ask_qty` are kept at that literal
value (10,00,000) in source** — the code matches the spec exactly.

This was tested against two full live Angel One sessions (2026-08-17,
2026-08-18; 43 NSE symbols in the ₹30–500 band, 5-level depth summed per
side). Finding, saved in full at
`reports/depth_threshold_analysis.json`:

> At the literal 10,00,000 threshold, **0 of 57** price-qualified
> crossovers passed on a full live trading day. Only 1 of 43 watchlist
> symbols (IDEA) ever cleared 10 lakh on both sides simultaneously, and
> only for brief windows. This appears to be an inherent tension in the
> assignment's own parameters — stocks priced ₹30–500 tend to be
> mid/small-cap names, which are typically the less liquid end of NSE, so
> a 10-lakh-both-sides bar is very rarely met by that universe regardless
> of how many depth levels are aggregated.

**For live runs**, an environment variable overrides the effective
threshold without touching the spec-matching default in source:

```bash
# Windows (cmd)
set SCREENER_MIN_DEPTH_QTY=2000
python main.py --broker angelone

# Windows (PowerShell)
$env:SCREENER_MIN_DEPTH_QTY=2000; python main.py --broker angelone

# Training likewise:
set SCREENER_MIN_DEPTH_QTY=2000
python -m ml.train --tick-csv data\store\ticks_2026-08-18.csv
```

2,000 was chosen because it's the smallest round threshold that clears
the 30-example training minimum on real data (38 crossovers passed) while
still filtering out the thinnest books — see the threshold sweep in
`reports/depth_threshold_analysis.json` for the full comparison across
1,000 / 2,000 / 5,000 / 10,000 / 20,000 / 50,000 / 100,000 / 10,00,000.
Omit the environment variable entirely to run at the literal spec value
(and see near-zero signals, as documented above) — useful if a grader
wants to reproduce that finding directly.


## Data collection log

- **2026-08-17 (partial day, ~12:22–16:08 IST)**: first live capture,
  before the depth-aggregation fix. 308,385 ticks, 43 symbols. Kept for
  reference/debugging only.
- **2026-08-18 (full day, ~9:20 AM–4:08 PM IST)**: 648,591 ticks, all 43
  symbols, with the 5-level depth aggregation in place. This is the
  session `reports/depth_threshold_analysis.json` is built from, and the
  session `ml/model_store/crossover_model.joblib` was trained on (38
  crossover examples, using `SCREENER_MIN_DEPTH_QTY=2000`).
- **Next trading day**: intended as the next-day validation run against
  the model trained above, per the assignment spec — run with
  `SCREENER_MIN_DEPTH_QTY=2000` still set, no retraining.

## On crossover frequency (important, read before demo day)

Verified empirically in this scaffold: with SMMA20/120 on a 1-minute
timeframe, a **single stock produces roughly 1–3 crossovers in a 6-hour
session** — SMMA120 is a slow, heavily-smoothed indicator by design. The
assignment's real NSE universe (all stocks with LTP ₹30–500 and sufficient
depth — likely 150–400+ names) will produce far more crossovers in
aggregate than a small watchlist does. Two practical implications:

1. **Widen `CONFIG.watchlist`** to the real screened NSE universe before
   a live run, not just a handful of names — `config.py` currently ships
   with 6 example symbols for the mock demo.
2. **`ml/train.py` refuses to train below 30 examples** on purpose — a
   model fit on a handful of crossovers won't generalize and would just
   be overfitting noise. If day 1 doesn't clear that bar, either widen
   the universe or extend collection across more of the session.

## What still needs a real broker

- `brokers/fyers_client.py` / `angelone_client.py` are written against
  each SDK's documented v3/SmartAPI surface (websocket depth mode,
  auth flow) but have not been run against a live account here — this
  sandbox has no network access. Before a real run: install the SDK,
  export credentials per `.env.example`, and smoke-test `connect()` +
  a few seconds of `subscribe()` against your actual account.
- **Angel One's instrument master is now wired in automatically**
  (`data/instrument_master.py`): running `python main.py --broker angelone`
  downloads Angel One's daily instrument list, resolves each
  `config.CONFIG.watchlist` symbol to its numeric exchange token, and
  passes that map into `AngelOneClient`. **Caveat: this fetch logic has
  not been run against the live URL** — this sandbox has no network
  access. The endpoint and JSON schema are Angel One's documented/
  commonly-used ones as of this writing, but Angel One has changed this
  URL before; if `build_token_map` comes back empty, check
  `INSTRUMENT_MASTER_URL` against Angel One's current docs first. Run
  `python -m data.instrument_master TATAMOTORS SBIN` standalone to debug
  token resolution before a full `main.py` run.
- `FyersClient.historical_candles` / a Fyers equivalent should be used to
  warm-start SMMA on startup so you're not waiting 2 hours into every
  session before SMMA120 is ready — currently the system seeds purely
  from live candles.

## Packaging the .exe deliverable

```bash
pip install pyinstaller
python build_exe.py
# -> dist/stockscreener.exe
```

Ship `dist/stockscreener.exe` alongside `dashboard/`, `run_dashboard.bat`,
and a `.env` (with real values, **not** `.env.example`) for the grader.
Remember to scrub `.env` of your personal credentials before final
submission per the assignment's instructions, and hand over a template
instead.

## Testing

```bash
python -m pytest tests/ -v
```

`tests/test_pipeline_end_to_end.py` drives ~4000 synthetic ticks through
the full tick→candle→SMMA→feature pipeline and asserts it never raises
and produces well-formed 11-dimensional feature vectors on every
crossover — this is what's been run in this environment (pytest wasn't
installable in this sandbox, so it was executed directly with `python3`;
verify equivalently on your machine).

