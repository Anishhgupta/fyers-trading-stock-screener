"""
Builds a labeled training set from a completed trading day's raw tick
CSV (as written by data/tick_store.py).

IMPORTANT (added 2026-09-02): build_training_set now accepts an
optional pre-built CrossoverEngine. Pass a warm-started engine (see
ml/rebuild_with_warmstart.py) to avoid the cold-start problem: without
warm-starting, this function has to spend the first ~2 hours of the
day's OWN tick data just re-warming SMMA20/120 from scratch before it
can detect any crossover -- silently losing every real crossover that
happened during that warm-up window, even though the live system (which
DOES warm-start at every startup) caught them in real time. This was
discovered directly: a live 2026-09-02 session reported 18 real
ML-scored crossovers, but a cold-start replay of that same day's CSV
reconstructed only 1.
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from brokers.base import Tick
from config import CONFIG
from signals.crossover import CrossoverEngine, CrossoverEvent

LOOKAHEAD_MINUTES = CONFIG.max_hold_minutes


@dataclass
class LabeledExample:
    symbol: str
    signal: str
    ts: float
    features: list
    outcome_pct: float
    label: int


def load_ticks_csv(path: Path) -> List[Tick]:
    """
    Skips malformed rows instead of crashing on the whole file -- these
    can occur if a live session's terminal was force-closed mid-write
    (a partial/truncated final row), leaving missing or empty fields.
    Prints a count of skipped rows so data quality issues stay visible
    rather than being silently swallowed.
    """
    ticks = []
    skipped = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ticks.append(Tick(
                    symbol=row["symbol"], ts=float(row["ts"]), ltp=float(row["ltp"]),
                    ltq=int(row["ltq"]), total_traded_qty=int(row["total_traded_qty"]),
                    bid_price=float(row["bid_price"]), bid_qty=int(row["bid_qty"]),
                    ask_price=float(row["ask_price"]), ask_qty=int(row["ask_qty"]),
                ))
            except (TypeError, ValueError, KeyError):
                skipped += 1
    if skipped:
        print(f"  (skipped {skipped} malformed row(s) in {path})")
    return ticks


def _forward_outcome(ticks_by_symbol, symbol, entry_idx, entry_price, signal):
    series = ticks_by_symbol[symbol]
    entry_ts = series[entry_idx].ts
    stop = CONFIG.stop_loss_pct
    target = CONFIG.target_pct
    direction = 1 if signal == "BUY" else -1
    for t in series[entry_idx + 1:]:
        if t.ts - entry_ts > LOOKAHEAD_MINUTES * 60:
            break
        move = direction * (t.ltp - entry_price) / entry_price
        if move >= target:
            return target
        if move <= -stop:
            return -stop
    last = series[entry_idx]
    for t in series[entry_idx + 1:]:
        if t.ts - entry_ts > LOOKAHEAD_MINUTES * 60:
            break
        last = t
    return direction * (last.ltp - entry_price) / entry_price


def build_training_set(tick_csv_path: Path, engine: Optional[CrossoverEngine] = None) -> List[LabeledExample]:
    """
    engine: optional pre-built CrossoverEngine. Pass a warm-started one
    (engine.warm_start() already called per symbol) to avoid losing
    crossovers to the cold-start warm-up problem described above. If
    None, a fresh (cold) engine is created, matching the original
    behavior -- kept as the default for backward compatibility.
    """
    ticks = load_ticks_csv(tick_csv_path)
    ticks.sort(key=lambda t: t.ts)

    ticks_by_symbol = {}
    for t in ticks:
        ticks_by_symbol.setdefault(t.symbol, []).append(t)

    if engine is None:
        engine = CrossoverEngine()
    events = []

    for t in ticks:
        ev = engine.on_tick(t)
        if ev is not None:
            events.append((ev, t.symbol, t.ts))

    examples = []
    for ev, symbol, entry_ts in events:
        series = ticks_by_symbol[symbol]
        entry_idx = next((i for i, t in enumerate(series) if t.ts >= entry_ts), None)
        if entry_idx is None:
            continue
        entry_price = series[entry_idx].ltp
        outcome_pct = _forward_outcome(ticks_by_symbol, symbol, entry_idx, entry_price, ev.signal)
        examples.append(LabeledExample(
            symbol=symbol, signal=ev.signal, ts=entry_ts,
            features=ev.features.to_feature_vector(),
            outcome_pct=outcome_pct,
            label=1 if outcome_pct > 0 else 0,
        ))

    return examples