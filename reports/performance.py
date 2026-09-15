"""
Builds the paper-trading performance report required by the assignment:
total signals, accepted/avoided counts, win/loss stats for both books,
total simulated P/L, and how many losing trades the ML filter avoided.
"""
from __future__ import annotations

import json
from pathlib import Path

from trading.paper_engine import PaperTradingEngine, Book


def build_report(engine: PaperTradingEngine) -> dict:
    basic = engine.summary(Book.BASIC)
    filtered = engine.summary(Book.FILTERED)

    total_signals = basic["total_trades"] + basic["open_trades"] + engine.avoided_count
    # Losing trades avoided: crossovers the filter skipped that WOULD have
    # lost money in the basic book. We approximate by symbol+ts pairing --
    # for exact accounting, cross-reference engine.trade_log entries by
    # (symbol, signal, ts) between AVOIDED and the basic book's LOSS outcomes.
    avoided_losses = _estimate_avoided_losses(engine)

    report = {
        "total_crossover_signals": total_signals,
        "signals_accepted": filtered["total_trades"] + filtered["open_trades"],
        "signals_avoided": engine.avoided_count,
        "basic_strategy": basic,
        "ml_filtered_strategy": filtered,
        "losing_trades_avoided_estimate": avoided_losses,
        "avoid_reasons_breakdown": engine.avoided_reasons,
    }
    return report


def _estimate_avoided_losses(engine: PaperTradingEngine) -> dict:
    """
    Cross-references AVOIDED signals against how the same signal fared in
    the basic (unfiltered) book, when it later closed.
    """
    basic_outcomes = {}
    for p in engine.closed_positions:
        if p.book == Book.BASIC:
            basic_outcomes[(p.symbol, p.signal, round(p.entry_ts))] = p.pnl

    avoided_that_would_have_lost = 0
    avoided_that_would_have_won = 0
    avoided_unresolved = 0

    avoided_keys = [
        (e.symbol, e.signal, round(e.ts))
        for e in engine.trade_log if e.outcome == "AVOIDED"
    ]
    for key in avoided_keys:
        # basic entries are logged at roughly the same ts; allow +-2s tolerance
        match = next((pnl for (s, sig, ts), pnl in basic_outcomes.items()
                      if s == key[0] and sig == key[1] and abs(ts - key[2]) <= 2), None)
        if match is None:
            avoided_unresolved += 1
        elif match <= 0:
            avoided_that_would_have_lost += 1
        else:
            avoided_that_would_have_won += 1

    total_resolved = avoided_that_would_have_lost + avoided_that_would_have_won
    return {
        "avoided_signals_resolved_in_basic_book": total_resolved,
        "would_have_lost": avoided_that_would_have_lost,
        "would_have_won": avoided_that_would_have_won,
        "pct_of_avoided_that_were_losses": (
            avoided_that_would_have_lost / total_resolved if total_resolved else None
        ),
        "avoided_unresolved_still_open_or_no_match": avoided_unresolved,
    }


def save_report(engine: PaperTradingEngine, out_path: Path) -> dict:
    report = build_report(engine)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=float))
    return report


def print_report(report: dict) -> None:
    print("=" * 60)
    print("PAPER TRADING PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Total crossover signals   : {report['total_crossover_signals']}")
    print(f"Signals accepted (ML)     : {report['signals_accepted']}")
    print(f"Signals avoided (ML)      : {report['signals_avoided']}")
    print()
    for key in ("basic_strategy", "ml_filtered_strategy"):
        s = report[key]
        print(f"-- {s['book'].upper()} --")
        print(f"   Trades: {s['total_trades']} (open: {s['open_trades']})")
        print(f"   Wins/Losses: {s['wins']}/{s['losses']}  Win rate: {s['win_rate']:.1%}")
        print(f"   Total P/L: Rs {s['total_pnl']:.2f}")
        print()
    avoided = report["losing_trades_avoided_estimate"]
    print("-- ML FILTER IMPACT --")
    print(f"   Of avoided signals resolved in basic book: {avoided['avoided_signals_resolved_in_basic_book']}")
    print(f"   Would have lost: {avoided['would_have_lost']}  Would have won: {avoided['would_have_won']}")
    if avoided["pct_of_avoided_that_were_losses"] is not None:
        print(f"   -> {avoided['pct_of_avoided_that_were_losses']:.1%} of avoided signals were losing trades")
    print()
    print("-- TOP AVOID REASONS --")
    for reason, count in sorted(report["avoid_reasons_breakdown"].items(), key=lambda x: -x[1]):
        print(f"   {count:>4}  {reason}")
