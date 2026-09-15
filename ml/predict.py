"""
Loads the trained model and turns a CrossoverEvent into an
ACCEPT / AVOID decision with probability + reasons. Also exposes
`check_deterioration` used by the position monitor on open paper trades.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from config import CONFIG, MODEL_DIR
from features.engineering import FeatureSnapshot, explain_signal
from signals.crossover import CrossoverEvent


@dataclass
class Decision:
    symbol: str
    signal: str
    probability: float
    decision: str        # "ACCEPT" or "AVOID"
    reasons: list[str]


class CrossoverModel:
    def __init__(self, model_path: Path = MODEL_DIR / "crossover_model.joblib"):
        self.model_path = model_path
        self._bundle = None

    def _ensure_loaded(self):
        if self._bundle is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"No trained model at {self.model_path}. Run "
                    f"`python -m ml.train --tick-csv <path-to-a-day-of-ticks.csv>` first."
                )
            self._bundle = joblib.load(self.model_path)

    @property
    def is_available(self) -> bool:
        return self.model_path.exists()

    def score(self, feat: FeatureSnapshot) -> float:
        self._ensure_loaded()
        model = self._bundle["model"]
        vec = np.array([feat.to_feature_vector()])
        return float(model.predict_proba(vec)[0, 1])

    def decide(self, event: CrossoverEvent) -> Decision:
        prob = self.score(event.features)
        reasons = explain_signal(event.signal, event.features)

        if prob >= CONFIG.accept_probability:
            decision = "ACCEPT"
        else:
            decision = "AVOID"
            if prob <= CONFIG.avoid_probability:
                reasons = ["Low model confidence"] + reasons
            else:
                reasons = ["Below acceptance threshold"] + reasons

        return Decision(
            symbol=event.symbol, signal=event.signal,
            probability=prob, decision=decision, reasons=reasons,
        )


def check_deterioration(model: CrossoverModel, entry_feat: FeatureSnapshot,
                         entry_prob: float, live_feat: FeatureSnapshot) -> Optional[str]:
    """
    Re-scores an OPEN position's current live features against its entry
    probability. Returns a human-readable "DETERIORATING" reason string,
    or None if the position still looks healthy.
    """
    live_prob = model.score(live_feat)
    drop = entry_prob - live_prob
    flags = []

    if drop >= CONFIG.deteriorate_prob_drop:
        flags.append(f"Model confidence fell {drop:.0%} since entry")

    imbalance_flip = (entry_feat.bid_ask_imbalance > 0.05 and live_feat.bid_ask_imbalance < -0.05) or \
                      (entry_feat.bid_ask_imbalance < -0.05 and live_feat.bid_ask_imbalance > 0.05)
    if imbalance_flip:
        flags.append("Bid/Ask imbalance flipped direction")

    if live_feat.ltq_change_pct < -0.3 and entry_feat.ltq_change_pct >= 0:
        flags.append("LTQ has sharply declined")

    if not flags:
        return None
    return "; ".join(flags)
