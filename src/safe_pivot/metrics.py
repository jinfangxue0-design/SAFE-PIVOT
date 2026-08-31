from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_metrics(frame: pd.DataFrame, accept: np.ndarray) -> dict[str, Any]:
    required = {"flat_correct", "generic_correct"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")
    if len(frame) == 0:
        raise ValueError("Cannot compute metrics on an empty frame")

    accept = np.asarray(accept, dtype=bool)
    if len(accept) != len(frame):
        raise ValueError("accept length does not match frame length")

    flat = frame["flat_correct"].to_numpy(int)
    generic = frame["generic_correct"].to_numpy(int)
    harmful = (flat == 1) & (generic == 0)
    beneficial = (flat == 0) & (generic == 1)
    chosen = np.where(accept, generic, flat)

    harmful_switches = int(np.sum(accept & harmful))
    beneficial_captured = int(np.sum(accept & beneficial))
    harmful_total = int(np.sum(harmful))
    beneficial_total = int(np.sum(beneficial))
    accepted = int(np.sum(accept))
    n = len(frame)
    hoc = harmful_switches / harmful_total if harmful_total else 0.0

    return {
        "n": n,
        "accepted": accepted,
        "accept_rate": accepted / n,
        "fallback_rate": 1.0 - accepted / n,
        "harmful_switches": harmful_switches,
        "risk": harmful_switches / n,
        "hoc": hoc,
        "resilience": 1.0 - hoc,
        "beneficial_captured": beneficial_captured,
        "benefit_capture": beneficial_captured / beneficial_total if beneficial_total else 0.0,
        "accuracy": float(np.mean(chosen)),
        "flat_accuracy": float(np.mean(flat)),
        "generic_accuracy": float(np.mean(generic)),
        "delta_vs_flat": float(np.mean(chosen) - np.mean(flat)),
        "harmful_total": harmful_total,
        "beneficial_total": beneficial_total,
    }

