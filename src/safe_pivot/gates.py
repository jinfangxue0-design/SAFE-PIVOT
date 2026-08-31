from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def normalize_decision(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"a", "f", "flat", "answer a", "y0", "y(0)"}:
        return "A"
    if text in {"b", "e", "evidence", "generic", "answer b", "y1", "y(1)"}:
        return "B"
    return "Tie"


def basic_accept(
    decision: Any,
    delta_hat: float,
    harm_hat: float,
    answer_equivalent: int,
    tau_delta: float = 0.10,
    tau_harm: float = 0.25,
) -> bool:
    return bool(
        normalize_decision(decision) == "B"
        and float(delta_hat) >= tau_delta
        and float(harm_hat) <= tau_harm
        and int(answer_equivalent) == 0
    )


def apply_basic_gate(
    frame: pd.DataFrame,
    tau_delta: float = 0.10,
    tau_harm: float = 0.25,
) -> pd.DataFrame:
    required = {
        "decision",
        "delta_hat",
        "harm_hat",
        "answer_equivalent",
        "flat_correct",
        "generic_correct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns for basic gate: {missing}")

    output = frame.copy()
    accept = np.array(
        [
            basic_accept(d, delta, harm, equivalent, tau_delta, tau_harm)
            for d, delta, harm, equivalent in zip(
                output["decision"],
                output["delta_hat"],
                output["harm_hat"],
                output["answer_equivalent"],
            )
        ],
        dtype=bool,
    )
    output["gate_accept"] = accept.astype(int)
    output["final_choice"] = np.where(accept, "B", "A")
    output["pivot_correct"] = np.where(
        accept,
        output["generic_correct"].to_numpy(int),
        output["flat_correct"].to_numpy(int),
    )
    return output


def threshold_accept(
    frame: pd.DataFrame,
    method: str,
    tau_delta: float,
    tau_reliability: float = math.nan,
) -> np.ndarray:
    if math.isinf(tau_delta):
        return np.zeros(len(frame), dtype=bool)
    if "delta_hat" not in frame:
        raise ValueError("Missing delta_hat")
    mask = frame["delta_hat"].to_numpy(float) >= tau_delta
    if method == "dual_r":
        if "evidence_reliability" not in frame:
            raise ValueError("Missing evidence_reliability")
        mask &= frame["evidence_reliability"].to_numpy(float) >= tau_reliability
    elif method != "scalar":
        raise ValueError(f"Unknown method: {method}")
    return mask

