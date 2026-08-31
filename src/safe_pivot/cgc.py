from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binom

from .gates import threshold_accept
from .metrics import compute_metrics

Config = tuple[float, float]


@dataclass(frozen=True)
class SelectionResult:
    method: str
    alpha: float
    delta: float
    config: Config
    config_label: str
    certified_set_nonempty: bool
    fallback_selected: bool
    n_certified_configs: int
    n_tested_configs: int
    calibration_p_value: float
    calibration_risk: float
    calibration_delta_vs_flat: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_configs(method: str, dual_grid: int = 11, scalar_grid: int = 121) -> list[Config]:
    if method == "scalar":
        thresholds = sorted(np.linspace(-1.0, 1.0, scalar_grid), reverse=True)
        return [(float(value), math.nan) for value in thresholds]
    if method == "dual_r":
        delta_thresholds = sorted(np.linspace(-1.0, 1.0, dual_grid), reverse=True)
        reliability_thresholds = sorted(np.linspace(0.0, 1.0, dual_grid), reverse=True)
        return [
            (float(tau_delta), float(tau_reliability))
            for tau_delta in delta_thresholds
            for tau_reliability in reliability_thresholds
        ]
    raise ValueError(f"Unknown method: {method}")


def fallback_config() -> Config:
    return (math.inf, math.nan)


def config_label(method: str, config: Config) -> str:
    tau_delta, tau_reliability = config
    if math.isinf(tau_delta):
        return "always-flat"
    if method == "scalar":
        return f"delta>={tau_delta:.6g}"
    return f"delta>={tau_delta:.6g};reliability>={tau_reliability:.6g}"


def exact_binomial_p_value(k: int, n: int, alpha: float) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0 <= k <= n:
        raise ValueError("k must lie in [0, n]")
    return float(binom.cdf(k, n, alpha))


def select_fixed_sequence(
    calibration: pd.DataFrame,
    method: str,
    configs: Sequence[Config],
    alpha: float,
    delta: float = 0.10,
    conserve_net_gain: bool = True,
) -> SelectionResult:
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    if len(calibration) == 0:
        raise ValueError("Calibration data is empty")

    certified: list[tuple[Config, dict[str, float], float]] = []
    tested = 0
    for config in configs:
        tested += 1
        accept = threshold_accept(calibration, method, *config)
        metrics = compute_metrics(calibration, accept)
        p_value = exact_binomial_p_value(
            int(metrics["harmful_switches"]), int(metrics["n"]), alpha
        )
        if p_value <= delta:
            certified.append((config, metrics, p_value))
        else:
            break

    if certified:
        certified.sort(
            key=lambda item: (
                item[1]["delta_vs_flat"],
                item[1]["accuracy"],
                item[1]["benefit_capture"],
                item[1]["accept_rate"],
                -item[2],
            ),
            reverse=True,
        )
        selected_config, selected_metrics, selected_p = certified[0]
        if not conserve_net_gain or selected_metrics["delta_vs_flat"] > 0:
            return SelectionResult(
                method=method,
                alpha=alpha,
                delta=delta,
                config=selected_config,
                config_label=config_label(method, selected_config),
                certified_set_nonempty=True,
                fallback_selected=False,
                n_certified_configs=len(certified),
                n_tested_configs=tested,
                calibration_p_value=selected_p,
                calibration_risk=float(selected_metrics["risk"]),
                calibration_delta_vs_flat=float(selected_metrics["delta_vs_flat"]),
            )

    zero_metrics = compute_metrics(calibration, np.zeros(len(calibration), dtype=bool))
    return SelectionResult(
        method=method,
        alpha=alpha,
        delta=delta,
        config=fallback_config(),
        config_label="always-flat",
        certified_set_nonempty=bool(certified),
        fallback_selected=True,
        n_certified_configs=len(certified),
        n_tested_configs=tested,
        calibration_p_value=0.0,
        calibration_risk=float(zero_metrics["risk"]),
        calibration_delta_vs_flat=float(zero_metrics["delta_vs_flat"]),
    )


def grouped_split(
    frame: pd.DataFrame,
    seed: int,
    calibration_fraction: float = 0.5,
    n_calibration_groups: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "question_id" not in frame:
        raise ValueError("Grouped splitting requires question_id")
    groups = np.array(sorted(frame["question_id"].astype(str).unique()))
    if len(groups) < 2:
        raise ValueError("At least two question groups are required")
    if n_calibration_groups is None:
        if not 0 < calibration_fraction < 1:
            raise ValueError("calibration_fraction must lie in (0, 1)")
        n_calibration_groups = int(round(len(groups) * calibration_fraction))
    n_calibration_groups = min(max(int(n_calibration_groups), 1), len(groups) - 1)

    rng = np.random.default_rng(seed)
    calibration_groups = set(rng.permutation(groups)[:n_calibration_groups])
    is_calibration = frame["question_id"].astype(str).isin(calibration_groups)
    calibration = frame[is_calibration].reset_index(drop=True)
    heldout = frame[~is_calibration].reset_index(drop=True)
    return calibration, heldout


def run_grouped_experiment(
    frame: pd.DataFrame,
    alphas: Iterable[float],
    methods: Iterable[str] = ("scalar", "dual_r"),
    delta: float = 0.10,
    n_seeds: int = 200,
    calibration_fraction: float = 0.5,
    n_calibration_groups: int | None = None,
    dual_grid: int = 11,
    scalar_grid: int = 121,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    config_map = {
        method: build_configs(method, dual_grid=dual_grid, scalar_grid=scalar_grid)
        for method in methods
    }
    for alpha in alphas:
        for seed in range(n_seeds):
            calibration, heldout = grouped_split(
                frame,
                seed,
                calibration_fraction=calibration_fraction,
                n_calibration_groups=n_calibration_groups,
            )
            for method, configs in config_map.items():
                selection = select_fixed_sequence(
                    calibration, method, configs, alpha=float(alpha), delta=delta
                )
                heldout_metrics = compute_metrics(
                    heldout, threshold_accept(heldout, method, *selection.config)
                )
                full_metrics = compute_metrics(
                    frame, threshold_accept(frame, method, *selection.config)
                )
                rows.append(
                    {
                        "alpha": float(alpha),
                        "delta": delta,
                        "seed": seed,
                        "method": method,
                        "n_total": len(frame),
                        "n_calibration": len(calibration),
                        "n_heldout": len(heldout),
                        "n_calibration_groups": calibration["question_id"].nunique(),
                        "n_heldout_groups": heldout["question_id"].nunique(),
                        **selection.to_dict(),
                        "heldout_risk": heldout_metrics["risk"],
                        "heldout_accuracy": heldout_metrics["accuracy"],
                        "heldout_delta_vs_flat": heldout_metrics["delta_vs_flat"],
                        "heldout_benefit_capture": heldout_metrics["benefit_capture"],
                        "heldout_hoc": heldout_metrics["hoc"],
                        "heldout_fallback_rate": heldout_metrics["fallback_rate"],
                        "full_risk": full_metrics["risk"],
                        "full_violation": int(full_metrics["risk"] > float(alpha)),
                    }
                )
    return pd.DataFrame.from_records(rows)


def summarize_runs(seed_rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for (alpha, method), group in seed_rows.groupby(["alpha", "method"], sort=True):
        violations = int(group["full_violation"].sum())
        runs = len(group)
        ci_low, ci_high = _wilson_interval(violations, runs)
        output.append(
            {
                "alpha": alpha,
                "method": method,
                "runs": runs,
                "runs_with_nonempty_certified_set": int(
                    group["certified_set_nonempty"].astype(bool).sum()
                ),
                "final_nonfallback_runs": int(
                    (~group["fallback_selected"].astype(bool)).sum()
                ),
                "fallback_rate": float(group["fallback_selected"].astype(float).mean()),
                "violation_count": violations,
                "empirical_violation_rate": violations / runs,
                "violation_ci95_low": ci_low,
                "violation_ci95_high": ci_high,
                "heldout_risk": float(group["heldout_risk"].mean()),
                "heldout_accuracy": float(group["heldout_accuracy"].mean()),
                "delta_vs_flat_pp": 100.0 * float(group["heldout_delta_vs_flat"].mean()),
                "benefit_capture": float(group["heldout_benefit_capture"].mean()),
                "hoc": float(group["heldout_hoc"].mean()),
                "heldout_fallback_rate": float(group["heldout_fallback_rate"].mean()),
            }
        )
    return pd.DataFrame.from_records(output)


def _wilson_interval(
    successes: int, n: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    proportion = successes / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)
    ) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))

