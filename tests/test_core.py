from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from safe_pivot.cgc import build_configs, grouped_split, select_fixed_sequence
from safe_pivot.gates import basic_accept, threshold_accept
from safe_pivot.metrics import compute_metrics


class GateTests(unittest.TestCase):
    def test_basic_gate_is_tie_to_flat(self) -> None:
        self.assertFalse(basic_accept("Tie", 0.8, 0.0, 0))
        self.assertFalse(basic_accept("B", 0.8, 0.0, 1))
        self.assertTrue(basic_accept("B", 0.8, 0.0, 0))

    def test_metrics_count_harm_and_benefit(self) -> None:
        frame = pd.DataFrame(
            {
                "flat_correct": [1, 0, 1, 0],
                "generic_correct": [0, 1, 1, 0],
            }
        )
        metrics = compute_metrics(frame, np.array([True, True, False, False]))
        self.assertEqual(metrics["harmful_switches"], 1)
        self.assertEqual(metrics["beneficial_captured"], 1)
        self.assertAlmostEqual(metrics["risk"], 0.25)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)


class CGCTests(unittest.TestCase):
    def test_positive_certified_gain_is_selected(self) -> None:
        frame = pd.DataFrame(
            {
                "flat_correct": [0] * 30,
                "generic_correct": [1] * 30,
                "delta_hat": [0.8] * 30,
                "evidence_reliability": [0.9] * 30,
            }
        )
        result = select_fixed_sequence(
            frame,
            "dual_r",
            build_configs("dual_r", dual_grid=3),
            alpha=0.10,
            delta=0.10,
        )
        self.assertTrue(result.certified_set_nonempty)
        self.assertFalse(result.fallback_selected)
        self.assertGreater(result.calibration_delta_vs_flat, 0)

    def test_nonpositive_gain_forces_final_fallback(self) -> None:
        frame = pd.DataFrame(
            {
                "flat_correct": [1] * 30,
                "generic_correct": [1] * 30,
                "delta_hat": [0.0] * 30,
                "evidence_reliability": [0.9] * 30,
            }
        )
        result = select_fixed_sequence(
            frame,
            "scalar",
            build_configs("scalar", scalar_grid=3),
            alpha=0.10,
            delta=0.10,
        )
        self.assertTrue(result.certified_set_nonempty)
        self.assertTrue(result.fallback_selected)
        self.assertTrue(math.isinf(result.config[0]))
        self.assertFalse(threshold_accept(frame, "scalar", *result.config).any())

    def test_grouped_split_has_no_question_leakage(self) -> None:
        frame = pd.DataFrame(
            {
                "question_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
                "flat_correct": [1] * 6,
                "generic_correct": [1] * 6,
                "delta_hat": [0.0] * 6,
                "evidence_reliability": [0.5] * 6,
            }
        )
        calibration, heldout = grouped_split(
            frame, seed=0, calibration_fraction=0.5
        )
        overlap = set(calibration["question_id"]) & set(heldout["question_id"])
        self.assertFalse(overlap)


if __name__ == "__main__":
    unittest.main()

