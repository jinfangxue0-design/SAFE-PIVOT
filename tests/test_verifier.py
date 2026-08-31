from __future__ import annotations

import unittest

from safe_pivot.verifier import build_user_prompt, canonicalize, extract_json


class VerifierTests(unittest.TestCase):
    def test_prompt_does_not_include_correctness_labels(self) -> None:
        row = {
            "question": "Question text",
            "grounding": "Evidence text",
            "flat_answer": "A",
            "generic_answer": "B",
            "flat_correct": 1,
            "generic_correct": 0,
        }
        prompt = build_user_prompt(row)
        self.assertIn("Question text", prompt)
        self.assertNotIn("flat_correct", prompt)
        self.assertNotIn("generic_correct", prompt)

    def test_canonicalization_recomputes_derived_scores(self) -> None:
        raw = {
            "decision": "B",
            "support_flat": 0.8,
            "support_evidence": 0.3,
            "answer_conflict": 1,
            "evidence_reliability": 0.25,
            "answer_equivalent": 0,
            "delta_hat": 999,
            "harm_hat": -999,
        }
        parsed = canonicalize(raw)
        self.assertAlmostEqual(parsed["delta_hat"], -0.5)
        self.assertAlmostEqual(parsed["harm_hat"], 0.375)

    def test_extract_json_tolerates_markdown_fence(self) -> None:
        value = extract_json(
            '```json\n{"decision":"Tie","support_flat":0.5,'
            '"support_evidence":0.5,"answer_conflict":0}\n```'
        )
        self.assertEqual(value["decision"], "Tie")


if __name__ == "__main__":
    unittest.main()

