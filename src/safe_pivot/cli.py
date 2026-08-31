from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cgc import run_grouped_experiment, summarize_runs
from .gates import apply_basic_gate
from .io import load_labeled_scores, write_jsonl
from .metrics import compute_metrics


def basic_main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the SAFE-PIVOT basic tie-to-flat gate"
    )
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--tau-delta", type=float, default=0.10)
    parser.add_argument("--tau-harm", type=float, default=0.25)
    parser.add_argument("--out-detail", required=True)
    parser.add_argument("--out-summary", required=True)
    args = parser.parse_args()

    frame = load_labeled_scores(args.judgments, args.verifier)
    detail = apply_basic_gate(
        frame, tau_delta=args.tau_delta, tau_harm=args.tau_harm
    )
    metrics = compute_metrics(detail, detail["gate_accept"].to_numpy(bool))
    summary = {
        "tau_delta": args.tau_delta,
        "tau_harm": args.tau_harm,
        **metrics,
    }
    write_jsonl(args.out_detail, detail.to_dict("records"))
    output = Path(args.out_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cgc_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run grouped SAFE-PIVOT CGC calibration"
    )
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--calibration-fraction", type=float, default=0.50)
    parser.add_argument("--n-calibration-groups", type=int)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["scalar", "dual_r"],
        default=["scalar", "dual_r"],
    )
    parser.add_argument("--dual-grid", type=int, default=11)
    parser.add_argument("--scalar-grid", type=int, default=121)
    parser.add_argument("--out-seed-csv", required=True)
    parser.add_argument("--out-summary-csv", required=True)
    args = parser.parse_args()

    frame = load_labeled_scores(args.judgments, args.verifier)
    seed_rows = run_grouped_experiment(
        frame,
        alphas=args.alphas,
        methods=args.methods,
        delta=args.delta,
        n_seeds=args.n_seeds,
        calibration_fraction=args.calibration_fraction,
        n_calibration_groups=args.n_calibration_groups,
        dual_grid=args.dual_grid,
        scalar_grid=args.scalar_grid,
    )
    summary = summarize_runs(seed_rows)
    for path in [Path(args.out_seed_csv), Path(args.out_summary_csv)]:
        path.parent.mkdir(parents=True, exist_ok=True)
    seed_rows.to_csv(args.out_seed_csv, index=False)
    summary.to_csv(args.out_summary_csv, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    basic_main()

