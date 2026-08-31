from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_labeled_scores(
    judgments_path: str | Path,
    verifier_path: str | Path,
) -> pd.DataFrame:
    judgments = read_jsonl(judgments_path)
    verifier_rows = read_jsonl(verifier_path)
    verifier_map: dict[str, dict[str, Any]] = {}
    for row in verifier_rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            raise ValueError("Verifier row without id")
        if row_id in verifier_map:
            raise ValueError(f"Duplicate verifier id: {row_id}")
        verifier_map[row_id] = row

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(judgments, start=1):
        row_id = str(row.get("id", index)).strip()
        if row_id in seen:
            raise ValueError(f"Duplicate judgment id: {row_id}")
        seen.add(row_id)
        if row_id not in verifier_map:
            raise ValueError(f"Missing verifier output for id={row_id}")
        verifier = verifier_map[row_id]
        if verifier.get("error"):
            raise ValueError(f"Verifier output has an error for id={row_id}; rerun it")

        flat = _binary(row.get("flat_correct"), "flat_correct", row_id)
        generic = _binary(row.get("generic_correct"), "generic_correct", row_id)
        question_id = str(
            row.get("question_id")
            or row.get("original_id")
            or (row_id.split("::", 1)[1] if "::" in row_id else row_id)
        )
        support_flat = _number(verifier.get("support_flat", 0.5), "support_flat", row_id)
        support_evidence = _number(
            verifier.get("support_evidence", 0.5), "support_evidence", row_id
        )
        delta_hat = _number(
            verifier.get("delta_hat", support_evidence - support_flat),
            "delta_hat",
            row_id,
        )
        reliability = _number(
            verifier.get("evidence_reliability", 0.5),
            "evidence_reliability",
            row_id,
        )

        records.append(
            {
                **row,
                "id": row_id,
                "question_id": question_id,
                "flat_correct": flat,
                "generic_correct": generic,
                "beneficial": int(flat == 0 and generic == 1),
                "harmful": int(flat == 1 and generic == 0),
                "decision": str(verifier.get("decision", "Tie")),
                "support_flat": support_flat,
                "support_evidence": support_evidence,
                "answer_conflict": _binary(
                    verifier.get("answer_conflict", 0), "answer_conflict", row_id
                ),
                "evidence_reliability": reliability,
                "answer_equivalent": _binary(
                    verifier.get("answer_equivalent", 0), "answer_equivalent", row_id
                ),
                "delta_hat": delta_hat,
                "harm_hat": _number(verifier.get("harm_hat", 0.0), "harm_hat", row_id),
                "rationale": str(verifier.get("rationale", "")),
            }
        )

    extra = sorted(set(verifier_map) - seen)
    if extra:
        raise ValueError(f"Verifier file has extra ids, examples: {extra[:5]}")
    if not records:
        raise ValueError("No aligned records")
    return pd.DataFrame.from_records(records)


def _binary(value: Any, field: str, row_id: str) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        result = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"id={row_id}: {field} is not binary: {value!r}") from exc
    if result not in (0, 1):
        raise ValueError(f"id={row_id}: {field} is not 0/1: {value!r}")
    return result


def _number(value: Any, field: str, row_id: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"id={row_id}: {field} is not numeric: {value!r}") from exc

