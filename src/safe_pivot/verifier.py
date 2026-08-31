from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from .io import append_jsonl, read_jsonl

SYSTEM_PROMPT = """You are SAFE-PIVOT, a structured evidence-intervention verifier.

You compare two candidate answers for the same question:
- Answer A = Y(0), the flat answer generated without external evidence.
- Answer B = Y(1), the evidence-grounded answer generated with external evidence.

Your job is NOT to always trust evidence. Your job is to estimate whether accepting the evidence-grounded answer is beneficial and safe.

Return ONLY a valid JSON object. Do not use markdown. Do not add extra text.

Required JSON schema:
{
  "decision": "A" | "B" | "Tie",
  "support_flat": number from 0 to 1,
  "support_evidence": number from 0 to 1,
  "answer_conflict": 0 or 1,
  "evidence_reliability": number from 0 to 1,
  "answer_equivalent": 0 or 1,
  "delta_hat": number,
  "harm_hat": number,
  "rationale": "short explanation"
}

Definitions:
- support_flat: how well Answer A is supported by the question, evidence, and factual consistency.
- support_evidence: how well Answer B is supported by the question, evidence, and factual consistency.
- answer_conflict: 1 if A and B make different factual claims; 0 if they are equivalent or compatible.
- evidence_reliability: reliability of the provided evidence for this question, considering relevance, specificity, freshness, and internal consistency.
- answer_equivalent: 1 if A and B are semantically equivalent for the question; otherwise 0.
- delta_hat = support_evidence - support_flat.
- harm_hat = answer_conflict * (1 - evidence_reliability) * max(0, support_flat - support_evidence).
- decision: choose B only when Answer B is clearly better and safely supported. Choose Tie when the answers are equivalent or the evidence is insufficient. When uncertain, be conservative.
"""


def build_user_prompt(row: dict[str, Any]) -> str:
    question = row.get("question", "")
    evidence = row.get("grounding", "") or row.get("evidence", "") or row.get("context", "")
    flat = row.get("flat_answer", "") or row.get("answer_a", "")
    generic = (
        row.get("generic_answer", "")
        or row.get("evidence_answer", "")
        or row.get("answer_b", "")
    )
    return f"""Question:
{_truncate(question, 4000)}

External evidence:
{_truncate(evidence, 12000)}

Answer A = Y(0), flat answer without evidence:
{_truncate(flat, 4000)}

Answer B = Y(1), evidence-grounded answer:
{_truncate(generic, 4000)}

Return the required JSON object only."""


def extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", (text or "").strip(), flags=re.I)
    cleaned = cleaned.replace("```", "").strip()
    if not cleaned:
        raise ValueError("Empty verifier response")
    candidates: list[dict[str, Any]] = []
    try:
        direct = json.loads(cleaned)
        if isinstance(direct, dict):
            candidates.append(direct)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    markers = {
        "decision",
        "support_flat",
        "support_evidence",
        "answer_conflict",
        "evidence_reliability",
        "answer_equivalent",
    }
    candidates.sort(key=lambda value: sum(key in value for key in markers), reverse=True)
    if candidates and sum(key in candidates[0] for key in markers) >= 3:
        return candidates[0]
    raise ValueError(f"No structured verifier JSON found: {cleaned[:300]}")


def canonicalize(value: dict[str, Any]) -> dict[str, Any]:
    support_flat = _clamp(value.get("support_flat"), 0.5)
    support_evidence = _clamp(value.get("support_evidence"), 0.5)
    conflict = _binary(value.get("answer_conflict"), 0)
    reliability = _clamp(value.get("evidence_reliability"), 0.5)
    equivalent = _binary(value.get("answer_equivalent"), 0)
    return {
        "decision": _decision(value.get("decision")),
        "support_flat": round(support_flat, 4),
        "support_evidence": round(support_evidence, 4),
        "answer_conflict": conflict,
        "evidence_reliability": round(reliability, 4),
        "answer_equivalent": equivalent,
        "delta_hat": round(support_evidence - support_flat, 4),
        "harm_hat": round(
            conflict
            * (1.0 - reliability)
            * max(0.0, support_flat - support_evidence),
            4,
        ),
        "rationale": str(value.get("rationale", ""))[:1200],
    }


def run(
    input_path: str | Path,
    output_path: str | Path,
    base_url: str,
    api_key: str,
    model: str,
    *,
    resume: bool = False,
    limit: int | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    max_retries: int = 4,
    sleep_seconds: float = 0.2,
    store_raw: bool = False,
) -> None:
    rows = read_jsonl(input_path)
    if limit is not None:
        rows = rows[:limit]
    output = Path(output_path)
    if output.exists() and not resume:
        raise FileExistsError(f"Output exists; use --resume or choose another path: {output}")
    done = _successful_ids(output) if resume else set()
    client = OpenAI(base_url=base_url, api_key=api_key)

    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id", index))
        if row_id in done:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ]
        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                request: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False,
                }
                if max_tokens is not None:
                    request["max_tokens"] = max_tokens
                response = client.chat.completions.create(**request)
                text = _response_text(response)
                parsed = canonicalize(extract_json(text))
                result = {
                    "id": row_id,
                    "question_id": row.get(
                        "question_id", row.get("original_id", "")
                    ),
                    "dataset": row.get("dataset", ""),
                    "answer_model": row.get("answer_model", ""),
                    "verifier_model": model,
                    **parsed,
                }
                if store_raw:
                    result["raw_verifier_output"] = text
                append_jsonl(output, result)
                break
            except Exception as exc:  # API failures may vary by provider.
                last_error = type(exc).__name__
                if attempt < max_retries:
                    time.sleep(min(60.0, 2.0**attempt))
        else:
            append_jsonl(
                output,
                {
                    "id": row_id,
                    "verifier_model": model,
                    "decision": "Tie",
                    "support_flat": 0.5,
                    "support_evidence": 0.5,
                    "answer_conflict": 0,
                    "evidence_reliability": 0.5,
                    "answer_equivalent": 0,
                    "delta_hat": 0.0,
                    "harm_hat": 0.0,
                    "rationale": "Fallback Tie due to verifier error.",
                    "error": last_error,
                },
            )
        if sleep_seconds:
            time.sleep(sleep_seconds)
        print(f"[{index}/{len(rows)}] {row_id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SAFE-PIVOT structured verifier")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--store-raw", action="store_true")
    args = parser.parse_args()

    base_url = os.environ.get("SAFE_PIVOT_BASE_URL", "")
    api_key = os.environ.get("SAFE_PIVOT_API_KEY", "")
    model = os.environ.get("SAFE_PIVOT_MODEL", "")
    if not base_url or not api_key or not model:
        raise RuntimeError(
            "Set SAFE_PIVOT_BASE_URL, SAFE_PIVOT_API_KEY, and SAFE_PIVOT_MODEL"
        )
    run(
        args.input,
        args.output,
        base_url,
        api_key,
        model,
        resume=args.resume,
        limit=args.limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep,
        store_raw=args.store_raw,
    )


def _truncate(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= max_chars else text[:max_chars] + "\n[TRUNCATED]"


def _clamp(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def _binary(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(float(value) >= 0.5)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    return default


def _decision(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"a", "flat", "f", "answer a", "y0", "y(0)"}:
        return "A"
    if text in {"b", "e", "evidence", "generic", "answer b", "y1", "y(1)"}:
        return "B"
    return "Tie"


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content", ""))
    raise TypeError(f"Unsupported API response type: {type(response).__name__}")


def _successful_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("id")) for row in read_jsonl(path) if not row.get("error")}


if __name__ == "__main__":
    main()
