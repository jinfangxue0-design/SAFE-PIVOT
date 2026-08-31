# Core protocol

## Candidate answers

For each question `x` and evidence `e`, an answer model produces:

- `Y(0) = M(x)`, the flat answer;
- `Y(1) = M(x, e)`, the evidence-grounded answer.

The structured verifier receives `x`, `e`, `Y(0)`, and `Y(1)`. It produces
support scores, answer conflict, evidence reliability, semantic equivalence,
an A/B/Tie preference, and a short rationale.

## Basic arbitration

The basic structured gate switches to `Y(1)` when:

```text
decision == B
and delta_hat >= tau_delta
and harm_hat <= tau_harm
and answer_equivalent == 0
```

All other cases keep `Y(0)`. In particular, Tie maps to flat.

## Risk-controlled candidate families

The Scalar family uses `delta_hat >= tau_delta`. The Dual-R family uses:

```text
delta_hat >= tau_delta and evidence_reliability >= tau_reliability
```

The paper protocol uses 121 candidates per family: 121 Scalar thresholds or an
11 by 11 Dual-R grid. Thresholds are fixed before labels are examined.

## Controlled loss

```text
L = 1[accept Y(1) and flat_correct = 1 and generic_correct = 0]
```

Thus `R = E[L]` is the unconditional probability of a harmful switch.

## Fixed-sequence certification

For each configuration, let `k` be harmful switches among `n` calibration rows.
At risk budget `alpha` and confidence parameter `delta`, compute:

```text
p = P[Binomial(n, alpha) <= k]
```

The configuration passes when `p <= delta`. Candidates are tested in a fixed,
pre-registered order and testing stops at the first failure. The selected point
maximizes calibration accuracy gain over always-flat inside the certified prefix.

## Conservation rule

Certification means a configuration is safe enough under the test; it does not
mean evidence adoption has positive expected utility. If the best certified
calibration gain is not strictly positive, the final deployment is always-flat.

## Grouped splitting

When several records share one underlying question, all of them must stay on the
same side of a calibration/held-out split. The public CLI groups by `question_id`.

