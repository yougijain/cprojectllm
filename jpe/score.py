"""The scorer: how right one predicted record is, field by field.

Written by hand (see the README's build rules). Every headline number is a
mean of what score_record returns, so its definition is the definition of
"accuracy" in this project. tests/test_score.py pins the parts the spec
fixes; the parts it leaves open are listed in data/gold/LABELING.md under
"Choices still open" and must be decided there before this is written.

Where this sits: scripts/evaluate.py calls score_record(prediction, gold)
for every post and every model, keeps every per-field score so models can
be compared post by post, and aggregates with jpe.stats. Gold annotators are
compared with the same function (scripts/agreement.py), so agreement and
accuracy are on one scale.
"""

from __future__ import annotations

from jpe.fields import SCORED_FIELDS  # noqa: F401  (for the implementation)
from jpe.normalize import canonical_stack  # noqa: F401

SALARY_TOLERANCE = 0.05


def salary_matches(pred: dict, gold: dict, tol: float = SALARY_TOLERANCE) -> bool:
    """Whether the predicted salary is right.

    Contract:
      * Right when min, max and currency are all right.
      * min (and separately max): both None, or both numbers with
        |pred - gold| <= tol * gold.
      * Currency: equal, where None equals None.
      * A gold record with no salary is matched only by a prediction with no
        salary: the "correct null". Inventing a salary is wrong.
    """
    raise NotImplementedError("jpe.score.salary_matches")


def stack_f1(pred: list[str], gold: list[str]) -> float:
    """Set F1 between two tech stacks.

    Contract:
      * Both sides go through canonical_stack first, so "Golang" and "go"
        are one item and duplicates count once.
      * Both empty: 1.0 (the model correctly found no stack). Exactly one
        empty: 0.0.
    """
    raise NotImplementedError("jpe.score.stack_f1")


def score_record(pred: dict | None, gold: dict) -> dict[str, float]:
    """Score one prediction against its gold record.

    Contract:
      * Returns one value per field in SCORED_FIELDS, each in [0, 1].
      * pred is None for a post the model failed on (invalid JSON after the
        retry). Every field scores 0.0: a failure counts against the model
        and is never dropped from the denominator.
      * seniority, remote, visa_sponsorship, new_grad_eligible: exact match.
      * salary: 1.0 if salary_matches, else 0.0.
      * tech_stack: stack_f1.
      * company, role_title, location: your comparison rule. Identical
        strings must match; beyond that, decide (case, punctuation, legal
        suffixes, "SF" vs "San Francisco") and record it in LABELING.md.
    """
    raise NotImplementedError("jpe.score.score_record")
