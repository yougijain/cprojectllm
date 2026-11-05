"""The extraction schema: what one job post becomes.

This module only names the fields and their allowed values. Checking a
model's output against them is jpe/validate.py.
"""

from __future__ import annotations

SCHEMA_VERSION = "1"

SENIORITY = ("intern", "new_grad", "mid", "senior", "unspecified")
REMOTE = ("yes", "no", "hybrid", "unspecified")
VISA = ("yes", "no", "unspecified")
NEW_GRAD = ("yes", "no", "unspecified")

ENUMS: dict[str, tuple[str, ...]] = {
    "seniority": SENIORITY,
    "remote": REMOTE,
    "visa_sponsorship": VISA,
    "new_grad_eligible": NEW_GRAD,
}

# Keys of one record, in the order prompts and files show them.
RECORD_KEYS = (
    "company",
    "role_title",
    "seniority",
    "location",
    "remote",
    "visa_sponsorship",
    "salary_min",
    "salary_max",
    "salary_currency",
    "tech_stack",
    "new_grad_eligible",
)

# What gets a score. Salary is one field made of three keys: it is right
# when min, max and currency all are.
SCORED_FIELDS = (
    "company",
    "role_title",
    "seniority",
    "location",
    "remote",
    "visa_sponsorship",
    "salary",
    "tech_stack",
    "new_grad_eligible",
)

# The fields a job seeker filters on, reported on their own.
USER_FIELDS = ("new_grad_eligible", "remote", "salary")

EMPTY_RECORD: dict = {
    "company": None,
    "role_title": None,
    "seniority": "unspecified",
    "location": None,
    "remote": "unspecified",
    "visa_sponsorship": "unspecified",
    "salary_min": None,
    "salary_max": None,
    "salary_currency": None,
    "tech_stack": [],
    "new_grad_eligible": "unspecified",
}

# JSON Schema for one record: shown to models in the prompt, and usable for
# constrained decoding (vLLM guided_json, llama.cpp json_schema) as an
# ablation. The headline numbers use unconstrained decoding, so that JSON
# validity is something measured rather than guaranteed.
_nullable_str = {"type": ["string", "null"]}
_nullable_num = {"type": ["number", "null"]}

JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": list(RECORD_KEYS),
    "properties": {
        "company": _nullable_str,
        "role_title": _nullable_str,
        "seniority": {"enum": list(SENIORITY)},
        "location": _nullable_str,
        "remote": {"enum": list(REMOTE)},
        "visa_sponsorship": {"enum": list(VISA)},
        "salary_min": _nullable_num,
        "salary_max": _nullable_num,
        "salary_currency": {"type": ["string", "null"], "pattern": "^[A-Z]{3}$"},
        "tech_stack": {"type": "array", "items": {"type": "string"}},
        "new_grad_eligible": {"enum": list(NEW_GRAD)},
    },
}
