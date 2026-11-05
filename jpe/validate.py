"""Schema validation: turn a model's reply into a record, or refuse to.

Written by hand (see the README's build rules). The contract below is what
the rest of the pipeline relies on, and tests/test_validate.py checks it.
Change the contract if you disagree with it, but change the test with it.

Where this sits: jpe/extract.py calls `parse_record` on every reply. On
InvalidRecord it retries once, showing the model the exception message, and
if the retry fails too the post is marked failed. Nothing downstream ever
fills in a field the model did not produce. Gold labels go through
`validate_record` when they are loaded, so a typo in a label fails loudly
instead of scoring as a model error.
"""

from __future__ import annotations

from jpe.fields import ENUMS, JSON_SCHEMA, RECORD_KEYS  # noqa: F401  (for the implementation)


class InvalidRecord(ValueError):
    """The reply cannot be turned into a record.

    The message is sent back to the model on its one retry, so it should say
    what is wrong in a sentence the model can act on ("salary_min is the
    string '120k'; it must be a number or null"), not just "invalid".
    """


def validate_record(data: object) -> dict:
    """Check one record and return it in canonical form.

    Contract:
      * `data` must be a dict with exactly the keys in RECORD_KEYS. A missing
        or extra key is InvalidRecord.
      * seniority, remote, visa_sponsorship and new_grad_eligible must be one
        of their values in ENUMS.
      * company, role_title and location are a non-empty string or None.
      * salary_min and salary_max are numbers (not strings, not booleans) or
        None, and min <= max when both are given.
      * salary_currency is a three-letter uppercase code or None, and is None
        whenever both salary numbers are.
      * tech_stack is a list of non-empty strings.
      * The returned dict has the keys in RECORD_KEYS order and is a new
        object; the input is not modified.

    Decide, and write down in data/gold/LABELING.md, anything the contract
    leaves open: whether enum values are case-insensitive, whether true/false
    count as "yes"/"no", and whether strings are stripped.

    Suggested shape: a Pydantic model `JobPost` with extra="forbid", Literal
    types for the enums and a model validator for the cross-field rules,
    wrapping pydantic.ValidationError in InvalidRecord.
    """
    raise NotImplementedError("jpe.validate.validate_record")


def parse_record(raw: str) -> dict:
    """Turn one model reply into a record with validate_record.

    Contract:
      * The reply must be one JSON object. A reply wrapped in a single
        Markdown code fence (```json ... ```) is accepted, since that is
        formatting rather than content. Any other text around the object,
        or two objects, is InvalidRecord.
      * Invalid JSON is InvalidRecord, never a partial record.
    """
    raise NotImplementedError("jpe.validate.parse_record")
