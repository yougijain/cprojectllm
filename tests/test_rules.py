"""B1, the rules baseline. Headers are real ones from the corpus."""

import pytest

from jpe.fields import ENUMS, RECORD_KEYS
from jpe.rules import extract, split_header, tech_stack


def test_split_header():
    assert split_header("Acme (YC W21) | Senior Engineer | NYC\n\nbody") == ["Acme (YC W21)", "Senior Engineer", "NYC"]
    assert split_header("\n\nAcme • Engineer • Remote") == ["Acme", "Engineer", "Remote"]


@pytest.mark.parametrize(
    "header,expected",
    [
        (
            "Sharecare | BI Engineer (Contract) | Remote – Mexico / South America",
            {"company": "Sharecare", "role_title": "BI Engineer (Contract)", "location": "Mexico / South America", "remote": "yes"},
        ),
        (
            "krea.ai | Distributed Systems Eng. & Data Engineers | San Francisco (on-site), CA | $200K-$400K",
            {"company": "krea.ai", "location": "San Francisco, CA", "remote": "no", "salary_min": 200_000, "salary_max": 400_000, "salary_currency": "USD"},
        ),
        (
            "Queueco | Research Engineer | HFT | London, UK | Full time on-site | Visa sponsor",
            {"role_title": "Research Engineer", "location": "London, UK", "remote": "no", "visa_sponsorship": "yes"},
        ),
        (
            "Blueberry Pediatrics | Full-stack Senior or Staff Engineer | REMOTE (US-only)",
            {"seniority": "senior", "location": "US", "remote": "yes", "new_grad_eligible": "no"},
        ),
        (
            "Konvu | https://konvu.com/ | Software Engineer | FULLTIME | ONSITE | Paris, France",
            {"company": "Konvu", "role_title": "Software Engineer", "location": "Paris, France", "remote": "no"},
        ),
        (
            "Apple | Full-Stack Software Engineer | San Diego, CA OR Sunnyvale, CA | ONSITE (3 days/week in office)",
            {"remote": "hybrid", "location": "San Diego, CA OR Sunnyvale, CA"},
        ),
        (
            "Aeqium | Product Engineer | San Francisco (SF) | HYBRID | $110k-$170k + Equity | US CITIZEN / GREEN CARD ONLY (NO VISA SPONSORSHIP)",
            {"remote": "hybrid", "visa_sponsorship": "no", "salary_min": 110_000, "salary_max": 170_000},
        ),
        (
            "PermitFlow (YC W22) | Staff, Fullstack & Frontend Software Engineers | New York, NY | ONSITE | Full-time | https://permitflow.com",
            {"company": "PermitFlow", "location": "New York, NY", "seniority": "senior"},
        ),
    ],
)
def test_headers(header, expected):
    r = extract(header + "\n\nWe build software.")
    for k, v in expected.items():
        assert r[k] == v, k


def test_output_is_always_a_schema_shaped_record():
    for text in ["", "just a comment, not a post", "Acme | Engineer | Remote\n\n" + "x " * 500]:
        r = extract(text)
        assert tuple(r) == RECORD_KEYS
        for field, values in ENUMS.items():
            assert r[field] in values


def test_tech_stack_word_boundaries_and_case():
    assert tech_stack("We use Python, Postgres and k8s on AWS.") == ["aws", "kubernetes", "postgresql", "python"]
    assert tech_stack("Go and Rust services; C++ and C# tooling") == ["c#", "c++", "go", "rust"]
    # Ordinary words are not technologies.
    assert tech_stack("Let's go! We rust-proof boats; a swift hire; Series C; spring launch") == []
    assert tech_stack("PostgreSQL") == ["postgresql"]  # not also "sql"


def test_new_grad_rules():
    assert extract("Acme | Software Engineer, New Grad | NYC")["new_grad_eligible"] == "yes"
    assert extract("Acme | Software Engineer | NYC\n\n5+ years of experience required")["new_grad_eligible"] == "no"
    assert extract("Acme | Software Engineer | NYC")["new_grad_eligible"] == "unspecified"
