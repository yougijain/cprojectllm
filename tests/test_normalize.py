"""Salary parsing, tech canonicalization and enum coercion.

Most salary cases are copied from real posts in the corpus, including the
funding rounds and market sizes that a naive "$ followed by a number" rule
would read as pay.
"""

import pytest

from jpe.normalize import canonical_stack, canonical_tech, coerce_enum, find_salaries, parse_salary


def sal(text):
    s = parse_salary(text)
    return None if s is None else (s.min, s.max, s.currency)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Senior Software Engineer | $80k-$110k+ (can go higher)", (80_000, 110_000, "USD")),
        ("Customer Success Engineer ($130–180K base, 0.05–0.15%)", (130_000, 180_000, "USD")),
        ("San Francisco, CA | ONSITE | Full-time | $150K - $220K + Equity", (150_000, 220_000, "USD")),
        ("Director, Privacy Engineering- $243,800 USD + equity", (243_800, 243_800, "USD")),
        ("REMOTE (USA only) | Full Time | $150-$300k", (150_000, 300_000, "USD")),
        ("Total Compensation (incl. Bonus) €80K – €100K | Equity", (80_000, 100_000, "EUR")),
        ("Total Salary Budget: $150,000 - $180,000 USD", (150_000, 180_000, "USD")),
        ("Full-time | https://freetrade.io | £75k-£140k", (75_000, 140_000, "GBP")),
        ("- $80,000–$160,000 / year cash - Local from Athens", (80_000, 160_000, "USD")),
        ("$165k–$250k base + $100k–$200k equity", (165_000, 250_000, "USD")),
        ("SF Bay Area | ONSITE | $200–250k + 1–2% equity", (200_000, 250_000, "USD")),
        ("ML Engineer x2 | Remote • Full time | $150-250k", (150_000, 250_000, "USD")),
        ("modash.io/careers/senior-data-engineer (€60k-140k)", (60_000, 140_000, "EUR")),
        ("Location: Waltham, MA — Pay Range $182,300.00 - $291,500.00", (182_300, 291_500, "USD")),
        ("Goody | Remote | $170-230K + equity and benefits", (170_000, 230_000, "USD")),
        ("REMOTE (US only) or NYC/Dallas | $130K to $170K base", (130_000, 170_000, "USD")),
        ("Frontend Web3) | REMOTE | Full-Time | >= $70-110k + 1-2% equity", (70_000, 110_000, "USD")),
        ("Toronto | CA$120k - CA$150k", (120_000, 150_000, "CAD")),
        ("Remote EU | EUR 70-90k", (70_000, 90_000, "EUR")),
        ("Zurich | CHF 140k–180k", (140_000, 180_000, "CHF")),
        ("Bangalore | 20-30 LPA", None),  # no currency and no "k": not read as pay
        ("Bangalore | ₹20-30 LPA", (2_000_000, 3_000_000, "INR")),
        ("Remote | 120-150k", (120_000, 150_000, None)),  # no currency stated
    ],
)
def test_ranges_and_points(text, expected):
    assert sal(text) == expected


def test_open_floor_and_ceiling():
    assert sal("- Salary: $100k+ depending on experience") == (100_000, None, "USD")
    assert sal("Comp: up to $200k") == (None, 200_000, "USD")
    assert sal("net new products | up to ~ $750k") == (None, 750_000, "USD")


def test_salary_word_must_be_a_whole_word():
    # "Remote" ends in "ote" (on-target earnings); that is not a pay keyword.
    assert sal("Palo Alto, CA & Remote | 100k+ and up") is None
    assert sal("OTE: 100k+") == (100_000, None, None)


def test_hourly_and_monthly_are_annualized():
    s = parse_salary("Contract | $100-$140/hr | Remote")
    assert (s.min, s.max, s.period) == (208_000, 291_200, "hour")
    s = parse_salary("REMOTE (worldwide) | Full-time | €5,000/month (full-time)")
    assert (s.min, s.max, s.currency, s.period) == (60_000, 60_000, "EUR", "month")
    s = parse_salary("the company wants to pay ~$2k a month for engineers")
    assert (s.min, s.period) == (24_000, "month")


@pytest.mark.parametrize(
    "text",
    [
        "We raised an $8M seed led by Kleiner Perkins and have paying customers",
        "a founding team behind multiple $1B+ ad products.",
        "We have raised $50 million in total funding",
        "problems in a $650B staffing industry",
        "We reached $4MM+ annual revenue before raising",
        "simplify construction permitting in the $1.6 trillion United States market",
        "a total of $71mil, including our $50mil Series B",
        "trading that has handled over £37 billion in volume since 2010",
        "Compensation (Pre-Funding): $150 - $300 setup/consulting fee upfront",
        "Benefits: health, dental, 401k matching",
        "We serve 100k+ users across 40 countries",
        "Our API handles 10-50k requests per second",
        "Founded in 2019, 2024-2025 was our best year",
        "",
    ],
)
def test_not_salaries(text):
    assert parse_salary(text) is None


def test_find_salaries_returns_every_one_in_order():
    found = find_salaries("Backend ($150k-$180k) and Frontend ($130k-$160k)")
    assert [(s.min, s.max) for s in found] == [(150_000, 180_000), (130_000, 160_000)]


def test_trailing_code_beats_dollar_sign():
    assert sal("Vancouver | $150k - $180k CAD") == (150_000, 180_000, "CAD")


def test_canonical_tech():
    assert canonical_tech("Golang") == "go"
    assert canonical_tech(" Postgres ") == "postgresql"
    assert canonical_tech("K8s") == "kubernetes"
    assert canonical_tech("React.js") == "react"
    assert canonical_tech("Zig") == "zig"  # unknown names still compare equal to themselves
    assert canonical_stack(["Postgres", "postgresql", "Python", "", None]) == ["postgresql", "python"]


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("remote", "REMOTE (US only)", "yes"),
        ("remote", "Hybrid, 2 days a week in SF", "hybrid"),
        ("remote", "ONSITE (3 days/week in office)", "hybrid"),
        ("remote", "ONSITE", "no"),
        ("remote", "No remote", "no"),
        ("remote", "Onsite or Remote", "yes"),
        ("remote", True, "yes"),
        ("remote", "", "unspecified"),
        ("seniority", "Senior/Staff Engineer", "senior"),
        ("seniority", "Software Engineering Intern", "intern"),
        ("seniority", "Entry-level Data Analyst", "new_grad"),
        ("seniority", "Software Engineer", "unspecified"),
        ("visa_sponsorship", "Visa sponsorship available", "yes"),
        ("visa_sponsorship", "no visa sponsorship", "no"),
        ("visa_sponsorship", "We are unable to sponsor visas", "no"),
        ("new_grad_eligible", "Yes", "yes"),
        ("new_grad_eligible", False, "no"),
    ],
)
def test_coerce_enum(field, value, expected):
    assert coerce_enum(field, value) == expected
