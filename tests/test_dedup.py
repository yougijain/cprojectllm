"""Near-duplicate filtering and the silver pool order."""

from jpe.dedup import company_key, jaccard, near_duplicates, shingles, training_pool
from jpe.silver import ordered_pool

POST = (
    "Acme | Senior Backend Engineer | New York, NY | ONSITE | $180k-$220k\n\n"
    "We build payment infrastructure for small businesses. Our stack is Python, Postgres and Kubernetes "
    "on AWS. You will own services end to end and work closely with product. Apply at acme.example/jobs."
)
REPOST = POST.replace("$180k-$220k", "$190k-$230k").replace("Apply at", "Email us or apply at")
OTHER = (
    "Beta | Designer | Remote (EU)\n\nWe make tools for architects. Figma, user research, and a love of "
    "typography. Small team, four-day week, no meetings on Wednesdays. Write to jobs@beta.example."
)


def test_shingles_and_jaccard():
    assert jaccard(shingles(POST), shingles(POST)) == 1.0
    assert jaccard(shingles(POST), shingles(OTHER)) < 0.05
    assert shingles("") == frozenset()
    assert shingles("too short") == frozenset({"too short"})


def test_monthly_repost_is_a_near_duplicate():
    assert jaccard(shingles(POST), shingles(REPOST)) > 0.5
    found = near_duplicates({10: REPOST, 11: OTHER}, {1: POST})
    assert list(found) == [10] and found[10][0] == 1


def test_company_key():
    assert company_key("PermitFlow (YC W22) | Engineer") == "permitflow"
    assert company_key("permitflow.com | Engineer") == "permitflow"
    assert company_key("PermitFlow, Inc. | Engineer") == "permitflow"
    assert company_key("") is None


def test_training_pool_excludes_gold_its_twins_and_its_companies():
    rewritten = "Acme | Staff Engineer | Remote (US)\n\nEntirely new words about a different team and role."
    texts = {
        1: POST,
        2: OTHER,
        10: REPOST,  # near-identical text
        13: rewritten,  # same company, new text
        11: OTHER + " Also hiring a PM.",
        12: "Gamma | ML Engineer | Remote\n\nPyTorch and JAX research.",
    }
    eligible, dropped = training_pool(texts, gold_ids=[1])
    assert 1 not in eligible
    assert dropped[10]["near"] == 1 and dropped[13] == {"company": "acme"}
    assert 2 in eligible and 12 in eligible
    # Near-duplicates among training posts themselves are fine; only gold's are removed.
    assert 11 in eligible


def test_pool_order_is_fixed_and_prefixes_nest():
    a = ordered_pool(list(range(100)))
    assert a == ordered_pool(list(reversed(range(100))))
    assert sorted(a) == list(range(100)) and a != list(range(100))
    assert a[:25] == ordered_pool(list(range(100)))[:25]
