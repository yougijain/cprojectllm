"""Normalization shared by labels, baselines and scoring.

Three things live here:
  * parse_salary: find a salary in free text and annualize it.
  * canonical_tech: one spelling per technology ("golang" -> "go").
  * coerce_enum: map loose wording ("on-site", "Entry level") to a schema
    value.

Everything here is deterministic and has no notion of a "right answer";
deciding what counts as correct is jpe/score.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- Salary ---------------------------------------------------------------

# Anything outside this, once annualized, is not a salary: a consulting fee,
# a funding round, a market size.
ANNUAL_FLOOR = 10_000
ANNUAL_CEILING = 2_000_000
# Currencies whose unit is small enough that a salary is in the millions.
_BOUNDS = {"INR": (100_000, 100_000_000), "JPY": (1_000_000, 300_000_000)}

HOURS_PER_YEAR = 2080
MONTHS_PER_YEAR = 12

_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}
_DOLLAR_PREFIXES = {"C": "CAD", "CA": "CAD", "A": "AUD", "AU": "AUD", "S": "SGD", "NZ": "NZD", "HK": "HKD", "US": "USD"}
_CODES = ("USD", "EUR", "GBP", "CAD", "AUD", "CHF", "INR", "SGD", "NZD", "HKD", "JPY", "SEK", "NOK", "DKK", "PLN", "BRL", "MXN", "ILS")

_NUM = r"\d{1,3}(?:[,.]\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d+)?"
_SUFFIX = r"k|K|mm|MM|m|M|mil|million|bn|b|B|billion|trillion|LPA|lpa|lakhs?|L"
_CUR = r"(?:(?P<pre>[A-Z]{1,2})?\$|[€£₹¥]|(?:" + "|".join(_CODES) + r")\s?)"
_AMOUNT_RE = re.compile(
    rf"(?P<cur>{_CUR})?\s?(?P<num>{_NUM})\s?(?P<suf>{_SUFFIX})?(?![A-Za-z])(?P<plus>\+)?"
)
_RANGE_SEP = re.compile(r"^\s*(?:-|–|—|to)\s*$", re.I)
_TRAILING_CODE = re.compile(r"^\s*(" + "|".join(_CODES) + r")\b")
_HOUR = re.compile(r"^\W{0,3}(?:/\s?(?:hr|hour|h)\b|per hour|an hour|hourly)", re.I)
_MONTH = re.compile(r"^\W{0,3}(?:/\s?(?:mo|month)\b|per month|a month|monthly|pm\b)", re.I)
_UP_TO = re.compile(r"(?:up to|upto|max(?:imum)?|<=?|under)[\s~≈]*$", re.I)
_SALARY_WORD = re.compile(r"\b(?:salary|comp|compensation|pay|base|ote|range)\b[^\n]{0,30}$", re.I)
_LARGE = {"mm", "m", "mil", "million", "bn", "b", "billion", "trillion"}
# "100k+ users" is a count, not a salary.
_COUNT_NOUN = re.compile(
    r"^\+?\s*(?:users|customers|clients|downloads|stars|requests|rps|qps|devices|members|people|"
    r"employees|companies|businesses|merchants|patients|students|developers|lines|transactions|orders)\b",
    re.I,
)


@dataclass(frozen=True)
class Salary:
    min: float | None
    max: float | None
    currency: str | None
    period: str  # "year", "month" or "hour", as written; min/max are annual
    span: tuple[int, int]


def _value(num: str) -> float:
    # "120,000", "120.000" (European grouping), "182,300.00", "1.5"
    if re.fullmatch(r"\d{1,3}(?:[,.]\d{3})+", num):
        return float(re.sub(r"[,.]", "", num))
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d{1,2}", num):
        return float(num.replace(",", ""))
    return float(num.replace(",", ""))


def _currency(m: re.Match, after: str) -> str | None:
    trailing = _TRAILING_CODE.match(after)
    if trailing:
        return trailing.group(1)
    cur = (m.group("cur") or "").strip()
    if not cur:
        return None
    if cur.endswith("$"):
        return _DOLLAR_PREFIXES.get(m.group("pre") or "", "USD")
    return _SYMBOLS.get(cur, cur if cur in _CODES else None)


def _scale(suf: str | None) -> float:
    if not suf:
        return 1.0
    s = suf.lower()
    if s == "k":
        return 1_000.0
    if s in ("lpa", "l", "lakh", "lakhs"):
        return 100_000.0
    return 0.0  # millions and up: never a salary


def _amounts(text: str):
    for m in _AMOUNT_RE.finditer(text):
        suf = m.group("suf")
        if suf and suf.lower() in _LARGE:
            continue
        yield m


def find_salaries(text: str) -> list[Salary]:
    """Every plausible salary in `text`, in order of appearance.

    A range is two amounts joined by a dash or "to", and a suffix or currency
    on either side applies to both ("$150-250k", "130–180K USD"). One amount
    is a point salary (min = max), an open floor if followed by "+", and a
    ceiling if preceded by "up to". Hourly and monthly figures are
    annualized at 2,080 hours and 12 months.
    """
    found: list[Salary] = []
    amounts = list(_amounts(text))
    i = 0
    while i < len(amounts):
        a = amounts[i]
        b = None
        if i + 1 < len(amounts) and _RANGE_SEP.match(text[a.end():amounts[i + 1].start()]):
            b = amounts[i + 1]
        last = b or a
        after = text[last.end(): last.end() + 25]

        suf_a, suf_b = a.group("suf"), b.group("suf") if b else None
        scale_a = _scale(suf_a or suf_b)
        scale_b = _scale(suf_b or suf_a) if b else scale_a
        if not scale_a or not scale_b:
            i += 2 if b else 1
            continue

        # A trailing code wins ("$150k - $180k CAD"), then a symbol.
        cur = _currency(last, after) or _currency(a, "")

        # "401k" is a retirement plan.
        if a.group("num") == "401" and (suf_a or "").lower() == "k":
            i += 2 if b else 1
            continue

        # With no currency, only a range in thousands ("120-150k") or an
        # amount the text calls pay counts, and never a count of things.
        if cur is None and not _SALARY_WORD.search(text[max(0, a.start() - 40): a.start()]):
            if not (b and (suf_a or suf_b)) or _COUNT_NOUN.match(after):
                i += 2 if b else 1
                continue

        lo = _value(a.group("num")) * scale_a
        hi = _value(b.group("num")) * scale_b if b else lo

        period = "year"
        if _HOUR.match(after):
            period, factor = "hour", HOURS_PER_YEAR
        elif _MONTH.match(after):
            period, factor = "month", MONTHS_PER_YEAR
        else:
            factor = 1
        lo, hi = lo * factor, hi * factor

        floor, ceiling = _BOUNDS.get(cur or "", (ANNUAL_FLOOR, ANNUAL_CEILING))
        if not (floor <= lo <= ceiling and floor <= hi <= ceiling) or lo > hi:
            i += 2 if b else 1
            continue

        smin, smax = lo, hi
        if not b:
            if last.group("plus"):
                smax = None
            elif _UP_TO.search(text[max(0, a.start() - 12): a.start()]):
                smin = None
        found.append(Salary(smin, smax, cur, period, (a.start(), last.end())))
        i += 2 if b else 1
    return found


def parse_salary(text: str) -> Salary | None:
    """The first plausible salary in `text`, or None."""
    found = find_salaries(text)
    return found[0] if found else None


# --- Tech stack -------------------------------------------------------------

# canonical name -> other spellings. Canonical names are lowercase.
TECH: dict[str, tuple[str, ...]] = {
    "python": ("py",),
    "javascript": ("js", "ecmascript"),
    "typescript": ("ts",),
    "go": ("golang",),
    "rust": (),
    "java": (),
    "kotlin": (),
    "scala": (),
    "c": (),
    "c++": ("cpp",),
    "c#": ("csharp", "c sharp"),
    ".net": ("dotnet", "asp.net"),
    "ruby": (),
    "rails": ("ruby on rails", "ror"),
    "php": (),
    "laravel": (),
    "elixir": (),
    "erlang": (),
    "haskell": (),
    "ocaml": (),
    "clojure": (),
    "swift": (),
    "objective-c": ("objc",),
    "dart": (),
    "flutter": (),
    "r": (),
    "julia": (),
    "sql": (),
    "react": ("reactjs", "react.js"),
    "react native": (),
    "next.js": ("nextjs", "next"),
    "vue": ("vuejs", "vue.js"),
    "angular": ("angularjs",),
    "svelte": ("sveltekit",),
    "node.js": ("node", "nodejs"),
    "django": (),
    "flask": (),
    "fastapi": (),
    "spring": ("spring boot",),
    "graphql": (),
    "postgresql": ("postgres", "psql"),
    "mysql": (),
    "sqlite": (),
    "mongodb": ("mongo",),
    "redis": (),
    "elasticsearch": ("elastic",),
    "kafka": (),
    "spark": ("pyspark", "apache spark"),
    "snowflake": (),
    "bigquery": (),
    "clickhouse": (),
    "dbt": (),
    "airflow": (),
    "aws": ("amazon web services",),
    "gcp": ("google cloud",),
    "azure": (),
    "kubernetes": ("k8s",),
    "docker": (),
    "terraform": (),
    "linux": (),
    "pytorch": ("torch",),
    "tensorflow": ("tf",),
    "jax": (),
    "cuda": (),
    "llms": ("llm",),
    "tailwind": ("tailwindcss",),
    "ios": (),
    "android": (),
    "unity": (),
    "verilog": ("systemverilog",),
    "fpga": (),
    "ros": (),
}

_ALIAS = {alias: canon for canon, aliases in TECH.items() for alias in aliases}


def canonical_tech(name: str) -> str:
    """One spelling per technology. Unknown names are lowercased and
    stripped, so they still compare equal to themselves."""
    key = re.sub(r"\s+", " ", (name or "").strip().lower())
    return _ALIAS.get(key, key)


def canonical_stack(names) -> list[str]:
    """Canonical, de-duplicated and sorted."""
    return sorted({canonical_tech(n) for n in names or [] if n and n.strip()})


# --- Enums ------------------------------------------------------------------

_ENUM_WORDS: dict[str, list[tuple[str, str]]] = {
    "remote": [
        # Order matters: a post offering remote *or* onsite is remote, but
        # "no remote" must be read before "remote".
        (r"\bhybrid\b|\bdays? ?(?:a |per |/ ?)week\b|\bin[- ]office \d", "hybrid"),
        (r"\bno remote\b|\bnot remote\b|\bremote (?:is )?not\b|\bon[- ]?site only\b|\bin[- ]office only\b", "no"),
        (r"\bremote\b|\bwfh\b|\bfully distributed\b", "yes"),
        (r"\bon[- ]?site\b|\bin[- ]office\b|\bin[- ]person\b", "no"),
    ],
    "seniority": [
        (r"\bintern(ship)?s?\b|\bco-?op\b", "intern"),
        (r"\bnew[- ]?grads?\b|\bentry[- ]level\b|\bjunior\b|\bjr\.?\b|\bgraduate\b|\bearly[- ]career\b", "new_grad"),
        (r"\bsenior\b|\bsr\.?\b|\bstaff\b|\bprincipal\b|\blead\b|\bdistinguished\b", "senior"),
        (r"\bmid[- ]?(level|senior)?\b|\bintermediate\b", "mid"),
    ],
    "visa_sponsorship": [
        (r"\bno (?:visa )?sponsorship\b|\bnot (?:able to |)sponsor\b|\bcannot sponsor\b|\bunable to sponsor\b|\bno visas?\b|\bwithout sponsorship\b", "no"),
        (r"\bvisa\b|\bsponsor(ship)?\b|\bh-?1b\b", "yes"),
    ],
}
_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}


def coerce_enum(field: str, value) -> str:
    """Map loose wording for an enum field to a schema value, or
    "unspecified" when nothing matches.

    Booleans map to yes/no for the yes/no fields. Order matters inside each
    list: "hybrid" is checked before "remote", "no sponsorship" before
    "sponsorship".
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value or "").strip().lower()
    if not text:
        return "unspecified"
    if field in ("remote", "visa_sponsorship", "new_grad_eligible"):
        if text in _YES:
            return "yes"
        if text in _NO:
            return "no"
    if text == "unspecified":
        return text
    if field == "seniority" and text in ("intern", "new_grad", "mid", "senior"):
        return text
    if field == "remote" and text == "hybrid":
        return text
    for pattern, out in _ENUM_WORDS.get(field, []):
        if re.search(pattern, text):
            return out
    return "unspecified"
