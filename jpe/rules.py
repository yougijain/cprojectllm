"""B1: the rules-and-regex baseline.

No model at all. Most posts follow the thread's header convention,
"Company | Role | Location | REMOTE | Salary | URL" (6,952 of 7,791 have at
least two pipes in the first line), so splitting that line and classifying
the pieces gets a long way. The rest is keyword lists and the salary parser.

It is cheap to run and harder to beat than it sounds on the fields with
fixed vocabulary (remote, visa, salary), and it is the floor every model
has to clear to justify its cost.
"""

from __future__ import annotations

import re

from jpe.fields import EMPTY_RECORD
from jpe.normalize import TECH, coerce_enum, parse_salary

VERSION = "rules-1"

# --- Header splitting -------------------------------------------------------

_SEP = re.compile(r"\s*\|\s*")
_FALLBACK_SEP = re.compile(r"\s+(?:•|·|—|–|-)\s+")

_URL = re.compile(r"https?://|www\.|^[\w-]+(?:\.[\w-]+)+(?:/\S*)?$", re.I)
_EMPLOYMENT = re.compile(r"\b(?:full[- ]?time|part[- ]?time|fulltime|contract(?:or)?|freelance|permanent|intern(?:ship)?s? only)\b", re.I)
_REMOTE_WORD = re.compile(r"\b(?:remote|on[- ]?site|onsite|hybrid|in[- ]office|in[- ]person|wfh)\b", re.I)
_ROLE = re.compile(
    r"\b(?:engineers?|eng\.?|developers?|devs?|scientists?|designers?|managers?|analysts?|architects?|"
    r"researchers?|leads?|directors?|sre|devops|interns?|recruiters?|marketers?|marketing|sales|"
    r"product|founding|cto|vp|head of|specialists?|consultants?|administrators?|technicians?|writers?|"
    r"operations|ops|accountants?|counsel|associates?|programmers?|hackers?|generalists?|"
    r"roles?|positions?|hiring)\b",
    re.I,
)
_US_STATE = re.compile(r",\s*(?:A[LKZR]|C[AOT]|D[EC]|FL|GA|HI|I[DLNA]|K[SY]|LA|M[EDAINSOT]|N[EVHJMYCD]|O[HKR]|PA|RI|S[CD]|T[NX]|UT|V[TA]|W[AVIY])\b")
_PLACES = re.compile(
    r"\b(?:San Francisco|SF|Bay Area|Silicon Valley|Palo Alto|Mountain View|Menlo Park|Oakland|San Jose|"
    r"Sunnyvale|Berkeley|NYC|New York|Brooklyn|Boston|Cambridge|Seattle|Austin|Chicago|Los Angeles|LA|"
    r"Denver|Boulder|Miami|Atlanta|Washington|DC|Philadelphia|Pittsburgh|Portland|San Diego|Dallas|"
    r"Houston|Salt Lake City|Toronto|Vancouver|Montreal|Waterloo|London|Cambridge|Oxford|Edinburgh|"
    r"Manchester|Dublin|Berlin|Munich|Hamburg|Paris|Amsterdam|Rotterdam|Brussels|Zurich|Geneva|"
    r"Lisbon|Porto|Madrid|Barcelona|Milan|Rome|Stockholm|Copenhagen|Oslo|Helsinki|Warsaw|Krakow|Prague|"
    r"Vienna|Athens|Tel Aviv|Singapore|Hong Kong|Tokyo|Seoul|Bangalore|Bengaluru|Mumbai|Delhi|"
    r"Hyderabad|Pune|Sydney|Melbourne|Auckland|Sao Paulo|Mexico City|Buenos Aires|"
    r"USA?|United States|UK|United Kingdom|EU|Europe|Canada|Germany|France|Spain|Portugal|Netherlands|"
    r"Switzerland|Sweden|Poland|Ireland|Israel|India|Japan|Australia|Brazil|Mexico|LatAm|LATAM|"
    r"North America|Americas|EMEA|APAC|Worldwide|Global|Anywhere)\b"
)
_PAREN = re.compile(r"\(([^)]*)\)")
_YC = re.compile(r"\s*[\(\[]\s*YC\s*[A-Z]?\d{2}\s*[\)\]]", re.I)


def split_header(text: str) -> list[str]:
    first = next((line for line in text.split("\n") if line.strip()), "")
    parts = _SEP.split(first) if first.count("|") >= 1 else _FALLBACK_SEP.split(first)
    return [p.strip(" []") for p in parts if p.strip(" []")]


def company_name(seg: str) -> str | None:
    seg = _YC.sub("", seg).strip()
    seg = re.sub(r"\s*\((?:FKA|formerly)[^)]*\)", "", seg, flags=re.I).strip()
    if not seg or len(seg) > 60 or len(seg.split()) > 6:
        return None
    if seg.lower().startswith(("http://", "https://", "www.")):
        seg = re.sub(r"^(?:https?://)?(?:www\.)?", "", seg, flags=re.I).split("/")[0]
    return seg


def _location(seg: str) -> str | None:
    """Clean a segment known to name a place: drop remote/onsite wording."""
    seg = _PAREN.sub(lambda m: "" if _REMOTE_WORD.search(m.group(1)) else m.group(0), seg)
    seg = _REMOTE_WORD.sub("", seg)
    seg = re.sub(r"\bonly\b|\bfriendly\b", "", seg, flags=re.I)
    seg = re.sub(r"^[\s\-–—:,/]+|[\s\-–—:,/]+$", "", seg)
    seg = re.sub(r"\s{2,}", " ", seg).replace(" ,", ",")
    seg = re.sub(r"\(\s*\)", "", seg).strip()
    return seg or None


def _region_from_remote(seg: str) -> str | None:
    """'REMOTE (US only)' -> 'US'; 'Remote – Mexico / South America' -> that."""
    m = _PAREN.search(seg)
    inner = m.group(1) if m else _REMOTE_WORD.sub("", seg)
    inner = re.sub(r"\b(?:only|preferred|friendly|time ?zones?|ok)\b|[-–—]only", "", inner, flags=re.I)
    inner = re.sub(r"^[\s\-–—:,/]+|[\s\-–—:,/]+$", "", inner).strip()
    return inner if inner and _PLACES.search(inner) else None


# --- Tech stack -------------------------------------------------------------

# Names that are also ordinary words (or too short to match safely) are
# matched case-sensitively, in the spellings listed. Empty means never: "C"
# and "R" collide with "Series C" and too much else for a keyword rule.
_CASED = {
    "go": ["Go", "Golang", "golang"], "rust": ["Rust"], "swift": ["Swift"], "spark": ["Spark", "PySpark", "Apache Spark"],
    "unity": ["Unity"], "dart": ["Dart"], "julia": ["Julia"], "ruby": ["Ruby"], "spring": ["Spring", "Spring Boot"],
    "ios": ["iOS"], "ros": ["ROS", "ROS2"], "node.js": ["Node", "Node.js", "NodeJS", "nodejs", "node.js"],
    "next.js": ["Next.js", "NextJS", "nextjs"], "flutter": ["Flutter"], "jax": ["JAX"], "dbt": ["dbt"],
    "elixir": ["Elixir"], "scala": ["Scala"], "java": ["Java"], "react": ["React", "React.js", "ReactJS", "react"],
    "sql": ["SQL"], "linux": ["Linux"], "c": [], "r": [], "llms": [],
}
_EDGE_L = r"(?<![\w.+#/-])"
_EDGE_R = r"(?![\w+#-])"


def _tech_patterns() -> list[tuple[str, re.Pattern]]:
    out = []
    for canon, aliases in TECH.items():
        if canon in _CASED:
            names, flags = _CASED[canon], 0
        else:
            names, flags = [canon, *[a for a in aliases if len(a) > 2 or a == "k8s"]], re.I
        if names:
            alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
            out.append((canon, re.compile(_EDGE_L + f"(?:{alt})" + _EDGE_R, flags)))
    return out


_TECH_PATTERNS = _tech_patterns()


def tech_stack(text: str) -> list[str]:
    return sorted({canon for canon, pat in _TECH_PATTERNS if pat.search(text)})


# --- New grads --------------------------------------------------------------

_NEW_GRAD_YES = re.compile(r"\bnew[- ]?grads?\b|\bentry[- ]level\b|\brecent (?:college )?grad(?:uate)?s?\b|\bjunior\b|\bearly[- ]career\b|\ball (?:experience )?levels\b", re.I)
_NEW_GRAD_NO = re.compile(r"\bno (?:new grads|juniors)\b|\bnot (?:a )?(?:new grad|junior|entry)|\b(?:[2-9]|1\d)\+?\s*(?:-\s*\d+\s*)?(?:years|yrs)\b", re.I)


def new_grad(text: str, seniority: str) -> str:
    if _NEW_GRAD_NO.search(text):
        return "no"
    if _NEW_GRAD_YES.search(text) or seniority in ("new_grad", "intern"):
        return "yes"
    if seniority == "senior":
        return "no"
    return "unspecified"


# --- The baseline -----------------------------------------------------------

def extract(text: str) -> dict:
    """One record from one post, by rules alone."""
    r = dict(EMPTY_RECORD)
    segs = split_header(text)
    header = " | ".join(segs)

    role_seg = loc_seg = remote_seg = None
    for i, seg in enumerate(segs):
        if i == 0:
            continue
        if _URL.search(seg) or parse_salary(seg):
            continue
        if remote_seg is None and _REMOTE_WORD.search(seg) and not _ROLE.search(seg):
            remote_seg = seg
            if loc_seg is None and (_US_STATE.search(seg) or _PLACES.search(_REMOTE_WORD.sub("", _PAREN.sub("", seg)))):
                loc_seg = seg
            continue
        # Role before employment type: "BI Engineer (Contract)" is a role.
        if role_seg is None and _ROLE.search(seg) and not _US_STATE.search(seg):
            role_seg = seg
            continue
        if _EMPLOYMENT.search(seg) and len(seg) < 25:
            continue
        if loc_seg is None and (_US_STATE.search(seg) or _PLACES.search(seg)):
            loc_seg = seg

    if segs:
        r["company"] = company_name(segs[0])
    r["role_title"] = role_seg
    if loc_seg:
        r["location"] = _location(loc_seg)
    elif remote_seg:
        r["location"] = _region_from_remote(remote_seg)

    r["remote"] = coerce_enum("remote", remote_seg or header)
    if r["remote"] == "unspecified":
        r["remote"] = coerce_enum("remote", text[:600])
    r["visa_sponsorship"] = coerce_enum("visa_sponsorship", text)
    r["seniority"] = coerce_enum("seniority", role_seg or header)

    s = parse_salary(header) or parse_salary(text)
    if s:
        r["salary_min"], r["salary_max"], r["salary_currency"] = s.min, s.max, s.currency

    r["tech_stack"] = tech_stack(text)
    r["new_grad_eligible"] = new_grad(text, r["seniority"])
    return r
