"""Keep the gold set's companies out of the training data.

Companies repost nearly the same text every month. If a test post's twin
from another month is in the training data, the fine-tuned model can score
well on it by remembering the company's name, location and stack, and the
test set stops measuring extraction.

Measured on the 7,604 posts of the frozen months, against the 390 gold
candidates: text similarity alone is not enough. Pairs at a Jaccard of 0.2
or even 0.15 are still the same company's post a few months apart, with
the wording drifted. So a training post is excluded if either
  * its company (the first header segment, normalized) is a gold
    candidate's company: 1,381 posts, or
  * its text is a near-duplicate of a gold candidate (Jaccard >= 0.3 on
    word 5-gram shingles): 923 posts, 48 of them not caught by company,
    mostly posts without a parseable header.
Together that is 1,429 posts, leaving 5,785 eligible: still more than the
4,000 silver labels needed.

An inverted index from shingle to gold post keeps the text comparison fast:
each training post is compared only with gold posts it shares a shingle with.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

from jpe.rules import company_name, split_header

SHINGLE = 5
THRESHOLD = 0.3

_WORD = re.compile(r"[a-z0-9]+")


def shingles(text: str, k: int = SHINGLE) -> frozenset[str]:
    words = _WORD.findall(text.lower())
    if len(words) < k:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[i:i + k]) for i in range(len(words) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def company_key(text: str) -> str | None:
    """'PermitFlow (YC W22)', 'permitflow.com' and 'PermitFlow, Inc.' all
    give 'permitflow'. None when the post has no parseable company."""
    segs = split_header(text)
    name = company_name(segs[0]) if segs else None
    if not name:
        return None
    k = re.sub(r"https?://|www\.", "", name.lower())
    k = re.sub(r"\(.*?\)|\[.*?\]", "", k)
    k = re.sub(r"\.(?:com|io|ai|dev|co|org|net|app|so|xyz)\b", "", k)
    k = re.sub(r"\b(?:inc|llc|ltd|gmbh|corp)\b\.?", "", k)
    k = re.sub(r"[^a-z0-9]+", " ", k).strip()
    return k or None


def near_duplicates(
    candidates: dict[int, str], reference: dict[int, str], threshold: float = THRESHOLD
) -> dict[int, tuple[int, float]]:
    """candidate id -> (most similar reference id, Jaccard) for every
    candidate at or above `threshold` against some reference post."""
    ref_sh = {rid: shingles(t) for rid, t in reference.items()}
    index: dict[str, list[int]] = defaultdict(list)
    for rid, sh in ref_sh.items():
        for s in sh:
            index[s].append(rid)

    out: dict[int, tuple[int, float]] = {}
    for cid, text in candidates.items():
        sh = shingles(text)
        shared = Counter(rid for s in sh for rid in index.get(s, ()))
        best = (None, 0.0)
        for rid, inter in shared.items():
            j = inter / (len(sh) + len(ref_sh[rid]) - inter)
            if j > best[1]:
                best = (rid, j)
        if best[0] is not None and best[1] >= threshold:
            out[cid] = (best[0], round(best[1], 3))
    return out


def training_pool(
    corpus_texts: dict[int, str], gold_ids: Iterable[int], threshold: float = THRESHOLD
) -> tuple[list[int], dict[int, dict]]:
    """Posts eligible for training: not gold, not from a gold company, and
    not a near-duplicate of a gold post.

    Returns the eligible ids (sorted) and, for each excluded post, why.
    """
    gold_set = set(gold_ids)
    gold_texts = {pid: corpus_texts[pid] for pid in gold_set if pid in corpus_texts}
    gold_companies = {k for k in (company_key(t) for t in gold_texts.values()) if k}
    rest = {pid: t for pid, t in corpus_texts.items() if pid not in gold_set}

    dropped: dict[int, dict] = {}
    for pid, text in rest.items():
        k = company_key(text)
        if k and k in gold_companies:
            dropped[pid] = {"company": k}
    for pid, (rid, j) in near_duplicates(rest, gold_texts, threshold).items():
        dropped.setdefault(pid, {}).update({"near": rid, "jaccard": j})
    return sorted(pid for pid in rest if pid not in dropped), dropped
