"""Read "Ask HN: Who is hiring?" threads from the public Hacker News API.

Every month the `whoishiring` account posts a thread, and each top-level
comment on it is one job post. Replies under a post are discussion, not
posts, so only the thread's direct children are read.

The API is https://github.com/HackerNews/API. It needs no key and has no
documented rate limit; this client still keeps a small pool and backs off on
errors, since nothing here is urgent.
"""

from __future__ import annotations

import html
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Iterable, Protocol

import requests

API = "https://hacker-news.firebaseio.com/v0"
POSTER = "whoishiring"
USER_AGENT = "cprojectllm-fetch (+https://github.com/yougijain/cprojectllm)"

MONTHS = {
    name: i
    for i, name in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"],
        start=1,
    )
}

# "Ask HN: Who is hiring? (November 2026)". The sibling threads, "Who wants
# to be hired?" and "Freelancer? Seeking freelancer?", must not match.
_TITLE = re.compile(r"^Ask HN: Who is hiring\?\s*\((\w+)\s+(\d{4})\)\s*$", re.I)


@dataclass(frozen=True)
class Thread:
    id: int
    month: str  # "YYYY-MM"
    title: str
    time: int


@dataclass(frozen=True)
class Post:
    id: int
    thread_id: int
    month: str
    author: str
    time: int
    html: str

    @property
    def text(self) -> str:
        return html_to_text(self.html)

    @property
    def url(self) -> str:
        return f"https://news.ycombinator.com/item?id={self.id}"

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Post":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def parse_thread_title(title: str) -> str | None:
    """Return the thread's month as "YYYY-MM", or None if it is not a hiring
    thread."""
    m = _TITLE.match(title or "")
    if not m:
        return None
    month = MONTHS.get(m.group(1).lower())
    if month is None:
        return None
    return f"{int(m.group(2)):04d}-{month:02d}"


_P = re.compile(r"<p>", re.I)
_LINK = re.compile(r'<a\s+[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t ]+")


def html_to_text(raw: str) -> str:
    """Turn HN comment HTML into plain text.

    HN separates paragraphs with a bare `<p>` and shortens the visible text of
    long links ("https://example.com/jo..."), so a link becomes its href.
    """
    if not raw:
        return ""
    text = _P.sub("\n\n", raw)
    text = _LINK.sub(lambda m: m.group(1), text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


class ItemSource(Protocol):
    def item(self, item_id: int) -> dict | None: ...
    def submitted(self, user: str) -> list[int]: ...


class HNClient:
    """Thin client over the Firebase API, with retries."""

    def __init__(self, base: str = API, timeout: float = 20.0, retries: int = 4):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def _get(self, path: str):
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                r = self.session.get(f"{self.base}/{path}.json", timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                if attempt == self.retries:
                    raise
                time.sleep(delay)
                delay *= 2

    def item(self, item_id: int) -> dict | None:
        return self._get(f"item/{item_id}")

    def submitted(self, user: str) -> list[int]:
        data = self._get(f"user/{user}") or {}
        return list(data.get("submitted", []))


def list_hiring_threads(
    source: ItemSource, since: str, until: str | None = None
) -> list[Thread]:
    """Hiring threads from `since` to `until` (inclusive, "YYYY-MM"), oldest
    first.

    `whoishiring` lists its submissions newest first, so the scan stops at the
    first thread older than `since`.
    """
    threads: list[Thread] = []
    for item_id in source.submitted(POSTER):
        item = source.item(item_id) or {}
        month = parse_thread_title(item.get("title", ""))
        if month is None:
            continue
        if month < since:
            break
        if until is not None and month > until:
            continue
        threads.append(Thread(item_id, month, item["title"], item.get("time", 0)))
    return sorted(threads, key=lambda t: t.month)


def _is_live_comment(item: dict | None) -> bool:
    return bool(
        item
        and item.get("type") == "comment"
        and not item.get("deleted")
        and not item.get("dead")
        and item.get("text")
    )


def fetch_posts(
    source: ItemSource,
    thread: Thread,
    skip_ids: Iterable[int] = (),
    workers: int = 8,
) -> list[Post]:
    """Top-level comments on `thread` that are not deleted or dead.

    `skip_ids` are posts already on disk; an incremental refresh only fetches
    what is new.
    """
    story = source.item(thread.id) or {}
    skip = set(skip_ids)
    kids = [k for k in story.get("kids", []) if k not in skip]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        items = list(pool.map(source.item, kids))

    posts = []
    for item in items:
        if not _is_live_comment(item) or item.get("parent") != thread.id:
            continue
        posts.append(
            Post(
                id=item["id"],
                thread_id=thread.id,
                month=thread.month,
                author=item.get("by", ""),
                time=item.get("time", 0),
                html=item["text"],
            )
        )
    return sorted(posts, key=lambda p: p.id)


def fetch_by_ids(
    source: ItemSource, ids: Iterable[int], threads: dict[int, Thread], workers: int = 8
) -> list[Post]:
    """Rebuild posts from committed IDs. Posts deleted since are dropped."""
    ids = list(ids)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        items = list(pool.map(source.item, ids))
    posts = []
    for item in items:
        if not _is_live_comment(item) or item.get("parent") not in threads:
            continue
        t = threads[item["parent"]]
        posts.append(
            Post(item["id"], t.id, t.month, item.get("by", ""), item.get("time", 0), item["text"])
        )
    return sorted(posts, key=lambda p: p.id)
