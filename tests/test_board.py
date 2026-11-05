"""The board's incremental build."""

from jpe.hn import Post
from scripts.build_board import build, read_store, write_store

POSTS = [
    Post(1, 100, "2026-10", "a", 1, "Acme | Backend Engineer | Remote (US) | $150k-$180k\n\nPython and Postgres."),
    Post(2, 100, "2026-10", "b", 2, "Beta | Designer | London, UK | ONSITE"),
]


def test_build_extracts_once_and_reuses_the_store(tmp_path):
    rows, store, info = build(POSTS, {}, "rules")
    assert info["extracted_now"] == 2
    assert [r["id"] for r in rows] == [2, 1]  # newest first
    acme = rows[1]
    assert acme["company"] == "Acme" and acme["remote"] == "yes" and acme["salary_min"] == 150_000
    assert acme["url"] == "https://news.ycombinator.com/item?id=1"

    path = tmp_path / "records.jsonl"
    write_store(store, path)
    rows2, _, info2 = build(POSTS + [Post(3, 100, "2026-10", "c", 3, "Gamma | ML Engineer | Remote")], read_store(path), "rules")
    assert info2["extracted_now"] == 1 and len(rows2) == 3


def test_failed_posts_stay_off_the_board():
    _, store, _ = build(POSTS, {}, "rules")
    store[2]["record"], store[2]["status"] = None, "failed"
    rows, _, info = build(POSTS, store, "rules")
    assert info["extracted_now"] == 0
    assert [r["id"] for r in rows] == [1]
