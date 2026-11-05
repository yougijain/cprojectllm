"""The HN reader, against a fake API. No network."""

from jpe import corpus
from jpe.hn import Post, Thread, fetch_by_ids, fetch_posts, html_to_text, list_hiring_threads, parse_thread_title


class FakeAPI:
    def __init__(self, items, submitted):
        self.items = items
        self._submitted = submitted
        self.calls = []

    def item(self, item_id):
        self.calls.append(item_id)
        return self.items.get(item_id)

    def submitted(self, user):
        return self._submitted


def test_parse_thread_title():
    assert parse_thread_title("Ask HN: Who is hiring? (November 2026)") == "2026-11"
    assert parse_thread_title("Ask HN: Who is hiring? (january 2025)") == "2025-01"
    assert parse_thread_title("Ask HN: Who wants to be hired? (November 2026)") is None
    assert parse_thread_title("Ask HN: Freelancer? Seeking freelancer? (November 2026)") is None
    assert parse_thread_title("") is None


def test_html_to_text_paragraphs_links_entities():
    raw = (
        "Acme | Senior Engineer | NYC &amp; REMOTE (US)<p>We build "
        '<i>things</i>. Apply: <a href="https:&#x2F;&#x2F;acme.example&#x2F;jobs&#x2F;123" '
        'rel="nofollow">https:&#x2F;&#x2F;acme.example&#x2F;jo...</a><p><p>Thanks'
    )
    assert html_to_text(raw) == (
        "Acme | Senior Engineer | NYC & REMOTE (US)\n\n"
        "We build things. Apply: https://acme.example/jobs/123\n\n"
        "Thanks"
    )


def test_html_to_text_empty():
    assert html_to_text("") == ""


def _api():
    items = {
        100: {"id": 100, "type": "story", "title": "Ask HN: Who is hiring? (October 2026)", "time": 1, "kids": [1, 2, 3, 4]},
        101: {"id": 101, "type": "story", "title": "Ask HN: Who wants to be hired? (October 2026)"},
        90: {"id": 90, "type": "story", "title": "Ask HN: Who is hiring? (September 2026)", "time": 0, "kids": []},
        80: {"id": 80, "type": "story", "title": "Ask HN: Who is hiring? (August 2026)", "time": 0},
        1: {"id": 1, "type": "comment", "parent": 100, "by": "a", "time": 5, "text": "Acme | Engineer"},
        2: {"id": 2, "type": "comment", "parent": 100, "deleted": True},
        3: {"id": 3, "type": "comment", "parent": 100, "dead": True, "text": "spam"},
        4: {"id": 4, "type": "comment", "parent": 100, "by": "b", "time": 6, "text": "Beta | Designer"},
    }
    return FakeAPI(items, submitted=[101, 100, 90, 80])


def test_list_threads_filters_titles_and_stops_at_since():
    api = _api()
    threads = list_hiring_threads(api, since="2026-09")
    assert [(t.id, t.month) for t in threads] == [(90, "2026-09"), (100, "2026-10")]
    assert 80 in api.calls  # read, found older than since, and the scan stopped


def test_fetch_posts_skips_deleted_dead_and_known():
    api = _api()
    thread = Thread(100, "2026-10", "t", 1)
    posts = fetch_posts(api, thread, workers=2)
    assert [p.id for p in posts] == [1, 4]
    assert posts[0].url == "https://news.ycombinator.com/item?id=1"

    api.calls.clear()
    again = fetch_posts(api, thread, skip_ids={1}, workers=2)
    assert [p.id for p in again] == [4]
    assert 1 not in api.calls


def test_fetch_by_ids_drops_posts_deleted_since():
    api = _api()
    threads = {100: Thread(100, "2026-10", "t", 1)}
    posts = fetch_by_ids(api, [1, 2, 4], threads, workers=2)
    assert [p.id for p in posts] == [1, 4]
    assert all(p.month == "2026-10" for p in posts)


def test_corpus_round_trip(tmp_path):
    raw, ids = tmp_path / "raw", tmp_path / "ids"
    a = Post(1, 100, "2026-10", "a", 5, "one")
    b = Post(2, 100, "2026-10", "b", 6, "two")
    corpus.write_posts("2026-10", [b], raw)
    corpus.write_posts("2026-10", [a, b], raw)  # merges, no duplicates
    assert [p.id for p in corpus.read_posts("2026-10", raw)] == [1, 2]
    corpus.write_ids("2026-10", [2, 1, 2], ids)
    assert corpus.read_ids("2026-10", ids) == [1, 2]
    assert set(corpus.load_corpus(raw_dir=raw)) == {1, 2}
