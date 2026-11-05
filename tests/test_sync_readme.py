import pytest

from scripts.sync_readme import blocks, fill

README = "# T\n\n<!-- results:table -->\n_pending_\n<!-- /results:table -->\n\n<!-- results:errors -->\n_pending_\n<!-- /results:errors -->\n"


def test_fill_replaces_only_the_marked_blocks():
    md = "| a | b |\n\n### Why fields were wrong\n\n| m | missed |\n"
    out = fill(README, blocks(md))
    assert "_pending_" not in out
    assert "<!-- results:table -->\n| a | b |\n<!-- /results:table -->" in out
    assert "<!-- results:errors -->\n| m | missed |\n<!-- /results:errors -->" in out
    assert fill(out, blocks(md)) == out  # idempotent


def test_errors_block_untouched_without_an_error_section():
    out = fill(README, blocks("| a | b |\n"))
    assert "<!-- results:errors -->\n_pending_" in out


def test_missing_marker_is_an_error():
    with pytest.raises(ValueError):
        fill("# no markers\n", {"table": "x"})
