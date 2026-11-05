"""Copy the generated results into the README.

    python scripts/sync_readme.py          # rewrite the marked blocks
    python scripts/sync_readme.py --check  # fail if the README is stale

The README quotes results/test_results.md between markers:
    <!-- results:table --> ... <!-- /results:table -->
    <!-- results:errors --> ... <!-- /results:errors -->
so no number in it is typed by hand. Until scripts/evaluate.py has written
the file, the blocks keep their "pending" text and --check passes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "results" / "test_results.md"
ERRORS_HEADING = "### Why fields were wrong"


def blocks(results_md: str) -> dict[str, str]:
    table, _, errors = results_md.partition(ERRORS_HEADING)
    out = {"table": table.strip()}
    if errors.strip():
        out["errors"] = errors.split("\n", 1)[1].strip() if "\n" in errors else errors.strip()
    return out


def fill(readme: str, content: dict[str, str]) -> str:
    for name, body in content.items():
        pattern = re.compile(rf"(<!-- results:{name} -->\n).*?(\n<!-- /results:{name} -->)", re.S)
        if not pattern.search(readme):
            raise ValueError(f"README has no results:{name} block")
        readme = pattern.sub(lambda m: m.group(1) + body + m.group(2), readme)
    return readme


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if not RESULTS.exists():
        print("no results/test_results.md yet; README blocks left as they are")
        return 0
    current = README.read_text(encoding="utf-8")
    updated = fill(current, blocks(RESULTS.read_text(encoding="utf-8")))
    if args.check:
        if updated != current:
            print("README results are stale: run scripts/sync_readme.py", file=sys.stderr)
            return 1
        return 0
    README.write_text(updated, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
