"""Read fixture files into memory for the v2 leak check.

The only I/O behind scoring_v2.no_fixture_leak, kept apart so scoring_v2
stays pure. The tests and the v2 runner both load fixtures through here, so
both derive the same forbidden strings from the same files.
"""

from __future__ import annotations

from pathlib import Path


def load_fixture_texts(*dirs: Path) -> dict[str, str]:
    """Every *.txt under the given dirs, keyed by path relative to its dir.

    Later dirs override earlier ones on the same relative name, matching an
    overlay copied on top of a base fixture set.
    """
    texts: dict[str, str] = {}
    for root in dirs:
        for path in sorted(root.rglob("*.txt")):
            if path.is_file():
                texts[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
    return texts
