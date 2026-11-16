"""Re-score every stored v1 answer under the v2 matcher and print what flips.

Read-only: reads eval/results/*.json, prints to the terminal, writes nothing.

For each stored answer (ERROR rows and BLOCK_EXPECTED rows are skipped, since
the first has no answer to score and the second is scored on the filesystem)
three statuses exist:

  stored   the status recorded when the run happened
  v1 now   what today's run_eval.matches() gives
  v2       what scoring_v2.matches_number() gives

A row is listed when stored != v2. The last column separates the cause:
rows where v1-now already differs from stored are matcher evolution that
predates v2 (the phrase lists were widened after those runs); a row where
v2 differs from v1-now is a change the v2 matcher itself introduces.

Run from the repo root:
    python eval/flip_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_eval
import scoring_v2

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EXCERPT_WIDTH = 60


def _runs(data: Any) -> list[tuple[str, list[dict[str, Any]]]]:
    """(run label suffix, entries) for a baseline list or a passk payload."""
    if isinstance(data, list):
        return [("", data)]
    return [(f"#run{i}", run) for i, run in enumerate(data["runs"], start=1)]


def _excerpt(answer: str) -> str:
    flat = " ".join(answer.split())
    return flat if len(flat) <= EXCERPT_WIDTH else flat[: EXCERPT_WIDTH - 1] + "…"


def main() -> None:
    rows: list[tuple[str, str, str, str, str, str]] = []
    rescored = 0
    v2_attributable = 0

    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for suffix, entries in _runs(data):
            for entry in entries:
                if entry["status"] == "ERROR" or entry["expect"] == run_eval.BLOCK_EXPECTED:
                    continue
                rescored += 1
                v1_now = "PASS" if run_eval.matches(entry["expect"], entry["answer"]) else "FAIL"
                v2 = "PASS" if scoring_v2.matches_number(entry["expect"], entry["answer"]) else "FAIL"
                if v2 == entry["status"]:
                    continue
                if v2 != v1_now:
                    v2_attributable += 1
                    cause = "v2 matcher"
                else:
                    cause = "pre-existing: v1 matcher today already gives this"
                rows.append(
                    (entry["id"], path.name + suffix, entry["status"], v2, _excerpt(entry["answer"]), cause)
                )

    print(f"Rescored {rescored} stored answers (ERROR and BLOCK_EXPECTED rows skipped).")
    print(f"Rows where stored status != v2 status: {len(rows)}")
    print(f"  of which caused by the v2 matcher itself: {v2_attributable}")
    print(f"  of which pre-existing (v1 matcher today already differs): {len(rows) - v2_attributable}")
    if not rows:
        return

    headers = ("TASK", "RUN FILE", "OLD", "NEW", "ANSWER EXCERPT", "CAUSE")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print()
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


if __name__ == "__main__":
    main()
