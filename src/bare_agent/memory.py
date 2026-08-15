"""Cross-run memory for the ReAct agent.

Model-initiated writes (via the `save_memory` tool in tools.py) are always
stored as DERIVED and expire after DERIVED_TTL_DAYS. This is not a default
that can be overridden by an argument the model supplies - there is no
parameter on the tool call that changes it. A model that read something
adversarial via run_bash could otherwise just claim "this came from the
user" and earn a trust tier it never should have - the same failure shape
as letting a model self-report its own permission tier. See OWASP ASI06
(memory & context poisoning).

VERIFIED entries exist for facts a human has actually confirmed, and are
reachable only through add_verified(), called from this module's __main__
block - never from a tool the model can invoke. There is deliberately no
code path from inside the agent loop to a VERIFIED entry. This is the
"never" tier from the permissions research applied to trust rather than
to actions: the constraint is structural, not a prompt instruction telling
the model not to claim VERIFIED.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

MEMORY_FILE = Path("memory.jsonl")
DERIVED_TTL_DAYS = 7

MemoryTier = Literal["VERIFIED", "DERIVED"]


@dataclass
class MemoryEntry:
    """One remembered fact, with enough provenance to judge whether to trust it.

    Attributes:
        content: The fact itself, as free text.
        tier: VERIFIED (human-confirmed, no expiry) or DERIVED (model-written
            during a run, expires after DERIVED_TTL_DAYS).
        basis: What the model or human is grounding this in, e.g. "output of
            'wc -l tests/'" or "user stated directly in conversation". Free
            text for a human reader's benefit; never used to decide tier.
        created_at: ISO 8601 UTC timestamp.
        expires_at: ISO 8601 UTC timestamp, or None for VERIFIED entries.
    """

    content: str
    tier: MemoryTier
    basis: str
    created_at: str
    expires_at: str | None

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        now = now or datetime.now(timezone.utc)
        return datetime.fromisoformat(self.expires_at) <= now


def _read_all_entries() -> list[MemoryEntry]:
    """Read every entry in MEMORY_FILE, tolerating corrupt lines.

    A hand-edited JSONL file can end up with a malformed line. This is
    harness startup, not a tool call the model sees - so a bad line is
    skipped with a printed warning rather than surfaced as a tool error
    or allowed to crash the run.
    """
    if not MEMORY_FILE.exists():
        return []

    entries: list[MemoryEntry] = []
    text = MEMORY_FILE.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            entries.append(MemoryEntry(**data))
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"  memory: skipping malformed line {line_number} in {MEMORY_FILE}: {exc}")
    return entries


def load_memory(now: datetime | None = None) -> list[MemoryEntry]:
    """Return all non-expired memory entries, most recent first."""
    entries = [e for e in _read_all_entries() if not e.is_expired(now)]
    return list(reversed(entries))


def _append_entry(entry: MemoryEntry) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def write_derived_memory(content: str, basis: str) -> str:
    """Persist a fact the agent observed during a run. Always DERIVED.

    Backing function for the `save_memory` tool in tools.py. Kept here,
    not there, because the JSONL I/O and tier logic is harness
    infrastructure shared with load_memory() and add_verified(), not
    something specific to being an LLM-facing tool.
    """
    if not content or not content.strip():
        return "No content given. Provide the fact to remember, e.g. 'tests run via pytest tests/ -v'."
    if not basis or not basis.strip():
        return "No basis given. Say what you're grounding this in, e.g. 'output of run_bash: wc -l tests/'."

    now = datetime.now(timezone.utc)
    entry = MemoryEntry(
        content=content.strip(),
        tier="DERIVED",
        basis=basis.strip(),
        created_at=now.isoformat(),
        expires_at=(now + timedelta(days=DERIVED_TTL_DAYS)).isoformat(),
    )
    _append_entry(entry)
    return f"Saved to memory (unverified, expires in {DERIVED_TTL_DAYS} days): {content.strip()}"


def add_verified(content: str) -> str:
    """Persist a human-confirmed fact. Never expires.

    Not reachable from the agent loop - see the module docstring. The only
    caller is this module's __main__ block, run directly from a terminal.
    """
    if not content or not content.strip():
        return "No content given."
    now = datetime.now(timezone.utc)
    entry = MemoryEntry(
        content=content.strip(),
        tier="VERIFIED",
        basis="human-confirmed via memory.py CLI",
        created_at=now.isoformat(),
        expires_at=None,
    )
    _append_entry(entry)
    return f"Saved as VERIFIED (no expiry): {content.strip()}"


def format_memory_for_prompt(entries: list[MemoryEntry]) -> str:
    """Render entries as a labeled block for injection into the system prompt.

    The label is the point: DERIVED entries are marked unverified and
    given an explicit expiry so the model treats them as notes to sanity-
    check, not instructions to obey. Returns "" when there's nothing to
    show, so callers can skip appending an empty section.
    """
    if not entries:
        return ""
    lines = ["Memory from previous runs of this agent:"]
    for e in entries:
        if e.tier == "VERIFIED":
            lines.append(f"[MEMORY - VERIFIED]: {e.content}")
        else:
            lines.append(f"[MEMORY - DERIVED, unverified, expires {e.expires_at}]: {e.content}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m bare_agent.memory "fact to remember"')
        raise SystemExit(1)
    print(add_verified(" ".join(sys.argv[1:])))
