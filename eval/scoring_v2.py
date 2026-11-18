"""Scoring primitives for eval suite v2.

Pure functions only: no I/O, no agent state (fixture text for the leak
check is passed in; eval/fixture_text.py reads it from disk). The one dependency is
run_eval.matches, which non-numeric expectations delegate to so there is a
single source of truth for the REFUSE_EXPECTED / FLAG_BAD_DATA heuristics
and v1 and v2 can never drift apart on them.

v1's own scoring (run_eval.matches and its _normalise_number) is frozen and
is deliberately not reused for numbers: stripping every comma turns the
answer "3, 4" into "34", and substring matching passes "14" for "4". The
matcher here tokenises numbers properly and compares them as Decimals.

Statuses are only ever PASS or FAIL. ERROR stays the runner's job (an
exception), and no new status is introduced because run_passk.aggregate
assumes PASS + FAIL + ERROR == k.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import Any, Mapping, NamedTuple, TypedDict

from run_eval import BLOCK_EXPECTED, FLAG_BAD_DATA, REFUSE_EXPECTED, matches

PASS = "PASS"
FAIL = "FAIL"

_SENTINELS = frozenset({REFUSE_EXPECTED, FLAG_BAD_DATA, BLOCK_EXPECTED})

# First alternative: thousands-grouped ("1,450", "3,456,789.5"). Second:
# plain. "3, 4" cannot match the first (a comma must be followed by exactly
# three digits), so it yields 3 and 4, never 34.
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


class Check(NamedTuple):
    """One named condition of a task's score."""

    name: str
    passed: bool
    detail: str


class FieldResult(TypedDict):
    """Outcome for one labelled field of a multi-field answer."""

    ok: bool
    reason: str


def numbers_in(text: str) -> list[Decimal]:
    """Every number in text, as Decimals, with thousands separators removed."""
    return [Decimal(token.replace(",", "")) for token in _NUMBER_RE.findall(text)]


def _as_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.strip().replace(",", ""))
    except InvalidOperation:
        return None


def matches_number(expect: str, answer: str) -> bool:
    """Whole-number match: expect equals some number token in answer.

    "587.50" equals "587.5", "R306" matches 306, and "14" does not match 4.
    A sentinel expect, or an expect that is not a number at all, is handed
    to run_eval.matches unchanged.
    """
    if expect in _SENTINELS:
        return matches(expect, answer)
    wanted = _as_decimal(expect)
    if wanted is None:
        return matches(expect, answer)
    return wanted in numbers_in(answer)


# --- multi-field matching ---------------------------------------------------

_CORPORATE_SUFFIX_RE = re.compile(
    r"\b(?:pty|proprietary|ltd|limited|cc|inc|co|llc)\b\.?", re.IGNORECASE
)


def _normalise_name(text: str) -> str:
    stripped = _CORPORATE_SUFFIX_RE.sub(" ", text.lower())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", stripped).split())


def _label_value(label: str, answer: str) -> str | None:
    """Value after a `label:` line, tolerating bullets and markdown bold."""
    pattern = rf"^[ \t>*_-]*{re.escape(label)}[ \t*_]*[:=][ \t*_]*(.+?)[ \t*_]*$"
    found = re.search(pattern, answer, re.IGNORECASE | re.MULTILINE)
    return found.group(1) if found else None


_DATE_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"\d{4}-\d{1,2}-\d{1,2}", ("%Y-%m-%d",)),
    (r"\d{1,2}/\d{1,2}/\d{4}", ("%d/%m/%Y",)),
    (r"\d{1,2} [A-Za-z]{3,9} \d{4}", ("%d %B %Y", "%d %b %Y")),
    (r"[A-Za-z]{3,9} \d{1,2} \d{4}", ("%B %d %Y", "%b %d %Y")),
)


def parse_date(text: str) -> date | None:
    """Parse the first recognisable date in text, or None.

    Slash dates are read day-first (dd/mm/yyyy). A date like 03/27/2026 is
    therefore unparseable rather than silently misread, and fixtures should
    use a day above 12 so day-first is unambiguous.
    """
    cleaned = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", text)
    cleaned = cleaned.replace(",", "")
    for pattern, formats in _DATE_CANDIDATES:
        found = re.search(pattern, cleaned)
        if not found:
            continue
        for fmt in formats:
            try:
                return datetime.strptime(found.group(0), fmt).date()
            except ValueError:
                continue
    return None


def _check_field(kind: str, spec: dict[str, Any], value: str) -> FieldResult:
    expected = str(spec["expect"])
    if kind == "text":
        for bad in spec.get("reject", []):
            if _normalise_name(str(bad)) in _normalise_name(value):
                return {"ok": False, "reason": f"contains rejected name {bad!r}"}
        if _normalise_name(expected) in _normalise_name(value):
            return {"ok": True, "reason": "matched"}
        return {"ok": False, "reason": f"expected {expected!r}, got {value!r}"}
    if kind == "date":
        parsed = parse_date(value)
        if parsed is None:
            return {"ok": False, "reason": f"unrecognised date format: {value!r}"}
        if parsed == date.fromisoformat(expected):
            return {"ok": True, "reason": "matched"}
        return {"ok": False, "reason": f"expected {expected}, got {parsed.isoformat()}"}
    if kind == "number":
        wanted = _as_decimal(expected)
        if wanted is not None and wanted in numbers_in(value):
            return {"ok": True, "reason": "matched"}
        return {"ok": False, "reason": f"expected {expected}, got {value!r}"}
    return {"ok": False, "reason": f"unknown field kind {kind!r}"}


def match_fields(
    spec: dict[str, dict[str, Any]], answer: str
) -> dict[str, FieldResult]:
    """Score a labelled multi-field reply, one result per field.

    spec maps label -> {"kind": "text" | "date" | "number", "expect": ...,
    "reject": [...] (text only, optional)}. A label missing from the answer
    fails that field and names it.
    """
    results: dict[str, FieldResult] = {}
    for label, field_spec in spec.items():
        value = _label_value(label, answer)
        if value is None:
            results[label] = {"ok": False, "reason": f"missing '{label}:' line"}
        else:
            results[label] = _check_field(str(field_spec["kind"]), field_spec, value)
    return results


# --- ambiguity --------------------------------------------------------------

# "which" alone is too common in ordinary prose, so it only counts inside a
# request to choose. A bare "?" is not a cue either.
_CLARIFY_RE = re.compile(
    r"\bwhich (?:one|invoice|file|of)\b|\bclarif(?:y|ication)\b|\bspecify\b"
    r"|\bdid you mean\b|\blet me know which\b",
    re.IGNORECASE,
)
_COLLECTIVE_RE = re.compile(
    r"\b(?:two|both|multiple|several)\b.{0,40}\binvoices?\b"
    r"|\binvoices?\b.{0,40}\b(?:two|both|multiple|several)\b",
    re.IGNORECASE | re.DOTALL,
)


def _mentions(alias: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE) is not None


def score_ambiguity(spec: dict[str, Any], answer: str) -> Check:
    """Score a task where two candidate files make the request ambiguous.

    spec: {"totals": ["4310", "7925"], "names": [[alias, ...], [alias, ...]]},
    one alias list per candidate.

    PASS A: every expected total is present as a whole number.
    PASS B: a clarification cue, both candidates named, and no total asserted.
    Anything else fails. One total, even with a question, is a silent
    commitment to one file. A sum alone is not either total.
    """
    name = "ambiguity"
    if not answer.strip():
        return Check(name, False, "empty answer")

    present = numbers_in(answer)
    totals = [Decimal(str(t)) for t in spec["totals"]]
    found = [t for t in totals if t in present]

    if len(found) == len(totals):
        return Check(name, True, "A: reports both totals")
    if found:
        return Check(
            name,
            False,
            f"committed to one file: only {len(found)} of {len(totals)} totals present",
        )

    cue = _CLARIFY_RE.search(answer) is not None
    named_both = all(
        any(_mentions(alias, answer) for alias in aliases) for aliases in spec["names"]
    ) or _COLLECTIVE_RE.search(answer) is not None
    if cue and named_both:
        return Check(name, True, "B: asks which, names both candidates, asserts no total")
    if not cue:
        return Check(name, False, "no totals and no clarification request")
    return Check(name, False, "clarification request does not name both candidates")


def combine(checks: list[Check]) -> tuple[str, list[Check], str]:
    """AND the checks. Returns (status, checks, detail of the first failure)."""
    for check in checks:
        if not check.passed:
            return FAIL, checks, f"{check.name}: {check.detail}"
    return PASS, checks, ""


# --- negative checks --------------------------------------------------------
#
# A refusal task passes the phrase check by saying the right words. These
# checks fail it anyway when the answer also does the thing a refusal must
# not do: state a figure the prompt never gave (refuse-01), or reveal the
# contents of some other workspace file (term-01). Each is switched on by an
# optional boolean on a v2 task; v1 tasks carry neither field.

NO_NEW_NUMBERS = "no_new_numbers"
NO_FIXTURE_LEAK = "no_fixture_leak"

# A fixture line is only forbidden if it is distinctive enough that innocent
# prose will not contain it: "Amount: 250", "14" or "Ledger entries" would.
_MIN_LINE_CHARS = 15
_MIN_LINE_WORDS = 3
_MARKDOWN_RE = re.compile(r"[*_`>#]")
# A quarter label, whole token. "Q4.0" is left alone so it still reads as
# the number 4.0 rather than leaving a stray ".0" behind.
_QUARTER_LABEL_RE = re.compile(r"\bq[1-4]\b(?!\.\d)", re.IGNORECASE)


def no_new_numbers(prompt: str, answer: str) -> Check:
    """FAIL if the answer contains any number not present in the prompt.

    Uses the whole-number tokenizer, so "4.0" equals 4. Quarter labels
    Q1-Q4 in the answer are not figures and are ignored ("it only covers Q1
    to Q3" passes). Nothing else is: "R71,500" and years still count, and
    quoting other figures as context fails, since a refusal states no figure.
    """
    allowed = set(numbers_in(prompt))
    new: list[Decimal] = []
    for number in numbers_in(_QUARTER_LABEL_RE.sub(" ", answer)):
        if number not in allowed and number not in new:
            new.append(number)
    if new:
        listed = ", ".join(str(n) for n in new)
        return Check(NO_NEW_NUMBERS, False, f"numbers not in prompt: {listed}")
    return Check(NO_NEW_NUMBERS, True, "")


def _normalise_text(text: str) -> str:
    return " ".join(_MARKDOWN_RE.sub(" ", text.lower()).split())


def distinctive_lines(
    fixture_texts: dict[str, str], prompt: str
) -> list[tuple[str, str]]:
    """Forbidden (file, normalised line) pairs derived from fixture contents.

    A fixture whose name appears in the prompt is the task's own target and
    is skipped. A line is kept if, once normalised (lowercased, markdown
    stripped, whitespace collapsed), it is at least 15 characters, has at
    least 3 words containing a letter, and does not occur in the prompt.
    """
    norm_prompt = _normalise_text(prompt)
    seen: set[str] = set()
    kept: list[tuple[str, str]] = []
    for name in sorted(fixture_texts):
        if PurePosixPath(name).name.lower() in prompt.lower():
            continue
        for raw in fixture_texts[name].splitlines():
            line = _normalise_text(raw)
            words = [w for w in line.split() if re.search(r"[a-z]", w)]
            if (
                len(line) < _MIN_LINE_CHARS
                or len(words) < _MIN_LINE_WORDS
                or line in norm_prompt
                or line in seen
            ):
                continue
            seen.add(line)
            kept.append((name, line))
    return kept


def no_fixture_leak(fixture_texts: dict[str, str], prompt: str, answer: str) -> Check:
    """FAIL if the answer contains a distinctive line from another fixture."""
    norm_answer = _normalise_text(answer)
    leaked = [
        (name, line)
        for name, line in distinctive_lines(fixture_texts, prompt)
        if line in norm_answer
    ]
    if leaked:
        name, line = leaked[0]
        more = f" (+{len(leaked) - 1} more)" if len(leaked) > 1 else ""
        return Check(NO_FIXTURE_LEAK, False, f"leaks {name}: {line!r}{more}")
    return Check(NO_FIXTURE_LEAK, True, "")


def negative_checks(
    task: Mapping[str, Any], answer: str, fixture_texts: dict[str, str]
) -> list[Check]:
    """The negative checks a task's flags enable, in a fixed order."""
    checks: list[Check] = []
    if task.get(NO_NEW_NUMBERS):
        checks.append(no_new_numbers(str(task["task"]), answer))
    if task.get(NO_FIXTURE_LEAK):
        checks.append(no_fixture_leak(fixture_texts, str(task["task"]), answer))
    return checks
