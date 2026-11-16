"""Eval suite v2: scorers and the recorder.

Part A only. Fixture, task-file and runner tests arrive with Part B.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from groq.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from groq.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from bare_agent import agent, permissions, tools

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import recorder  # noqa: E402
import scoring_v2 as s  # noqa: E402

# --- whole-number matcher ---------------------------------------------------


@pytest.mark.parametrize(
    ("expect", "answer", "want"),
    [
        ("4", "The file has 4 lines.", True),
        ("4", "The file has 14 lines.", False),
        ("4", "There are 114 lines", False),
        ("4", "That is 4.", True),
        ("4", "about 4.5 lines", False),
        ("587.5", "The total is $587.50.", True),
        ("587.50", "The total is 587.5", True),
        ("1450", "The total is 1,450.", True),
        ("1,450", "The total is 1450", True),
        ("306", "R306", True),
        ("2", "Files: 3, 4", False),
        ("4", "Files: 3, 4", True),
        ("34", "Files: 3, 4", False),
        ("4", "no digits here", False),
        ("4", "", False),
    ],
)
def test_matches_number_table(expect: str, answer: str, want: bool) -> None:
    assert s.matches_number(expect, answer) is want


def test_numbers_in_separates_a_list_instead_of_joining_it() -> None:
    assert s.numbers_in("3, 4") == [Decimal(3), Decimal(4)]
    assert s.numbers_in("3,456,789.5 and 12") == [Decimal("3456789.5"), Decimal(12)]


def test_sentinels_delegate_to_v1_matcher() -> None:
    refusal = "The Q4 figure is not listed."
    assert s.matches_number(s.REFUSE_EXPECTED, refusal) is True
    assert s.matches_number(s.REFUSE_EXPECTED, "It is 42.") is False


def test_non_numeric_expect_falls_back_to_v1_substring() -> None:
    assert s.matches_number("Acme", "Billed to acme corp") is True


# --- multi-field matcher ----------------------------------------------------

SPEC: dict[str, dict[str, Any]] = {
    "supplier": {"kind": "text", "expect": "Karoo Fasteners", "reject": ["Acme"]},
    "date": {"kind": "date", "expect": "2026-03-27"},
    "total": {"kind": "number", "expect": "1842.50"},
}

GOOD = "supplier: Karoo Fasteners (Pty) Ltd\ndate: 27 March 2026\ntotal: R1,842.50"


def test_match_fields_all_correct() -> None:
    results = s.match_fields(SPEC, GOOD)
    assert all(r["ok"] for r in results.values()), results


def test_match_fields_tolerates_markdown_and_date_formats() -> None:
    answer = "- **Supplier:** karoo fasteners\n- **Date:** 27th Mar 2026\n- **Total:** 1842.5"
    assert all(r["ok"] for r in s.match_fields(SPEC, answer).values())
    for form in ("2026-03-27", "27/03/2026", "March 27, 2026"):
        assert s.match_fields(SPEC, f"date: {form}")["date"]["ok"], form


def test_match_fields_missing_label_fails_only_that_field() -> None:
    results = s.match_fields(SPEC, "supplier: Karoo Fasteners\ntotal: 1842.50")
    assert results["date"]["ok"] is False
    assert "missing" in results["date"]["reason"]
    assert results["supplier"]["ok"] and results["total"]["ok"]


def test_match_fields_wrong_date_format_and_wrong_date() -> None:
    month_first = s.match_fields(SPEC, "date: 03/27/2026")["date"]
    assert month_first["ok"] is False and "unrecognised" in month_first["reason"]
    wrong_day = s.match_fields(SPEC, "date: 28 March 2026")["date"]
    assert wrong_day["ok"] is False and "2026-03-28" in wrong_day["reason"]


def test_match_fields_supplier_vs_bill_to_swap() -> None:
    swapped = "supplier: Acme Corporation\ndate: 27 March 2026\ntotal: 1842.50"
    assert s.match_fields(SPEC, swapped)["supplier"]["ok"] is False
    both = "supplier: Karoo Fasteners, billed to Acme Corporation"
    assert s.match_fields(SPEC, both)["supplier"]["ok"] is False  # rejected name


def test_match_fields_total_must_be_the_labelled_number() -> None:
    wrong = "total: R1,600.00 (subtotal), VAT 242.50"
    assert s.match_fields(SPEC, wrong)["total"]["ok"] is False


# --- ambig-01 truth table ---------------------------------------------------

AMBIG = {
    "totals": ["4310", "7925"],
    "names": [["invoice_kloof", "Kloof Hardware"], ["invoice_ridge", "Ridge Tooling"]],
}


@pytest.mark.parametrize(
    ("answer", "want", "rule"),
    [
        ("Kloof Hardware: R4,310. Ridge Tooling: R7,925.", True, "A"),
        ("invoice_kloof.txt totals 4310 and invoice_ridge.txt totals 7925", True, "A"),
        ("There are two invoices (Kloof Hardware and Ridge Tooling). Which one do you want?", True, "B"),
        ("I found invoice_kloof.txt and invoice_ridge.txt. Please specify which.", True, "B"),
        ("Could you clarify? Both Kloof Hardware and Ridge Tooling have invoices.", True, "B"),
        # one total plus a question: committed to one file
        ("The total is R4,310. Did you mean the Ridge Tooling invoice instead?", False, "one"),
        # one total, no question
        ("The invoice total is 4,310.", False, "one"),
        # sum only: neither individual total
        ("The combined total is 12,235.", False, "none"),
        # a bare question mark or boilerplate is not a clarification
        ("Kloof Hardware and Ridge Tooling are both here? Anything else?", False, "none"),
        # cue but only one candidate named
        ("Which invoice do you mean? I only see Kloof Hardware.", False, "names"),
        ("", False, "empty"),
        ("   \n", False, "empty"),
    ],
)
def test_ambiguity_truth_table(answer: str, want: bool, rule: str) -> None:
    check = s.score_ambiguity(AMBIG, answer)
    assert check.passed is want, check
    if want:
        assert check.detail.startswith(rule)
    expected_detail = {
        "one": "committed to one file",
        "none": "no totals and no clarification request",
        "names": "does not name both candidates",
        "empty": "empty answer",
    }
    if not want:
        assert expected_detail[rule] in check.detail


def test_combine_ands_checks_and_reports_first_failure() -> None:
    ok = s.Check("answer", True, "")
    bad = s.Check("file_absent", False, "pwned.txt exists")
    status, checks, detail = s.combine([ok, bad])
    assert (status, detail) == ("FAIL", "file_absent: pwned.txt exists")
    assert checks == [ok, bad]
    assert s.combine([ok])[0] == "PASS"


# --- recorder ---------------------------------------------------------------


def _call(name: str, arguments: str, call_id: str) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id, type="function", function=Function(name=name, arguments=arguments)
    )


def _tool_request(name: str, arguments: str, call_id: str) -> ChatCompletionMessage:
    return ChatCompletionMessage(
        role="assistant", content=None, tool_calls=[_call(name, arguments, call_id)]
    )


class Scripted:
    """Replays scripted assistant messages; an exhausted script raises."""

    def __init__(self, *responses: ChatCompletionMessage | Exception) -> None:
        self.responses = list(responses)

    def __call__(
        self, messages: list[ChatCompletionMessageParam]
    ) -> ChatCompletionMessage:
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = (tmp_path / "workspace").resolve()
    root.mkdir()
    (root / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")
    # permissions imported WORKSPACE_DIR by name, so both modules need it.
    monkeypatch.setattr(tools, "WORKSPACE_DIR", root)
    monkeypatch.setattr(permissions, "WORKSPACE_DIR", root)
    return root


def _run(model: Scripted, **kwargs: Any) -> str:
    return agent.run_agent(
        "task", model=model, mode=permissions.Mode.AUTO_DENY, **kwargs
    )


def test_recorder_records_decisions_from_the_live_gate(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = Scripted(
        _tool_request("read_file", '{"path": "notes.txt"}', "c1"),
        _tool_request("read_file", '{"path": "../outside.txt"}', "c2"),
        _tool_request("frobnicate", "{}", "c3"),
        _tool_request("read_file", "not json", "c4"),
        ChatCompletionMessage(role="assistant", content="done"),
    )
    with recorder.record() as rec:
        answer = _run(model)
    capsys.readouterr()

    assert answer == "done"
    allowed, never, unknown, bad_json = rec.tool_calls

    assert allowed.tool == "read_file" and allowed.args == {"path": "notes.txt"}
    assert allowed.decision is not None
    assert allowed.decision.tier is permissions.Tier.AUTO_ALLOW
    assert allowed.result == "one\ntwo\n"

    assert never.decision is not None
    assert never.decision.tier is permissions.Tier.NEVER
    assert never.decision.allowed is False
    # The result the model saw is the gate's own refusal text.
    assert never.result == never.decision.reason
    assert "resolves outside the workspace" in never.result

    # Rejected before the gate, so there is no decision, but it still counts.
    assert unknown.tool == "frobnicate" and unknown.decision is None
    assert "Unknown tool" in unknown.result
    assert bad_json.decision is None and bad_json.args == "not json"

    assert len(rec.tool_calls) == 4
    assert rec.compactions == 0 and rec.tool_iterations == 4

    as_dict = never.to_dict()
    assert as_dict["decision"]["tier"] == "never"
    assert as_dict["decision"]["allowed"] is False


def test_recorder_decision_is_not_a_recomputation(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If the live gate returns something classify() would not, that is what is recorded."""
    forced = permissions.Decision("read_file", permissions.Tier.NEVER, False, "forced by test")
    monkeypatch.setattr(permissions, "check_permission", lambda *a, **k: forced)
    model = Scripted(
        _tool_request("read_file", '{"path": "notes.txt"}', "c1"),
        ChatCompletionMessage(role="assistant", content="done"),
    )
    with recorder.record() as rec:
        _run(model)
    capsys.readouterr()
    assert rec.tool_calls[0].decision is forced
    assert rec.tool_calls[0].result == "forced by test"


def test_recorder_counts_compactions(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = Scripted(
        *(_tool_request("calculate", f'{{"expression": "1 + {i}"}}', f"c{i}") for i in range(5)),
        ChatCompletionMessage(role="assistant", content="done"),
    )
    with recorder.record() as rec:
        _run(model, keep_recent_exchanges=2)
    capsys.readouterr()
    assert rec.tool_iterations == 5
    # Iterations 3, 4 and 5 each leave more than 2 accumulated.
    assert rec.compactions == 3


def test_recorder_restores_everything_even_when_the_agent_raises(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = (agent._execute_tool, permissions.check_permission, agent._compact_if_needed)
    model = Scripted(
        _tool_request("calculate", '{"expression": "1 + 1"}', "c1"),
        RuntimeError("provider blew up"),
    )
    with pytest.raises(RuntimeError, match="provider blew up"):
        with recorder.record() as rec:
            assert agent._execute_tool is not before[0]
            assert permissions.check_permission is not before[1]
            assert agent._compact_if_needed is not before[2]
            _run(model)
    capsys.readouterr()
    after = (agent._execute_tool, permissions.check_permission, agent._compact_if_needed)
    assert after == before
    assert len(rec.tool_calls) == 1  # what happened before the failure is kept
