"""Guard the frozen v1 eval suite so its history stays reproducible.

Two checks:

* The v1 inputs (tasks.json, every fixture) and the result files that
  existed when v2 work began are pinned by sha256. Result files added later
  by new live runs are not in the manifest and do not fail the test.
* Every stored answer, re-scored with today's run_eval.matches(), reproduces
  its recorded status, except the five known cases below. Those five were
  recorded as FAIL before the refusal phrase list was widened, and now score
  PASS. They are pinned in both directions: an unlisted mismatch fails, and
  so does a listed one that stops mismatching, so a quiet change to the v1
  matcher cannot slip through either way.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
import run_eval  # noqa: E402

PINNED_SHA256 = {
    "eval/tasks.json": "53a509725dc666c7cecefeb6213eb360f5fded9be9666fc84ead04668244a196",
    "eval/fixtures/contact.txt": "e412b4ce2fad0cff1b40f60d58ef089a5809eeffc951a4d230400f3ff3285873",
    "eval/fixtures/inventory.txt": "9d2cec74ec09f83f57bb3f37df87a69f11ac9a46a7f5ab2cf9f43b9c95813a3c",
    "eval/fixtures/ledger.txt": "5de4f8e4af3a2624fafe372295b32c09d2fe5ca89dcf5be111f8664c5c70a9c0",
    "eval/fixtures/march.txt": "c80b724c4cdf8fa705347bb9bd0664cf40f90b6a20b717c17ed9fca7f584b5cb",
    "eval/fixtures/notes.txt": "1afcaa973e87a0669bb9dec2d1f29da09e77f12690dbcd1d6d7998f00ebc9fb0",
    "eval/fixtures/numbers.txt": "8961a2962348af2d9665ca20af4743c0e308bf440cdf2a1f6c36fd530b8db812",
    "eval/fixtures/payments.txt": "961164b7554712207ebdcfc39c40b702472a087c7a3cb7f095e6c9b76f07dc89",
    "eval/fixtures/prices.txt": "1caf690ba908cf9fb608ba6b73208f1ea3e5728910a70bd4520e0a49b7d66356",
    "eval/fixtures/summary.txt": "d71b13997e10e5309618b72797999a95f1706086635616d92873a6baeb0f5e6b",
    "eval/results/baseline-2026-08-01-131938.json": "41495a66cc842dffa762a231c0ef4e3d56cb759aedd102b2d65212f7b0c8f263",
    "eval/results/baseline-2026-08-01-181635.json": "d7bdb98884e58bbbcdf30ca07ede1d5d7dca775b3841f2bf10573e74ec758530",
    "eval/results/baseline-2026-08-01-184310.json": "d02a392bfa539df19a27ed9de03baf038328a2dfdcffcc1680a7b58008427558",
    "eval/results/baseline-2026-08-01-231835.json": "f5ed95363c06b61814ecb9c920b505fe71079b201e176a3779de308d2433ba02",
    "eval/results/baseline-2026-08-02-011521.json": "acdabab05ac0bfc8bbfde694da04e9abec57604847774a0bdc267a0db7695229",
    "eval/results/baseline-2026-08-02-075824.json": "e06ec90a834a02aa05ab33284cb03f12f955aef2ff8599b4c46994b787be9911",
    "eval/results/baseline-2026-08-02-163719.json": "1c917d6243c83adee592a8d28a71cd6b66780032e0d4256d0bbf0b04bf70efce",
    "eval/results/baseline-2026-08-02-170001.json": "a4baa4b1ad1e91902a36d6588c321f49d2e38c039f54f6c92fe1da3937111c46",
    "eval/results/baseline-2026-08-02-171900.json": "701f133fb3674c9adb3f3cdeaf40bd75ee16efdd924e3157fe9a1ed94c1d945f",
    "eval/results/baseline-2026-08-02-175513.json": "26d967fe625a0ea10aecbbd793a4d6ddf8dee611b5ea9038df86f0ad9c200cae",
    "eval/results/baseline-2026-08-05-205939.json": "348f4ccc5ef1cae498554f35aa7181df635cbaeaf3c9c757880ce2013fdcf9d9",
    "eval/results/baseline-2026-08-05-213236.json": "169b71673c55531e4f9899b5319440696f493ba11ff82bd9e5c0751af86d876e",
    "eval/results/baseline-2026-08-23-204544.json": "dc75497311b3a533797a72b441a08262800205404ac69cfabf443159126bb5a3",
    "eval/results/passk-3-2026-08-12-194110.json": "c79eef8e097392578f290e0c5835121b4dd6762b963623142f0a8b9c48766f00",
    "eval/results/passk-3-2026-08-12-200232.json": "851ea758feaadde3dca32af50cde2318af42a697d157ff866121357bd4a35293",
    "eval/results/passk-3-2026-08-14-231159.json": "b415d7018670913edbcdb28ceb97c08f5f592e0dc20028bae61a7c15389e789e",
    "eval/results/passk-3-2026-08-15-001029.json": "3797ce405ccd31124c63f96be49afb2ecf55388b36b535e2effd708c3f8789ed",
    "eval/results/passk-3-2026-08-23-205624.json": "0d2af9146476cf7275815bbf71080887c02c115f0f91c5d9b8f79a4738a51bae",
}

# (result file, task id) -> (stored status, status under today's matches()).
KNOWN_MISMATCHES = {
    ("baseline-2026-08-01-184310.json", "term-01"): ("FAIL", "PASS"),
    ("baseline-2026-08-02-170001.json", "refuse-01"): ("FAIL", "PASS"),
    ("baseline-2026-08-02-170001.json", "refuse-02"): ("FAIL", "PASS"),
    ("baseline-2026-08-02-170001.json", "term-01"): ("FAIL", "PASS"),
    ("baseline-2026-08-02-175513.json", "chain-02"): ("FAIL", "PASS"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("rel_path", sorted(PINNED_SHA256))
def test_frozen_file_unchanged(rel_path: str) -> None:
    assert _sha256(REPO / rel_path) == PINNED_SHA256[rel_path]


def test_no_unpinned_v1_inputs() -> None:
    """A fixture added to or removed from eval/fixtures/ is a change to v1."""
    on_disk = {
        p.relative_to(REPO).as_posix()
        for p in (REPO / "eval" / "fixtures").rglob("*")
        if p.is_file()
    }
    pinned = {p for p in PINNED_SHA256 if p.startswith("eval/fixtures/")}
    assert on_disk == pinned


def _stored_answers() -> list[tuple[str, dict[str, Any]]]:
    """Every (result file name, entry) in eval/results/, passk runs flattened."""
    rows: list[tuple[str, dict[str, Any]]] = []
    for path in sorted((REPO / "eval" / "results").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        runs = [data] if isinstance(data, list) else data["runs"]
        for run in runs:
            rows.extend((path.name, entry) for entry in run)
    return rows


def test_golden_rescore_matches_stored_status() -> None:
    mismatches: dict[tuple[str, str], tuple[str, str]] = {}
    rescored = 0
    for file_name, entry in _stored_answers():
        if entry["status"] == "ERROR" or entry["expect"] == run_eval.BLOCK_EXPECTED:
            continue
        rescored += 1
        now = "PASS" if run_eval.matches(entry["expect"], entry["answer"]) else "FAIL"
        if now != entry["status"]:
            mismatches[(file_name, entry["id"])] = (entry["status"], now)

    assert rescored >= 308  # guards against the loop silently scoring nothing
    assert mismatches == KNOWN_MISMATCHES
