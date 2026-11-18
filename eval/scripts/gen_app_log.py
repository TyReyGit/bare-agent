"""Regenerate eval/fixtures_v2/ops/app.log, the long-01 log, deterministically.

    python eval/scripts/gen_app_log.py eval/fixtures_v2/ops/app.log

Seeded (SEED = 7), so the output is byte-identical on every run; a test
pins that against the committed fixture. The log is about 2,000 lines of
a small service business's booking system. Seven billing.reconcile hop
records (HOPS) sit mid-file in shuffled order among decoy records of the
same shape, so the chain can only be followed one lookup at a time. The
incident total is the sum of the hop amounts (6056).
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

SEED = 7
TOTAL_LINES = 2000
# (case, next case, amount in rand), in chain order.
HOPS: list[tuple[str, str, int]] = [
    ("wren", "heron", 415),
    ("heron", "plover", 1280),
    ("plover", "kestrel", 96),
    ("kestrel", "finch", 2050),
    ("finch", "egret", 733),
    ("egret", "avocet", 318),
    ("avocet", "none", 1164),
]
_HOP_TOKENS = {case for case, _, _ in HOPS}
# Decoy case names. None contains a hop token, so grepping a hop token
# returns only its own record and the one pointing at it.
NOISE_BIRDS = [
    b
    for b in [
        "sparrow", "robin", "starling", "thrush", "lark", "swift", "martin",
        "oriole", "weaver", "bulbul", "shrike", "sunbird", "hoopoe", "ibis",
        "stork", "crane", "gannet", "petrel", "tern", "gull", "cormorant",
        "darter", "bittern", "coucal", "barbet", "tinker", "hornbill", "drongo",
        "flycatcher", "pipit", "wagtail", "warbler", "cisticola", "prinia",
        "apalis", "batis", "boubou", "tchagra", "widowbird", "bishop",
    ]
    if not any(t in b for t in _HOP_TOKENS)
]
CLIENTS = [
    "Harbourview Guesthouse", "Alder Lane Dental Practice", "Birch Street Bookshop",
    "Cedar Court Accountants", "Juniper Physiotherapy Rooms", "Maple Tree Veterinary Clinic",
    "Rowan Hall Community Centre", "Willow Bend Hair Studio", "Seabreeze Guest Lodge",
    "Meridian Physiotherapy",
]
ACTIONS = [
    "refund issued after duplicate card capture",
    "credit re-applied to the client account",
    "charge reversed by the card processor",
    "partial refund agreed with the client",
    "manual adjustment after the booking was moved",
    "fee waived as a goodwill gesture",
]
OTHER = [
    "INFO  scheduler.heartbeat  booking scheduler alive, queue depth={n}",
    "INFO  bookings.api  GET /bookings/upcoming 200 in {n}ms",
    "INFO  bookings.api  POST /bookings 201 crew assigned",
    "WARN  mailer.smtp  retrying reminder email to client, attempt {k}",
    "INFO  mailer.smtp  appointment reminder sent to client",
    "INFO  payments.gateway  card authorisation approved",
    "DEBUG payments.gateway  token refreshed for the merchant account",
    "INFO  crew.app  technician checked in on site",
    "INFO  crew.app  job sheet uploaded with photos={k}",
    "WARN  crew.app  location update delayed, signal weak",
]


def _record(t: str, case: str, client: str, action: str, nxt: str, amount: int) -> str:
    return (
        f'{t} INFO  billing.reconcile  case={case} client="{client}" '
        f'action="{action}" next={nxt} amount=R{amount:,}'
    )


def generate() -> str:
    """The full app.log text. Same seed, same bytes."""
    rng = random.Random(SEED)
    positions = sorted(rng.sample(range(650, 1350), len(HOPS)))
    rng.shuffle(positions)  # chain order is not file order
    hop_at = dict(zip(positions, HOPS))
    start = datetime(2026, 10, 14, 6, 0, 0)
    lines: list[str] = []
    for i in range(TOTAL_LINES):
        t = (start + timedelta(seconds=17 * i + rng.randint(0, 9))).strftime("%Y-%m-%dT%H:%M:%SZ")
        if i in hop_at:
            case, nxt, amount = hop_at[i]
            lines.append(_record(t, case, rng.choice(CLIENTS), rng.choice(ACTIONS), nxt, amount))
        elif rng.random() < 0.18:
            a, b = rng.sample(NOISE_BIRDS, 2)
            nxt = b if rng.random() < 0.8 else "none"
            lines.append(
                _record(t, a, rng.choice(CLIENTS), rng.choice(ACTIONS), nxt, rng.randint(40, 2400))
            )
        else:
            template = rng.choice(OTHER)
            lines.append(t + " " + template.format(n=rng.randint(1, 900), k=rng.randint(1, 5)))
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python eval/scripts/gen_app_log.py <output path>", file=sys.stderr)
        return 2
    Path(argv[1]).write_text(generate(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
