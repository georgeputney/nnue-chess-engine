"""Checks the numba-jitted sliding attack functions in attacks.py against the plain-Python
reference implementations they replaced, across every square and a large sample of occupancy
patterns (not just a few hand-picked ones - occupancy is what makes a sliding attack differ from
its empty-board mask, so it's the part most worth stress-testing).
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import attacks as attacks  # noqa: E402  (path setup above must run first)

SAMPLES_PER_SQUARE = 300


def main() -> None:
    random.seed(7)
    mismatches = 0
    checked = 0

    for sq in range(64):
        occupancies = [0, (1 << 64) - 1]
        occupancies += [random.getrandbits(64) for _ in range(SAMPLES_PER_SQUARE)]

        for occ in occupancies:
            checked += 1
            got = attacks.bishop_attacks(sq, occ)
            want = attacks._bishop_attacks_reference(sq, occ)
            if got != want:
                mismatches += 1
                print(f"BISHOP MISMATCH sq={sq} occ={occ:#x} got={got:#x} want={want:#x}")

            checked += 1
            got = attacks.rook_attacks(sq, occ)
            want = attacks._rook_attacks_reference(sq, occ)
            if got != want:
                mismatches += 1
                print(f"ROOK MISMATCH sq={sq} occ={occ:#x} got={got:#x} want={want:#x}")

    print(f"checked {checked}, {mismatches} mismatches")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
