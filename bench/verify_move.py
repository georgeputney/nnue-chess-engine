"""Round-trip check for move.py's packed-int move encoding: every field that goes in via
encode_move must come back out exactly the same way through the move_* accessors.
"""

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.move import (  # noqa: E402
    encode_move,
    move_from_square,
    move_is_capture,
    move_is_castle,
    move_is_double_push,
    move_is_en_passant,
    move_piece,
    move_promotion,
    move_to_square,
)


def main() -> None:
    checked = 0
    mismatches = 0

    squares = (0, 1, 6, 27, 62, 63)
    pieces = range(6)
    promotions = (None, 1, 2, 3, 4)
    flags = list(itertools.product((False, True), repeat=4))

    for frm, to, piece, promo, (cap, ep, castle, dbl) in itertools.product(
        squares, squares, pieces, promotions, flags
    ):
        checked += 1
        m = encode_move(frm, to, piece, promo, cap, ep, castle, dbl)
        got = (
            move_from_square(m), move_to_square(m), move_piece(m), move_promotion(m),
            move_is_capture(m), move_is_en_passant(m), move_is_castle(m), move_is_double_push(m),
        )
        want = (frm, to, piece, promo, cap, ep, castle, dbl)
        if got != want:
            mismatches += 1
            print(f"MISMATCH encoded={m:#08x} got={got} want={want}")

    print(f"checked {checked}, {mismatches} mismatches")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
