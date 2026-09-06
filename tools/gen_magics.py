"""Search for the rook / bishop magic constants that attacks.py checks in.

Slow in pure Python (tens of seconds), which is why the result is a checked-in table rather
than something attacks.py rebuilds at import. The search is deterministic - a fixed xorshift
seed - so re-running this prints the exact arrays that are in attacks.py now. Change a seed
here and paste the new output over ROOK_MAGICS / BISHOP_MAGICS to rotate them.

    uv run python tools/gen_magics.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks import (  # noqa: E402
    BISHOP_DIRECTIONS,
    ROOK_DIRECTIONS,
    ray_attacks_reference,
    relevant_mask,
    subsets,
)

MASK64 = (1 << 64) - 1
ROOK_SEED = 0x00C0_FFEE_1234_5678
BISHOP_SEED = 0x0BAD_F00D_DEAD_BEEF


# xorshift64: a few sparse (three draws ANDed together) candidates per try is the standard way
# to land a magic quickly, and a fixed seed keeps the whole table reproducible.
class Xorshift64:
    def __init__(self, seed: int) -> None:
        self.state = seed

    def next(self) -> int:
        x = self.state
        x ^= (x << 13) & MASK64
        x ^= x >> 7
        x ^= (x << 17) & MASK64
        self.state = x
        return x

    def sparse(self) -> int:
        return self.next() & self.next() & self.next()


# The smallest constant that maps every blocker submask of `square`'s relevant mask onto its own
# slot (or one already holding the same attack set).
def find_magic(
    square: int, directions: tuple[tuple[int, int], ...], rng: Xorshift64
) -> int:
    mask = relevant_mask(square, directions)
    bits = bin(mask).count("1")
    shift = 64 - bits
    blockers = subsets(mask)
    attacks = [ray_attacks_reference(square, directions, occ) for occ in blockers]

    for _ in range(100_000_000):
        magic = rng.sparse()
        if bin((mask * magic) & 0xFF00_0000_0000_0000).count("1") < 6:
            continue

        table: list[int | None] = [None] * (1 << bits)
        for occ, attack in zip(blockers, attacks, strict=True):
            index = ((occ * magic) & MASK64) >> shift
            if table[index] is None:
                table[index] = attack
            elif table[index] != attack:
                break
        else:
            return magic

    raise RuntimeError(f"no magic found for square {square}")


def dump(name: str, directions: tuple[tuple[int, int], ...], seed: int) -> None:
    rng = Xorshift64(seed)
    magics = [find_magic(square, directions, rng) for square in range(64)]
    print(f"{name} = (")
    for row in range(0, 64, 4):
        print("    " + ", ".join(f"0x{m:016X}" for m in magics[row:row + 4]) + ",")
    print(")")


def main() -> None:
    dump("ROOK_MAGICS", ROOK_DIRECTIONS, ROOK_SEED)
    dump("BISHOP_MAGICS", BISHOP_DIRECTIONS, BISHOP_SEED)


if __name__ == "__main__":
    main()
