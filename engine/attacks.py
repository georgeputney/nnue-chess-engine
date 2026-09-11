"""Attack generation: which squares a piece on `square` could move to or capture on, given
what is occupied. Pawn / knight / king attacks never depend on occupancy, so they are
precomputed once per square at import. Sliding-piece attacks do, and they are the most
call-heavy thing in the engine, so they go through magic bitboards: one multiply-shift-index
into a table built at import (see the Magic bitboards section below), not a ray walk. The
classical outward-stepping walk is kept as ray_attacks_reference - build_attack_tables()
checks every table entry against it at import, and tools/verify_attacks.py re-checks the
compiled lookup, so the fast path can never silently disagree with the slow one.

This is the bitboard layer's stand-in for python-chess's attack tables (`chess.BB_DIAG_ATTACKS`
and friends, `chess.Board.attacks_mask`): bishop / rook / queen attacks are numba-jitted.
`ray_attacks_reference` / `bishop_attacks_reference` / `rook_attacks_reference` are the
plain-Python originals, kept as the reference the magics are verified against.
"""

import numba as nb
import numpy as np
from numba import njit

from engine.bitboard import (
    BLACK,
    FULL_BB,
    NOT_FILE_A,
    NOT_FILE_H,
    WHITE,
    bit,
    square_file,
    square_rank,
)

# each direction as (file_step, rank_step); first four are bishop-like, last four rook-like
BISHOP_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
ROOK_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
KNIGHT_STEPS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))


# The squares a pawn of `colour` on `square` attacks - the two forward diagonals, off-board
# files dropped. Built for every square, so the white table is masked back to 64 bits: a pawn
# on rank 8 never attacks from there, but bit 63 << 9 would overflow, and a numpy uint64 slot
# for a jitted caller cannot hold an out-of-range value.
def mask_pawn_attacks(colour: int, square: int) -> int:
    origin = bit(square)

    if colour == WHITE:
        return ((origin & NOT_FILE_A) << 7 | (origin & NOT_FILE_H) << 9) & FULL_BB
    
    return ((origin & NOT_FILE_H) >> 7) | ((origin & NOT_FILE_A) >> 9)


# The squares a knight on `square` reaches - each of the eight L-steps that stays on the board.
def mask_knight_attacks(square: int) -> int:
    attacks = 0
    file, rank = square_file(square), square_rank(square)

    for file_step, rank_step in KNIGHT_STEPS:
        next_file, next_rank = file + file_step, rank + rank_step

        if 0 <= next_file < 8 and 0 <= next_rank < 8:
            attacks |= bit(next_rank * 8 + next_file)

    return attacks


# The squares a king on `square` reaches - the eight neighbours that stay on the board.
def mask_king_attacks(square: int) -> int:
    attacks = 0
    file, rank = square_file(square), square_rank(square)

    for file_step in (-1, 0, 1):
        for rank_step in (-1, 0, 1):
            if file_step == 0 and rank_step == 0:
                continue

            next_file, next_rank = file + file_step, rank + rank_step

            if 0 <= next_file < 8 and 0 <= next_rank < 8:
                attacks |= bit(next_rank * 8 + next_file)

    return attacks


# Plain-Python sliding attacks: walk each direction from `square` one step at a time, adding
# every square passed and stopping on (but including) the first occupied one. bishop_attacks /
# rook_attacks below are the jitted twins verify_attacks.py checks against this.
def ray_attacks_reference(
    square: int, directions: tuple[tuple[int, int], ...], occupied: int
) -> int:
    
    attacks = 0
    start_file, start_rank = square_file(square), square_rank(square)

    for file_step, rank_step in directions:
        file, rank = start_file + file_step, start_rank + rank_step

        while 0 <= file < 8 and 0 <= rank < 8:
            target = rank * 8 + file
            attacks |= bit(target)

            if occupied & bit(target):
                break  # blocker: as far as the ray goes, but still a legal target

            file, rank = file + file_step, rank + rank_step
            
    return attacks


def bishop_attacks_reference(square: int, occupied: int) -> int:
    return ray_attacks_reference(square, BISHOP_DIRECTIONS, occupied)


def rook_attacks_reference(square: int, occupied: int) -> int:
    return ray_attacks_reference(square, ROOK_DIRECTIONS, occupied)


PAWN_ATTACKS = [[mask_pawn_attacks(colour, sq) for sq in range(64)] for colour in (WHITE, BLACK)]
KNIGHT_ATTACKS = [mask_knight_attacks(sq) for sq in range(64)]
KING_ATTACKS = [mask_king_attacks(sq) for sq in range(64)]

# numpy-array copies of the three tables above, for jitted callers - a plain Python list is not
# something numba's nopython mode can use as a captured global (it is untyped as far as numba is
# concerned), but a numpy array is. Filled one element at a time rather than via
# np.array(nested_list, dtype=np.uint64): numpy infers a signed intermediate type from a nested
# Python list first and only casts to the requested dtype after, and a bitboard with bit 63 set
# (any position with a piece on the a8-h8 rank) overflows that intermediate type.
PAWN_ATTACKS_NB = np.zeros((2, 64), dtype=np.uint64)
KNIGHT_ATTACKS_NB = np.zeros(64, dtype=np.uint64)
KING_ATTACKS_NB = np.zeros(64, dtype=np.uint64)
for colour in (WHITE, BLACK):
    for square in range(64):
        PAWN_ATTACKS_NB[colour, square] = np.uint64(PAWN_ATTACKS[colour][square])
for square in range(64):
    KNIGHT_ATTACKS_NB[square] = np.uint64(KNIGHT_ATTACKS[square])
    KING_ATTACKS_NB[square] = np.uint64(KING_ATTACKS[square])


# --- Magic bitboards -------------------------------------------------------------------------
# Classical ray-walking (kept above as the *_reference pair) costs one iteration per square the
# ray crosses, and it sits on the hottest path there is here - move generation, is_check and the
# mobility term in evaluate all lean on it. A magic bitboard replaces the walk with one
# multiply-shift-index: AND the occupancy down to just the squares that could block a slider
# from `square`, multiply by a constant chosen so every distinct blocker layout lands on a
# distinct index once shifted down, and read the attack set straight out of a table.
#
# The two magic arrays below were found offline by tools/gen_magics.py (a deterministic xorshift
# search - re-run it to reproduce them exactly). Searching is slow in Python and would eat the
# platform's init budget, so the constants are checked in; build_attack_tables() still verifies
# every one square by square against ray_attacks_reference at import, and tools/verify_attacks.py
# re-checks the compiled lookup, so a wrong constant cannot reach the search.
# ref: https://www.chessprogramming.org/Magic_Bitboards
MASK64 = (1 << 64) - 1

ROOK_MAGICS = (
    0x8280004002218050, 0x2100210080400011, 0x0200201008420080, 0x0200100804204200,
    0xC88004008800804E, 0x130004000813000A, 0x1080208002000100, 0x0100084021001082,
    0x4002800240008038, 0x0010802000804000, 0x2001001020004100, 0xD011001000A10188,
    0x0449000408011100, 0x0108012004104008, 0x1104001014014288, 0x000100010030408A,
    0x0040008002984160, 0x2060004000403002, 0x0010002008002402, 0x1801010008201002,
    0x2900808004000800, 0x0004008004020080, 0x0000040030812208, 0x200002002040810C,
    0x200080008020400B, 0x0060002040100040, 0x8CA8200080100880, 0x0000100080080080,
    0x4000100500080100, 0x0800040080800200, 0x0028100400010208, 0x2000088200110044,
    0x0044400028800080, 0x60C0804000802002, 0x4020088020801000, 0x1110000800808010,
    0x1000804402800800, 0x4008800400800200, 0x0200A2180C000190, 0x2001408052000104,
    0x0180002000414000, 0x0000500020044000, 0x9020200010008080, 0x0100081200420020,
    0xA080080004008080, 0x0100020004008080, 0x0000521001040028, 0x08A0008400420001,
    0x0400210040801100, 0x0281002098400100, 0x0000420010248200, 0x0000300108008180,
    0x0000080100455100, 0x020A001008040200, 0x8000525008210400, 0x0040040849008A00,
    0x0000210080004011, 0x0480400080210015, 0x28000A0010204082, 0x1202001040040822,
    0x00A1001028000423, 0x6101000802040001, 0x400A9001020800E4, 0x0590010C00408426,
)
BISHOP_MAGICS = (
    0x814050020A012120, 0x0004040404002848, 0x0210194600220000, 0x042404028400D480,
    0x0004030801040130, 0x5001042085009000, 0x00020104024000C0, 0x4218802088044000,
    0x8020886224044400, 0x0084200801410420, 0x1000082800448009, 0x8440041042000008,
    0x0000611040404110, 0x00000A0882080800, 0x0024810809242010, 0x0802142404424800,
    0x00040420580A8800, 0x0204011090008920, 0x0210028A0C084088, 0x4008008082004500,
    0x0000801404A03300, 0x042A014501009200, 0x0000428202022004, 0x8202050022010480,
    0x1002088012203808, 0x8034110844016800, 0x0988080001004100, 0x0004004004010003,
    0x4001001091004000, 0x0080450000900800, 0x84021C042A008211, 0x0000802002020228,
    0x0802201060200300, 0x080801B080080200, 0x2804004400182520, 0x4908020080880080,
    0x0262008400020202, 0x8070004080191000, 0x0A14040400006700, 0x8882004201004200,
    0x8401411010114010, 0x0000440220004800, 0x00800A0802001404, 0x00C0174202002020,
    0x0012084100410400, 0x1010200800410020, 0x0312020843020610, 0x8801410C01000880,
    0x804A012420040404, 0x0008422804420280, 0x1010104200900000, 0x020800004202100C,
    0x0400001002088520, 0x2200600811084100, 0x0240234404008009, 0x0249B00882810040,
    0x0001010050240400, 0x0804064120882000, 0x0300080516880400, 0x0000023000420202,
    0x0002220411820200, 0x021C103020718505, 0x4800403002224042, 0x0404440C820E0200,
)


# Every submask of `mask`, including 0 and `mask` itself - the Carry-Rippler walk over subsets.
def subsets(mask: int) -> list[int]:
    out = [0]
    sub = (0 - mask) & mask
    while sub:
        out.append(sub)
        sub = (sub - mask) & mask
    return out


# The squares that can block a slider leaving `square` along `directions`: the ray squares, but
# not the last one before each edge (a man there blocks nothing beyond it) and not `square`.
def relevant_mask(square: int, directions: tuple[tuple[int, int], ...]) -> int:
    file0, rank0 = square & 7, square >> 3
    mask = 0
    for file_step, rank_step in directions:
        file, rank = file0 + file_step, rank0 + rank_step
        while 0 <= file + file_step <= 7 and 0 <= rank + rank_step <= 7:
            mask |= 1 << (rank * 8 + file)
            file, rank = file + file_step, rank + rank_step
    return mask


# Turn the checked-in magics into what the jitted lookup needs: a flat attack table plus the
# per-square magic / mask / shift / base-offset arrays that index it. Every magic is verified
# here - if one ever failed to map two blocker layouts apart, this raises at import rather than
# letting a wrong attack set through.
def build_attack_tables(
    directions: tuple[tuple[int, int], ...], magics: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    magic = np.zeros(64, dtype=np.uint64)
    mask = np.zeros(64, dtype=np.uint64)
    shift = np.zeros(64, dtype=np.uint64)
    offset = np.zeros(64, dtype=np.int64)
    flat: list[int] = []

    for square in range(64):
        square_mask = relevant_mask(square, directions)
        bits = bin(square_mask).count("1")
        square_shift = 64 - bits
        table: list[int | None] = [None] * (1 << bits)

        for occ in subsets(square_mask):
            index = ((occ * magics[square]) & MASK64) >> square_shift
            attack = ray_attacks_reference(square, directions, occ)
            if table[index] is None or table[index] == attack:
                table[index] = attack
            else:
                raise RuntimeError(f"magic {magics[square]:#018x} collides on square {square}")

        magic[square] = np.uint64(magics[square])
        mask[square] = np.uint64(square_mask)
        shift[square] = np.uint64(square_shift)
        offset[square] = len(flat)
        flat.extend(0 if entry is None else entry for entry in table)

    flat_nb = np.zeros(len(flat), dtype=np.uint64)
    for i, entry in enumerate(flat):
        flat_nb[i] = np.uint64(entry)
    return magic, mask, shift, offset, flat_nb


ROOK_MAGIC, ROOK_MASK, ROOK_SHIFT, ROOK_OFFSET, ROOK_TABLE = build_attack_tables(
    ROOK_DIRECTIONS, ROOK_MAGICS
)
BISHOP_MAGIC, BISHOP_MASK, BISHOP_SHIFT, BISHOP_OFFSET, BISHOP_TABLE = build_attack_tables(
    BISHOP_DIRECTIONS, BISHOP_MAGICS
)


# The magic lookup, typed for nopython mode: `square` and `occupied` have to be explicitly
# unsigned (nb.uint64), or a bitboard with bit 63 set would read as negative and every shift on
# it would be wrong. The multiply is meant to overflow - numba's uint64 wraps like C's, which is
# exactly what the magic relies on.
# ref: https://numba.readthedocs.io/en/stable/reference/types.html
@njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def bishop_attacks(square: int, occupied: int) -> int:
    blockers = occupied & BISHOP_MASK[square]
    index = (blockers * BISHOP_MAGIC[square]) >> BISHOP_SHIFT[square]
    return int(BISHOP_TABLE[BISHOP_OFFSET[square] + np.int64(index)])


@njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def rook_attacks(square: int, occupied: int) -> int:
    blockers = occupied & ROOK_MASK[square]
    index = (blockers * ROOK_MAGIC[square]) >> ROOK_SHIFT[square]
    return int(ROOK_TABLE[ROOK_OFFSET[square] + np.int64(index)])


# A queen reaches everything a bishop and a rook on the same square would.
@njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def queen_attacks(square: int, occupied: int) -> int:
    return bishop_attacks(square, occupied) | rook_attacks(square, occupied)


# compile now, inside the init budget, not on the first real call
def warm_up() -> None:
    bishop_attacks(0, 0)
    rook_attacks(0, 0)
    queen_attacks(0, 0)


warm_up()
