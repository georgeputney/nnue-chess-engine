"""Evaluation weights, read by agent.evaluate.

Currently the PeSTO piece-square set (public domain, already Texel-tuned by its authors)
with the scalar terms at their pre-tuning values, so the eval matches the hand-built
version it replaced. tools/tune.py writes this file; a real tuning pass would overwrite
the tables and the scalars together.
"""

import chess

# fmt: off
# PeSTO piece-square tables, midgame and endgame, pawn..king. published a8-first; _a1()
# below flips to python-chess order (a1 = 0). placement only - material is added in where
# MIDGAME_TABLE / ENDGAME_TABLE are built.
# ref: https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function

# base piece value per phase, pawn..king, indexed [piece_type - 1]. king is 0: both sides
# always have one.
MIDGAME_VALUE = [82, 337, 365, 477, 1025, 0]
ENDGAME_VALUE = [94, 281, 297, 512,  936, 0]

MIDGAME_PAWN = [
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
]

ENDGAME_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]

MIDGAME_KNIGHT = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
]

ENDGAME_KNIGHT = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]

MIDGAME_BISHOP = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]

ENDGAME_BISHOP = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
]

MIDGAME_ROOK = [
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
]

ENDGAME_ROOK = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
]

MIDGAME_QUEEN = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
]

ENDGAME_QUEEN = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]

MIDGAME_KING = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]

ENDGAME_KING = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]
# fmt: on


# PeSTO tables are published a8-first; return them in a1-first (python-chess) order.
def _a1(table: list[int]) -> list[int]:
    return [table[sq ^ 56] for sq in range(64)]


_MIDGAME_RAW = [MIDGAME_PAWN, MIDGAME_KNIGHT, MIDGAME_BISHOP,
                MIDGAME_ROOK, MIDGAME_QUEEN, MIDGAME_KING]
_ENDGAME_RAW = [ENDGAME_PAWN, ENDGAME_KNIGHT, ENDGAME_BISHOP,
                ENDGAME_ROOK, ENDGAME_QUEEN, ENDGAME_KING]

# material only, one flat number per virtual piece type (0 = far pawn, 1..6 = pawn..king).
# king is 0: it is never captured, both sides always have exactly one, so its "material"
# never contributes a white-minus-black difference. far pawn is seeded from pawn.
MATERIAL_MG: dict[int, int] = {pt: MIDGAME_VALUE[pt - 1] for pt in range(1, 7)}
MATERIAL_MG[0] = MATERIAL_MG[1]
MATERIAL_EG: dict[int, int] = {pt: ENDGAME_VALUE[pt - 1] for pt in range(1, 7)}
MATERIAL_EG[0] = MATERIAL_EG[1]

# [piece_type][square] -> placement only, white's view, a1-first. key 0 is FAR_PAWN, agent.py's
# virtual piece type for a pawn on the far side of the board from its own king; seeded as a
# copy of PAWN's row (1) so it starts identical until tools/tune.py fits it separately.
MIDGAME_PST: dict[int, list[int]] = {pt: _a1(_MIDGAME_RAW[pt - 1]) for pt in range(1, 7)}
MIDGAME_PST[0] = list(MIDGAME_PST[1])
ENDGAME_PST: dict[int, list[int]] = {pt: _a1(_ENDGAME_RAW[pt - 1]) for pt in range(1, 7)}
ENDGAME_PST[0] = list(ENDGAME_PST[1])

# scalar terms, split midgame / endgame. these values reproduce the hand-built eval: mobility
# flat across phases, king exposure midgame only, tempo flat.
MOBILITY_WEIGHT_MG: dict[chess.PieceType, int] = {chess.BISHOP: 4, chess.ROOK: 2, chess.QUEEN: 1}
MOBILITY_WEIGHT_EG: dict[chess.PieceType, int] = {chess.BISHOP: 4, chess.ROOK: 2, chess.QUEEN: 1}

KING_EXPOSURE_MG = -2
KING_EXPOSURE_EG = 0
TEMPO_MG = 10
TEMPO_EG = 10

# per virtual piece type (0 = far pawn, 1..6 = pawn..king): weight per friendly pawn ahead of
# this piece on its file. only the two pawn rows start non-zero - the old flat doubled-pawn
# penalty, -12 either phase; tools/tune.py is free to find a value for the rest.
PAWN_AHEAD_MG: dict[int, int] = {0: -12, 1: -12, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
PAWN_AHEAD_EG: dict[int, int] = {0: -12, 1: -12, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

# passed pawn: no enemy pawn on its own file or either adjacent file, on any rank ahead of it.
# bonus by how far it has advanced toward promotion (relative rank, 1..6 - a pawn is never on
# its own back rank or the promotion rank, so slots 0 and 7 never apply). The endgame PST
# already ramps steeply for an advanced pawn, so these seeds are only the *marginal* worth of
# it being passed rather than merely far up the board; a tuning pass that fits the two together
# would set them properly. ref: https://www.chessprogramming.org/Passed_Pawn
PASSED_PAWN_MG = [0, 1, 3, 7, 13, 24, 40, 0]
PASSED_PAWN_EG = [0, 6, 10, 17, 30, 52, 82, 0]

# king activity around an advanced passed pawn (relative rank >= 4 only): Chebyshev distance
# from each king to the square directly in front of the pawn, scored in the endgame only. Own
# king close in is worth having (negative weight); the enemy king kept away is too (positive).
# ref: https://www.chessprogramming.org/King_Activity
KING_PASSER_OWN_EG = -5
KING_PASSER_ENEMY_EG = 5
