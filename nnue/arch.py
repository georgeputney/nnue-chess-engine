"""Shared, dependency-free description of the NNUE architecture.

Both the trainer (nnue/model.py, torch) and the runtime (nnue/net.py, numpy, and the numba
agent that loads it) import their shape from here so the three can never drift apart. Nothing
in this module imports torch or numba - it is plain constants and one pure-Python index helper.

Feature set: dual-perspective, piece placement only (no king bucketing). One feature is a
triple (relative_colour, piece_type, square):

  - relative_colour is 0 for the perspective side's own pieces, 1 for the other side's
  - piece_type is 0..5 (pawn..king), the bitboard-layer numbering
  - square is a1=0..h8=63 from white's view, mirrored (square ^ 56) for black's perspective

so a perspective sees FEATURES = 2 * 6 * 64 = 768 inputs. The accumulator holds one
FT_OUT-wide vector per colour; evaluate() concatenates [own, other] into the 2*FT_OUT tail.
"""

# one perspective's input width and the transformer's output width
COLOURS = 2
PIECE_TYPES = 6
SQUARES = 64
FEATURES = COLOURS * PIECE_TYPES * SQUARES  # 768

FT_OUT = 256          # accumulator width per colour
L1_OUT = 32           # first tail layer, fed the 2 * FT_OUT concatenation
L2_OUT = 32           # second tail layer
L1_IN = 2 * FT_OUT    # 512

# the net regresses an internal score on a "pawns / 4" scale; runtime multiplies by this to
# get centipawns, training compares sigmoid(pred) against the win-probability label. keep the
# two in step: SCALE == 1 / K in tools/label.py (K = 1/400).
CP_SCALE = 400

WHITE = 0
BLACK = 1


# flat feature index for a piece of `piece_colour` (0/1) of `piece_type` on `square`, seen from
# `perspective` (0 = white to move's own view, 1 = black's). Mirrors nnue/net.py's vectorised
# permutation and the numba refresh loop; all three must agree.
def feature_index(perspective: int, piece_colour: int, piece_type: int, square: int) -> int:
    relative_colour = 0 if piece_colour == perspective else 1
    relative_square = square if perspective == WHITE else square ^ 56
    return relative_colour * (PIECE_TYPES * SQUARES) + piece_type * SQUARES + relative_square


# the length-768 column permutation that turns a white-perspective plane into a black-perspective
# one: black_plane[:, i] = white_plane[:, BLACK_PERSPECTIVE_PERM[i]]. Built from feature_index so
# it cannot disagree with it.
def black_perspective_perm() -> list[int]:
    perm = [0] * FEATURES
    for piece_colour in range(COLOURS):
        for piece_type in range(PIECE_TYPES):
            for square in range(SQUARES):
                white_i = feature_index(WHITE, piece_colour, piece_type, square)
                black_i = feature_index(BLACK, piece_colour, piece_type, square)
                perm[black_i] = white_i
    return perm
