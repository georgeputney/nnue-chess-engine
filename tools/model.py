"""The NNUE network as a torch module - training only. nnue/net.py is the numpy twin the
engine actually runs; tools/export_nn.py quantises the weights this trains into the .npz that
one loads, and checks the two agree.

Shape (see nnue/arch.py for the feature set):

    per-perspective input   768
    feature transformer     768 -> 256   shared across both perspectives
    accumulator             [own 256, other 256], clipped to [0, 1]
    tail                     512 -> 32 -> 32 -> {OUTPUT_BUCKETS}, clipped ReLU between

The last layer produces one score per output bucket; forward() gathers the column the caller's
piece count selects (nnue.arch.output_bucket). The forward here takes the two perspective planes
already assembled (own-to-move first) plus that bucket index; the trainer builds all three from
the packed bitboards. Output is a raw score on a pawns/4 scale - the loss sigmoids it before
comparing to the win-probability label, matching tools/label.py's K.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nnue.arch import FEATURES, FT_OUT, L1_OUT, L2_OUT, OUTPUT_BUCKETS


# clipped ReLU: the activation NNUE quantisation assumes everywhere - a plain ReLU's unbounded
# top has no fixed-point range. Training in float with the same clamp keeps the quantised net
# faithful.
def clipped_relu(x: Tensor) -> Tensor:
    return torch.clamp(x, 0.0, 1.0)


class NNUE(nn.Module):
    # ft_out is the accumulator width per colour. The default matches nnue/arch.py; a narrower
    # net (e.g. 128) roughly halves the per-leaf tail cost in the engine. The runtime derives
    # the width from the exported net.npz, so only the trainer needs to know it.
    def __init__(self, ft_out: int = FT_OUT) -> None:
        super().__init__()
        self.ft_out = ft_out
        self.transformer = nn.Linear(FEATURES, ft_out)
        self.l1 = nn.Linear(2 * ft_out, L1_OUT)
        self.l2 = nn.Linear(L2_OUT, L2_OUT)
        self.out = nn.Linear(L2_OUT, OUTPUT_BUCKETS)

    # own_plane / other_plane: (batch, 768) float 0/1 feature planes, side-to-move's own view
    # first. out_bucket: (batch,) long, the piece-count bucket for each position. Returns
    # (batch,) raw scores - the selected bucket's head only.
    def forward(self, own_plane: Tensor, other_plane: Tensor, out_bucket: Tensor) -> Tensor:
        own = clipped_relu(self.transformer(own_plane))
        other = clipped_relu(self.transformer(other_plane))
        x = torch.cat((own, other), dim=1)
        x = clipped_relu(self.l1(x))
        x = clipped_relu(self.l2(x))
        heads = self.out(x)  # (batch, OUTPUT_BUCKETS)
        return heads.gather(1, out_bucket.unsqueeze(1)).squeeze(1)
