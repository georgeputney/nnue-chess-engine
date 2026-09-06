"""Quantise a trained nnue/model.pt into nnue/net.npz - the file the engine loads.

The feature transformer goes to int16 with a single shared scale, so the accumulator is int32
and its incremental column updates in make_move are exact. The tail (512 -> 32 -> 32 -> 1)
stays float32: it is ~17k MACs, off the hot path, and keeping it float removes any
torch-vs-numba quant-matching risk.

    uv run python tools/export_nn.py --model nnue/model.pt --out nnue/net.npz

Prints the quantisation error against the float model on a sample of real positions; a few cp
of RMSE is expected and harmless.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nnue import net as netmod  # noqa: E402
from nnue.arch import CP_SCALE  # noqa: E402
from nnue.model import NNUE  # noqa: E402

INT16_MAX = 32767


def quantise_transformer(
    weight: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """weight [ft_out, 768] float, bias [ft_out] float -> (int16 [768, ft_out] transposed,
    int32 [ft_out], scale). scale is picked so the largest weight lands near the int16 ceiling."""
    peak = float(np.max(np.abs(weight)))
    scale = peak / (INT16_MAX * 0.98) if peak > 0 else 1.0
    weight_int = np.clip(np.round(weight / scale), -INT16_MAX, INT16_MAX).astype(np.int16)
    bias_int = np.round(bias / scale).astype(np.int32)
    return weight_int.T.copy(), bias_int, scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "nnue" / "model.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "nnue" / "net.npz")
    parser.add_argument("--npz", type=Path, default=ROOT / "data" / "lichess.npz",
                        help="positions to measure quantisation error on")
    parser.add_argument("--sample", type=int, default=50000)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    ft_out = int(checkpoint.get("ft_out", 256))
    model = NNUE(ft_out=ft_out)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"loaded {args.model}  (ft_out {ft_out}, val mse "
          f"{checkpoint.get('val_mse', float('nan')):.5f})")

    ft_w = model.transformer.weight.detach().numpy()
    ft_b = model.transformer.bias.detach().numpy()
    ft_weight_t, ft_bias, ft_scale = quantise_transformer(ft_w, ft_b)
    print(f"transformer |w| peak {np.abs(ft_w).max():.4f}  ft_scale {ft_scale:.3e}  "
          f"int16 range used [{ft_weight_t.min()}, {ft_weight_t.max()}]")

    blob = {
        "ft_weight_t": ft_weight_t,
        "ft_bias": ft_bias,
        "ft_scale": np.float32(ft_scale),
        "l1_weight": model.l1.weight.detach().numpy().astype(np.float32),
        "l1_bias": model.l1.bias.detach().numpy().astype(np.float32),
        "l2_weight": model.l2.weight.detach().numpy().astype(np.float32),
        "l2_bias": model.l2.bias.detach().numpy().astype(np.float32),
        "out_weight": model.out.weight.detach().numpy().astype(np.float32),
        "out_bias": model.out.bias.detach().numpy().astype(np.float32),
        "cp_scale": np.float32(CP_SCALE),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **blob)
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out}  {size_kb:.0f} KB")

    # quantisation error: the int16-transformer numpy forward (what the engine runs) vs the
    # float torch model, on real positions
    source = np.load(args.npz)
    total = len(source["stm"])
    pick = np.random.default_rng(0).choice(total, min(args.sample, total), replace=False)
    packed = source["packed"][pick]
    stm = source["stm"][pick]

    weights = netmod.load(args.out)
    quant_cp = netmod.forward_packed(packed, stm, weights)

    own, other = netmod.perspective_planes(packed, stm)
    with torch.no_grad():
        float_cp = model(torch.from_numpy(own), torch.from_numpy(other)).numpy() * CP_SCALE

    diff = quant_cp - float_cp
    print(
        f"quant vs torch float on {len(pick):,} positions:  "
        f"rmse {np.sqrt(np.mean(diff ** 2)):.2f} cp  max |err| {np.max(np.abs(diff)):.2f} cp  "
        f"corr {np.corrcoef(quant_cp, float_cp)[0, 1]:.5f}"
    )
    label_cp = source["cp"][pick].astype(np.float32)
    print(
        f"quant vs Stockfish label:  rmse {np.sqrt(np.mean((quant_cp - label_cp) ** 2)):.1f} cp"
        f"  corr {np.corrcoef(quant_cp, label_cp)[0, 1]:.4f}"
    )


if __name__ == "__main__":
    main()
