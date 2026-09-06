"""Train the NNUE evaluation on a labelled .npz (tools/ingest_lichess.py / tools/label.py
schema: packed bitboards, side-to-move flag, Stockfish cp, win-probability wdl).

Stockfish is a labeller only - nothing it produced ships in the zip. The net trained here is
ours; tools/export_nn.py quantises it into nnue/net.npz, which is what the engine runs.

    uv run python tools/train_nn.py --npz data/lichess.npz --out nnue/model.pt
    uv run python tools/train_nn.py --npz data/lichess.npz --limit 400000 --epochs 3   # smoke

Device defaults to MPS on Apple Silicon, else CPU. The exported net is identical either way;
the platform runs it on CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nnue.arch import CP_SCALE, FEATURES  # noqa: E402
from nnue.model import NNUE  # noqa: E402
from nnue.net import PERM  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Batches:
    """Yields (own_plane, other_plane, wdl, cp) tensors on `device`. The packed bitboards stay
    a uint8 CPU tensor; per batch they are unpacked to a white-perspective plane, the black
    perspective is the fixed column permutation of it, and side-to-move picks which is 'own'."""

    def __init__(
        self, packed: np.ndarray, stm: np.ndarray, wdl: np.ndarray, cp: np.ndarray,
        index: np.ndarray, batch: int, device: torch.device, shuffle: bool,
    ) -> None:
        self.packed = torch.from_numpy(packed)          # [N, 96] uint8, CPU
        self.stm = torch.from_numpy(stm.astype(np.bool_))
        self.wdl = torch.from_numpy(wdl.astype(np.float32))
        self.cp = torch.from_numpy(cp.astype(np.float32))
        self.index = index
        self.batch = batch
        self.device = device
        self.shuffle = shuffle
        self.perm = torch.from_numpy(PERM).to(device)   # [768] long
        self.bit = (1 << torch.arange(8, dtype=torch.uint8)).to(device)  # little-endian unpack

    def __len__(self) -> int:
        return (len(self.index) + self.batch - 1) // self.batch

    def __iter__(self):
        order = self.index.copy()
        if self.shuffle:
            np.random.shuffle(order)
        for start in range(0, len(order), self.batch):
            rows = torch.from_numpy(order[start : start + self.batch]).long()
            packed = self.packed[rows].to(self.device)                 # [B, 96] uint8
            bits = (packed.unsqueeze(-1) & self.bit).ne(0).to(torch.float32)  # [B, 96, 8]
            white = bits.reshape(packed.shape[0], -1)[:, :FEATURES]    # [B, 768]
            black = white[:, self.perm]
            white_to_move = self.stm[rows].to(self.device).unsqueeze(1)
            own = torch.where(white_to_move, white, black)
            other = torch.where(white_to_move, black, white)
            yield (
                own, other,
                self.wdl[rows].to(self.device),
                self.cp[rows].to(self.device),
            )


def sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def evaluate_split(model: NNUE, batches: Batches) -> tuple[float, float]:
    """(wdl MSE, cp RMSE) over a split."""
    model.eval()
    wdl_se = 0.0
    cp_se = 0.0
    n = 0
    with torch.no_grad():
        for own, other, wdl, cp in batches:
            pred = model(own, other)
            wdl_se += torch.sum((sigmoid(pred) - wdl) ** 2).item()
            cp_se += torch.sum((pred * CP_SCALE - cp) ** 2).item()
            n += own.shape[0]
    return wdl_se / n, (cp_se / n) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=ROOT / "data" / "lichess.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "nnue" / "model.pt")
    parser.add_argument("--device", default="auto", help="auto | mps | cpu | cuda")
    parser.add_argument("--ft-out", type=int, default=256, help="accumulator width per colour")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="cap positions (0 = all), for smoke")
    parser.add_argument("--patience", type=int, default=6, help="stop after N flat val epochs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    print(f"device {device}")

    blob = np.load(args.npz)
    packed, stm, cp, wdl = blob["packed"], blob["stm"], blob["cp"], blob["wdl"]
    count = len(cp) if args.limit <= 0 else min(args.limit, len(cp))
    packed, stm, cp, wdl = packed[:count], stm[:count], cp[:count], wdl[:count]
    print(f"{count:,} positions from {args.npz.name}")

    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(count)
    val_n = int(args.val_frac * count)
    val_index, train_index = shuffled[:val_n], shuffled[val_n:]
    print(f"{len(train_index):,} train / {len(val_index):,} val")

    train_batches = Batches(packed, stm, wdl, cp, train_index, args.batch, device, shuffle=True)
    val_batches = Batches(packed, stm, wdl, cp, val_index, args.batch, device, shuffle=False)

    model = NNUE(ft_out=args.ft_out).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        seen = 0
        for own, other, wdl_batch, _cp_batch in train_batches:
            optimiser.zero_grad(set_to_none=True)
            pred = model(own, other)
            loss = torch.mean((sigmoid(pred) - wdl_batch) ** 2)
            loss.backward()
            optimiser.step()
            running += loss.item() * own.shape[0]
            seen += own.shape[0]
        schedule.step()

        val_mse, val_rmse_cp = evaluate_split(model, val_batches)
        improved = val_mse < best_val * (1.0 - 1e-4)
        if improved:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        print(
            f"epoch {epoch:3d}  train {running / seen:.5f}  "
            f"val {val_mse:.5f}  ({val_rmse_cp:6.1f} cp rmse)  "
            f"{time.time() - started:5.1f}s{'  *' if improved else ''}"
        )
        if args.patience and stale >= args.patience:
            print(f"early stop: no val gain for {args.patience} epochs")
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "val_mse": best_val,
            "ft_out": args.ft_out,
            "arch": f"nnue-v1-768-{args.ft_out}-32-32",
        },
        args.out,
    )
    print(f"wrote {args.out}  (ft_out {args.ft_out}, best val mse {best_val:.5f})")


if __name__ == "__main__":
    main()
