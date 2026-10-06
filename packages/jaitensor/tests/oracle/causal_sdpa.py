#!/usr/bin/env python3
"""Record what PyTorch's causal SDPA produces, for test_attention.jai to replay.

Inputs are rebuilt on both sides from the same formula, `sin(i * 0.731 + phase)`
in float64 rounded to float32, so nothing but the answers has to be stored.
Each case is packed `[T, heads * hd]` attention with `is_causal=True`, run in
float64 on the CPU, and the loss is `sum(y * w)`. A case records the loss and,
at 64 fixed positions, the output and the three input gradients.

    ~/.venvs/scratch/bin/python packages/jaitensor/tests/oracle/causal_sdpa.py
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent / "causal_sdpa.txt"
HEADS = 2
LENGTHS = [1, 63, 64, 65, 256, 1024]
HEAD_DIMS = [32, 64]
SAMPLES = 64


def wave(count: int, phase: float) -> torch.Tensor:
    values = [math.sin(float(i) * 0.731 + phase) for i in range(count)]
    return torch.tensor(values, dtype=torch.float32).to(torch.float64)


def positions(size: int) -> list[int]:
    return [(j * 7919 + 13) % size for j in range(SAMPLES)]


def case(seq: int, hd: int) -> list[str]:
    dim = HEADS * hd
    q = wave(seq * dim, 0.1).reshape(seq, dim).requires_grad_(True)
    k = wave(seq * dim, 0.9).reshape(seq, dim).requires_grad_(True)
    v = wave(seq * dim, 1.7).reshape(seq, dim).requires_grad_(True)
    w = wave(seq * dim, 2.5).reshape(seq, dim)

    def heads_first(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(seq, HEADS, hd).permute(1, 0, 2).unsqueeze(0)

    y = torch.nn.functional.scaled_dot_product_attention(
        heads_first(q), heads_first(k), heads_first(v), is_causal=True
    )
    y = y.squeeze(0).permute(1, 0, 2).reshape(seq, dim)
    loss = (y * w).sum()
    loss.backward()
    picks = positions(seq * dim)

    def sample(t: torch.Tensor) -> str:
        flat = t.detach().reshape(-1)
        return " ".join(repr(float(flat[i])) for i in picks)

    return [
        f"case {seq} {hd} {HEADS} {float(loss.detach())!r}",
        sample(y),
        sample(q.grad),
        sample(k.grad),
        sample(v.grad),
    ]


def main() -> None:
    lines: list[str] = []
    for hd in HEAD_DIMS:
        for seq in LENGTHS:
            lines.extend(case(seq, hd))
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines) // 5} cases to {OUT}")


if __name__ == "__main__":
    main()
