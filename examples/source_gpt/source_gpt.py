"""The torch MPS peer for examples/source_gpt: the same model, data and protocol.

    ~/.venvs/scratch/bin/python examples/source_gpt/source_gpt.py

Run from the repository root. Same shapes (4 layers, d256, 4 heads, T128,
batch 64, vocab 256), the same initialisation ranges, Adam with the same
learning rate and eps, the same training files and held-out windows, and the
loss downloaded every step (`loss.item()`), as train.jai does.

MODE=bench times the step (first step, WARM seconds of steps, then REPS x
STEPS); MODE=train runs STEPS (2000) steps with the held-out evaluation;
MODE=gen times cached batch-1 decoding (PROMPT, NEW, REPS, WARM).
"""
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = torch.device("mps")


def env_int(name, fallback):
    return int(os.environ.get(name, fallback))


L = env_int("GPT_LAYERS", 4)
D = env_int("GPT_DIM", 256)
H = env_int("GPT_HEADS", 4)
T = env_int("GPT_SEQ", 128)
B = env_int("GPT_BATCH", 64)
VOCAB = 256


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h
        s.ln1 = nn.LayerNorm(d)
        s.ln2 = nn.LayerNorm(d)
        s.wq = nn.Linear(d, d, bias=False)
        s.wk = nn.Linear(d, d, bias=False)
        s.wv = nn.Linear(d, d, bias=False)
        s.wo = nn.Linear(d, d, bias=False)
        s.fc1 = nn.Linear(d, 4 * d)
        s.fc2 = nn.Linear(4 * d, d, bias=False)

    def attn(s, x, cache=None, at=0):
        b, t, d = x.shape
        q, k, v = s.wq(x), s.wk(x), s.wv(x)
        if cache is not None:
            ck, cv = cache
            ck[:, at:at + t] = k
            cv[:, at:at + t] = v
            k, v = ck[:, :at + t], cv[:, :at + t]
        sp = lambda z: z.view(b, z.shape[1], s.h, d // s.h).transpose(1, 2)
        if cache is not None and t == 1:
            y = F.scaled_dot_product_attention(sp(q), sp(k), sp(v))
        else:
            y = F.scaled_dot_product_attention(sp(q), sp(k), sp(v), is_causal=True)
        return s.wo(y.transpose(1, 2).reshape(b, t, d))

    def forward(s, x, cache=None, at=0):
        x = x + s.attn(s.ln1(x), cache, at)
        return x + s.fc2(F.gelu(s.fc1(s.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(s, layers, d, h, vocab, max_t):
        super().__init__()
        s.wte = nn.Embedding(vocab, d)
        s.wpe = nn.Embedding(max_t, d)
        s.blocks = nn.ModuleList(Block(d, h) for _ in range(layers))
        s.lnf = nn.LayerNorm(d)
        with torch.no_grad():
            s.wte.weight.uniform_(-0.035, 0.035)
            s.wpe.weight.uniform_(-0.035, 0.035)
            for b in s.blocks:
                for w in (b.wq, b.wk, b.wv, b.wo, b.fc1):
                    w.weight.uniform_(-math.sqrt(3 / d), math.sqrt(3 / d))
                b.fc1.bias.zero_()
                b.fc2.weight.uniform_(-math.sqrt(3 / (4 * d)) / 2, math.sqrt(3 / (4 * d)) / 2)

    def forward(s, idx, caches=None, at=0):
        _, t = idx.shape
        x = s.wte(idx) + s.wpe(torch.arange(at, at + t, device=idx.device))
        for i, b in enumerate(s.blocks):
            x = b(x, None if caches is None else caches[i], at)
        return s.lnf(x) @ s.wte.weight.t()


def source_files(held_out):
    found = glob.glob("lib/**/*.jai", recursive=True) + glob.glob("packages/*/src/**/*.jai", recursive=True)
    out = []
    for rel in found:
        held = rel.startswith("packages/jaiyaml/") or rel.startswith("packages/jaitoml/")
        if held == held_out:
            out.append(rel)
    return sorted(out)


class Corpus:
    """Files as one byte array, and where windows of T + 1 bytes may start."""

    def __init__(s, paths, t):
        chunks, starts, base, s.size = [], [], 0, 0
        for rel in paths:
            data = open(rel, "rb").read()
            s.size += len(data)
            if len(data) < t + 1:
                continue
            chunks.append(np.frombuffer(data, dtype=np.uint8))
            starts.append(base + np.arange(len(data) - t))
            base += len(data)
        s.data = np.concatenate(chunks).astype(np.int64)
        s.starts = np.concatenate(starts)
        s.files = len(chunks)
        s.lengths = [len(c) for c in chunks]

    def batch(s, picks, t):
        rows = picks[:, None] + np.arange(t)[None, :]
        x = torch.from_numpy(s.data[rows]).to(dev)
        y = torch.from_numpy(s.data[rows + 1]).to(dev)
        return x, y


def held_out_picks(c, t):
    """Every file cut into back-to-back windows, the order train.jai uses."""
    picks, base = [], 0
    for n in c.lengths:
        off = 0
        while off + t + 1 <= n:
            picks.append(base + off)
            off += t
        base += n
    return np.array(picks, dtype=np.int64)


def held_out_bits(model, c, picks, limit=0):
    wanted = min(limit * B, len(picks)) if limit > 0 else len(picks)
    total, batches = 0.0, 0
    with torch.no_grad():
        for start in range(0, wanted - B + 1, B):
            x, y = c.batch(picks[start:start + B], T)
            total += F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1)).item()
            batches += 1
    return total / max(batches, 1) / math.log(2)


def stats(label, xs):
    s = sorted(xs)
    print(f"{label} median={s[len(s) // 2]:.2f} min={s[0]:.2f} max={s[-1]:.2f} n={len(s)}")
    return s[len(s) // 2]


def main():
    mode = os.environ.get("MODE", "train")
    t_read = time.perf_counter()
    train_set = Corpus(source_files(False), T)
    held = Corpus(source_files(True), T)
    print(f"corpus: {train_set.files} files, {train_set.size} bytes; held out {held.files} files, "
          f"{held.size} bytes; read in {(time.perf_counter() - t_read) * 1e3:.0f} ms")
    torch.manual_seed(1)
    model = GPT(L, D, H, VOCAB, T).to(dev)
    print(f"model: {L} layers, d{D}, {H} heads, T{T}, batch {B}, "
          f"{sum(p.numel() for p in model.parameters())} parameters")
    if mode == "gen":
        return generate(model, train_set)
    opt = torch.optim.Adam(model.parameters(), lr=float(os.environ.get("LR", "1e-3")), eps=1e-7)
    rng = np.random.default_rng(env_int("SEED", 7))

    def step():
        picks = train_set.starts[rng.integers(0, len(train_set.starts), B)]
        x, y = train_set.batch(picks, T)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.item()

    if mode == "bench":
        t0 = time.perf_counter()
        l0 = step()
        first = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        step()
        print(f"first_step_ms={first:.1f} second_step_ms={(time.perf_counter() - t1) * 1e3:.1f} loss0={l0:.4f}")
        warm = float(os.environ.get("WARM", "2"))
        tw, n = time.perf_counter(), 0
        while time.perf_counter() - tw < warm:
            step()
            n += 1
        reps, steps = env_int("REPS", 7), env_int("STEPS", 5)
        times, last = [], 0.0
        for _ in range(reps):
            ts = time.perf_counter()
            for _ in range(steps):
                last = step()
            times.append((time.perf_counter() - ts) * 1e3 / steps)
        m = stats("train_ms_per_step", times)
        print(f"tokens_per_s={B * T / m * 1000:.0f} warm_steps={n} loss_last={last:.4f}")
        return
    picks = held_out_picks(held, T)
    steps = env_int("STEPS", 2000)
    every, eval_batches = env_int("EVAL_EVERY", 250), env_int("EVAL_BATCHES", 8)
    log_every = env_int("LOG_EVERY", 100)
    print(f"held out: {len(picks)} windows of {T}; evaluating {eval_batches} batches every {every} steps")
    eval_s, recent = 0.0, 0.0
    t0 = time.perf_counter()
    for i in range(1, steps + 1):
        loss = step()
        recent += loss
        if i == 1:
            print(f"step 1 loss {loss:.4f} ({(time.perf_counter() - t0) * 1e3:.0f} ms)")
        if i % log_every == 0:
            print(f"step {i} loss {recent / log_every:.4f} ({time.perf_counter() - t0:.1f} s)")
            recent = 0.0
        if i % every == 0 and i < steps:
            te = time.perf_counter()
            bits = held_out_bits(model, held, picks, eval_batches)
            eval_s += time.perf_counter() - te
            print(f"step {i} held-out {bits:.4f} bits/byte")
    wall = time.perf_counter() - t0
    print(f"trained {steps} steps in {wall:.1f} s ({(wall - eval_s) * 1e3 / steps:.2f} ms/step outside "
          f"{eval_s:.1f} s of evaluation)")
    tf = time.perf_counter()
    bits = held_out_bits(model, held, picks)
    print(f"final held-out {bits:.4f} bits/byte over {len(picks) // B * B} windows ({time.perf_counter() - tf:.1f} s)")


def generate(model, corpus):
    """Cached batch-1 greedy decoding into a preallocated KV cache."""
    text = b"fn parse_scalar("
    prompt_len, new = len(text), env_int("NEW", 112)
    total = prompt_len + new
    prompt = torch.tensor(list(text), dtype=torch.long, device=dev).view(1, prompt_len)
    caches = [(torch.zeros(1, total, D, device=dev), torch.zeros(1, total, D, device=dev)) for _ in range(L)]
    reps, warm = env_int("REPS", 5), float(os.environ.get("WARM", "2"))
    per, pre, early, late = [], [], [], []
    tw, timing = time.perf_counter(), False
    with torch.no_grad():
        while True:
            if not timing and time.perf_counter() - tw >= warm:
                timing = True
            tp = time.perf_counter()
            tok = model(prompt, caches, 0)[0, -1].argmax().item()
            pms = (time.perf_counter() - tp) * 1e3
            td, t16 = time.perf_counter(), 0.0
            for st in range(1, new):
                x = torch.tensor([[tok]], device=dev)
                tok = model(x, caches, prompt_len + st - 1)[0, -1].argmax().item()
                if st == 16:
                    t16 = (time.perf_counter() - td) * 1e3
            dms = (time.perf_counter() - td) * 1e3
            if timing:
                per.append(dms / (new - 1))
                pre.append(pms)
                early.append(t16 / 16)
                late.append((dms - t16) / (new - 17))
                if len(per) >= reps:
                    break
    m = stats("gen_cached_ms_per_token", per)
    stats("gen_cached_prefill_ms", pre)
    stats("gen_cached_tokens_1_16_ms", early)
    stats("gen_cached_tokens_17_end_ms", late)
    print(f"bytes_per_s={1000 / m:.0f}")


if __name__ == "__main__":
    main()
