"""The torch MPS peer for examples/story_chat: the same model, tokens and step.

    ~/.venvs/scratch/bin/python examples/story_chat/story_chat.py

Run from the repository root, after prepare.jai. Same shape (GPT_LAYERS,
GPT_DIM, GPT_HEADS, GPT_SEQ, GPT_BATCH, VOCAB; 8 / 512 / 8 / 256 / 32 / 4096),
the same initialisation (uniform at GPT-2's 0.02, the residual projections
scaled by 1/sqrt(2 * layers)), AdamW with betas (0.9, 0.95), eps 1e-8 and
decay 0.1 on the matrices only, the gradient clipped to norm 1, and random
windows of the same token file. The loss is read every step (`loss.item()`),
as pretrain.jai's MODE=bench does.

MODE=bench (default) times the step: WARM seconds of steps by the wall clock,
then REPS x BENCH_STEPS. MODE=train runs STEPS steps with the held-out loss
every EVAL_EVERY, for a quality comparison at equal steps. MODE=tokcheck
re-encodes the first COUNT stories of the valid split with a Python BPE that
reads the merges tokenizer.jai saved, and checks the ids against valid-<V>.u16.

PEER_FUSED=1 uses torch's fused AdamW.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def env_int(name, fallback):
    return int(os.environ.get(name, fallback))


L = env_int("GPT_LAYERS", 8)
D = env_int("GPT_DIM", 512)
H = env_int("GPT_HEADS", 8)
T = env_int("GPT_SEQ", 256)
B = env_int("GPT_BATCH", 32)
V = env_int("VOCAB", 4096)
CACHE = os.environ.get("STORY_CACHE", os.path.expanduser("~/.cache/jaithon/story_chat"))
LIMIT = 0.02 * math.sqrt(3.0)


class Block(nn.Module):
    def __init__(s, d, h, layers):
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
        residual = LIMIT / math.sqrt(2.0 * layers)
        with torch.no_grad():
            for w in (s.wq, s.wk, s.wv, s.fc1):
                w.weight.uniform_(-LIMIT, LIMIT)
            for w in (s.wo, s.fc2):
                w.weight.uniform_(-residual, residual)
            s.fc1.bias.zero_()

    def forward(s, x):
        b, t, d = x.shape
        h = s.ln1(x)
        sp = lambda z: z.view(b, t, s.h, d // s.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(sp(s.wq(h)), sp(s.wk(h)), sp(s.wv(h)), is_causal=True)
        x = x + s.wo(y.transpose(1, 2).reshape(b, t, d))
        return x + s.fc2(F.gelu(s.fc1(s.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.wte = nn.Embedding(V, D)
        s.wpe = nn.Embedding(T, D)
        s.blocks = nn.ModuleList(Block(D, H, L) for _ in range(L))
        s.lnf = nn.LayerNorm(D)
        with torch.no_grad():
            s.wte.weight.uniform_(-LIMIT, LIMIT)
            s.wpe.weight.uniform_(-LIMIT, LIMIT)

    def forward(s, idx):
        _, t = idx.shape
        x = s.wte(idx) + s.wpe(torch.arange(t, device=idx.device))
        for b in s.blocks:
            x = b(x)
        return s.lnf(x) @ s.wte.weight.t()


def optimiser(model, lr):
    decayed, plain = [], []
    for name, p in model.named_parameters():
        is_matrix = name == "wte.weight" or (p.dim() == 2 and not name.startswith("wpe"))
        (decayed if is_matrix else plain).append(p)
    groups = [{"params": decayed, "weight_decay": 0.1}, {"params": plain, "weight_decay": 0.0}]
    extra = {"fused": True} if os.environ.get("PEER_FUSED") == "1" else {}
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, **extra)


def tokens(split):
    path = os.path.join(CACHE, f"{split}-{V}.u16")
    return np.memmap(path, dtype="<u2", mode="r")


def batch(data, rng, dev):
    starts = rng.integers(0, len(data) - T - 1, size=B)
    rows = starts[:, None] + np.arange(T + 1)[None, :]
    w = torch.from_numpy(data[rows].astype(np.int64)).to(dev)
    return w[:, :-1], w[:, 1:]


def step(model, opt, x, y, lr):
    for g in opt.param_groups:
        g["lr"] = lr
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss


def bench():
    dev = torch.device("mps")
    torch.manual_seed(7)
    model = GPT().to(dev)
    opt = optimiser(model, 1e-4)
    data = tokens("train")
    rng = np.random.default_rng(7)
    n = sum(p.numel() for p in model.parameters())
    print(f"torch {torch.__version__}, {L} layers d{D} {H} heads T{T} batch {B} vocab {V}: {n} parameters")
    t0 = time.perf_counter()
    loss0 = step(model, opt, *batch(data, rng, dev), 1e-4).item()
    print(f"first_step_ms={(time.perf_counter() - t0) * 1000:.1f} loss0={loss0:.4f}")
    warm = float(os.environ.get("WARM", 3.0))
    tw = time.perf_counter()
    warm_steps = 0
    while time.perf_counter() - tw < warm:
        step(model, opt, *batch(data, rng, dev), 1e-4).item()
        warm_steps += 1
    times = []
    last = 0.0
    for _ in range(env_int("REPS", 5)):
        ts = time.perf_counter()
        for _ in range(env_int("BENCH_STEPS", 5)):
            last = step(model, opt, *batch(data, rng, dev), 1e-4).item()
        times.append((time.perf_counter() - ts) * 1000 / env_int("BENCH_STEPS", 5))
    s = sorted(times)
    med = s[len(s) // 2]
    print(f"train_ms_per_step median={med:.1f} min={s[0]:.1f} max={s[-1]:.1f} n={len(s)} (after {warm_steps} warm steps)")
    print(f"tokens_per_s={B * T / med * 1000:.0f} loss_last={last:.4f}")


def train():
    dev = torch.device("mps")
    torch.manual_seed(7)
    model = GPT().to(dev)
    peak, warmup, total = float(os.environ.get("LR", 6e-4)), env_int("WARMUP", 200), env_int("STEPS", 2000)
    opt = optimiser(model, peak)
    data, valid = tokens("train"), tokens("valid")
    rng = np.random.default_rng(7)
    span = len(valid) - T - 1
    count = env_int("EVAL_WINDOWS", 256)
    if count < B or count % B:
        raise SystemExit(f"EVAL_WINDOWS={count}: it has to be a multiple of the batch, {B}")
    held = [i * (span // count) for i in range(count)]
    t0 = time.perf_counter()
    recent = []
    for s in range(total):
        if s < warmup:
            lr = peak * (s + 1) / warmup
        else:
            p = min(1.0, s / total)  # as pretrain.jai: the cosine runs over the whole schedule
            lr = peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
        recent.append(step(model, opt, *batch(data, rng, dev), lr).item())
        if (s + 1) % env_int("EVAL_EVERY", 250) == 0 or s + 1 == total:
            model.eval()
            with torch.no_grad():
                losses = []
                for i in range(0, count - B + 1, B):
                    rows = np.array(held[i:i + B])[:, None] + np.arange(T + 1)[None, :]
                    w = torch.from_numpy(valid[rows].astype(np.int64)).to(dev)
                    lg = model(w[:, :-1])
                    losses.append(F.cross_entropy(lg.view(-1, V), w[:, 1:].reshape(-1)).item())
            model.train()
            el = time.perf_counter() - t0
            print(f"step {s + 1} loss {sum(recent) / len(recent):.4f} held-out {sum(losses) / len(losses):.4f} ({el:.0f} s, {(s + 1) * B * T / el:.0f} tokens/s)")
            recent = []


SPACE, BLANK, LETTER, DIGIT, MARK = 0, 1, 2, 3, 4


def classes():
    out = []
    for b in range(256):
        c = MARK
        if b == 32:
            c = SPACE
        elif b in (9, 10, 11, 12, 13):
            c = BLANK
        elif 65 <= b <= 90 or 97 <= b <= 122 or b >= 128:
            c = LETTER
        elif 48 <= b <= 57:
            c = DIGIT
        out.append(c)
    return out


def pieces(data):
    """The same cut as tokenizer.jai's piece_end."""
    cls = classes()
    n, i = len(data), 0
    while i < n:
        j, c = i, cls[data[i]]
        if c == SPACE and j + 1 < n and cls[data[j + 1]] >= LETTER:
            j += 1
            c = cls[data[j]]
        if c == LETTER:
            j += 1
            while j < n:
                if cls[data[j]] == LETTER:
                    j += 1
                elif data[j] == 39 and j + 1 < n and cls[data[j + 1]] == LETTER:
                    j += 2
                else:
                    break
        elif c in (DIGIT, MARK):
            j += 1
            while j < n and cls[data[j]] == c:
                j += 1
        else:
            j += 1
            while j < n and cls[data[j]] <= BLANK:
                j += 1
            if j < n and j - i > 1 and data[j - 1] == 32:
                j -= 1
        yield data[i:j]
        i = j


def load_merges():
    lines = open(os.path.join(CACHE, f"tokenizer-{V}.txt")).read().split("\n")
    assert lines[0].startswith("story_chat bpe 1")
    ranks = {}
    for k, line in enumerate(l for l in lines[1:] if l):
        a, b = map(int, line.split())
        ranks[(a, b)] = 259 + k
    return ranks


def encode_piece(piece, ranks):
    ids = list(piece)
    while len(ids) >= 2:
        best = min(((ranks.get((ids[i], ids[i + 1]), 1 << 30), i) for i in range(len(ids) - 1)))
        if best[0] == 1 << 30:
            break
        made, a, b = best[0], ids[best[1]], ids[best[1] + 1]
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(made)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids


def tokcheck():
    ranks = load_merges()
    root = os.environ.get("STORY_DATA", os.path.expanduser("~/Developer/datasets/tinystories"))
    text = open(os.path.join(root, "TinyStoriesV2-GPT4-valid.txt"), "rb").read()
    stories = [s.strip() for s in text.split(b"<|endoftext|>")]
    stories = [s for s in stories if s]
    count = env_int("COUNT", 2000)
    ids = [256]
    cache = {}
    for s in stories[:count]:
        for p in pieces(s):
            if p not in cache:
                cache[p] = encode_piece(p, ranks)
            ids.extend(cache[p])
        ids.append(256)
    saved = tokens("valid")[: len(ids)].tolist()
    same = sum(1 for a, b in zip(ids, saved) if a == b)
    print(f"python BPE from the saved merges: {len(ids)} ids for {count} stories, {same} equal to valid-{V}.u16")
    print("tokenizer agrees" if same == len(ids) else "TOKENIZER DIFFERS")
    return 0 if same == len(ids) else 1


if __name__ == "__main__":
    mode = os.environ.get("MODE", "bench")
    if mode == "train":
        train()
    elif mode == "tokcheck":
        raise SystemExit(tokcheck())
    else:
        bench()
