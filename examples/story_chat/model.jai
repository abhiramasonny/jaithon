#: The model, the token files, the batches, evaluation and checkpoints, shared
#: by every program in this example.
#:
#: A GPT over the BPE ids of `tokenizer.jai`: pre-LN blocks of causal
#: multi-head attention and a GELU MLP, learned positions, and an output head
#: tied to the token embedding -- `examples/source_gpt`'s model, wider, deeper
#: and over a vocabulary of thousands rather than 256 bytes. Every piece is a
#: jaitensor layer or op.
#:
#: Activations are sequence-first, as in source_gpt: row `t * batch + b` is
#: step `t` of sequence `b`, the order batched attention reads without a copy.
import jaitensor as jt
import std.gpu as gpu
import std.math
import std.os
from jaitensor import AdamW, Embedding, LayerNorm, MultiHeadAttention, PositionalEmbedding, Tensor
from jaitensor.ops import copy_into
from jaitensor.optimizers import moment_buffers, set_steps_taken
from std.io import Path, read_bytes
from std.random import Random
from .tokenizer import Tokenizer, cache_dir

#: The model's shape and the batch it trains on.
pub class Config {
    pub let layers: int
    pub let dim: int
    pub let heads: int
    pub let seq: int
    pub let batch: int
    pub let vocab: int

    pub fn init(self, layers: int, dim: int, heads: int, seq: int, batch: int, vocab: int) {
        self.layers = layers
        self.dim = dim
        self.heads = heads
        self.seq = seq
        self.batch = batch
        self.vocab = vocab
    }

    fn __str__(self) -> str {
        return f"{self.layers} layers, d{self.dim}, {self.heads} heads, T{self.seq}, batch {self.batch}, vocab {self.vocab}"
    }
}

pub fn env_int(name: str, fallback: int) -> int { return int(os.env_or(name, f"{fallback}")) }

pub fn env_float(name: str, fallback: float) -> float {
    return float(os.env_or(name, f"{fallback}"))
}

#: The shape the full run trains: `GPT_LAYERS`, `GPT_DIM`, `GPT_HEADS`,
#: `GPT_SEQ` and `GPT_BATCH` override it. `VOCAB` picks the tokenizer.
pub fn config_from_env() -> Config {
    return Config(
        env_int("GPT_LAYERS", 8),
        env_int("GPT_DIM", 512),
        env_int("GPT_HEADS", 8),
        env_int("GPT_SEQ", 256),
        env_int("GPT_BATCH", 32),
        env_int("VOCAB", 4096)
    )
}

#: When queued work is handed to the device during a step, as in source_gpt:
#: 2 commits after every block so the device starts while the host encodes
#: the rest; 0 commits only at the loss download. Read once.
pub let FLUSH = env_int("GPT_FLUSH", 2)

#: Uniform values in `[-limit, limit]` drawn on the host. jaitensor's
#: `fill_uniform` gives correlated tensors for different seeds (one xorshift
#: round over a Weyl sequence; the decode track is fixing it), so a deep
#: model's weights are drawn here instead, from one generator.
fn uniform_tensor(shape: list[int], limit: float, rng: Random) -> Tensor {
    var n = 1
    for d in shape { n *= d }
    var values: list[float] = []
    let span = 2.0 * limit
    for _i in 0..n { values.push(rng.next_float() * span - limit) }
    return Tensor(shape, values, requires_grad: true)
}

fn overwrite(t: Tensor, limit: float, rng: Random) -> void {
    let fresh = uniform_tensor(t.shape, limit, rng)
    copy_into(fresh, t)
    fresh.free()
}

#: The standard deviation GPT-2 starts its weights at, as a uniform limit.
let LIMIT = 0.02 * 1.7320508075688772

#: One pre-LN transformer block.
pub class Block {
    pub let ln1: LayerNorm
    pub let attn: MultiHeadAttention
    pub let ln2: LayerNorm
    pub let fc1: Tensor
    pub let bias1: Tensor
    pub let fc2: Tensor

    #: Weights at GPT-2's 0.02, and the two projections that write the
    #: residual stream (`attn.wo`, `fc2`) scaled down by `sqrt(2 * layers)`.
    pub fn init(self, dim: int, heads: int, layers: int, rng: Random) {
        self.ln1 = LayerNorm(dim)
        self.ln1.build(dim, rng)
        self.attn = MultiHeadAttention(dim, heads: heads, causal: true, batch_first: false)
        self.attn.build(dim, rng)
        let residual = LIMIT / math.sqrt(2.0 * float(layers))
        let ws = self.attn.parameters()
        for i in 0..3 { overwrite(ws[i], LIMIT, rng) }
        overwrite(ws[3], residual, rng)
        self.ln2 = LayerNorm(dim)
        self.ln2.build(dim, rng)
        self.fc1 = uniform_tensor([dim, 4 * dim], LIMIT, rng)
        self.bias1 = Tensor.zeros([4 * dim])
        self.bias1.requires_grad = true
        self.fc2 = uniform_tensor([4 * dim, dim], residual, rng)
    }

    pub fn params(self) -> list[Tensor] {
        var out = self.ln1.parameters()
        for t in self.attn.parameters() { out.push(t) }
        for t in self.ln2.parameters() { out.push(t) }
        for t in [self.fc1, self.bias1, self.fc2] { out.push(t) }
        return out
    }

    pub fn names(self, prefix: str) -> list[str] {
        var out: list[str] = []
        for part in ["ln1.g", "ln1.b", "attn.wq", "attn.wk", "attn.wv", "attn.wo"] {
            out.push(prefix + part)
        }
        for part in ["ln2.g", "ln2.b", "fc1.w", "fc1.b", "fc2.w"] { out.push(prefix + part) }
        return out
    }

    #: `x` is the residual stream as `[seq * batch, dim]`; attention sees it
    #: as the `[seq, batch, dim]` it is.
    pub fn forward(self, x: Tensor, batch: int, seq: int) -> Tensor {
        let d = x.shape[1]
        let h = self.ln1.forward(x)
        let a = self.attn.forward(jt.reshape(h, [seq, batch, d]))
        let x2 = jt.add(x, jt.reshape(a, [batch * seq, d]))
        let h2 = self.ln2.forward(x2)
        let f = jt.linear_gelu(h2, self.fc1, self.bias1)
        return jt.linear_residual(f, self.fc2, x2)
    }

    #: The same block on the new rows of one sequence through the attention
    #: cache, with nothing recorded and every intermediate freed as it is used.
    pub fn forward_cached(self, x: Tensor) -> Tensor {
        let h = self.ln1.forward(x)
        let x2 = self.attn.forward_add(h, x)
        h.free()
        let h2 = self.ln2.forward(x2)
        let f = jt.linear_gelu(h2, self.fc1, self.bias1)
        h2.free()
        let out = jt.linear_residual(f, self.fc2, x2)
        f.free()
        x2.free()
        return out
    }
}

pub class GPT {
    pub let cfg: Config
    pub let wte: Embedding
    pub let wpe: PositionalEmbedding
    pub let blocks: list[Block]
    pub let lnf: LayerNorm

    pub fn init(self, cfg: Config, seed: int = 1) {
        let rng = Random.with_seed(seed)
        self.cfg = cfg
        self.wte = Embedding(cfg.vocab, cfg.dim)
        self.wte.build(1, rng)
        overwrite(self.wte.parameters()[0], LIMIT, rng)
        self.wpe = PositionalEmbedding(cfg.seq, cfg.dim, init_limit: LIMIT, batch_first: false)
        self.wpe.build(cfg.dim, rng)
        overwrite(self.wpe.parameters()[0], LIMIT, rng)
        var bs: list[Block] = []
        for _i in 0..cfg.layers { bs.push(Block(cfg.dim, cfg.heads, cfg.layers, rng)) }
        self.blocks = bs
        self.lnf = LayerNorm(cfg.dim)
        self.lnf.build(cfg.dim, rng)
    }

    #: Every trainable tensor, in the order `names` gives them.
    pub fn params(self) -> list[Tensor] {
        var out = [self.wte.parameters()[0], self.wpe.parameters()[0]]
        for b in self.blocks {
            for t in b.params() { out.push(t) }
        }
        for t in self.lnf.parameters() { out.push(t) }
        return out
    }

    pub fn names(self) -> list[str] {
        var out = ["wte", "wpe"]
        for (i, b) in self.blocks.enumerate() {
            for name in b.names(f"h{i}.") { out.push(name) }
        }
        out.push("lnf.g")
        out.push("lnf.b")
        return out
    }

    pub fn parameter_count(self) -> int {
        var n = 0
        for t in self.params() { n += t.size }
        return n
    }

    #: Training on or off. Off, nothing records a graph, which generation
    #: needs: the cached forward frees as it goes.
    pub fn set_training(self, on: bool) -> void {
        for t in self.params() { t.requires_grad = on }
    }

    #: `idx` is `[seq, batch]` token ids. The logits are one row per token in
    #: the same order, `[seq * batch, vocab]` -- or, given `rows`, only those
    #: rows of it, `[rows.size, vocab]`: fine-tuning scores the bot's turns
    #: alone, and gathers them before the head rather than after it.
    pub fn forward(self, idx: Tensor, rows: Tensor? = null) -> Tensor {
        let seq = idx.shape[0]
        let batch = idx.shape[1]
        let d = self.cfg.dim
        let x3 = self.wpe.forward(self.wte.lookup(idx))
        var x = jt.reshape(x3, [batch * seq, d])
        for b in self.blocks {
            x = b.forward(x, batch, seq)
            if FLUSH >= 2 { gpu.flush() }
        }
        var hf = self.lnf.forward(x)
        if rows is not null { hf = jt.gather_rows(hf, rows) }
        return jt.matmul(hf, self.wte.parameters()[0], trans_b: true)
    }

    #: Feed `ids` -- the next tokens of one sequence, at positions
    #: `start..start + len` -- through every layer's cache, and return the
    #: logits of the last one, `[1, vocab]`. Needs `start_cache` and training
    #: off.
    pub fn forward_cached(self, ids: list[int], start: int) -> Tensor {
        let rows = ids.len()
        let d = self.cfg.dim
        let idx = Tensor([rows], [float(t) for t in ids])
        let tok = self.wte.lookup(idx)
        idx.free()
        var x = self.wpe.forward_at(tok, start)
        tok.free()
        for b in self.blocks {
            let next = b.forward_cached(x)
            x.free()
            x = next
        }
        let last = if rows == 1 { x } else { x.view((rows - 1) * d, [1, d]) }
        let hf = self.lnf.forward(last)
        if rows != 1 { last.free() }
        x.free()
        let logits = jt.matmul(hf, self.wte.parameters()[0], trans_b: true)
        hf.free()
        return logits
    }

    #: A cache for one sequence of up to `max_seq` tokens in every layer.
    pub fn start_cache(self, max_seq: int) -> void {
        for b in self.blocks { b.attn.start_cache(max_seq) }
    }

    pub fn reset_cache(self) -> void {
        for b in self.blocks { b.attn.reset_cache() }
    }

    pub fn free_cache(self) -> void {
        for b in self.blocks { b.attn.free_cache() }
    }

    pub fn free(self) -> void {
        self.wte.free()
        self.wpe.free()
        for b in self.blocks {
            b.ln1.free()
            b.attn.free()
            b.ln2.free()
            for t in [b.fc1, b.bias1, b.fc2] { t.free() }
        }
        self.lnf.free()
    }
}

pub fn free_all(ts: list[Tensor]) -> void {
    for t in ts { if not t.is_freed() { t.free() } }
}

#: Drop every parameter's gradient, so the next backward adopts its
#: contributions instead of adding them into zeroed buffers.
pub fn drop_grads(params: list[Tensor]) -> void {
    for p in params {
        let g = p.grad
        if g is not null {
            g.free()
            p.grad = null
        }
    }
}

#: AdamW for the matrices -- the embedding, attention and MLP weights -- and
#: plain Adam (no decay) for the gains, biases and positions, as GPT-2 and
#: nanoGPT train.
pub class Optimisers {
    pub let decayed: AdamW
    pub let plain: AdamW
    pub let decayed_params: list[Tensor]
    pub let plain_params: list[Tensor]
    pub let decayed_names: list[str]
    pub let plain_names: list[str]

    pub fn init(self, model: GPT, lr: float, weight_decay: float) {
        self.decayed = AdamW(
            learning_rate: lr,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1.0e-8,
            weight_decay: weight_decay
        )
        self.plain = AdamW(
            learning_rate: lr,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1.0e-8,
            weight_decay: 0.0
        )
        var dp: list[Tensor] = []
        var pp: list[Tensor] = []
        var dn: list[str] = []
        var pn: list[str] = []
        for (name, t) in model.names().zip(model.params()) {
            let decays = name == "wte" or name.ends_with(".w") or name.contains("attn.")
            if decays {
                dp.push(t)
                dn.push(name)
            } else {
                pp.push(t)
                pn.push(name)
            }
        }
        self.decayed_params = dp
        self.plain_params = pp
        self.decayed_names = dn
        self.plain_names = pn
    }

    pub fn set_learning_rate(self, lr: float) -> void {
        self.decayed.set_learning_rate(lr)
        self.plain.set_learning_rate(lr)
    }

    pub fn step(self) -> void {
        self.decayed.step(self.decayed_params)
        self.plain.step(self.plain_params)
    }

    #: The two moment buffers of every parameter as tensors viewing them, with
    #: names, so a checkpoint carries them. The views are never freed: the
    #: buffers belong to the optimiser.
    pub fn state(self) -> (list[str], list[Tensor]) {
        var names: list[str] = []
        var ts: list[Tensor] = []
        for (group, kind) in [(0, self.decayed.kind()), (1, self.plain.kind())] {
            let params = if group == 0 { self.decayed_params } else { self.plain_params }
            let pnames = if group == 0 { self.decayed_names } else { self.plain_names }
            for (i, p) in params.enumerate() {
                let bufs = moment_buffers(p, kind)
                names.push("opt.m." + pnames[i])
                ts.push(Tensor([p.size], storage: bufs[0]))
                names.push("opt.v." + pnames[i])
                ts.push(Tensor([p.size], storage: bufs[1]))
            }
        }
        return (names, ts)
    }

    #: Bias correction reads how many steps each parameter has taken.
    pub fn set_steps(self, count: int) -> void {
        set_steps_taken(self.decayed_params, count)
        set_steps_taken(self.plain_params, count)
    }
}

#: A file of uint16 token ids, held in memory as its bytes.
pub class TokenFile {
    pub let data: bytes
    pub let count: int

    pub fn init(self, data: bytes) {
        self.data = data
        self.count = data.len() // 2
    }

    pub fn at(self, i: int) -> int { return self.data[2 * i] + 256 * self.data[2 * i + 1] }

    #: Ids `start..stop` as a list.
    pub fn ids(self, start: int, stop: int) -> list[int] {
        let data = self.data
        var out: list[int] = []
        for i in start..stop { out.push(data[2 * i] + 256 * data[2 * i + 1]) }
        return out
    }
}

#: The token file `prepare.jai` wrote, read whole.
pub fn read_token_file(path: str) -> TokenFile {
    if not Path(path).exists() {
        throw RuntimeError(
            f"missing {path}: run examples/story_chat/prepare.jai first (it writes the token files)"
        )
    }
    return TokenFile(read_bytes(path))
}

pub fn token_path(split: str, vocab: int) -> str { return cache_dir() + f"/{split}-{vocab}.u16" }

pub fn tokenizer_path(vocab: int) -> str { return cache_dir() + f"/tokenizer-{vocab}.txt" }

#: Windows of `seq + 1` tokens starting at `starts`, as `[seq, batch]` inputs
#: and the next-token targets beside them, sequence-first.
pub fn make_batch(file: TokenFile, starts: list[int], seq: int) -> (Tensor, Tensor) {
    let batch = starts.len()
    let data = file.data
    var xs: list[float] = []
    var ys: list[float] = []
    for t in 0..seq {
        for b in 0..batch {
            let at = 2 * (starts[b] + t)
            xs.push(float(data[at] + 256 * data[at + 1]))
            ys.push(float(data[at + 2] + 256 * data[at + 3]))
        }
    }
    return (Tensor([seq, batch], xs), Tensor([seq, batch], ys))
}

#: Uniformly random windows, the training batches. Reseeded from the step on
#: a resume, so a run split into chunks draws different windows each chunk.
pub class Sampler {
    pub let file: TokenFile
    pub var rng: Random

    pub fn init(self, file: TokenFile, seed: int) {
        self.file = file
        self.rng = Random.with_seed(seed)
    }

    pub fn next(self, batch: int, seq: int) -> (Tensor, Tensor) {
        let span = self.file.count - seq - 1
        var starts: list[int] = []
        for _b in 0..batch { starts.push(int(self.rng.next_float() * float(span)) % span) }
        return make_batch(self.file, starts, seq)
    }
}

#: Held-out windows: `count` windows spread evenly over the file, the same
#: ones every evaluation.
pub fn held_out_starts(file: TokenFile, seq: int, count: int) -> list[int] {
    let span = file.count - seq - 1
    var out: list[int] = []
    for i in 0..count { out.push(i * (span // count)) }
    return out
}

#: How many bytes of text the targets of these windows are: the denominator
#: of bits per byte. `<|endoftext|>` counts as one byte, the newline it stands
#: in for.
pub fn target_bytes(file: TokenFile, tok: Tokenizer, starts: list[int], seq: int) -> int {
    var n = 0
    for s in starts {
        for i in s + 1..s + seq + 1 {
            let t = file.at(i)
            n += if t >= 256 and t < 259 { 1 } else { tok.pieces[t].len() }
        }
    }
    return n
}

#: Mean held-out loss in nats a token over the windows at `starts`, a batch
#: at a time. The graph is recorded and discarded, as training frees it.
pub fn held_out_loss(model: GPT, file: TokenFile, starts: list[int]) -> float {
    let cfg = model.cfg
    let ce = jt.SparseCategoricalCrossentropy(from_logits: true)
    var total = 0.0
    var batches = 0
    var i = 0
    while i + cfg.batch <= starts.len() {
        let (x, y) = make_batch(file, starts[i:i + cfg.batch], cfg.seq)
        let logits = model.forward(x)
        let loss = ce.forward(logits, y)
        total += loss.download()[0]
        free_all(loss.discard_graph())
        free_all([loss, logits, x, y])
        batches += 1
        i += cfg.batch
    }
    return total / float(max(batches, 1))
}

#: The metadata a checkpoint carries: the shape, and where training is.
pub fn checkpoint_meta(cfg: Config, step: int, extra: dict[str, any]) -> dict[str, any] {
    var meta: dict[str, any] = {
        "layers": cfg.layers,
        "dim": cfg.dim,
        "heads": cfg.heads,
        "seq": cfg.seq,
        "batch": cfg.batch,
        "vocab": cfg.vocab,
        "step": step,
    }
    for (k, v) in extra.items() { meta[k] = v }
    return meta
}

pub fn config_of(meta: dict[str, any], batch: int = 1) -> Config {
    let layers: int = meta["layers"]
    let dim: int = meta["dim"]
    let heads: int = meta["heads"]
    let seq: int = meta["seq"]
    let vocab: int = meta["vocab"]
    return Config(layers, dim, heads, seq, batch, vocab)
}

#: Rebuild a model from a checkpoint: the header says what shape to build,
#: and the saved weights are copied into it on the device. Optimiser state in
#: the file is ignored.
pub fn load_model(path: str, batch: int = 1) -> (GPT, dict[str, any]) {
    if not Path(path).exists() {
        throw RuntimeError(f"no checkpoint at {path}: train one first (pretrain.jai, finetune.jai)")
    }
    let loaded = jt.load_checkpoint(path)
    let meta = loaded.meta
    let model = GPT(config_of(meta, batch))
    for (name, param) in model.names().zip(model.params()) { copy_into(loaded.get(name), param) }
    loaded.free()
    return (model, meta)
}

pub fn pretrain_checkpoint() -> str {
    return os.env_or("PRETRAIN_CKPT", cache_dir() + "/pretrain.ckpt")
}

pub fn chat_checkpoint() -> str { return os.env_or("CHAT_CKPT", cache_dir() + "/chat.ckpt") }
