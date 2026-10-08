#: The model, the corpus and the batches, shared by `train.jai`,
#: `generate.jai` and `check.jai`.
#:
#: A byte-level GPT: pre-LN blocks of causal multi-head attention and a GELU
#: MLP, learned positions, and a head tied to the token embedding. Every piece
#: is a jaitensor layer or op -- the attention layer takes the whole batch,
#: layer norm takes any rank -- so nothing here touches a kernel.
#:
#: Activations are sequence-first: row `t * batch + b` is step `t` of
#: sequence `b`. Every op but attention works row by row and does not care,
#: and attention folds that order into its heads without a copy (see
#: `MultiHeadAttention`'s `batch_first`). `GPT_BATCH_FIRST=1` trains in the
#: batch-first order instead, for an A/B in one binary; read once. It is for
#: training: batched cached decoding is written sequence-first and refuses it.
import jaitensor as jt
import std.gpu as gpu
import std.math
import std.os
from jaitensor import Embedding, LayerNorm, MultiHeadAttention, PositionalEmbedding, Tensor
from jaitensor.ops import copy_into
from std.io import Path, read_bytes
from std.random import Random

#: The model's shape and the batch it trains on.
pub class Config {
    pub let layers: int
    pub let dim: int
    pub let heads: int
    pub let seq: int
    pub let batch: int
    pub let vocab: int

    pub fn init(self, layers: int, dim: int, heads: int, seq: int, batch: int, vocab: int = 256) {
        self.layers = layers
        self.dim = dim
        self.heads = heads
        self.seq = seq
        self.batch = batch
        self.vocab = vocab
    }

    fn __str__(self) -> str {
        return f"{self.layers} layers, d{self.dim}, {self.heads} heads, T{self.seq}, batch {self.batch}"
    }
}

#: 4 layers, d256, 4 heads, T128, batch 64. `GPT_LAYERS`, `GPT_DIM`,
#: `GPT_HEADS`, `GPT_SEQ` and `GPT_BATCH` override it.
pub fn config_from_env() -> Config {
    return Config(
        env_int("GPT_LAYERS", 4),
        env_int("GPT_DIM", 256),
        env_int("GPT_HEADS", 4),
        env_int("GPT_SEQ", 128),
        env_int("GPT_BATCH", 64)
    )
}

pub fn env_int(name: str, fallback: int) -> int { return int(os.env_or(name, f"{fallback}")) }

pub let BATCH_FIRST = env_int("GPT_BATCH_FIRST", 0) == 1

#: When the queued work is handed to the device during a step. jaitensor
#: commits a step's work when something waits on it -- here the loss download
#: at the end -- so the device sits idle while the host encodes the whole
#: step. `GPT_FLUSH=1` commits the forward before encoding the backward, and
#: 2 (the default) also commits after every block, so the device starts on
#: the first block while the host is still encoding the rest. 0 waits for the
#: download. Read once.
pub let FLUSH = env_int("GPT_FLUSH", 2)

#: Whether a decoded token adds each attention output to the residual stream
#: inside the output product (`MultiHeadAttention.forward_add`) rather than
#: with a separate add. `GPT_FUSED_RESIDUAL=0` keeps the add, for an A/B;
#: read once.
pub let FUSED_RESIDUAL = env_int("GPT_FUSED_RESIDUAL", 1) == 1

pub fn env_float(name: str, fallback: float) -> float {
    return float(os.env_or(name, f"{fallback}"))
}

fn param(shape: list[int], limit: float, rng: Random) -> Tensor {
    let t = Tensor(shape, requires_grad: true)
    t.buffer().fill_uniform(-limit, limit, rng.next_int())
    return t
}

#: One pre-LN transformer block.
pub class Block {
    pub let ln1: LayerNorm
    pub let attn: MultiHeadAttention
    pub let ln2: LayerNorm
    pub let fc1: Tensor
    pub let bias1: Tensor
    pub let fc2: Tensor

    pub fn init(self, dim: int, heads: int, rng: Random) {
        self.ln1 = LayerNorm(dim)
        self.ln1.build(dim, rng)
        self.attn = MultiHeadAttention(dim, heads: heads, causal: true, batch_first: BATCH_FIRST)
        self.attn.build(dim, rng)
        self.ln2 = LayerNorm(dim)
        self.ln2.build(dim, rng)
        self.fc1 = param([dim, 4 * dim], math.sqrt(3.0 / float(dim)), rng)
        self.bias1 = Tensor.zeros([4 * dim])
        self.bias1.requires_grad = true
        self.fc2 = param([4 * dim, dim], math.sqrt(3.0 / float(4 * dim)) / 2.0, rng)
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
        let order = if BATCH_FIRST { [batch, seq, d] } else { [seq, batch, d] }
        let a = self.attn.forward(jt.reshape(h, order))
        let x2 = jt.add(x, jt.reshape(a, [batch * seq, d]))
        let h2 = self.ln2.forward(x2)
        let f = jt.linear_gelu(h2, self.fc1, self.bias1)
        return jt.linear_residual(f, self.fc2, x2)
    }

    #: The same block on the new rows of `batch` sequences through the
    #: attention cache, `[rows * batch, dim]` sequence-first, with nothing
    #: recorded: every intermediate is freed as soon as it is used.
    pub fn forward_cached(self, x: Tensor, batch: int) -> Tensor {
        let d = x.shape[1]
        let h = self.ln1.forward(x)
        var a = h
        if batch == 1 {
            a = self.attn.forward(h)
        } else {
            let steps = h.view(0, [x.shape[0] // batch, batch, d])
            let out = self.attn.forward(steps)
            steps.free()
            a = Tensor([x.shape[0], d], storage: out.buffer())
        }
        h.free()
        let x2 = jt.add(x, a)
        a.free()
        return self.finish_cached(x2)
    }

    #: Through the attention with the residual added in its output product.
    pub fn forward_cached_fused(self, x: Tensor, batch: int) -> Tensor {
        let d = x.shape[1]
        let h = self.ln1.forward(x)
        var x2 = h
        if batch == 1 {
            x2 = self.attn.forward_add(h, x)
        } else {
            let steps = h.view(0, [x.shape[0] // batch, batch, d])
            let out = self.attn.forward_add(steps, x)
            steps.free()
            x2 = Tensor([x.shape[0], d], storage: out.buffer())
        }
        h.free()
        return self.finish_cached(x2)
    }

    #: The MLP half of a cached block; frees `x2`.
    fn finish_cached(self, x2: Tensor) -> Tensor {
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
        self.wte.parameters()[0].buffer().fill_uniform(-0.035, 0.035, rng.next_int())
        self.wpe = PositionalEmbedding(
            cfg.seq,
            cfg.dim,
            init_limit: 0.035,
            batch_first: BATCH_FIRST
        )
        self.wpe.build(cfg.dim, rng)
        var bs: list[Block] = []
        for _i in 0..cfg.layers { bs.push(Block(cfg.dim, cfg.heads, rng)) }
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

    #: `idx` is `[seq, batch]` byte values (`[batch, seq]` batch-first).
    #: Logits are one row per byte in the same order, `[seq * batch, vocab]`.
    pub fn forward(self, idx: Tensor) -> Tensor {
        let batch = if BATCH_FIRST { idx.shape[0] } else { idx.shape[1] }
        let seq = if BATCH_FIRST { idx.shape[1] } else { idx.shape[0] }
        let d = self.cfg.dim
        let x3 = self.wpe.forward(self.wte.lookup(idx))
        var x = jt.reshape(x3, [batch * seq, d])
        for b in self.blocks {
            x = b.forward(x, batch, seq)
            if FLUSH >= 2 { gpu.flush() }
        }
        let hf = self.lnf.forward(x)
        return jt.matmul(hf, self.wte.parameters()[0], trans_b: true)
    }

    #: Feed `idx` -- `rows` new bytes of each of `batch` sequences,
    #: sequence-first, at positions `start..start + rows` -- through every
    #: layer's cache, and return the logits of each sequence's last row,
    #: `[batch, vocab]`. Needs `start_cache` and training off.
    pub fn forward_cached(self, idx: Tensor, start: int, batch: int = 1) -> Tensor {
        let rows = idx.size // batch
        let d = self.cfg.dim
        let tok = self.wte.lookup(idx)
        var x = tok
        if batch == 1 {
            x = self.wpe.forward_at(tok, start)
        } else {
            let steps = tok.view(0, [rows, batch, d])
            let placed = self.wpe.forward_at(steps, start)
            steps.free()
            x = Tensor([rows * batch, d], storage: placed.buffer())
        }
        tok.free()
        for b in self.blocks {
            let next = if FUSED_RESIDUAL {
                b.forward_cached_fused(x, batch)
            } else {
                b.forward_cached(x, batch)
            }
            x.free()
            x = next
        }
        let last = if rows == 1 { x } else { x.view((rows - 1) * batch * d, [batch, d]) }
        let hf = self.lnf.forward(last)
        if rows != 1 { last.free() }
        x.free()
        let logits = jt.matmul(hf, self.wte.parameters()[0], trans_b: true)
        hf.free()
        return logits
    }

    #: A cache for `batch` sequences of up to `max_seq` bytes in every layer.
    pub fn start_cache(self, max_seq: int, batch: int = 1) -> void {
        if BATCH_FIRST and batch > 1 {
            throw ValueError(
                "GPT_BATCH_FIRST=1 is for training: batched cached decoding is sequence-first"
            )
        }
        for b in self.blocks { b.attn.start_cache(max_seq, batch: batch) }
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

#: The repository's own Jaithon source. Training is `lib/**/*.jai` and
#: `packages/*/src/**/*.jai` without jaiyaml and jaitoml; those two are the
#: held-out set, a parser the model has never read.
pub fn source_files(root: str, held_out: bool) -> list[str] {
    var out: list[str] = []
    let top = Path(root)
    var found: list[Path] = top.join("lib").glob("**/*.jai")
    for p in top.join("packages").glob("*/src/**/*.jai") { found.push(p) }
    for p in found {
        let text = p.to_str()
        let rel = text[root.len() + 1:]
        let is_held = rel.starts_with("packages/jaiyaml/") or rel.starts_with("packages/jaitoml/")
        if is_held == held_out { out.push(rel) }
    }
    return out.sorted()
}

#: A set of files, each read once as bytes, and where windows of `seq + 1`
#: bytes can start in them. A window never crosses from one file into the next.
pub class Corpus {
    pub let files: list[bytes]
    #: `ends[i]` is how many window starts files `0..=i` hold together.
    pub let ends: list[int]
    pub let windows: int
    pub let size: int

    pub fn init(self, root: str, paths: list[str], seq: int) {
        var files: list[bytes] = []
        var ends: list[int] = []
        var total = 0
        var size = 0
        for rel in paths {
            let data = read_bytes(root + "/" + rel)
            size += data.len()
            if data.len() < seq + 1 { continue }
            files.push(data)
            total += data.len() - seq
            ends.push(total)
        }
        self.files = files
        self.ends = ends
        self.windows = total
        self.size = size
    }

    #: Window number `n`, as (file, offset).
    pub fn locate(self, n: int) -> (int, int) {
        var lo = 0
        var hi = self.ends.len() - 1
        while lo < hi {
            let mid = (lo + hi) // 2
            if self.ends[mid] > n { hi = mid } else { lo = mid + 1 }
        }
        let before = if lo == 0 { 0 } else { self.ends[lo - 1] }
        return (lo, n - before)
    }
}

#: `count` windows as `[seq, count]` inputs and the next-byte targets beside
#: them (`[count, seq]` batch-first). Each window goes straight from the file
#: bytes into its row (`upload_u8`), with no host list, and one transpose on
#: the device puts them sequence-first.
pub fn make_batch(corpus: Corpus, picks: list[int], seq: int) -> (Tensor, Tensor) {
    let count = picks.len()
    let x = Tensor([count, seq])
    let y = Tensor([count, seq])
    for (row, n) in picks.enumerate() {
        let (file, offset) = corpus.locate(n)
        let data = corpus.files[file]
        x.buffer().upload_u8(data, source_offset: offset, count: seq, offset: row * seq)
        y.buffer().upload_u8(data, source_offset: offset + 1, count: seq, offset: row * seq)
    }
    if BATCH_FIRST { return (x, y) }
    let xt = jt.transpose(x)
    let yt = jt.transpose(y)
    x.free()
    y.free()
    return (xt, yt)
}

#: Uniformly random windows, the training batches.
pub class Sampler {
    pub let corpus: Corpus
    pub let rng: Random

    pub fn init(self, corpus: Corpus, seed: int) {
        self.corpus = corpus
        self.rng = Random.with_seed(seed)
    }

    pub fn next(self, batch: int, seq: int) -> (Tensor, Tensor) {
        var picks: list[int] = []
        for _b in 0..batch {
            picks.push(
                int(self.rng.next_float() * float(self.corpus.windows)) % self.corpus.windows
            )
        }
        return make_batch(self.corpus, picks, seq)
    }
}

#: Held-out windows: every file cut into back-to-back windows of `seq`, the
#: same windows in the same order as the peer reads, so the two report bits
#: per byte over identical text.
pub fn held_out_windows(corpus: Corpus, seq: int) -> list[int] {
    var picks: list[int] = []
    var base = 0
    for (i, data) in corpus.files.enumerate() {
        var offset = 0
        while offset + seq + 1 <= data.len() {
            picks.push(base + offset)
            offset += seq
        }
        base = corpus.ends[i]
    }
    return picks
}

#: Mean cross-entropy over the held-out windows, in bits per byte. The graph
#: is recorded and discarded rather than skipped, so this needs no inference
#: path of its own and frees exactly what training frees.
pub fn held_out_bits(model: GPT, corpus: Corpus, picks: list[int], limit: int = 0) -> float {
    let cfg = model.cfg
    let ce = jt.SparseCategoricalCrossentropy(from_logits: true)
    var total = 0.0
    var batches = 0
    var start = 0
    let wanted = if limit > 0 { min(limit * cfg.batch, picks.len()) } else { picks.len() }
    while start + cfg.batch <= wanted {
        let (x, y) = make_batch(corpus, picks[start:start + cfg.batch], cfg.seq)
        let logits = model.forward(x)
        let loss = ce.forward(logits, y)
        total += loss.download()[0]
        free_all(loss.discard_graph())
        free_all([loss, logits, x, y])
        batches += 1
        start += cfg.batch
    }
    return total / float(max(batches, 1)) / math.log(2.0)
}

#: The checkpoint metadata that says what shape of model the weights fit.
pub fn checkpoint_meta(cfg: Config, step: int) -> dict[str, any] {
    return {
        "layers": cfg.layers,
        "dim": cfg.dim,
        "heads": cfg.heads,
        "seq": cfg.seq,
        "vocab": cfg.vocab,
        "step": step,
    }
}

pub fn default_checkpoint() -> str {
    return os.env_or("GPT_CKPT", Path(os.temp_dir()).join("source_gpt.ckpt").to_str())
}

#: Rebuild a model from a checkpoint written by `train.jai`: the header says
#: what shape to build, and the saved tensors are copied into it on the device.
pub fn load_model(path: str) -> GPT {
    let loaded = jt.load_checkpoint(path)
    let meta = loaded.meta
    let layers: int = meta["layers"]
    let dim: int = meta["dim"]
    let heads: int = meta["heads"]
    let seq: int = meta["seq"]
    let vocab: int = meta["vocab"]
    let model = GPT(Config(layers, dim, heads, seq, 1, vocab: vocab))
    for (name, param) in model.names().zip(model.params()) { copy_into(loaded.get(name), param) }
    loaded.free()
    return model
}
