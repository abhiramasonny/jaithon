#: The quick acceptance check for the example, about half a minute.
#:
#:     jaithon run examples/source_gpt/check.jai
#:
#: From the repository root. It trains a fresh model for 200 steps and
#: requires that
#:
#: - the loss falls from about 5.6 (ln 256 is 5.55) to below 3.0, averaged
#:   over the last 20 steps;
#: - training leaks nothing: `gpu.live_buffers()` is the same at step 200 as at
#:   step 10;
#: - greedy decoding through the attention cache, a prompt and then one byte at
#:   a time, picks the same 64 bytes as running the whole prefix through the
#:   model for every byte, and the cached decoding leaks nothing either;
#: - a checkpoint saved and loaded gives back the same logits, bit for bit.
#:
#: It exits non-zero if any of them fails.
import jaitensor as jt
import std.gpu as gpu
import std.math
import std.os
from jaitensor import Adam, SparseCategoricalCrossentropy, Tensor
from std.io import Path
from std.time import Instant
from .model import BATCH_FIRST, Corpus, GPT, Sampler, checkpoint_meta, config_from_env, drop_grads, free_all
from .model import load_model, source_files

var failures: list[str] = []

fn expect(ok: bool, what: str) -> void {
    let mark = if ok { "PASS" } else { "FAIL" }
    print(f"{mark}  {what}")
    if not ok { failures.push(what) }
}

fn argmax(row: list[float]) -> int {
    var best = 0
    for i in 1..row.len() {
        if row[i] > row[best] { best = i }
    }
    return best
}

#: The next byte for the whole prefix, with no cache: the training forward,
#: recorded and then discarded.
fn greedy_uncached(model: GPT, tokens: list[int]) -> int {
    let n = tokens.len()
    let shape = if BATCH_FIRST { [1, n] } else { [n, 1] }
    let idx = Tensor(shape, [float(t) for t in tokens])
    let logits = model.forward(idx)
    let row = logits.view((n - 1) * 256, [256])
    let next = argmax(row.download())
    row.free()
    free_all(logits.discard_graph())
    free_all([logits, idx])
    return next
}

fn greedy_cached(model: GPT, prompt: list[int], count: int) -> list[int] {
    model.reset_cache()
    let idx = Tensor([prompt.len()], [float(t) for t in prompt])
    let first = model.forward_cached(idx, 0)
    idx.free()
    var out = [argmax(first.download())]
    first.free()
    for step in 1..count {
        let one = Tensor([1], [float(out[out.len() - 1])])
        let logits = model.forward_cached(one, prompt.len() + step - 1)
        one.free()
        out.push(argmax(logits.download()))
        logits.free()
    }
    return out
}

fn main() -> int {
    let cfg = config_from_env()
    let train_set = Corpus(".", source_files(".", false), cfg.seq)
    let held = Corpus(".", source_files(".", true), cfg.seq)
    if train_set.files.len() == 0 { throw RuntimeError("run this from the repository root") }
    let model = GPT(cfg)
    let params = model.params()
    let opt = Adam(learning_rate: 1.0e-3, epsilon: 1.0e-7)
    let ce = SparseCategoricalCrossentropy(from_logits: true)
    let sampler = Sampler(train_set, 7)
    let steps = 200
    var losses: list[float] = []
    var live = -1
    let t0 = Instant.now()
    for step in 1..steps + 1 {
        let (x, y) = sampler.next(cfg.batch, cfg.seq)
        let logits = model.forward(x)
        let loss = ce.forward(logits, y)
        let graph = loss.backward()
        opt.step(params)
        drop_grads(params)
        free_all(graph)
        free_all([logits, x, y])
        losses.push(loss.download()[0])
        loss.free()
        if step == 10 {
            gpu.synchronize()
            live = gpu.live_buffers()
        }
    }
    gpu.synchronize()
    let drift = gpu.live_buffers() - live
    var tail = 0.0
    for value in losses[steps - 20:] { tail += value }
    tail /= 20.0
    print(f"{steps} steps in {t0.elapsed().to_seconds():.1f} s: loss {losses[0]:.3f} -> {tail:.3f}")
    expect(
        losses[0] > 5.0 and tail < 3.0,
        f"loss falls below 3.0 (first {losses[0]:.3f}, last 20 {tail:.3f})"
    )
    expect(drift == 0, f"training leaks no device buffers (drift {drift})")

    #: Cached against uncached greedy decoding, 64 bytes after a held-out
    #: prompt.
    let source = held.files[0]
    var prompt: list[int] = []
    for i in 0..16 { prompt.push(source[200 + i]) }
    let count = 64
    var uncached: list[int] = []
    var prefix = prompt.copy()
    for _n in 0..count {
        let next = greedy_uncached(model, prefix)
        uncached.push(next)
        prefix.push(next)
    }
    model.set_training(false)
    model.start_cache(prompt.len() + count)
    let cached = greedy_cached(model, prompt, count)
    gpu.synchronize()
    let before = gpu.live_buffers()
    let again = greedy_cached(model, prompt, count)
    gpu.synchronize()
    let decode_drift = gpu.live_buffers() - before
    var same = 0
    for i in 0..count {
        if cached[i] == uncached[i] { same += 1 }
    }
    print(f"cached:   {repr(bytes(cached))}")
    print(f"uncached: {repr(bytes(uncached))}")
    expect(
        same == count and again == cached,
        f"cached greedy decoding matches uncached ({same} of {count} bytes)"
    )
    expect(decode_drift == 0, f"cached decoding leaks no device buffers (drift {decode_drift})")
    model.free_cache()

    #: A checkpoint round trip changes no logit.
    let path = Path(os.temp_dir()).join("source_gpt_check.ckpt").to_str()
    jt.save_checkpoint(path, model.names(), params, meta: checkpoint_meta(cfg, steps))
    let restored = load_model(path)
    restored.set_training(false)
    model.start_cache(cfg.seq)
    restored.start_cache(cfg.seq)
    let idx = Tensor([16], [float(t) for t in prompt])
    let a = model.forward_cached(idx, 0)
    let b = restored.forward_cached(idx, 0)
    expect(a.download() == b.download(), "a saved and loaded checkpoint gives the same logits")
    free_all([idx, a, b])
    model.free_cache()
    restored.free_cache()
    restored.free()
    Path(path).remove()
    model.free()
    gpu.synchronize()
    if failures.len() > 0 {
        print(f"{failures.len()} check(s) failed")
        return 1
    }
    print("all checks passed")
    return 0
}
