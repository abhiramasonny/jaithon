#: Write Jaithon with the GPT that `train.jai` saved.
#:
#:     jaithon run examples/source_gpt/generate.jai -- "fn parse_"
#:
#: Loads `GPT_CKPT` (default `$TMPDIR/source_gpt.ckpt`), feeds the prompt
#: through the attention cache once, then samples `NEW` (600) bytes one at a
#: time at temperature `TEMP` (0.8), keeping the `TOPK` (40) likeliest bytes.
#: Each byte costs the same whatever its position: the keys and values of
#: everything before it are already in the cache. The context is the model's
#: 128 bytes; past that the last 64 are fed through again and decoding goes on.
#:
#: `MODE=bench` times greedy batch-1 decoding instead (`WARM` seconds, then
#: `REPS` samples of `NEW` bytes) and reports bytes per second.
#: `MODE=checker` samples `SAMPLES` (50) top-level functions -- all at once, as
#: one batch through a batched cache -- writes each to a file, and reports how
#: many `jaithon check` accepts.
import jaitensor as jt
import std.gpu as gpu
import std.math
import std.os
from jaitensor import Tensor
from std.io import Path, write_file
from std.random import Random
from std.time import Instant
from .model import GPT, default_checkpoint, env_float, env_int, load_model

#: Pick a byte from one row of logits: greedy at temperature 0, otherwise a
#: draw from the softmax of the `top_k` largest at that temperature.
fn pick(row: list[float], temperature: float, top_k: int, rng: Random) -> int {
    for value in row {
        if math.is_nan(value) {
            #: Comparing NaN throws here. The fused GELU product epilogue gave
            #: NaN from a pre-activation of about 20 until gemm_tanh clamped
            #: its argument (thin-gemms-fused-batchnorm); say so plainly.
            throw RuntimeError(
                "the model produced NaN logits; this needs the GEMM epilogue tanh clamp"
            )
        }
    }
    var best = 0
    for i in 1..row.len() {
        if row[i] > row[best] { best = i }
    }
    if temperature <= 0.0 { return best }
    var floor = -1.0e30
    if top_k > 0 and top_k < row.len() {
        let ordered = row.sorted()
        floor = ordered[row.len() - top_k]
    }
    var weights: list[float] = []
    var total = 0.0
    for value in row {
        let w = if value >= floor { math.exp((value - row[best]) / temperature) } else { 0.0 }
        weights.push(w)
        total += w
    }
    var target = rng.next_float() * total
    for (i, w) in weights.enumerate() {
        target -= w
        if target <= 0.0 and w > 0.0 { return i }
    }
    return best
}

#: Bytes as the float indices the model reads.
fn byte_tensor(values: list[int]) -> Tensor {
    return Tensor([values.len()], [float(v) for v in values])
}

#: Decode `count` bytes after `prompt` through the cache. `on_byte` sees each
#: one as it is chosen; the result is every byte, prompt first.
fn decode(
    model: GPT,
    prompt: list[int],
    count: int,
    temperature: float,
    top_k: int,
    rng: Random
) -> list[int] {
    let window = model.cfg.seq
    let keep = window // 2
    var out = prompt.copy()
    model.reset_cache()
    var context = if prompt.len() > window { prompt[prompt.len() - window:] } else { prompt.copy() }
    var at = 0
    var idx = byte_tensor(context)
    for _n in 0..count {
        let logits = model.forward_cached(idx, at)
        idx.free()
        at += context.len()
        let next = pick(logits.download(), temperature, top_k, rng)
        logits.free()
        out.push(next)
        if at >= window {
            #: The positions are used up: start again from the recent half.
            model.reset_cache()
            context = out[out.len() - keep:]
            at = 0
        } else {
            context = [next]
        }
        idx = byte_tensor(context)
    }
    idx.free()
    return out
}

fn text_of(values: list[int]) -> str {
    return bytes(values).decode()
}

fn median(xs: list[float]) -> float {
    let s = xs.sorted()
    return s[s.len() // 2]
}

#: Greedy batch-1 decoding, timed. The prompt is the first 16 bytes of the
#: held-out jaiyaml parser, as in the torch peer.
fn bench(model: GPT) -> void {
    let prompt = [int(b) for b in "fn parse_scalar(".bytes()]
    let count = env_int("NEW", 112)
    let warm = env_float("WARM", 2.0)
    let reps = env_int("REPS", 5)
    let rng = Random.with_seed(1)
    var per_byte: list[float] = []
    var first: list[float] = []
    let tw = Instant.now()
    while tw.elapsed().to_seconds() < warm { let _o = decode(model, prompt, count, 0.0, 0, rng) }
    var produced: list[int] = []
    for _r in 0..reps {
        model.reset_cache()
        let tp = Instant.now()
        let idx = byte_tensor(prompt)
        let logits = model.forward_cached(idx, 0)
        idx.free()
        var tok = pick(logits.download(), 0.0, 0, rng)
        logits.free()
        first.push(tp.elapsed().to_millis())
        let td = Instant.now()
        for step in 1..count {
            let one = byte_tensor([tok])
            let lg = model.forward_cached(one, prompt.len() + step - 1)
            one.free()
            tok = pick(lg.download(), 0.0, 0, rng)
            lg.free()
            if step < 12 { produced.push(tok) }
        }
        per_byte.push(td.elapsed().to_millis() / float(count - 1))
    }
    let s = per_byte.sorted()
    print(
        f"gen_cached_ms_per_byte median={median(per_byte):.3f} min={s[0]:.3f} max={s[s.len() - 1]:.3f} n={reps}"
    )
    print(f"prefill_ms median={median(first):.2f}")
    print(f"bytes_per_s={1000.0 / median(per_byte):.0f}")
}

#: Sample many top-level functions at once through a batched cache and ask
#: the checker about each.
fn checker(model: GPT) -> void {
    let samples = env_int("SAMPLES", 50)
    let count = env_int("NEW", 600)
    let temperature = env_float("TEMP", 0.8)
    let top_k = env_int("TOPK", 40)
    let rng = Random.with_seed(env_int("SEED", 3))
    let prompt = [int(b) for b in "\nfn ".bytes()]
    let window = model.cfg.seq
    let keep = window // 2
    model.start_cache(window, batch: samples)
    var streams: list[list[int]] = [prompt.copy() for _ in 0..samples]
    var done: list[bool] = [false for _ in 0..samples]
    var context_len = prompt.len()
    var at = 0
    let t0 = Instant.now()
    var steps = 0
    while steps < count {
        var flat: list[float] = []
        for t in 0..context_len {
            for b in 0..samples {
                let stream = streams[b]
                flat.push(float(stream[stream.len() - context_len + t]))
            }
        }
        let idx = Tensor([context_len * samples], flat)
        let logits = model.forward_cached(idx, at, batch: samples)
        idx.free()
        at += context_len
        let values = logits.download()
        logits.free()
        var all_done = true
        for b in 0..samples {
            let row = values[b * 256:(b + 1) * 256]
            let next = pick(row, temperature, top_k, rng)
            streams[b].push(next)
            if not done[b] {
                let s = streams[b]
                let n = s.len()
                #: A top-level body ends at a closing brace in column zero.
                if n > 3 and s[n - 1] == 10 and s[n - 2] == 125 and s[n - 3] == 10 {
                    done[b] = true
                }
            }
            if not done[b] { all_done = false }
        }
        steps += 1
        if all_done { break }
        if at >= window {
            model.reset_cache()
            context_len = keep
            at = 0
        } else {
            context_len = 1
        }
    }
    let gen_s = t0.elapsed().to_seconds()
    model.free_cache()
    let dir = Path(os.temp_dir()).join("source_gpt_samples")
    dir.mkdir(true, true)
    let jaithon = os.env_or("JAITHON", "./jaithon")
    var passed = 0
    var finished = 0
    for (b, stream) in streams.enumerate() {
        var body = stream[1:]
        var end = body.len()
        for i in 2..body.len() {
            if body[i] == 10 and body[i - 1] == 125 and body[i - 2] == 10 {
                end = i + 1
                break
            }
        }
        if done[b] { finished += 1 }
        let path = dir.join(f"sample_{b}.jai").to_str()
        var text = ""
        try {
            text = text_of(body[0:end])
        } catch _e: Error {
            text = "!! not UTF-8"
        }
        write_file(path, text)
        let result = os.run([jaithon, "check", path])
        if result.exit_code == 0 { passed += 1 }
        if b < env_int("SHOW", 2) { print(f"--- sample {b} ({result.exit_code == 0}) ---\n{text}") }
    }
    print(
        f"sampled {samples} functions, {steps} bytes each at most, in {gen_s:.1f} s; {finished} closed their body"
    )
    print(
        f"jaithon check accepted {passed} of {samples} ({100.0 * float(passed) / float(samples):.0f}%); files in {dir.to_str()}"
    )
}

fn main() -> int {
    let path = default_checkpoint()
    let tl = Instant.now()
    let model = load_model(path)
    gpu.synchronize()
    print(
        f"loaded {path}: {model.parameter_count()} parameters in {tl.elapsed().to_millis():.0f} ms"
    )
    model.set_training(false)
    let mode = os.env_or("MODE", "sample")
    if mode == "bench" {
        model.start_cache(model.cfg.seq)
        bench(model)
    } elif mode == "checker" {
        checker(model)
    } else {
        model.start_cache(model.cfg.seq)
        let args = os.argv()
        let prompt_text = if args.len() > 1 { args[args.len() - 1] } else { "fn main() -> int {\n" }
        let prompt = [int(b) for b in prompt_text.bytes()]
        let rng = Random.with_seed(env_int("SEED", 1))
        let t0 = Instant.now()
        let count = env_int("NEW", 600)
        let out = decode(model, prompt, count, env_float("TEMP", 0.8), env_int("TOPK", 40), rng)
        let ms = t0.elapsed().to_millis()
        var text = ""
        try {
            text = text_of(out)
        } catch _e: Error {
            text = text_of([b for b in out if b < 128])
        }
        print(text)
        print(f"--- {count} bytes in {ms:.0f} ms ({float(count) / ms * 1000.0:.0f} bytes/s)")
    }
    model.free_cache()
    model.free()
    gpu.synchronize()
    return 0
}
