#: Train a byte-level GPT on this repository's own Jaithon source.
#:
#:     jaithon run examples/source_gpt/train.jai
#:
#: Run it from the repository root. It reads `lib/` and `packages/*/src/`,
#: holds jaiyaml and jaitoml back, trains for `STEPS` (2000) steps, reports
#: held-out bits per byte as it goes, and saves a binary checkpoint to
#: `GPT_CKPT` (default `$TMPDIR/source_gpt.ckpt`) for `generate.jai`.
#:
#: `MODE=bench` times the step instead: the first step, `WARM` (2) seconds of
#: steps by the wall clock, then `REPS` (7) samples of `STEPS` (5) steps each.
#: `LR` (1e-3) and `SEED` (7) set the rest. Both modes download the loss every
#: step, as the torch peer `source_gpt.py` does.
import jaitensor as jt
import std.gpu as gpu
import std.math
import std.os
from jaitensor import Adam, SparseCategoricalCrossentropy, Tensor
from std.time import Instant
from .model import FLUSH, Config, Corpus, GPT, Sampler, checkpoint_meta, config_from_env
from .model import default_checkpoint, drop_grads, env_float, env_int, free_all, held_out_bits
from .model import held_out_windows, source_files

#: One optimisation step; returns the loss, downloaded.
fn train_step(
    model: GPT,
    opt: Adam,
    ce: SparseCategoricalCrossentropy,
    params: list[Tensor],
    sampler: Sampler
) -> float {
    let cfg = model.cfg
    let started = Instant.now()
    let (x, y) = sampler.next(cfg.batch, cfg.seq)
    BATCH_MS.push(started.elapsed().to_millis())
    let logits = model.forward(x)
    let loss = ce.forward(logits, y)
    if FLUSH >= 1 { gpu.flush() }
    let held = loss.backward()
    opt.step(params)
    drop_grads(params)
    free_all(held)
    free_all([logits, x, y])
    ENCODE_MS.push(started.elapsed().to_millis())
    let value = loss.download()[0]
    loss.free()
    return value
}

#: Host time per step: building the batch, and everything up to the loss
#: download (the encode the device waits on).
var BATCH_MS: list[float] = []
var ENCODE_MS: list[float] = []

fn median(xs: list[float]) -> float {
    let s = xs.sorted()
    return s[s.len() // 2]
}

fn spread(label: str, xs: list[float]) -> str {
    let s = xs.sorted()
    return f"{label} median={median(xs):.2f} min={s[0]:.2f} max={s[s.len() - 1]:.2f} n={xs.len()}"
}

fn bench(model: GPT, opt: Adam, ce: SparseCategoricalCrossentropy, sampler: Sampler) -> void {
    let params = model.params()
    let t0 = Instant.now()
    let loss0 = train_step(model, opt, ce, params, sampler)
    let first = t0.elapsed().to_millis()
    let t1 = Instant.now()
    let _second = train_step(model, opt, ce, params, sampler)
    print(
        f"first_step_ms={first:.1f} second_step_ms={t1.elapsed().to_millis():.1f} loss0={loss0:.4f}"
    )
    let warm = env_float("WARM", 2.0)
    let tw = Instant.now()
    var warm_steps = 0
    while tw.elapsed().to_seconds() < warm {
        let _l = train_step(model, opt, ce, params, sampler)
        warm_steps += 1
    }
    gpu.synchronize()
    let live = gpu.live_buffers()
    let reps = env_int("REPS", 7)
    let steps = env_int("STEPS", 5)
    var times: list[float] = []
    var last = 0.0
    BATCH_MS.clear()
    ENCODE_MS.clear()
    for _r in 0..reps {
        let ts = Instant.now()
        for _s in 0..steps { last = train_step(model, opt, ce, params, sampler) }
        times.push(ts.elapsed().to_millis() / float(steps))
    }
    gpu.synchronize()
    let tokens = float(model.cfg.batch * model.cfg.seq)
    print(spread("train_ms_per_step", times))
    print(
        f"tokens_per_s={tokens / median(times) * 1000.0:.0f} warm_steps={warm_steps} loss_last={last:.4f}"
    )
    print(f"live_buffers_drift={gpu.live_buffers() - live}")
    print(spread("host_batch_ms", BATCH_MS))
    print(spread("host_encode_ms", ENCODE_MS))
}

fn train(
    model: GPT,
    opt: Adam,
    ce: SparseCategoricalCrossentropy,
    sampler: Sampler,
    held: Corpus
) -> void {
    let cfg = model.cfg
    let params = model.params()
    let picks = held_out_windows(held, cfg.seq)
    let steps = env_int("STEPS", 2000)
    let every = env_int("EVAL_EVERY", 250)
    let eval_batches = env_int("EVAL_BATCHES", 8)
    let log_every = env_int("LOG_EVERY", 100)
    print(
        f"held out: {picks.len()} windows of {cfg.seq}; evaluating {eval_batches} batches every {every} steps"
    )
    var eval_ms = 0.0
    var recent = 0.0
    var live = -1
    let t0 = Instant.now()
    for step in 1..steps + 1 {
        let loss = train_step(model, opt, ce, params, sampler)
        recent += loss
        if step == 1 { print(f"step 1 loss {loss:.4f} ({t0.elapsed().to_millis():.0f} ms)") }
        if step % log_every == 0 {
            let s = t0.elapsed().to_seconds()
            print(f"step {step} loss {recent / float(log_every):.4f} ({s:.1f} s)")
            recent = 0.0
        }
        if step % every == 0 and step < steps {
            let te = Instant.now()
            let bits = held_out_bits(model, held, picks, limit: eval_batches)
            eval_ms += te.elapsed().to_millis()
            print(f"step {step} held-out {bits:.4f} bits/byte")
        }
        if step == 10 {
            gpu.synchronize()
            live = gpu.live_buffers()
        }
    }
    gpu.synchronize()
    let wall = t0.elapsed().to_seconds()
    print(
        f"trained {steps} steps in {wall:.1f} s ({(wall * 1000.0 - eval_ms) / float(steps):.2f} ms/step outside {eval_ms / 1000.0:.1f} s of evaluation)"
    )
    print(f"live_buffers_drift={gpu.live_buffers() - live}")
    let tf = Instant.now()
    let bits = held_out_bits(model, held, picks)
    print(
        f"final held-out {bits:.4f} bits/byte over {picks.len() // cfg.batch * cfg.batch} windows ({tf.elapsed().to_seconds():.1f} s)"
    )
    let path = default_checkpoint()
    let ts = Instant.now()
    jt.save_checkpoint(path, model.names(), params, meta: checkpoint_meta(cfg, steps))
    let save_ms = ts.elapsed().to_millis()
    let tl = Instant.now()
    let _meta = jt.restore_checkpoint(path, model.names(), params)
    gpu.synchronize()
    let load_ms = tl.elapsed().to_millis()
    print(
        f"checkpoint {path}: save {save_ms:.0f} ms, load {load_ms:.0f} ms, {model.parameter_count()} parameters"
    )
}

fn main() -> int {
    let cfg = config_from_env()
    let t_read = Instant.now()
    let train_set = Corpus(".", source_files(".", false), cfg.seq)
    let held = Corpus(".", source_files(".", true), cfg.seq)
    if train_set.files.len() == 0 {
        throw RuntimeError("no source found: run this from the repository root")
    }
    print(
        f"corpus: {train_set.files.len()} files, {train_set.size} bytes; held out {held.files.len()} files, {held.size} bytes; read in {t_read.elapsed().to_millis():.0f} ms"
    )
    let model = GPT(cfg)
    print(f"model: {cfg}, {model.parameter_count()} parameters")
    let opt = Adam(learning_rate: env_float("LR", 1.0e-3), epsilon: 1.0e-7)
    let ce = SparseCategoricalCrossentropy(from_logits: true)
    let sampler = Sampler(train_set, env_int("SEED", 7))
    if os.env_or("MODE", "train") == "bench" {
        bench(model, opt, ce, sampler)
    } else {
        train(model, opt, ce, sampler, held)
    }
    model.free()
    gpu.synchronize()
    return 0
}
