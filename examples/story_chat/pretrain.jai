#: Pretrain the GPT on TinyStories.
#:
#:     jaithon run examples/story_chat/pretrain.jai -- --steps 20000 --minutes 5 --resume
#:
#: Reads the token files `prepare.jai` wrote. Random windows of the train
#: split, AdamW (decay on the matrices only), a linear warm-up of `WARMUP`
#: (200) steps to `LR` (6e-4) and a cosine decay to a tenth of it at the last
#: step, the gradient clipped to a norm of `CLIP` (1.0).
#:
#: `--steps N` is the length of the whole schedule; `--minutes M` stops this
#: run after M minutes of training and saves where it got to, and `--resume`
#: carries on from that checkpoint -- weights, optimiser moments, step and
#: tokens seen -- so a long run can be split into chunks that each hold the
#: GPU only for a few minutes. Every `EVAL_EVERY` (250) steps it reports the
#: held-out loss and bits per byte on `EVAL_WINDOWS` (256) fixed windows of
#: the valid split, prints a story from a fixed prompt, and appends a line to
#: `pretrain_log.tsv` in the cache. It checkpoints every `CKPT_EVERY` (1000)
#: steps and at the end, to `PRETRAIN_CKPT` (default
#: `~/.cache/jaithon/story_chat/pretrain.ckpt`).
#:
#: `MODE=bench` times the step instead: `WARM` (3) seconds of steps by the
#: wall clock, then `REPS` (5) samples of `BENCH_STEPS` (5) steps, and reports
#: tokens a second.
import jaitensor as jt
import std.gpu as gpu
import std.math
import std.os
from jaitensor import SparseCategoricalCrossentropy, Tensor
from std.io import Path, append_file
from std.random import Random
from std.time import Instant
from .model import FLUSH, Config, GPT, Optimisers, Sampler, TokenFile, checkpoint_meta, config_from_env
from .model import config_of, drop_grads, env_float, env_int, free_all, held_out_loss, held_out_starts
from .model import pretrain_checkpoint, read_token_file, target_bytes, token_path, tokenizer_path
from .sample import Decoder, Sampling, generate
from .tokenizer import EOT, Tokenizer, cache_dir, load_tokenizer, text_of

#: The flag values after `--`.
pub fn flag(name: str, fallback: str) -> str {
    let args = os.argv()
    for (i, a) in args.enumerate() {
        if a == name {
            if i + 1 < args.len() and not args[i + 1].starts_with("--") { return args[i + 1] }
            return "1"
        }
    }
    return fallback
}

pub fn has_flag(name: str) -> bool { return os.argv().contains(name) }

#: Warm-up then cosine decay to `floor * peak` at `total`.
pub fn learning_rate(step: int, total: int, peak: float, warmup: int, floor: float) -> float {
    if step < warmup { return peak * float(step + 1) / float(warmup) }
    let progress = float(step - warmup) / float(max(1, total - warmup))
    let p = min(1.0, progress)
    return peak * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.PI * p)))
}

pub class Trainer {
    pub let model: GPT
    pub let opt: Optimisers
    pub let params: list[Tensor]
    pub let ce: SparseCategoricalCrossentropy
    pub let clip: float

    pub fn init(self, model: GPT, lr: float, weight_decay: float, clip: float) {
        self.model = model
        self.opt = Optimisers(model, lr, weight_decay)
        self.params = model.params()
        self.ce = SparseCategoricalCrossentropy(from_logits: true)
        self.clip = clip
    }

    #: One optimisation step on `(x, y)`, which it frees; returns the loss as
    #: a tensor the caller downloads and frees, so the device can run ahead.
    #: `rows`, when given, picks the logits rows the loss is over.
    pub fn step(self, x: Tensor, y: Tensor, lr: float, rows: Tensor? = null) -> Tensor {
        self.opt.set_learning_rate(lr)
        let logits = self.model.forward(x, rows: rows)
        let loss = self.ce.forward(logits, y)
        if FLUSH >= 1 { gpu.flush() }
        let held = loss.backward()
        if self.clip > 0.0 {
            let norm = jt.clip_grad_norm(self.params, self.clip)
            norm.free()
        }
        self.opt.step()
        drop_grads(self.params)
        free_all(held)
        free_all([logits, x, y])
        if rows is not null { rows.free() }
        return loss
    }

    #: Save weights, optimiser moments and `meta` to `path`, through a
    #: temporary file so an interrupted save never leaves half a checkpoint.
    pub fn save(self, path: str, meta: dict[str, any]) -> void {
        var names = self.model.names()
        var tensors = self.params.copy()
        let (onames, otensors) = self.opt.state()
        for n in onames { names.push(n) }
        for t in otensors { tensors.push(t) }
        let partial = path + ".partial"
        jt.save_checkpoint(partial, names, tensors, meta: meta)
        if Path(path).exists() { Path(path).remove() }
        Path(partial).rename(Path(path))
    }

    #: Carry on from a checkpoint `save` wrote; returns its metadata.
    pub fn restore(self, path: str) -> dict[str, any] {
        var names = self.model.names()
        var tensors = self.params.copy()
        let (onames, otensors) = self.opt.state()
        for n in onames { names.push(n) }
        for t in otensors { tensors.push(t) }
        let meta = jt.restore_checkpoint(path, names, tensors)
        let step: int = meta["step"]
        self.opt.set_steps(step)
        return meta
    }
}

#: A story from the fixed prompt, through the cache.
pub fn sample_story(model: GPT, tok: Tokenizer, prompt: str, count: int, seed: int) -> str {
    model.set_training(false)
    let decoder = Decoder(model)
    var ids = [EOT]
    for t in tok.encode(prompt) { ids.push(t) }
    let out = generate(
        decoder,
        ids,
        count,
        Sampling(temperature: 0.8, top_k: 40, top_p: 0.95, repetition_penalty: 1.0),
        Random.with_seed(seed),
        [EOT],
        fn(_t: int) -> void {}
    )
    decoder.free()
    model.set_training(true)
    return prompt + tok.decode(out)
}

fn median(xs: list[float]) -> float {
    let s = xs.sorted()
    return s[s.len() // 2]
}

fn main() -> int {
    let cfg = config_from_env()
    let tok = load_tokenizer(tokenizer_path(cfg.vocab))
    if tok.vocab_size() != cfg.vocab {
        throw RuntimeError(f"tokenizer has {tok.vocab_size()} ids, the model wants {cfg.vocab}")
    }
    let t_read = Instant.now()
    let train_file = read_token_file(token_path("train", cfg.vocab))
    let valid_file = read_token_file(token_path("valid", cfg.vocab))
    print(
        f"tokens: train {train_file.count}, valid {valid_file.count}; read in {t_read.elapsed().to_seconds():.1f} s"
    )
    let peak = env_float("LR", 6.0e-4)
    let total = int(flag("--steps", os.env_or("STEPS", "20000")))
    let minutes = float(flag("--minutes", os.env_or("MINUTES", "0")))
    let warmup = env_int("WARMUP", 200)
    let floor = env_float("MIN_LR_FRAC", 0.1)
    let seed = env_int("SEED", 7)
    let path = pretrain_checkpoint()
    let model = GPT(cfg, seed: seed)
    let trainer = Trainer(model, peak, env_float("WD", 0.1), env_float("CLIP", 1.0))
    var step = 0
    var tokens_seen = 0
    var train_seconds = 0.0
    if has_flag("--resume") and Path(path).exists() {
        let meta = trainer.restore(path)
        let at: int = meta["step"]
        let seen: int = meta["tokens"]
        let secs: float = meta["train_seconds"]
        step = at
        tokens_seen = seen
        train_seconds = secs
        print(f"resumed {path} at step {step}, {tokens_seen} tokens seen")
    }
    print(f"model: {cfg}, {model.parameter_count()} parameters; schedule {total} steps")
    let sampler = Sampler(train_file, seed * 1000003 + step)
    if os.env_or("MODE", "train") == "bench" {
        run_bench(trainer, sampler, cfg)
        model.free()
        return 0
    }
    let every = env_int("EVAL_EVERY", 250)
    let ckpt_every = env_int("CKPT_EVERY", 1000)
    let log_every = env_int("LOG_EVERY", 50)
    let held = held_out_starts(valid_file, cfg.seq, env_int("EVAL_WINDOWS", 256))
    let held_bytes = target_bytes(valid_file, tok, held, cfg.seq)
    let held_tokens = held.len() // cfg.batch * cfg.batch * cfg.seq
    let bytes_per_token = float(held_bytes) / float(held.len() * cfg.seq)
    let log_path = cache_dir() + "/pretrain_log.tsv"
    if step == 0 {
        append_file(log_path, "step\ttokens\ttrain_loss\tval_loss\tval_bpb\ttrain_seconds\n")
    }
    let prompt = os.env_or("SAMPLE_PROMPT", "Once upon a time")
    let per_step = cfg.batch * cfg.seq
    var recent = 0.0
    var recent_n = 0
    var last_train = 0.0
    var live = -1
    let t0 = Instant.now()
    let start_seconds = train_seconds
    let started_at = step
    var pending: Tensor? = null
    while step < total {
        if minutes > 0.0 and t0.elapsed().to_seconds() >= minutes * 60.0 { break }
        let lr = learning_rate(step, total, peak, warmup, floor)
        let (x, y) = sampler.next(cfg.batch, cfg.seq)
        let loss = trainer.step(x, y, lr)
        #: Download the previous step's loss, not this one's: the device is
        #: still working on this step while the host reads the last.
        let before = pending
        if before is not null {
            recent += before.download()[0]
            recent_n += 1
            before.free()
        }
        pending = loss
        step += 1
        tokens_seen += per_step
        if step == started_at + 10 {
            gpu.synchronize()
            live = gpu.live_buffers() - 1
        }
        if step % log_every == 0 and recent_n > 0 {
            let s = start_seconds + t0.elapsed().to_seconds()
            last_train = recent / float(recent_n)
            let rate = float((step - started_at) * per_step) / t0.elapsed().to_seconds()
            print(f"step {step} loss {last_train:.4f} lr {lr:.2e} ({s:.0f} s, {rate:.0f} tokens/s)")
            recent = 0.0
            recent_n = 0
        }
        let at_end = step == total
        if step % every == 0 or at_end {
            let p = pending
            if p is not null {
                p.free()
                pending = null
            }
            let te = Instant.now()
            let val = held_out_loss(model, valid_file, held)
            let bpb = val * float(held.len() * cfg.seq) / float(held_bytes) / math.log(2.0)
            let s = start_seconds + t0.elapsed().to_seconds()
            print(
                f"step {step} held-out loss {val:.4f} ({bpb:.4f} bits/byte, {bytes_per_token:.2f} bytes/token) in {te.elapsed().to_seconds():.1f} s"
            )
            append_file(
                log_path,
                f"{step}\t{tokens_seen}\t{last_train:.4f}\t{val:.4f}\t{bpb:.4f}\t{s:.0f}\n"
            )
            let story = sample_story(model, tok, prompt, 80, 1)
            print(f"  sample: {story.replace("\n", " / ")}")
        }
        if step % ckpt_every == 0 and not at_end {
            trainer.save(
                path,
                checkpoint_meta(
                    cfg,
                    step,
                    {
                        "tokens": tokens_seen,
                        "total_steps": total,
                        "train_seconds": start_seconds + t0.elapsed().to_seconds(),
                    }
                )
            )
        }
    }
    let p = pending
    if p is not null { p.free() }
    gpu.synchronize()
    let wall = t0.elapsed().to_seconds()
    #: Both counts leave out the one loss still waiting to be read.
    if live >= 0 { print(f"live_buffers_drift={gpu.live_buffers() - live}") }
    let rate = float((step - started_at) * per_step) / max(wall, 1.0e-9)
    print(
        f"ran steps {started_at}..{step} in {wall:.1f} s ({rate:.0f} tokens/s with evaluation); {tokens_seen} tokens seen in all"
    )
    let ts = Instant.now()
    trainer.save(
        path,
        checkpoint_meta(
            cfg,
            step,
            {"tokens": tokens_seen, "total_steps": total, "train_seconds": start_seconds + wall}
        )
    )
    print(f"checkpoint {path} at step {step} ({ts.elapsed().to_seconds():.1f} s)")
    model.free()
    gpu.synchronize()
    return 0
}

fn run_bench(trainer: Trainer, sampler: Sampler, cfg: Config) -> void {
    let lr = 1.0e-4
    let t0 = Instant.now()
    let (x0, y0) = sampler.next(cfg.batch, cfg.seq)
    let l0 = trainer.step(x0, y0, lr)
    let loss0 = l0.download()[0]
    l0.free()
    print(f"first_step_ms={t0.elapsed().to_millis():.1f} loss0={loss0:.4f}")
    let warm = env_float("WARM", 3.0)
    let tw = Instant.now()
    var warm_steps = 0
    while tw.elapsed().to_seconds() < warm {
        let (x, y) = sampler.next(cfg.batch, cfg.seq)
        let l = trainer.step(x, y, lr)
        let _v = l.download()
        l.free()
        warm_steps += 1
    }
    gpu.synchronize()
    let live = gpu.live_buffers()
    let reps = env_int("REPS", 5)
    let steps = env_int("BENCH_STEPS", 5)
    var times: list[float] = []
    var last = 0.0
    for _r in 0..reps {
        let ts = Instant.now()
        for _s in 0..steps {
            let (x, y) = sampler.next(cfg.batch, cfg.seq)
            let l = trainer.step(x, y, lr)
            last = l.download()[0]
            l.free()
        }
        times.push(ts.elapsed().to_millis() / float(steps))
    }
    gpu.synchronize()
    let s = times.sorted()
    let tokens = float(cfg.batch * cfg.seq)
    print(
        f"train_ms_per_step median={median(times):.1f} min={s[0]:.1f} max={s[s.len() - 1]:.1f} n={reps} (x{steps} steps, after {warm_steps} warm steps)"
    )
    print(f"tokens_per_s={tokens / median(times) * 1000.0:.0f} loss_last={last:.4f}")
    print(f"live_buffers_drift={gpu.live_buffers() - live}")
}
