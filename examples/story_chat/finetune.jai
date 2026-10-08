#: Fine-tune the pretrained GPT into a chat model on DailyDialog.
#:
#:     jaithon run examples/story_chat/finetune.jai -- --epochs 3
#:
#: Starts from `PRETRAIN_CKPT`, formats every dialogue as alternating
#: `<|user|>` / `<|bot|>` turns (each speaker taking a turn as the bot), and
#: trains on the bot's tokens only: the hidden states at those positions are
#: gathered with `gather_rows` before the output head, so the loss is the
#: ordinary cross-entropy over just them. AdamW at half the pretraining
#: rate, `LR` (3e-4; 1e-4 reached 3.235 after two epochs where 3e-4 reached
#: 2.979), after `WARMUP` (50) steps, cosine to a tenth over all `--epochs`
#: (3).
#: Reports the masked loss on `dialogues_validation` every epoch and saves the
#: best to `CHAT_CKPT` (default `~/.cache/jaithon/story_chat/chat.ckpt`,
#: weights only), and the latest, optimiser and all, beside it as
#: `chat.ckpt.state`.
#: `--one-epoch` stops after an epoch (exit status 10 while epochs remain) and
#: `--resume` carries on from the latest, so `run_full.sh` can give each epoch
#: its own turn on the GPU.
import std.gpu as gpu
import std.math
import std.os
from jaitensor import SparseCategoricalCrossentropy, Tensor
from std.io import Path, append_file
from std.random import Random
from std.time import Instant
from .dialog import Batch, Row, chat_prompt, epoch_batches, split_rows
from .model import GPT, checkpoint_meta, chat_checkpoint, env_float, env_int, free_all, load_model
from .model import pretrain_checkpoint, run_dir, tokenizer_path
from .pretrain import Trainer, flag, has_flag, learning_rate
from .sample import Decoder, Sampling, generate
from .tokenizer import BOT, EOT, USER, Tokenizer, dialog_file, load_tokenizer, read_dialogues

#: Mean loss over every target of `rows`, in nats a token.
pub fn masked_loss(model: GPT, rows: list[Row], batch: int, quantum: int) -> float {
    let ce = SparseCategoricalCrossentropy(from_logits: true)
    var total = 0.0
    var count = 0
    for (length, picked) in epoch_batches(rows, batch, model.cfg.seq, quantum, null) {
        let b = Batch(picked, length)
        let logits = model.forward(b.x, rows: b.rows)
        let loss = ce.forward(logits, b.y)
        total += loss.download()[0] * float(b.targets)
        count += b.targets
        free_all(loss.discard_graph())
        free_all([loss, logits, b.x, b.rows, b.y])
    }
    return total / float(max(count, 1))
}

#: The bot's reply to `turns`, sampled through the cache.
pub fn reply(
    model: GPT,
    tok: Tokenizer,
    turns: list[str],
    s: Sampling,
    rng: Random,
    max_new: int
) -> str {
    let decoder = Decoder(model)
    let prompt = chat_prompt(tok, turns, model.cfg.seq - max_new)
    let out = generate(decoder, prompt, max_new, s, rng, [USER, EOT, BOT], fn(_t: int) -> void {})
    decoder.free()
    return tok.decode(out).strip()
}

let PROBES = [
    ["Hi! How are you today?"],
    ["What do you like to do on weekends?"],
    ["I'm looking for a cheap hotel near the station."],
]

fn show_replies(model: GPT, tok: Tokenizer) -> void {
    model.set_training(false)
    let rng = Random.with_seed(1)
    for turns in PROBES {
        let text = reply(
            model,
            tok,
            turns,
            Sampling(temperature: 0.7, top_k: 40, top_p: 0.9),
            rng,
            48
        )
        print(f"  user: {turns[0]}\n  bot:  {text}")
    }
    model.set_training(true)
}

#: `--eval`: the masked loss of the saved chat model on the validation and
#: test dialogues, which training never reads.
fn evaluate() -> int {
    let batch = env_int("FT_BATCH", 32)
    let (model, meta) = load_model(chat_checkpoint(), batch: batch)
    let tok = load_tokenizer(tokenizer_path(model.cfg.vocab))
    let quantum = env_int("QUANTUM", 64)
    let epoch: int = meta["finetune_epoch"]
    for split in ["validation", "test"] {
        let rows = split_rows(tok, read_dialogues(dialog_file(split)), model.cfg.seq)
        let loss = masked_loss(model, rows, batch, quantum)
        print(
            f"chat model (fine-tuned {epoch} epochs) on dialogues_{split}: {rows.len()} rows, masked loss {loss:.4f}, perplexity {math.exp(loss):.2f}"
        )
    }
    model.free()
    return 0
}

fn main() -> int {
    if has_flag("--eval") { return evaluate() }
    let state_path = chat_checkpoint() + ".state"
    let resuming = has_flag("--resume") and Path(state_path).exists()
    let source = if resuming { state_path } else { os.env_or("FROM_CKPT", pretrain_checkpoint()) }
    let tl = Instant.now()
    let batch = env_int("FT_BATCH", 32)
    let (model, meta) = load_model(source, batch: batch)
    let cfg = model.cfg
    let tok = load_tokenizer(tokenizer_path(cfg.vocab))
    let pre_step: int = meta["step"]
    print(
        f"loaded {source} (pretrained {pre_step} steps): {cfg}, {model.parameter_count()} parameters, {tl.elapsed().to_seconds():.1f} s"
    )
    let quantum = env_int("QUANTUM", 64)
    let tr = Instant.now()
    let train_rows = split_rows(tok, read_dialogues(dialog_file("train")), cfg.seq)
    let valid_rows = split_rows(tok, read_dialogues(dialog_file("validation")), cfg.seq)
    var targets = 0
    var tokens = 0
    for r in train_rows {
        targets += r.targets()
        tokens += r.ids.len()
    }
    print(
        f"dialogues: {train_rows.len()} train rows ({tokens} tokens, {targets} scored), {valid_rows.len()} validation rows; {tr.elapsed().to_seconds():.1f} s"
    )
    var epochs = int(flag("--epochs", os.env_or("EPOCHS", "3")))
    let one_epoch = has_flag("--one-epoch")
    let seed = env_int("SEED", 11)
    let per_epoch = epoch_batches(train_rows, batch, cfg.seq, quantum, null).len()
    let peak = env_float("LR", 3.0e-4)
    let warmup = env_int("WARMUP", 50)
    let trainer = Trainer(model, peak, env_float("WD", 0.1), env_float("CLIP", 1.0))
    let path = chat_checkpoint()
    let log_path = run_dir() + "/finetune_log.tsv"
    var step = 0
    var first_epoch = 1
    var best = 1.0e30
    var seconds = 0.0
    if resuming {
        let _m = trainer.restore(state_path)
        let done: int = meta["finetune_epoch"]
        let at: int = meta["finetune_step"]
        let was: int = meta["finetune_epochs"]
        let best_so_far: float = meta["best_validation_loss"]
        let secs: float = meta["train_seconds"]
        trainer.opt.set_steps(at)
        first_epoch = done + 1
        step = at
        epochs = was
        best = best_so_far
        seconds = secs
        print(f"resumed {state_path} after epoch {done} of {epochs}")
    } else {
        append_file(log_path, "epoch\tstep\ttrain_loss\tval_loss\ttrain_seconds\n")
        let before = masked_loss(model, valid_rows, batch, quantum)
        print(f"validation loss before fine-tuning {before:.4f}")
        show_replies(model, tok)
        append_file(log_path, f"0\t0\t\t{before:.4f}\t0\n")
    }
    let total = per_epoch * epochs
    if first_epoch > epochs {
        print("fine-tuning already finished")
        model.free()
        return 0
    }
    for epoch in first_epoch..epochs + 1 {
        let rng = Random.with_seed(seed * 1000 + epoch)
        var sum = 0.0
        var n = 0
        var seen_tokens = 0
        var pending: Tensor? = null
        let t0 = Instant.now()
        for (length, picked) in epoch_batches(train_rows, batch, cfg.seq, quantum, rng) {
            let b = Batch(picked, length)
            let lr = learning_rate(step, total, peak, warmup, 0.1)
            let loss = trainer.step(b.x, b.y, lr, rows: b.rows)
            let before = pending
            if before is not null {
                sum += before.download()[0]
                n += 1
                before.free()
            }
            pending = loss
            step += 1
            seen_tokens += length * picked.len()
            if step % 100 == 0 {
                let rate = float(seen_tokens) / t0.elapsed().to_seconds()
                print(
                    f"epoch {epoch} step {step}/{total} loss {sum / float(max(n, 1)):.4f} lr {lr:.2e} ({rate:.0f} tokens/s)"
                )
            }
        }
        let last = pending
        if last is not null {
            sum += last.download()[0]
            n += 1
            last.free()
        }
        seconds += t0.elapsed().to_seconds()
        let val = masked_loss(model, valid_rows, batch, quantum)
        print(
            f"epoch {epoch}: train loss {sum / float(max(n, 1)):.4f}, validation loss {val:.4f} ({seconds:.0f} s)"
        )
        append_file(
            log_path,
            f"{epoch}\t{step}\t{sum / float(max(n, 1)):.4f}\t{val:.4f}\t{seconds:.0f}\n"
        )
        show_replies(model, tok)
        let improved = val < best
        if improved { best = val }
        let m = checkpoint_meta(
            cfg,
            pre_step,
            {
                "tokens": 0,
                "train_seconds": seconds,
                "finetune_epoch": epoch,
                "finetune_epochs": epochs,
                "finetune_step": step,
                "validation_loss": val,
                "best_validation_loss": best,
            }
        )
        trainer.save(state_path, m)
        if improved {
            trainer.save_weights(path, m)
            print(f"saved {path} (best validation loss so far)")
        }
        if one_epoch and epoch < epochs {
            print(f"epoch {epoch} of {epochs} done; run again with --resume to go on")
            model.free()
            return 10
        }
    }
    gpu.synchronize()
    model.free()
    print("fine-tuning finished")
    return 0
}
