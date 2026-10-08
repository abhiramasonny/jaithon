#: Continue a story with the pretrained model.
#:
#:     jaithon run examples/story_chat/generate.jai -- "Once upon a time, a little fox"
#:
#: Loads `PRETRAIN_CKPT` (default `~/.cache/jaithon/story_chat/pretrain.ckpt`),
#: starts from `<|endoftext|>` and the prompt, and samples until the story
#: ends (`<|endoftext|>`) or `NEW` (300) tokens, printing as it goes.
#: `SAMPLES` (1) stories, at temperature `TEMP` (0.8), `TOPK` (40), `TOPP`
#: (0.95), repetition penalty `REP` (1.0), from `SEED` (1). Past the model's
#: context the most recent half is fed through again and the story goes on.
#: `CKPT` loads another checkpoint, the chat model for example.
import std.gpu as gpu
import std.os
from std.io import stdout
from std.random import Random
from std.time import Instant
from .model import env_float, env_int, load_model, pretrain_checkpoint, tokenizer_path
from .sample import Decoder, Sampling, generate
from .tokenizer import EOT, load_tokenizer

fn main() -> int {
    let path = os.env_or("CKPT", pretrain_checkpoint())
    let tl = Instant.now()
    let (model, meta) = load_model(path)
    let tok = load_tokenizer(tokenizer_path(model.cfg.vocab))
    let step: int = meta["step"]
    print(
        f"loaded {path}: {model.parameter_count()} parameters, step {step}, {tl.elapsed().to_seconds():.1f} s"
    )
    model.set_training(false)
    let args = os.argv()
    let prompt = if args.len() > 1 and not args[args.len() - 1].ends_with(".jai") {
        args[args.len() - 1]
    } else {
        "Once upon a time"
    }
    let s = Sampling(
        temperature: env_float("TEMP", 0.8),
        top_k: env_int("TOPK", 40),
        top_p: env_float("TOPP", 0.95),
        repetition_penalty: env_float("REP", 1.0)
    )
    let rng = Random.with_seed(env_int("SEED", 1))
    let count = env_int("NEW", 300)
    let decoder = Decoder(model)
    for n in 0..env_int("SAMPLES", 1) {
        decoder.reset()
        var ids = [EOT]
        for t in tok.encode(prompt) { ids.push(t) }
        let _w = stdout.write(f"--- story {n + 1}\n{prompt}")
        stdout.flush()
        let t0 = Instant.now()
        let out = generate(decoder, ids, count, s, rng, [EOT], fn(t: int) -> void {
            let _n = stdout.write_bytes(tok.pieces[t])
            stdout.flush()
        })
        let ms = t0.elapsed().to_millis()
        print(
            f"\n--- {out.len()} tokens in {ms:.0f} ms ({float(out.len()) / ms * 1000.0:.0f} tokens/s)"
        )
    }
    decoder.free()
    model.free()
    gpu.synchronize()
    return 0
}
