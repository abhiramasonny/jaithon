#: Talk to the fine-tuned model in the terminal.
#:
#:     jaithon run examples/story_chat/chat.jai
#:     jaithon run examples/story_chat/chat.jai -- --prompt "Hi! What are you doing tonight?"
#:     printf 'Hello!\nWhere do you live?\n' | jaithon run examples/story_chat/chat.jai
#:
#: Loads `CHAT_CKPT` (default `~/.cache/jaithon/story_chat/chat.ckpt`). Each
#: line you type is a `<|user|>` turn; the conversation so far, oldest turns
#: dropped first when it outgrows the context, is fed through the attention
#: cache and the reply is sampled a token at a time and printed as it comes,
#: until the model hands the turn back (`<|user|>`) or ends the conversation
#: (`<|endoftext|>`).
#:
#: Commands: `/reset` forgets the conversation, `/temp N` sets the
#: temperature, `/topk N`, `/topp P` and `/rep R` the other knobs, `/quit`
#: (or end of input) leaves. `--prompt "..."` answers one line and exits.
#: `TEMP` (0.7), `TOPK` (40), `TOPP` (0.9), `REP` (1.1), `MAX_NEW` (64) and
#: `SEED` set the starting values; `CHAT_ECHO=1` repeats each input line,
#: which makes a piped transcript readable.
import std.gpu as gpu
import std.os
from std.io import read_line, stdout
from std.random import Random
from .dialog import chat_prompt
from .model import chat_checkpoint, env_float, env_int, load_model, tokenizer_path
from .pretrain import flag
from .sample import Decoder, Sampling, generate
from .tokenizer import BOT, EOT, USER, load_tokenizer

fn say(text: str) -> void {
    let _n = stdout.write(text)
    stdout.flush()
}

fn main() -> int {
    let path = chat_checkpoint()
    let (model, meta) = load_model(path)
    let cfg = model.cfg
    let tok = load_tokenizer(tokenizer_path(cfg.vocab))
    model.set_training(false)
    let s = Sampling(
        temperature: env_float("TEMP", 0.7),
        top_k: env_int("TOPK", 40),
        top_p: env_float("TOPP", 0.9),
        repetition_penalty: env_float("REP", 1.1)
    )
    let max_new = env_int("MAX_NEW", 64)
    let rng = Random.with_seed(env_int("SEED", 1))
    let echo = os.env_or("CHAT_ECHO", "0") == "1"
    let decoder = Decoder(model)
    var turns: list[str] = []
    let one = flag("--prompt", "")
    if one == "" {
        say(
            f"story_chat: {model.parameter_count()} parameters, {cfg.layers} layers, context {cfg.seq} tokens. /reset, /temp N, /quit\n"
        )
    }
    while true {
        var line = one
        if one == "" {
            say("you> ")
            let got = read_line()
            if got is null {
                say("\n")
                break
            }
            line = got.strip()
            if echo { say(line + "\n") }
        }
        if line.len() == 0 { continue }
        if line.starts_with("/") {
            let parts = line.split(" ")
            let cmd = parts[0]
            if cmd == "/quit" or cmd == "/exit" { break }
            if cmd == "/reset" {
                turns = []
                say("(conversation forgotten)\n")
            } elif cmd == "/temp" and parts.len() > 1 {
                s.temperature = float(parts[1])
                say(f"(temperature {s.temperature})\n")
            } elif cmd == "/topk" and parts.len() > 1 {
                s.top_k = int(parts[1])
                say(f"(top-k {s.top_k})\n")
            } elif cmd == "/topp" and parts.len() > 1 {
                s.top_p = float(parts[1])
                say(f"(top-p {s.top_p})\n")
            } elif cmd == "/rep" and parts.len() > 1 {
                s.repetition_penalty = float(parts[1])
                say(f"(repetition penalty {s.repetition_penalty})\n")
            } else {
                say("commands: /reset, /temp N, /topk N, /topp P, /rep R, /quit\n")
            }
            continue
        }
        turns.push(line)
        let prompt = chat_prompt(tok, turns, cfg.seq - max_new)
        decoder.reset()
        say("bot> ")
        let out = generate(decoder, prompt, max_new, s, rng, [USER, EOT, BOT], fn(t: int) -> void {
            let _n = stdout.write_bytes(tok.pieces[t])
            stdout.flush()
        })
        say("\n")
        turns.push(tok.decode(out).strip())
        if one != "" { break }
    }
    decoder.free()
    model.free()
    gpu.synchronize()
    return 0
}
