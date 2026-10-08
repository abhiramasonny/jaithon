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
#: (or end of input) leaves. `--prompt "..."` is a session of that one line,
#: as if it had been piped in: a message is answered, a command is carried
#: out, and then the session ends. `TEMP` (0.7), `TOPK` (40), `TOPP` (0.9),
#: `REP` (1.1), `MAX_NEW` (64) and `SEED` set the starting values;
#: `CHAT_ECHO=1` repeats each input line, which makes a piped transcript
#: readable.
import std.gpu as gpu
import std.os
from std.io import read_line, stdout
from std.random import Random
from .dialog import chat_prompt
from .model import GPT, chat_checkpoint, env_float, env_int, load_model, tokenizer_path
from .pretrain import flag
from .sample import Decoder, Sampling, generate
from .tokenizer import BOT, EOT, USER, Tokenizer, load_tokenizer

pub let COMMANDS = "commands: /reset, /temp N, /topk N, /topp P, /rep R, /quit\n"

fn write_out(data: bytes) -> void {
    let _n = stdout.write_bytes(data)
    stdout.flush()
}

#: The lines of a session: the `pending` ones first, then, when
#: `interactive`, standard input until it ends.
pub class Lines {
    var pending: list[str]
    let interactive: bool
    let echo: bool

    pub fn init(self, pending: list[str], interactive: bool, echo: bool = false) {
        self.pending = pending
        self.interactive = interactive
        self.echo = echo
    }

    #: The next line, stripped, or null at the end of the session.
    pub fn next(self) -> str? {
        if self.pending.len() > 0 {
            let line = self.pending[0]
            self.pending = self.pending[1:]
            return line.strip()
        }
        if not self.interactive { return null }
        write_out(bytes("you> "))
        let got = read_line()
        if got is null {
            write_out(bytes("\n"))
            return null
        }
        let line = got.strip()
        if self.echo { write_out(bytes(line + "\n")) }
        return line
    }
}

fn number(text: str) -> float? {
    try {
        return float(text)
    } catch _e: Error {
        return null
    }
}

#: One conversation: the turns so far and the sampling knobs. Everything it
#: prints goes through `emit`.
pub class Chat {
    let model: GPT
    let tok: Tokenizer
    let decoder: Decoder
    let max_new: int
    let budget: int
    let rng: Random
    let emit: fn(bytes) -> void
    pub var s: Sampling
    pub var turns: list[str]

    pub fn init(
        self,
        model: GPT,
        tok: Tokenizer,
        s: Sampling,
        max_new: int,
        rng: Random,
        emit: fn(bytes) -> void
    ) {
        self.model = model
        self.tok = tok
        self.s = s
        self.max_new = max(1, max_new)
        #: The prompt keeps at least half the context, however long the reply.
        self.budget = max(model.cfg.seq - self.max_new, model.cfg.seq // 2)
        self.rng = rng
        self.emit = emit
        self.turns = []
        self.decoder = Decoder(model)
    }

    fn say(self, text: str) -> void { self.emit(bytes(text)) }

    #: Carry out a `/` command; false when it ends the session.
    pub fn command(self, line: str) -> bool {
        let parts = line.split(" ")
        let cmd = parts[0]
        if cmd == "/quit" or cmd == "/exit" { return false }
        if cmd == "/reset" {
            self.turns = []
            self.say("(conversation forgotten)\n")
            return true
        }
        let knobs = ["/temp", "/topk", "/topp", "/rep"]
        if not knobs.contains(cmd) or parts.len() < 2 {
            self.say(COMMANDS)
            return true
        }
        let value = number(parts[1])
        if value is null {
            self.say(f"(not a number: {parts[1]})\n")
            return true
        }
        let v: float = value
        if cmd == "/temp" {
            self.s.temperature = v
            self.say(f"(temperature {self.s.temperature})\n")
        } elif cmd == "/topk" {
            self.s.top_k = int(v)
            self.say(f"(top-k {self.s.top_k})\n")
        } elif cmd == "/topp" {
            self.s.top_p = v
            self.say(f"(top-p {self.s.top_p})\n")
        } elif v <= 0.0 {
            self.say("(the repetition penalty has to be above 0)\n")
        } else {
            self.s.repetition_penalty = v
            self.say(f"(repetition penalty {self.s.repetition_penalty})\n")
        }
        return true
    }

    #: Answer one user line, streaming the reply; returns the reply's ids.
    pub fn answer(self, line: str) -> list[int] {
        self.turns.push(line)
        let prompt = chat_prompt(self.tok, self.turns, self.budget)
        self.decoder.reset()
        self.say("bot> ")
        let emit = self.emit
        let pieces = self.tok.pieces
        let out = generate(self.decoder, prompt, self.max_new, self.s, self.rng, [USER, EOT, BOT], fn(t: int) -> void {
            emit(pieces[t])
        })
        self.say("\n")
        self.turns.push(self.tok.decode(out).strip())
        return out
    }

    #: Read `lines` to the end of the session.
    pub fn run(self, lines: Lines) -> void {
        while true {
            let got = lines.next()
            if got is null { break }
            let line: str = got
            if line.len() == 0 { continue }
            if line.starts_with("/") {
                if not self.command(line) { break }
                continue
            }
            let _ids = self.answer(line)
        }
    }

    pub fn free(self) -> void { self.decoder.free() }
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
    let chat = Chat(
        model,
        tok,
        s,
        env_int("MAX_NEW", 64),
        Random.with_seed(env_int("SEED", 1)),
        write_out
    )
    let one = flag("--prompt", "")
    if one == "" {
        write_out(
            bytes(
                f"story_chat: {model.parameter_count()} parameters, {cfg.layers} layers, context {cfg.seq} tokens. /reset, /temp N, /quit\n"
            )
        )
        chat.run(Lines([], true, echo: os.env_or("CHAT_ECHO", "0") == "1"))
    } else {
        chat.run(Lines([one], false))
    }
    chat.free()
    model.free()
    gpu.synchronize()
    return 0
}
