#: The quick acceptance check for the example, well under a minute.
#:
#:     jaithon run examples/story_chat/check.jai
#:
#: Needs the data (TinyStories valid, DailyDialog validation) but nothing from
#: the cache: it trains its own small tokenizer and model and writes only to
#: the temporary directory. It requires that
#:
#: - the tokenizer round-trips: 1000 random stories and 1000 random dialogue
#:   turns decode to exactly their bytes, a formatted conversation decodes to
#:   its text with the special tokens spelled out, and a saved and loaded
#:   tokenizer encodes the same ids;
#: - a tiny GPT trained 200 steps on a slice of the valid split has its loss
#:   fall from about ln(vocab) to well below it, and leaks no device buffers;
#: - greedy decoding through the attention cache picks the same 32 tokens as
#:   running the whole prefix through the model for every token;
#: - a checkpoint saved and loaded gives back the same logits, bit for bit,
#:   and a trainer restored from it takes the same next step as the original;
#: - the fine-tuning loss, gathered at every row, equals the plain loss, and
#:   a dialogue batch scores exactly the bot's tokens;
#: - a save cut off at any step leaves a checkpoint that loads;
#: - the repetition penalty counts a token once however often it recurs, and
#:   leaves the role and end tokens alone;
#: - bytes that are not UTF-8 decode to U+FFFD one byte each, and the
#:   characters around them are kept;
#: - a chat session of one line ends after that line even when it is a
#:   command (`--prompt "/help"` once printed the commands for ever), a bad
#:   command argument is answered rather than thrown, and a reply as long as
#:   the context still leaves the prompt room.
#:
#: It exits non-zero if any of them fails.
import std.gpu as gpu
import std.math
import std.os
from jaitensor import SparseCategoricalCrossentropy, Tensor
from std.io import Path, write_file
from std.random import Random
from std.time import Instant
from .chat import COMMANDS, Chat, Lines
from .dialog import Batch, dialogue_rows, format_turns
from .model import Config, GPT, Sampler, TokenFile, checkpoint_meta, free_all, load_model, settle_checkpoint
from .prepare import u16_bytes
from .pretrain import Trainer, learning_rate
from .sample import Sampling, pick
from .tokenizer import BOT, EOT, USER, Tokenizer, count_pieces, dialog_file, load_tokenizer, read_dialogues
from .tokenizer import read_range, story_file, text_of, train_bpe

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

#: Story offsets in `data`, split on the marker.
fn story_spans(data: bytes) -> list[(int, int)] {
    let marker = bytes("<|endoftext|>")
    var out: list[(int, int)] = []
    var at = 0
    while at < data.len() {
        var found = data.find(marker, at)
        if found < 0 { found = data.len() }
        var a = at
        var b = found
        while a < b and data[a] <= 32 { a += 1 }
        while b > a and data[b - 1] <= 32 { b -= 1 }
        if b > a { out.push((a, b)) }
        at = found + marker.len()
    }
    return out
}

fn greedy_uncached(model: GPT, tokens: list[int]) -> int {
    let n = tokens.len()
    let idx = Tensor([n, 1], [float(t) for t in tokens])
    let logits = model.forward(idx)
    let row = logits.view((n - 1) * model.cfg.vocab, [model.cfg.vocab])
    let next = argmax(row.download())
    row.free()
    free_all(logits.discard_graph())
    free_all([logits, idx])
    return next
}

fn greedy_cached(model: GPT, prompt: list[int], count: int) -> list[int] {
    model.reset_cache()
    let first = model.forward_cached(prompt, 0)
    var out = [argmax(first.download())]
    first.free()
    for step in 1..count {
        let logits = model.forward_cached([out[out.len() - 1]], prompt.len() + step - 1)
        out.push(argmax(logits.download()))
        logits.free()
    }
    return out
}

fn check_tokenizer(sample: bytes, dialogues: list[list[str]]) -> Tokenizer {
    let t0 = Instant.now()
    var counts: dict[bytes, int] = {}
    let spans = story_spans(sample)
    for (a, b) in spans { count_pieces(sample, a, b, counts) }
    for turns in dialogues {
        for turn in turns {
            let raw = bytes(turn)
            count_pieces(raw, 0, raw.len(), counts)
        }
    }
    let tok = train_bpe(counts, 1024)
    print(
        f"tokenizer: {tok.vocab_size()} ids from {sample.len()} bytes in {t0.elapsed().to_seconds():.1f} s"
    )
    let rng = Random.with_seed(5)
    var stories_ok = 0
    for _i in 0..1000 {
        let (a, b) = spans[rng.int_range(0, spans.len())]
        var ids: list[int] = []
        tok.encode_into(sample, a, b, ids)
        if tok.decode_bytes(ids) == sample.slice(a, b) { stories_ok += 1 }
    }
    expect(stories_ok == 1000, f"1000 random stories round-trip ({stories_ok} exact)")
    var turns_ok = 0
    var all_turns: list[str] = []
    for d in dialogues {
        for t in d { all_turns.push(t) }
    }
    for _i in 0..1000 {
        let turn = all_turns[rng.int_range(0, all_turns.len())]
        if tok.decode(tok.encode(turn)) == turn { turns_ok += 1 }
    }
    expect(turns_ok == 1000, f"1000 random dialogue turns round-trip ({turns_ok} exact)")
    let convo = dialogues[0]
    let f = format_turns(tok, convo)
    var spelled = ""
    for (i, turn) in convo.enumerate() {
        spelled += (if i % 2 == 0 { "<|user|>" } else { "<|bot|>" }) + turn
    }
    spelled += "<|endoftext|>"
    expect(
        tok.decode(f.ids) == spelled,
        "a formatted dialogue decodes to its turns and role tokens"
    )
    let path = Path(os.temp_dir()).join("story_chat_check_tokenizer.txt").to_str()
    tok.save(path)
    let again = load_tokenizer(path)
    Path(path).remove()
    let (a, b) = spans[0]
    var x: list[int] = []
    var y: list[int] = []
    tok.encode_into(sample, a, b, x)
    again.encode_into(sample, a, b, y)
    expect(
        x == y and again.vocab_size() == tok.vocab_size(),
        "a saved and loaded tokenizer encodes the same ids"
    )
    return tok
}

#: The logits `path` gives for `prompt`, through the cache.
fn logits_of(path: str, prompt: list[int]) -> list[float] {
    let (m, _meta) = load_model(path)
    m.set_training(false)
    m.start_cache(m.cfg.seq)
    let out = m.forward_cached(prompt, 0)
    let row = out.download()
    out.free()
    m.free_cache()
    m.free()
    return row
}

fn leftovers(path: str) -> bool {
    return Path(path + ".partial").exists() or Path(path + ".whole").exists()
}

#: Each way a save can be cut off, made by hand from a whole checkpoint at
#: `path`, then settled and loaded.
fn check_interrupted_saves(path: str, prompt: list[int], want: list[float]) -> void {
    let whole = path + ".copy"
    Path(path).copy(Path(whole))
    #: Cut off while copying the new file over the old.
    Path(whole).copy(Path(path + ".partial"))
    write_file(path + ".whole", "")
    write_file(path, "not a checkpoint")
    settle_checkpoint(path)
    let a = logits_of(path, prompt) == want and not leftovers(path)
    #: Cut off between removing the old file and copying the new.
    Path(whole).copy(Path(path + ".partial"))
    write_file(path + ".whole", "")
    Path(path).remove()
    settle_checkpoint(path)
    let b = logits_of(path, prompt) == want and not leftovers(path)
    #: Cut off while writing the new file: the old one stands.
    write_file(path + ".partial", "half a checkpoint")
    settle_checkpoint(path)
    let c = logits_of(path, prompt) == want and not leftovers(path)
    Path(whole).remove()
    expect(
        a and b and c,
        f"a save cut off while copying ({a}), after removing the old file ({b}) or while writing ({c}) settles to a checkpoint that loads"
    )
}

fn check_text_of() -> void {
    #: "café", the lead byte of a three-byte character cut off, "ok".
    let cut = text_of(bytes([99, 97, 102, 195, 169, 226, 111, 107]))
    #: An overlong form, a surrogate and a stray continuation byte.
    let bad = text_of(bytes([224, 128, 128, 237, 160, 128, 128, 65]))
    expect(
        cut == "caf\u{e9}\u{fffd}ok" and bad == "\u{fffd}".repeat(7) + "A",
        f"bytes that are not UTF-8 become U+FFFD and the rest is kept ({repr(cut)}, {repr(bad)})"
    )
}

#: Greedy picks that the penalty decides.
fn check_penalty() -> void {
    let s = Sampling(temperature: 0.0, repetition_penalty: 2.0)
    let rng = Random.with_seed(1)
    var row: list[float] = []
    for _i in 0..300 { row.push(0.0) }
    #: 5 seen three times: divided by 2 once (2.0) it still beats 1.5;
    #: divided three times (0.5) it would not.
    row[5] = 4.0
    row[8] = 1.5
    let once = pick(row, s, [5, 5, 5], rng)
    #: <|user|> is in every chat prompt; penalised (1.5) it would lose to 2.0.
    row[5] = 0.0
    row[USER] = 3.0
    row[9] = 2.0
    let role = pick(row, s, [EOT, USER, 9, BOT, USER], rng)
    expect(
        once == 5 and role == USER,
        f"the repetition penalty counts a token once ({once}, want 5) and spares <|user|> ({role}, want {USER})"
    )
}

#: A chat session against the small model, everything it prints captured.
fn check_chat(model: GPT, tok: Tokenizer) -> void {
    model.set_training(false)
    var printed: list[bytes] = []
    let capture = fn(b: bytes) -> void {
        printed.push(b)
        if printed.len() > 2000 {
            throw RuntimeError("the chat session did not end")
        }
    }
    let chat = Chat(model, tok, Sampling(temperature: 0.0), 8, Random.with_seed(1), capture)
    var ended = true
    try {
        chat.run(Lines(["/help"], false))
    } catch _e: RuntimeError {
        ended = false
    }
    var text = ""
    for b in printed { text += text_of(b) }
    expect(
        ended and text == COMMANDS,
        f"a one-line session of /help prints the commands once and ends ({printed.len()} writes)"
    )
    printed.clear()
    chat.run(Lines(["/temp hot", "/temp 0.5", "/rep 0", "Hello there", "/quit", "not read"], false))
    text = ""
    for b in printed { text += text_of(b) }
    expect(
        chat.s.temperature == 0.5 and
            chat.s.repetition_penalty == 1.1 and
            chat.turns.len() == 2 and
            text.contains("(not a number: hot)"),
        f"bad command arguments are answered, one reply is made, /quit ends the session ({chat.turns.len()} turns)"
    )
    chat.free()
    let long = Chat(
        model,
        tok,
        Sampling(temperature: 0.0),
        model.cfg.seq,
        Random.with_seed(1),
        capture
    )
    printed.clear()
    var answered = true
    try {
        let _reply = long.answer("Tell me a long story, please.")
    } catch _e: Error {
        answered = false
    }
    long.free()
    expect(
        answered,
        f"a reply allowed the whole context ({model.cfg.seq} tokens) still leaves the prompt room"
    )
    model.set_training(true)
}

fn main() -> int {
    let t_all = Instant.now()
    let sample = read_range(story_file("valid"), 0, 4 * 1048576)
    let dialogues = read_dialogues(dialog_file("validation"))
    let tok = check_tokenizer(sample, dialogues)

    #: The slice of the valid split as a token file, in memory.
    var ids = [EOT]
    for (a, b) in story_spans(sample) {
        tok.encode_into(sample, a, b, ids)
        ids.push(EOT)
    }
    let file = TokenFile(u16_bytes(ids))
    let cfg = Config(2, 128, 4, 64, 16, tok.vocab_size())
    let model = GPT(cfg, seed: 3)
    let trainer = Trainer(model, 2.0e-3, 0.1, 1.0)
    let sampler = Sampler(file, 7)
    let steps = 200
    var losses: list[float] = []
    var live = -1
    let t0 = Instant.now()
    for step in 0..steps {
        let (x, y) = sampler.next(cfg.batch, cfg.seq)
        let loss = trainer.step(x, y, learning_rate(step, steps, 2.0e-3, 20, 0.1))
        losses.push(loss.download()[0])
        loss.free()
        if step == 9 {
            gpu.synchronize()
            live = gpu.live_buffers()
        }
    }
    gpu.synchronize()
    let drift = gpu.live_buffers() - live
    var tail = 0.0
    for v in losses[steps - 20:] { tail += v }
    tail /= 20.0
    let start = math.log(float(cfg.vocab))
    print(
        f"{steps} steps of {cfg} in {t0.elapsed().to_seconds():.1f} s: loss {losses[0]:.3f} -> {tail:.3f} (ln {cfg.vocab} = {start:.3f})"
    )
    expect(
        losses[0] > start - 0.5 and tail < start - 2.0,
        f"loss falls more than 2 nats below ln(vocab) (first {losses[0]:.3f}, last 20 {tail:.3f})"
    )
    expect(drift == 0, f"training leaks no device buffers (drift {drift})")

    #: Cached against uncached greedy decoding.
    let prompt = ids[100:116]
    let count = 32
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
    for i in 0..count { if cached[i] == uncached[i] { same += 1 } }
    print(f"cached:   {repr(tok.decode(cached))}")
    print(f"uncached: {repr(tok.decode(uncached))}")
    expect(
        same == count and again == cached,
        f"cached greedy decoding matches uncached ({same} of {count} tokens)"
    )
    expect(decode_drift == 0, f"cached decoding leaks no device buffers (drift {decode_drift})")
    model.free_cache()
    model.set_training(true)

    #: Checkpoint round trip: logits, then the optimiser's next step.
    let path = Path(os.temp_dir()).join("story_chat_check.ckpt").to_str()
    trainer.save(path, checkpoint_meta(cfg, steps, {"tokens": 0, "train_seconds": 0.0}))
    let (restored, _meta) = load_model(path)
    restored.set_training(false)
    model.set_training(false)
    model.start_cache(cfg.seq)
    restored.start_cache(cfg.seq)
    let a = model.forward_cached(prompt, 0)
    let b = restored.forward_cached(prompt, 0)
    let want = a.download()
    expect(want == b.download(), "a saved and loaded checkpoint gives the same logits")
    expect(not leftovers(path), "a finished save leaves no partial file or mark")
    free_all([a, b])
    model.free_cache()
    restored.free_cache()
    restored.free()
    model.set_training(true)
    check_interrupted_saves(path, prompt, want)
    let twin = GPT(cfg, seed: 99)
    let twin_trainer = Trainer(twin, 2.0e-3, 0.1, 1.0)
    let _m = twin_trainer.restore(path)
    Path(path).remove()
    let probe = Sampler(file, 1234)
    let (x1, y1) = probe.next(cfg.batch, cfg.seq)
    let probe2 = Sampler(file, 1234)
    let (x2, y2) = probe2.next(cfg.batch, cfg.seq)
    var gap = 0.0
    for _k in 0..2 {
        let (xa, ya) = probe.next(cfg.batch, cfg.seq)
        let (xb, yb) = probe2.next(cfg.batch, cfg.seq)
        let la = trainer.step(xa, ya, 1.0e-3)
        let lb = twin_trainer.step(xb, yb, 1.0e-3)
        gap = max(gap, math.fabs(la.download()[0] - lb.download()[0]))
        free_all([la, lb])
    }
    free_all([x1, y1, x2, y2])
    expect(gap < 1.0e-4, f"a restored trainer steps as the original does (loss gap {gap:.2e})")
    twin.free()

    #: The fine-tuning loss.
    let (x, y) = sampler.next(cfg.batch, cfg.seq)
    var every: list[float] = []
    for i in 0..cfg.batch * cfg.seq { every.push(float(i)) }
    let rows = Tensor([every.len()], every)
    let flat_y = Tensor([y.size], y.download())
    let ce = SparseCategoricalCrossentropy(from_logits: true)
    model.set_training(false)
    let full = model.forward(x)
    let gathered = model.forward(x, rows: rows)
    let l_full = ce.forward(full, y)
    let l_gathered = ce.forward(gathered, flat_y)
    let diff = math.fabs(l_full.download()[0] - l_gathered.download()[0])
    expect(diff < 1.0e-5, f"the loss gathered at every row is the plain loss (gap {diff:.2e})")
    free_all([x, y, rows, flat_y, full, gathered, l_full, l_gathered])
    let dialogue_row = dialogue_rows(tok, dialogues[1], cfg.seq)[0]
    let batch = Batch([dialogue_row], dialogue_row.ids.len() - 1)
    var bot_tokens = 0
    var in_bot = false
    for t in dialogue_row.ids {
        if t == BOT {
            in_bot = true
        } elif t == USER or t == EOT {
            if in_bot { bot_tokens += 1 }
            in_bot = false
        } elif in_bot {
            bot_tokens += 1
        }
    }
    expect(
        batch.targets == bot_tokens and batch.targets == dialogue_row.targets(),
        f"a dialogue batch scores the bot's {bot_tokens} tokens and their turn ends ({batch.targets})"
    )
    free_all([batch.x, batch.rows, batch.y])
    check_penalty()
    check_text_of()
    check_chat(model, tok)
    model.free()
    gpu.synchronize()
    print(f"check took {t_all.elapsed().to_seconds():.1f} s")
    if failures.len() > 0 {
        print(f"{failures.len()} check(s) failed")
        return 1
    }
    print("all checks passed")
    return 0
}
