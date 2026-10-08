#: Train the tokenizer and encode TinyStories with it, once.
#:
#:     jaithon run examples/story_chat/prepare.jai
#:
#: Writes three files to `STORY_CACHE` (default `~/.cache/jaithon/story_chat`),
#: each skipped when it is already there:
#:
#: - `tokenizer-<V>.txt`, the merges, learned from the first `SAMPLE_MB` (64)
#:   megabytes of the TinyStories train split plus all of DailyDialog's train
#:   dialogues, to a vocabulary of `VOCAB` (4096) ids;
#: - `valid-<V>.u16` and `train-<V>.u16`, every story encoded and followed by
#:   `<|endoftext|>`, as little-endian uint16 ids. The train split is read
#:   `BLOCK_MB` (16) megabytes at a time, so the 2.2 GB file is never in
#:   memory at once: with 16 MB blocks the encoder peaks under 4 GB
#:   resident, against about 14 GB with 64, and writes the same token file
#:   in the same time.
#:
#: `ONLY=tokenizer` or `ONLY=valid` stops early; `check.jai` and the vocabulary
#: comparison use that.
import std.os
from std.io import Path, open
from std.time import Instant
from .tokenizer import EOT, TRAIN_BYTES, Tokenizer, cache_dir, count_pieces, dialog_file, file_size
from .tokenizer import load_tokenizer, read_dialogues, read_range, story_file, train_bpe

let MARKER = bytes("<|endoftext|>")

fn env_int(name: str, fallback: int) -> int { return int(os.env_or(name, f"{fallback}")) }

#: The text of each story in `data[start:stop]`, as (start, stop) offsets with
#: the white space around it trimmed, handed to `each`. Returns where the
#: last complete story's marker ended: the next block starts there.
fn stories_in(data: bytes, stop: int, final: bool, each: fn(int, int) -> void) -> int {
    var at = 0
    while at < stop {
        let found = data.find(MARKER, at)
        var end = stop
        if found < 0 or found >= stop {
            if not final { return at }
        } else {
            end = found
        }
        var a = at
        var b = end
        while a < b and data[a] <= 32 { a += 1 }
        while b > a and data[b - 1] <= 32 { b -= 1 }
        if b > a { each(a, b) }
        if found < 0 or found >= stop { return stop }
        at = found + MARKER.len()
    }
    return at
}

#: Ids as little-endian uint16 bytes.
pub fn u16_bytes(ids: list[int]) -> bytes {
    var raw: list[int] = []
    for t in ids {
        raw.push(t & 255)
        raw.push(t >> 8)
    }
    return bytes(raw)
}

fn train_tokenizer(path: str, vocab: int) -> Tokenizer {
    let t0 = Instant.now()
    let sample = read_range(story_file("train"), 0, env_int("SAMPLE_MB", 64) * 1048576)
    var counts: dict[bytes, int] = {}
    var stories = 0
    let _end = stories_in(sample, sample.len(), false, fn(a: int, b: int) -> void {
        count_pieces(sample, a, b, counts)
        stories += 1
    })
    var turns = 0
    for dialogue in read_dialogues(dialog_file("train")) {
        for turn in dialogue {
            let raw = bytes(turn)
            count_pieces(raw, 0, raw.len(), counts)
            turns += 1
        }
    }
    print(
        f"tokenizer sample: {sample.len()} bytes, {stories} stories and {turns} dialogue turns, {counts.len()} distinct pieces ({t0.elapsed().to_seconds():.1f} s)"
    )
    let t1 = Instant.now()
    let tok = train_bpe(counts, vocab)
    tok.save(path)
    print(f"learned {tok.lefts.len()} merges in {t1.elapsed().to_seconds():.1f} s -> {path}")
    return tok
}

#: Encode one TinyStories file to `out_path`, a block at a time.
fn encode_split(tok: Tokenizer, source: str, out_path: str) -> void {
    let size = file_size(source)
    let block = env_int("BLOCK_MB", 16) * 1048576
    let partial = out_path + ".partial"
    let out = open(partial, "w")
    let t0 = Instant.now()
    var offset = 0
    var tokens = 1
    var stories = 0
    let _w = out.write_bytes(u16_bytes([EOT]))
    while offset < size {
        let data = read_range(source, offset, block)
        let final = offset + data.len() >= size
        var ids: list[int] = []
        let used = stories_in(data, data.len(), final, fn(a: int, b: int) -> void {
            tok.encode_into(data, a, b, ids)
            ids.push(EOT)
            stories += 1
        })
        if used == 0 and not final {
            throw RuntimeError(f"no story ends in {block} bytes at offset {offset}; raise BLOCK_MB")
        }
        let _n = out.write_bytes(u16_bytes(ids))
        tokens += ids.len()
        offset += used
        let s = t0.elapsed().to_seconds()
        print(
            f"  {offset * 100 // size}%: {offset} bytes, {stories} stories, {tokens} tokens, {float(offset) / s / 1048576.0:.1f} MB/s"
        )
    }
    out.close()
    Path(partial).rename(Path(out_path))
    let s = t0.elapsed().to_seconds()
    print(
        f"encoded {size} bytes into {tokens} tokens ({float(size) / float(tokens):.3f} bytes a token) in {s:.1f} s ({float(size) / s / 1048576.0:.1f} MB/s), {tok.cache.len()} distinct pieces -> {out_path}"
    )
}

fn main() -> int {
    let vocab = env_int("VOCAB", 4096)
    let dir = cache_dir()
    Path(dir).mkdir(true, true)
    let tok_path = dir + f"/tokenizer-{vocab}.txt"
    var tok = if Path(tok_path).exists() {
        load_tokenizer(tok_path)
    } else {
        train_tokenizer(tok_path, vocab)
    }
    let only = os.env_or("ONLY", "")
    if only == "tokenizer" { return 0 }
    let valid_path = dir + f"/valid-{vocab}.u16"
    if not Path(valid_path).exists() { encode_split(tok, story_file("valid"), valid_path) }
    if only == "valid" { return 0 }
    let train_path = dir + f"/train-{vocab}.u16"
    if not Path(train_path).exists() {
        let source = story_file("train")
        let size = file_size(source)
        if size != TRAIN_BYTES {
            throw RuntimeError(
                f"{source} is {size} bytes, not {TRAIN_BYTES}: is the download finished?"
            )
        }
        encode_split(tok, source, train_path)
    }
    return 0
}
