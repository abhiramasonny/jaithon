#: A byte-level BPE tokenizer, trained here, in Jaithon.
#:
#: Ids 0-255 are the bytes themselves, so any input encodes and every id
#: decodes. 256, 257 and 258 are the three special tokens -- `<|endoftext|>`,
#: `<|user|>` and `<|bot|>` -- and 259 onwards are the merges, in the order
#: they were learned: merge `k` makes id `259 + k` out of two earlier ids.
#:
#: Text is first cut into pieces the way GPT-2 cuts it -- a word with the one
#: space before it, a run of digits, a run of punctuation, a run of other
#: white space -- and no merge crosses a piece. That keeps the merges to
#: whole words and word parts, and it is what makes encoding a 2.2 GB corpus
#: cheap: TinyStories is written in a small vocabulary, so nearly every piece
#: has been seen before and its ids come out of a dictionary.
#:
#: The merges are saved as text, one `left right` pair a line, which the
#: torch peer reads too (`story_chat.py`).
import std.os
from std.io import open, read_bytes, read_file, write_file

pub let EOT = 256
pub let USER = 257
pub let BOT = 258
pub let FIRST_MERGE = 259
pub let SPECIAL_NAMES = ["<|endoftext|>", "<|user|>", "<|bot|>"]

#: Character classes of a byte for the piece cutter.
let SPACE = 0
let BLANK = 1
let LETTER = 2
let DIGIT = 3
let MARK = 4

#: The class of every byte value. Bytes 128 and up are letters, so a UTF-8
#: character never splits from the word it is in.
pub fn byte_classes() -> list[int] {
    var out: list[int] = []
    for b in 0..256 {
        var c = MARK
        if b == 32 {
            c = SPACE
        } elif b == 9 or b == 10 or b == 11 or b == 12 or b == 13 {
            c = BLANK
        } elif b >= 65 and b <= 90 or b >= 97 and b <= 122 or b >= 128 {
            c = LETTER
        } elif b >= 48 and b <= 57 {
            c = DIGIT
        }
        out.push(c)
    }
    return out
}

#: Where the piece that starts at `i` ends. One space before a word, number
#: or run of punctuation belongs to it; an apostrophe between two letters is
#: part of the word (`don't` is one piece); a run of white space gives its
#: last space to the word after it.
pub fn piece_end(data: bytes, cls: list[int], i: int, n: int) -> int {
    var j = i
    var c = cls[data[j]]
    if c == SPACE and j + 1 < n {
        let next = cls[data[j + 1]]
        if next >= LETTER {
            j += 1
            c = next
        }
    }
    if c == LETTER {
        j += 1
        while j < n {
            let k = cls[data[j]]
            if k == LETTER {
                j += 1
            } elif data[j] == 39 and j + 1 < n and cls[data[j + 1]] == LETTER {
                j += 2
            } else {
                break
            }
        }
        return j
    }
    if c == DIGIT or c == MARK {
        j += 1
        while j < n and cls[data[j]] == c { j += 1 }
        return j
    }
    #: White space: the whole run, less a final space that a word follows.
    j += 1
    while j < n and cls[data[j]] <= BLANK { j += 1 }
    if j < n and j - i > 1 and data[j - 1] == 32 { j -= 1 }
    return j
}

#: Every piece of `data[start:stop]` and how often it occurs, added into
#: `counts`.
pub fn count_pieces(data: bytes, start: int, stop: int, counts: dict[bytes, int]) -> void {
    let cls = byte_classes()
    var i = start
    while i < stop {
        let j = piece_end(data, cls, i, stop)
        let piece = data.slice(i, j)
        counts[piece] = counts.get(piece, 0) + 1
        i = j
    }
}

fn pair_key(a: int, b: int) -> int { return a * 65536 + b }

pub class Tokenizer {
    #: `lefts[k]` and `rights[k]` make id `FIRST_MERGE + k`.
    pub let lefts: list[int]
    pub let rights: list[int]
    #: Pair key (`a * 65536 + b`) to the id the merge makes; a lower id is an
    #: earlier merge, which is the order encoding applies them in.
    pub let ranks: dict[int, int]
    #: The bytes every id stands for.
    pub let pieces: list[bytes]
    #: Encoded pieces already seen.
    pub let cache: dict[bytes, list[int]]
    let cls: list[int]

    pub fn init(self, lefts: list[int], rights: list[int]) {
        self.lefts = lefts
        self.rights = rights
        self.ranks = {}
        self.cache = {}
        self.cls = byte_classes()
        var pieces: list[bytes] = []
        for b in 0..256 { pieces.push(bytes([b])) }
        for name in SPECIAL_NAMES { pieces.push(bytes(name)) }
        for (k, a) in lefts.enumerate() {
            let b = rights[k]
            self.ranks[pair_key(a, b)] = FIRST_MERGE + k
            pieces.push(pieces[a].concat(pieces[b]))
        }
        self.pieces = pieces
    }

    pub fn vocab_size(self) -> int { return FIRST_MERGE + self.lefts.len() }

    #: The ids of one piece, merges applied lowest id first.
    pub fn encode_piece(self, piece: bytes) -> list[int] {
        var ids: list[int] = []
        for i in 0..piece.len() { ids.push(piece[i]) }
        let ranks = self.ranks
        while ids.len() >= 2 {
            var best = 1 << 30
            var at = -1
            for i in 0..ids.len() - 1 {
                let r = ranks.get(ids[i] * 65536 + ids[i + 1], 1 << 30)
                if r < best {
                    best = r
                    at = i
                }
            }
            if at < 0 { break }
            let a = ids[at]
            let b = ids[at + 1]
            var merged: list[int] = []
            var i = 0
            let n = ids.len()
            while i < n {
                if i + 1 < n and ids[i] == a and ids[i + 1] == b {
                    merged.push(best)
                    i += 2
                } else {
                    merged.push(ids[i])
                    i += 1
                }
            }
            ids = merged
        }
        return ids
    }

    #: Append the ids of `data[start:stop]` -- plain text, no special tokens
    #: -- to `out`.
    pub fn encode_into(self, data: bytes, start: int, stop: int, out: list[int]) -> void {
        let cls = self.cls
        let cache = self.cache
        var i = start
        while i < stop {
            let j = piece_end(data, cls, i, stop)
            let piece = data.slice(i, j)
            var ids = cache.get(piece, null)
            if ids is null {
                let fresh = self.encode_piece(piece)
                cache[piece] = fresh
                ids = fresh
            }
            for t in ids { out.push(t) }
            i = j
        }
    }

    #: The ids of `text`, which holds no special tokens.
    pub fn encode(self, text: str) -> list[int] {
        let data = bytes(text)
        var out: list[int] = []
        self.encode_into(data, 0, data.len(), out)
        return out
    }

    #: The bytes `ids` stand for, special tokens spelled out.
    pub fn decode_bytes(self, ids: list[int]) -> bytes {
        var parts: list[int] = []
        for t in ids {
            let p = self.pieces[t]
            for i in 0..p.len() { parts.push(p[i]) }
        }
        return bytes(parts)
    }

    #: The text `ids` stand for. Bytes that are not UTF-8 -- a sampled
    #: sequence can stop half way through a character -- become U+FFFD.
    pub fn decode(self, ids: list[int]) -> str { return text_of(self.decode_bytes(ids)) }

    #: Write the merges to `path`, one `left right` pair a line.
    pub fn save(self, path: str) -> void {
        var lines = [f"story_chat bpe 1 {self.vocab_size()}"]
        for (k, a) in self.lefts.enumerate() { lines.push(f"{a} {self.rights[k]}") }
        write_file(path, "\n".join(lines) + "\n")
    }
}

#: Read a tokenizer `Tokenizer.save` wrote.
pub fn load_tokenizer(path: str) -> Tokenizer {
    let lines = read_file(path).split("\n")
    if lines.len() == 0 or not lines[0].starts_with("story_chat bpe 1") {
        throw ValueError(f"{path} is not a story_chat tokenizer")
    }
    var lefts: list[int] = []
    var rights: list[int] = []
    for line in lines[1:] {
        if line.len() == 0 { continue }
        let parts = line.split(" ")
        lefts.push(int(parts[0]))
        rights.push(int(parts[1]))
    }
    return Tokenizer(lefts, rights)
}

#: Bytes as text, with anything that is not UTF-8 replaced byte by byte.
pub fn text_of(data: bytes) -> str {
    try {
        return data.decode()
    } catch _e: Error {
        var kept: list[int] = []
        for i in 0..data.len() {
            let b = data[i]
            if b < 128 {
                kept.push(b)
            } else {
                for x in [239, 191, 189] { kept.push(x) }
            }
        }
        return bytes(kept).decode()
    }
}

#: The pair counts of the merge search, kept in slots so the search for the
#: most frequent pair is one pass over a list of ints.
class PairTable {
    pub let slot_of: dict[int, int]
    pub let keys: list[int]
    pub let counts: list[int]
    #: The words each pair was ever seen in; a word may be listed twice, or
    #: no longer hold the pair. Merging checks.
    pub let words: list[list[int]]

    pub fn init(self) {
        self.slot_of = {}
        self.keys = []
        self.counts = []
        self.words = []
    }

    pub fn add(self, key: int, amount: int, word: int) -> void {
        var slot = self.slot_of.get(key, -1)
        if slot < 0 {
            slot = self.keys.len()
            self.slot_of[key] = slot
            self.keys.push(key)
            self.counts.push(0)
            self.words.push([])
        }
        self.counts[slot] = self.counts[slot] + amount
        if amount > 0 { self.words[slot].push(word) }
    }

    #: The slot with the highest count, the earliest seen on a tie.
    pub fn best(self) -> int {
        let counts = self.counts
        var best = 0
        var at = -1
        for i in 0..counts.len() {
            if counts[i] > best {
                best = counts[i]
                at = i
            }
        }
        return at
    }
}

fn add_pairs(table: PairTable, word: list[int], count: int, index: int) -> void {
    for i in 0..word.len() - 1 { table.add(pair_key(word[i], word[i + 1]), count, index) }
}

fn merge_word(word: list[int], a: int, b: int, made: int) -> list[int] {
    var out: list[int] = []
    var i = 0
    let n = word.len()
    while i < n {
        if i + 1 < n and word[i] == a and word[i + 1] == b {
            out.push(made)
            i += 2
        } else {
            out.push(word[i])
            i += 1
        }
    }
    return out
}

fn holds_pair(word: list[int], a: int, b: int) -> bool {
    for i in 0..word.len() - 1 {
        if word[i] == a and word[i + 1] == b { return true }
    }
    return false
}

#: Learn merges from piece counts until the vocabulary is `vocab` ids. Each
#: merge updates the pair counts of only the words it touches.
pub fn train_bpe(counts: dict[bytes, int], vocab: int, report: bool = false) -> Tokenizer {
    var words: list[list[int]] = []
    var freq: list[int] = []
    for (piece, n) in counts.items() {
        if piece.len() < 2 { continue }
        var ids: list[int] = []
        for i in 0..piece.len() { ids.push(piece[i]) }
        words.push(ids)
        freq.push(n)
    }
    let table = PairTable()
    for (w, word) in words.enumerate() { add_pairs(table, word, freq[w], w) }
    var seen: list[int] = [-1 for _w in 0..words.len()]
    var lefts: list[int] = []
    var rights: list[int] = []
    let wanted = vocab - FIRST_MERGE
    while lefts.len() < wanted {
        let slot = table.best()
        if slot < 0 { break }
        let key = table.keys[slot]
        let a = key // 65536
        let b = key % 65536
        let made = FIRST_MERGE + lefts.len()
        let round = lefts.len()
        for w in table.words[slot] {
            if seen[w] == round { continue }
            seen[w] = round
            let word = words[w]
            if not holds_pair(word, a, b) { continue }
            add_pairs(table, word, -freq[w], w)
            let merged = merge_word(word, a, b, made)
            words[w] = merged
            if merged.len() >= 2 { add_pairs(table, merged, freq[w], w) }
        }
        table.words[slot] = []
        lefts.push(a)
        rights.push(b)
        if report and lefts.len() % 500 == 0 {
            print(f"  {lefts.len()} merges, last made from {table.counts.len()} pairs seen")
        }
    }
    return Tokenizer(lefts, rights)
}

#: DailyDialog's text has a space before every punctuation mark (`Say , Jim
#: , how about ...?`); put them back where English writes them.
pub fn normalise_turn(text: str) -> str {
    var t = text.strip()
    for mark in [" ,", " .", " ?", " !", " ;", " :", " %", " )"] {
        t = t.replace(mark, mark[1:])
    }
    t = t.replace("( ", "(")
    t = t.replace(" ' s ", "'s ").replace(" n't", "n't").replace(" 's ", "'s ").replace(" ’ ", "'")
    t = t.replace("’", "'")
    return t
}

#: A DailyDialog file as dialogues, each a list of turns.
pub fn read_dialogues(path: str) -> list[list[str]] {
    var out: list[list[str]] = []
    for line in read_file(path).split("\n") {
        if line.strip().len() == 0 { continue }
        var turns: list[str] = []
        for part in line.split("__eou__") {
            let t = normalise_turn(part)
            if t.len() > 0 { turns.push(t) }
        }
        if turns.len() >= 2 { out.push(turns) }
    }
    return out
}

#: Where the data and the caches are: `STORY_DATA` (the TinyStories folder),
#: `DIALOG_DATA` (the DailyDialog folder) and `STORY_CACHE`.
pub fn story_dir() -> str {
    return os.env_or("STORY_DATA", os.home_dir() + "/Developer/datasets/tinystories")
}

pub fn dialog_dir() -> str {
    return os.env_or("DIALOG_DATA", os.home_dir() + "/Developer/datasets/dailydialog")
}

pub fn cache_dir() -> str {
    return os.env_or("STORY_CACHE", os.home_dir() + "/.cache/jaithon/story_chat")
}

#: A data file that must exist, or a message that says where it was looked
#: for and which variable moves it.
pub fn require_file(path: str, variable: str) -> str {
    let f = open_or_null(path)
    if f is null {
        throw RuntimeError(
            f"missing {path}: download the dataset there, or point {variable} at its folder"
        )
    }
    return path
}

fn open_or_null(path: str) -> any {
    try {
        let f = open(path, "r")
        f.close()
        return true
    } catch _e: Error {
        return null
    }
}

pub fn story_file(split: str) -> str {
    return require_file(story_dir() + f"/TinyStoriesV2-GPT4-{split}.txt", "STORY_DATA")
}

pub fn dialog_file(split: str) -> str {
    return require_file(dialog_dir() + f"/{split}/dialogues_{split}.txt", "DIALOG_DATA")
}

#: The sizes the encoded corpus must be checked against: the train file is
#: 2,227,753,162 bytes when its download is complete.
pub let TRAIN_BYTES = 2227753162

#: Read `count` bytes of `path` from `offset`.
pub fn read_range(path: str, offset: int, count: int) -> bytes {
    let f = open(path, "r")
    defer { f.close() }
    let _p = f.seek(offset)
    return f.read_bytes(count)
}

pub fn file_size(path: str) -> int {
    let f = open(path, "r")
    defer { f.close() }
    return f.size()
}

pub fn whole_file(path: str) -> bytes { return read_bytes(path) }
