#: LZ77: dictionary compression into copy-then-literal tokens.
#:
#: The other classic answer to the compression problem `std.algo.bwt` and
#: `std.algo.huffman` solve together — a back-reference into what was already
#: written, rather than a rearrangement of the alphabet.

# Shortest match LZ77 will encode as a back-reference: a two-character match
# costs more to store than the literals it replaces.
const MIN_MATCH = 3

# Back-references examined per position. Bounding the hash chain is what keeps
# compression linear on input like "aaaa..." where every position is a
# candidate; raising it buys a slightly better ratio for a lot of time.
const MAX_CHAIN = 64

#: A single LZ77 instruction: copy, then emit one literal.
#:
#: Copy `length` characters starting `distance` characters back from the end of
#: the output, then append `literal`. `length` is 0 when no back-reference was
#: found, and `literal` is empty when the input ended on a match. `distance`
#: may be smaller than `length`: the copy is character by character, so
#: `LZ77Token(1, 5, "")` repeats the previous character five times.
pub class LZ77Token {
    #: How far back the copy starts, in characters; 0 when there is no copy.
    pub let distance: int
    #: How many characters to copy; 0 when there is no copy.
    pub let length: int
    #: The character to append after the copy; empty at end of input.
    pub let literal: str

    #: Record one copy-then-literal step.
    pub fn init(self, distance: int, length: int, literal: str) {
        self.distance = distance
        self.length = length
        self.literal = literal
    }

    fn __str__(self) -> str {
        return f"LZ77Token({self.distance}, {self.length}, \"{self.literal}\")"
    }
}

#: Compress `text` into LZ77 copy-then-literal tokens.
#:
#: Matches are found through a hash chain over three-character prefixes: `head`
#: holds the most recent position of each trigram and `previous` chains the
#: older ones, so a position only ever examines places that really do share
#: three characters. Each walk is capped at `MAX_CHAIN` links and at `window`
#: characters, which bounds the whole compression at O(n) for repetitive input
#: instead of O(n^2).
#:
#: `window` is how far back a copy may reach and `max_length` how long one may
#: be; both trade ratio against the size of an encoded token.
#:
#: ```
#: let tokens = lz77_compress("abcabcabc")
#: tokens.len()              # 4
#: lz77_decompress(tokens)   # "abcabcabc"
#: ```
pub fn lz77_compress(text: str, window: int = 4096, max_length: int = 18) -> list[LZ77Token] {
    if window < 1 { throw ValueError(f"window must be positive, got {window}") }
    if max_length < MIN_MATCH {
        throw ValueError(f"max_length must be at least {MIN_MATCH}, got {max_length}")
    }
    let chars = _chars(text)
    let n = chars.len()
    let tokens: list[LZ77Token] = []
    let head: dict[str, int] = {}
    let previous = [-1 for _ in 0..n]
    var position = 0
    while position < n {
        var best_length = 0
        var best_distance = 0
        if position + MIN_MATCH <= n {
            let limit = min(max_length, n - position)
            var candidate = head.get(_trigram(chars, position), -1)
            var probes = 0
            while candidate >= 0 and probes < MAX_CHAIN and position - candidate <= window {
                var length = 0
                while length < limit and chars[candidate + length] == chars[position + length] {
                    length += 1
                }
                if length > best_length {
                    best_length = length
                    best_distance = position - candidate
                    if length == limit { break }
                }
                candidate = previous[candidate]
                probes += 1
            }
        }
        if best_length < MIN_MATCH {
            best_length = 0
            best_distance = 0
        }
        # Index every position the token covers, the one the literal consumes
        # included, or a later match would never see inside this copy.
        let covered = min(position + max(best_length, 1), n)
        for j in position..covered {
            if j + MIN_MATCH <= n {
                let key = _trigram(chars, j)
                previous[j] = head.get(key, -1)
                head[key] = j
            }
        }
        let literal = position + best_length < n ? chars[position + best_length] : ""
        tokens.push(LZ77Token(best_distance, best_length, literal))
        position += best_length + 1
    }
    return tokens
}

#: Rebuild the text that `lz77_compress` produced these tokens from.
#:
#: O(output length). Raises `ValueError` on a distance reaching before the
#: start of the output, which is the one way a token stream can be malformed.
#:
#: ```
#: lz77_decompress([LZ77Token(0, 0, "a"), LZ77Token(1, 3, "")])   # "aaaa"
#: ```
pub fn lz77_decompress(tokens: list[LZ77Token]) -> str {
    let out: list[str] = []
    for token in tokens {
        if token.length > 0 {
            if token.distance < 1 or token.distance > out.len() {
                throw ValueError(f"distance must be in 1..={out.len()}, got {token.distance}")
            }
            let start = out.len() - token.distance
            # Reading through the tail as it grows is deliberate: that is how a
            # distance shorter than the length encodes a repeating run.
            for offset in 0..token.length { out.push(out[start + offset]) }
        }
        if token.literal.len() > 0 { out.push(token.literal) }
    }
    return "".join(out)
}

# Duplicated in every sibling file that needs O(1) random access into a string
# rather than repeated O(n) slicing: too small to be worth a shared import for.
fn _chars(text: str) -> list[str] {
    let out: list[str] = []
    for ch in text { out.push(ch) }
    return out
}

fn _trigram(chars: list[str], position: int) -> str {
    return chars[position] + chars[position + 1] + chars[position + 2]
}
