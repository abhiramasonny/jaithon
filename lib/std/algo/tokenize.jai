#: Breaking text into countable units: word tokens, their frequencies, and
#: character/word n-grams — the standard features for language detection and
#: fuzzy matching.
from std.algo.sort import sort_by, natural_order

#: Split `text` into word tokens.
#:
#: A token is a maximal run of letters and digits, keeping an apostrophe that
#: stands between two of them, so `don't` is one token. Case is preserved;
#: `word_frequency` folds it.
#:
#: ```
#: tokenize("Don't stop, it's fine!")   # ["Don't", "stop", "it's", "fine"]
#: ```
pub fn tokenize(text: str) -> list[str] {
    let chars = _chars(text)
    let tokens: list[str] = []
    let current: list[str] = []
    for (index, ch) in chars.enumerate() {
        if _is_token_char(ch) {
            current.push(ch)
        } elif ch == "'" and
            current.len() > 0 and
            index + 1 < chars.len() and
            _is_token_char(chars[index + 1]) {
            current.push(ch)
        } elif current.len() > 0 {
            tokens.push("".join(current))
            current.clear()
        }
    }
    if current.len() > 0 { tokens.push("".join(current)) }
    return tokens
}

#: Count how often each lower-cased token occurs in `text`.
#:
#: ```
#: word_frequency("the cat the hat")   # {"the": 2, "cat": 1, "hat": 1}
#: ```
pub fn word_frequency(text: str) -> dict[str, int] {
    let counts: dict[str, int] = {}
    for token in tokenize(text) {
        let word = token.lower()
        counts[word] = counts.get(word, 0) + 1
    }
    return counts
}

#: Return the `count` most frequent words of `text`, most frequent first.
#:
#: Ties break alphabetically, so the answer is the same on every run.
#:
#: ```
#: top_words("the cat the hat the bat", 2)   # [("the", 3), ("bat", 1)]
#: ```
pub fn top_words(text: str, count: int) -> list[tuple[str, int]] {
    if count < 0 { throw ValueError(f"count must be non-negative, got {count}") }
    let counts = word_frequency(text)
    let out: list[tuple[str, int]] = []
    for word in sort_by(counts.keys(), |a, b| _by_count_then_word(counts, a, b)) {
        if out.len() >= count { break }
        out.push((word, counts[word]))
    }
    return out
}

#: Return every contiguous run of `size` characters in `text`.
#:
#: Character n-grams are the standard feature for language detection and fuzzy
#: matching: two strings sharing many of them are similar however their words
#: are ordered.
#:
#: ```
#: char_ngrams("hello", 2)   # ["he", "el", "ll", "lo"]
#: ```
pub fn char_ngrams(text: str, size: int) -> list[str] {
    if size < 1 { throw ValueError(f"size must be positive, got {size}") }
    let out: list[str] = []
    var start = 0
    while start + size <= text.len() {
        out.push(text[start:start + size])
        start += 1
    }
    return out
}

#: Return every run of `size` consecutive tokens of `text`, space-joined.
#:
#: Joining makes an n-gram usable as a dict key, which is what a phrase count
#: or a language model needs. Tokens never contain a space, so `split(" ")`
#: takes one apart again.
#:
#: ```
#: word_ngrams("a b c d", 2)   # ["a b", "b c", "c d"]
#: ```
pub fn word_ngrams(text: str, size: int) -> list[str] {
    if size < 1 { throw ValueError(f"size must be positive, got {size}") }
    let tokens = tokenize(text)
    let out: list[str] = []
    var start = 0
    while start + size <= tokens.len() {
        out.push(" ".join(tokens[start:start + size]))
        start += 1
    }
    return out
}

# Duplicated in every sibling file that needs O(1) random access into a string
# rather than repeated O(n) slicing: too small to be worth a shared import for.
fn _chars(text: str) -> list[str] {
    let out: list[str] = []
    for ch in text { out.push(ch) }
    return out
}

fn _is_token_char(ch: str) -> bool {
    let code = ord(ch)
    if code >= 48 and code <= 57 { return true }
    # A character with two cases is a letter; CJK scalars have one case but are
    # still words.
    if ch.lower() != ch.upper() { return true }
    return code >= 0x3041
}

fn _by_count_then_word(counts: dict[str, int], a: str, b: str) -> int {
    let by_count = natural_order(counts[b], counts[a])
    return by_count != 0 ? by_count : natural_order(a, b)
}
