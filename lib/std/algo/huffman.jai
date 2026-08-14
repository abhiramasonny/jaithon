#: `HuffmanCode`: a canonical Huffman code, an optimal prefix code over a
#: known alphabet.
#:
#: The code lengths come from the two-queue construction, which is O(k log k)
#: for the initial sort and O(k) after it — no heap, because merged weights
#: come out in non-decreasing order and so stay ordered in a plain queue. The
#: codes themselves are then assigned canonically: shortest first, and
#: alphabetically within one length. That makes the table reproducible across
#: runs and means a decoder needs only the code lengths, never the tree.
#:
#: ```
#: let code = HuffmanCode.from_text("abracadabra")
#: code.table()["a"]                         # "0"
#: code.encode("abracadabra")                # "01001110101011001001110"
#: code.decode(code.encode("abracadabra"))   # "abracadabra"
#: ```
from std.algo.sort import sort_by, natural_order

pub class HuffmanCode {
    var codes: dict[str, str]
    var by_length: list[dict[int, str]]

    #: Build the code for a symbol-to-frequency table.
    #:
    #: Raises `ValueError` on an empty table, a symbol that is not exactly one
    #: character, or a frequency below 1.
    pub fn init(self, frequencies: dict[str, int]) {
        self.codes = {}
        self.by_length = [{}]
        let alphabet = sort_by(frequencies.keys(), natural_order)
        if alphabet.len() == 0 { throw ValueError("a Huffman code needs at least one symbol") }
        for symbol in alphabet {
            if symbol.len() != 1 {
                throw ValueError(f"symbol must be one character, got '{symbol}'")
            }
            if frequencies[symbol] < 1 {
                throw ValueError(
                    f"frequency of '{symbol}' must be positive, got {frequencies[symbol]}"
                )
            }
        }
        let lengths = _huffman_lengths(alphabet, frequencies)
        var value = 0
        var previous = 0
        for symbol in sort_by(alphabet, |a, b| _by_length_then_symbol(lengths, a, b)) {
            let length = lengths[symbol]
            if previous > 0 { value = value + 1 << length - previous }
            previous = length
            self.codes[symbol] = _to_bits(value, length)
            while self.by_length.len() <= length { self.by_length.push({}) }
            self.by_length[length][value] = symbol
        }
    }

    #: Build the code from the character frequencies of `text`.
    pub static fn from_text(text: str) -> HuffmanCode {
        let frequencies: dict[str, int] = {}
        for ch in text { frequencies[ch] = frequencies.get(ch, 0) + 1 }
        return HuffmanCode(frequencies)
    }

    #: The symbol-to-code table, as a copy.
    pub fn table(self) -> dict[str, str] {
        return {symbol: bits for (symbol, bits) in self.codes.items()}
    }

    #: Encode `text` as a string of "0" and "1".
    #:
    #: Raises `KeyError` on a character the code was not built for.
    pub fn encode(self, text: str) -> str {
        let out: list[str] = []
        for ch in text {
            if ch not in self.codes { throw KeyError(f"no code for '{ch}'") }
            out.push(self.codes[ch])
        }
        return "".join(out)
    }

    #: Decode a string of "0" and "1" produced by `encode`.
    #:
    #: Raises `ValueError` on a character other than "0" or "1", or on trailing
    #: bits that complete no code.
    pub fn decode(self, bits: str) -> str {
        let out: list[str] = []
        var value = 0
        var length = 0
        for bit in bits {
            if bit != "0" and bit != "1" { throw ValueError(f"expected a bit, got '{bit}'") }
            value = value * 2 + (bit == "1" ? 1 : 0)
            length += 1
            if length < self.by_length.len() and value in self.by_length[length] {
                out.push(self.by_length[length][value])
                value = 0
                length = 0
            }
        }
        if length > 0 {
            throw ValueError(f"{length} trailing bits complete no code")
        }
        return "".join(out)
    }

    fn __str__(self) -> str { return f"HuffmanCode({self.codes.len()} symbols)" }
}

fn _to_bits(value: int, width: int) -> str {
    let out: list[str] = []
    var position = width - 1
    while position >= 0 {
        out.push(value >> position & 1 == 1 ? "1" : "0")
        position -= 1
    }
    return "".join(out)
}

fn _by_frequency(frequencies: dict[str, int], a: str, b: str) -> int {
    return natural_order(frequencies[a], frequencies[b])
}

fn _by_length_then_symbol(lengths: dict[str, int], a: str, b: str) -> int {
    let by_length = natural_order(lengths[a], lengths[b])
    return by_length != 0 ? by_length : natural_order(a, b)
}

#: Return the code length of every symbol under an optimal prefix code.
#:
#: The two-queue construction: leaves in one queue in non-decreasing weight,
#: merged nodes in the other. A merge always produces a weight at least as
#: large as the last one, so merged nodes stay ordered without a heap and the
#: two smallest nodes are always at one head or the other. Every merge deepens
#: both groups by one, which yields the lengths — all a canonical code needs.
#:
#: `alphabet` must be sorted and `sort_by` is stable, so equal frequencies keep
#: alphabetical order and the code is the same on every run.
fn _huffman_lengths(alphabet: list[str], frequencies: dict[str, int]) -> dict[str, int] {
    let lengths = {symbol: 0 for symbol in alphabet}
    if alphabet.len() == 1 {
        lengths[alphabet[0]] = 1
        return lengths
    }
    let leaves = sort_by(alphabet, |a, b| _by_frequency(frequencies, a, b))
    let leaf_weights = [frequencies[symbol] for symbol in leaves]
    let merged_weights: list[int] = []
    let merged_groups: list[list[str]] = []
    var leaf_index = 0
    var merged_index = 0
    while leaves.len() - leaf_index + (merged_weights.len() - merged_index) > 1 {
        var weight = 0
        let combined: list[str] = []
        for _ in 0..2 {
            var take_leaf = merged_index >= merged_weights.len()
            if not take_leaf and leaf_index < leaf_weights.len() {
                take_leaf = leaf_weights[leaf_index] <= merged_weights[merged_index]
            }
            if take_leaf {
                weight += leaf_weights[leaf_index]
                combined.push(leaves[leaf_index])
                leaf_index += 1
            } else {
                weight += merged_weights[merged_index]
                for symbol in merged_groups[merged_index] { combined.push(symbol) }
                merged_index += 1
            }
        }
        for symbol in combined { lengths[symbol] += 1 }
        merged_weights.push(weight)
        merged_groups.push(combined)
    }
    return lengths
}
