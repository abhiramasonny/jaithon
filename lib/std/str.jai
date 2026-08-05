#: String algorithms that the built-in `str` methods do not cover.
#:
#: The methods on `str` already split, join, trim, search and case-fold whole
#: strings. This module adds the layout, case-style, distance, escaping and
#: parsing operations that programs otherwise rewrite by hand, plus
#: `StringBuilder` for accumulating text in linear time.
#:
#: Every function treats a string as a sequence of Unicode scalar values, so all
#: indices and lengths agree with `str.len()`.
const _INT_MIN = -9223372036854775807 - 1
const _HEX_DIGITS = "0123456789ABCDEF"

let _ESCAPES: dict[str, str] = {
    "\\": "\\\\",
    "\"": "\\\"",
    "\n": "\\n",
    "\t": "\\t",
    "\r": "\\r",
    "\0": "\\0",
}

let _UNESCAPES: dict[str, str] = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "0": "\0",
    "\\": "\\",
    "\"": "\"",
    "'": "'",
}

let _HTML_ESCAPES: dict[str, str] = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
}

# Accented letters `to_ascii` folds, as two parallel strings of equal length.
let _ACCENT_FROM = "àáâãäåçèéêëìíîïñòóôõöùúûüýÿÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝŸ"
let _ACCENT_TO = "aaaaaaceeeeiiiinooooouuuuyyAAAAAACEEEEIIIINOOOOOUUUUYY"

let _ACCENT_MAP: dict[str, str] = {source: target for (source, target) in zip(_ACCENT_FROM, _ACCENT_TO)}

let _SYMBOL_MAP: dict[str, str] = {
    "æ": "ae",
    "Æ": "AE",
    "œ": "oe",
    "Œ": "OE",
    "ß": "ss",
    "ø": "o",
    "Ø": "O",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "Th",
    "ł": "l",
    "Ł": "L",
    "–": "-",
    "—": "-",
    "‑": "-",
    "…": "...",
    "•": "*",
    "·": ".",
    "“": "\"",
    "”": "\"",
    "„": "\"",
    "«": "\"",
    "»": "\"",
    "‘": "'",
    "’": "'",
    "‚": "'",
    "×": "x",
    "÷": "/",
    "°": "deg",
    "©": "(c)",
    "®": "(r)",
    "™": "(tm)",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "¢": "c",
    "\u{00A0}": " ",
}

#: Accumulate strings in a list and join them once.
#:
#: Repeated `text = text + piece` copies the whole accumulated string on every
#: append, so building `n` pieces costs O(n^2) bytes copied. A builder appends
#: to a list in O(1) amortised time and pays a single copy in `build()`, which
#: is O(total length).
#:
#: ```
#: StringBuilder.from_parts(["a", "b", "c"]).build()   # "abc"
#: ```
pub class StringBuilder {
    var parts: list[str]
    var total: int

    #: Create an empty builder.
    pub fn init(self) {
        self.parts = []
        self.total = 0
    }

    #: Create a builder already holding `parts`.
    pub static fn from_parts(parts: list[str]) -> StringBuilder {
        let builder = StringBuilder()
        builder.extend(parts)
        return builder
    }

    #: Append `text`.
    pub fn push(self, text: str) -> void {
        self.parts.push(text)
        self.total += text.len()
    }

    #: Append `text` followed by a newline.
    pub fn push_line(self, text: str = "") -> void {
        self.push(text)
        self.push("\n")
    }

    #: Append every element of `texts`, in order.
    pub fn extend(self, texts: list[str]) -> void {
        for text in texts { self.push(text) }
    }

    #: Total number of characters appended so far.
    pub fn len(self) -> int { return self.total }

    #: Whether nothing has been appended.
    pub fn is_empty(self) -> bool { return self.total == 0 }

    #: Discard everything appended so far.
    pub fn clear(self) -> void {
        self.parts = []
        self.total = 0
    }

    #: Join everything appended so far. The builder stays usable afterwards.
    pub fn build(self) -> str { return "".join(self.parts) }

    fn __len__(self) -> int { return self.total }

    fn __str__(self) -> str { return self.build() }
}

#: Return `text` repeated `times` times, using O(log times) concatenations.
#:
#: ```
#: repeat("ab", 3)   # "ababab"
#: ```
pub fn repeat(text: str, times: int) -> str {
    if times < 0 { throw ValueError(f"times must be non-negative, got {times}") }
    if times == 0 or text.len() == 0 { return "" }
    var result = ""
    var chunk = text
    var remaining = times
    while remaining > 0 {
        if remaining % 2 == 1 { result = result + chunk }
        remaining = remaining // 2
        if remaining > 0 { chunk = chunk + chunk }
    }
    return result
}

#: Break `text` into lines of at most `width` characters, splitting on spaces.
#:
#: Existing newlines start a new line and runs of whitespace collapse to one
#: space. A word longer than `width` is split rather than allowed to overflow.
#: Width counts characters; for terminal output measured in display cells use
#: `std.fmt.display_width`.
#:
#: ```
#: wrap("the quick brown fox", 10)   # ["the quick", "brown fox"]
#: ```
pub fn wrap(text: str, width: int) -> list[str] {
    if width <= 0 { throw ValueError(f"width must be positive, got {width}") }
    if text.strip().len() == 0 { return [] }
    let lines: list[str] = []
    for paragraph in text.split("\n") {
        let pieces = _split_whitespace(paragraph)
        if pieces.len() == 0 {
            lines.push("")
            continue
        }
        var current = ""
        for word in pieces {
            var piece = word
            while piece.len() > width {
                if current.len() > 0 {
                    lines.push(current)
                    current = ""
                }
                lines.push(piece[0:width])
                piece = piece[width:piece.len()]
            }
            if current.len() == 0 {
                current = piece
            } elif current.len() + 1 + piece.len() <= width {
                current = current + " " + piece
            } else {
                lines.push(current)
                current = piece
            }
        }
        if current.len() > 0 { lines.push(current) }
    }
    return lines
}

#: Prefix every non-blank line of `text` with `prefix`.
#:
#: Pass `blank_lines: true` to prefix whitespace-only lines too, which is wrong
#: for most output because it leaves trailing whitespace behind.
#:
#: ```
#: indent("a\nb", "  ")   # "  a\n  b"
#: ```
pub fn indent(text: str, prefix: str, blank_lines: bool = false) -> str {
    let out: list[str] = []
    for line in text.split("\n") {
        if line.strip().len() == 0 and not blank_lines {
            out.push(line)
        } else {
            out.push(prefix + line)
        }
    }
    return "\n".join(out)
}

#: Remove the whitespace prefix that every non-blank line of `text` shares.
#:
#: Blank lines become empty and do not constrain the common prefix, so a
#: triple-quoted block written inside a function comes back flush left.
#:
#: ```
#: dedent("    a\n      b")   # "a\n  b"
#: ```
pub fn dedent(text: str) -> str {
    var margin = ""
    var seen = false
    for line in text.split("\n") {
        if line.strip().len() == 0 { continue }
        let leading = _leading_whitespace(line)
        if not seen {
            margin = leading
            seen = true
        } else {
            margin = common_prefix(margin, leading)
        }
    }
    if margin.len() == 0 { return text }
    let out: list[str] = []
    for line in text.split("\n") {
        if line.strip().len() == 0 {
            out.push("")
        } else {
            out.push(line[margin.len():line.len()])
        }
    }
    return "\n".join(out)
}

#: Return the longest string that both `a` and `b` start with.
#:
#: ```
#: common_prefix("prefix", "president")   # "pre"
#: ```
pub fn common_prefix(a: str, b: str) -> str {
    let limit = min(a.len(), b.len())
    var index = 0
    while index < limit and a[index] == b[index] { index += 1 }
    return a[0:index]
}

#: Return the longest string that both `a` and `b` end with.
#:
#: ```
#: common_suffix("running", "jumping")   # "ing"
#: ```
pub fn common_suffix(a: str, b: str) -> str {
    let limit = min(a.len(), b.len())
    var index = 0
    while index < limit and a[a.len() - 1 - index] == b[b.len() - 1 - index] { index += 1 }
    return a[a.len() - index:a.len()]
}

#: Split `text` into words, honouring separators and camel-case humps.
#:
#: A word ends at any character that is neither a letter nor a digit, before an
#: uppercase letter following a lowercase letter or digit, and at the end of an
#: acronym. Words keep their original case; the `to_*` converters fold them.
#:
#: ```
#: words("parseHTTPResponse")   # ["parse", "HTTP", "Response"]
#: words("user_id-2")           # ["user", "id", "2"]
#: ```
pub fn words(text: str) -> list[str] {
    let letters = _chars(text)
    let out: list[str] = []
    var start = -1
    var index = 0
    while index < letters.len() {
        if not _is_word_char(letters[index]) {
            if start >= 0 {
                out.push("".join(letters[start:index]))
                start = -1
            }
            index += 1
            continue
        }
        if start < 0 {
            start = index
        } elif _word_boundary(letters, index) {
            out.push("".join(letters[start:index]))
            start = index
        }
        index += 1
    }
    if start >= 0 { out.push("".join(letters[start:letters.len()])) }
    return out
}

#: Convert `text` to `snake_case`.
#:
#: ```
#: to_snake("parseHTTPResponse")   # "parse_http_response"
#: ```
pub fn to_snake(text: str) -> str {
    return "_".join([word.lower() for word in words(text)])
}

#: Convert `text` to `SCREAMING_SNAKE_CASE`.
#:
#: ```
#: to_screaming_snake("maxRetries")   # "MAX_RETRIES"
#: ```
pub fn to_screaming_snake(text: str) -> str {
    return "_".join([word.upper() for word in words(text)])
}

#: Convert `text` to `kebab-case`.
#:
#: ```
#: to_kebab("XMLHttpRequest")   # "xml-http-request"
#: ```
pub fn to_kebab(text: str) -> str {
    return "-".join([word.lower() for word in words(text)])
}

#: Convert `text` to `PascalCase`.
#:
#: ```
#: to_pascal("user-id")   # "UserId"
#: ```
pub fn to_pascal(text: str) -> str {
    return "".join([_capitalize(word) for word in words(text)])
}

#: Convert `text` to `camelCase`.
#:
#: ```
#: to_camel("user_id")   # "userId"
#: ```
pub fn to_camel(text: str) -> str {
    let parts = words(text)
    if parts.len() == 0 { return "" }
    let out = [parts[0].lower()]
    var index = 1
    while index < parts.len() {
        out.push(_capitalize(parts[index]))
        index += 1
    }
    return "".join(out)
}

#: Return the number of single-character edits that turn `a` into `b`.
#:
#: Insertions, deletions and substitutions cost one each. O(n*m) time,
#: O(min(n, m)) space.
#:
#: ```
#: levenshtein("kitten", "sitting")   # 3
#: ```
pub fn levenshtein(a: str, b: str) -> int {
    let source = _chars(a)
    let target = _chars(b)
    if source.len() == 0 { return target.len() }
    if target.len() == 0 { return source.len() }
    var previous = [column for column in 0..=target.len()]
    var current = [0 for _ in 0..=target.len()]
    for row in 1..=source.len() {
        current[0] = row
        for column in 1..=target.len() {
            let cost = source[row - 1] == target[column - 1] ? 0 : 1
            let step = min(current[column - 1] + 1, previous[column] + 1)
            current[column] = min(step, previous[column - 1] + cost)
        }
        let spare = previous
        previous = current
        current = spare
    }
    return previous[target.len()]
}

#: Return the edit distance between `a` and `b` allowing transpositions.
#:
#: This is the unrestricted Damerau-Levenshtein distance: insertion, deletion,
#: substitution and the swap of two adjacent characters each cost one, and the
#: result obeys the triangle inequality. O(n*m) time and space.
#:
#: ```
#: damerau_levenshtein("ca", "abc")   # 2
#: levenshtein("ca", "abc")           # 3
#: ```
pub fn damerau_levenshtein(a: str, b: str) -> int {
    let source = _chars(a)
    let target = _chars(b)
    let rows = source.len()
    let columns = target.len()
    if rows == 0 { return columns }
    if columns == 0 { return rows }
    let far = rows + columns
    let table = [[0 for _ in 0..=columns + 1] for _ in 0..=rows + 1]
    table[0][0] = far
    for row in 0..=rows {
        table[row + 1][0] = far
        table[row + 1][1] = row
    }
    for column in 0..=columns {
        table[0][column + 1] = far
        table[1][column + 1] = column
    }
    # Last row in which each character of `a` was seen, and, per row, the last
    # column that matched. Together they locate the transposition to undo.
    let last_row: dict[str, int] = {}
    for row in 1..=rows {
        var last_column = 0
        for column in 1..=columns {
            let swap_row = last_row.get(target[column - 1], 0)
            let swap_column = last_column
            var cost = 1
            if source[row - 1] == target[column - 1] {
                cost = 0
                last_column = column
            }
            let substitute = table[row][column] + cost
            let insert = table[row + 1][column] + 1
            let delete = table[row][column + 1] + 1
            let skipped = row - swap_row - 1 + (column - swap_column - 1)
            let transpose = table[swap_row][swap_column] + skipped + 1
            table[row + 1][column + 1] = min(min(substitute, insert), min(delete, transpose))
        }
        last_row[source[row - 1]] = row
    }
    return table[rows + 1][columns + 1]
}

#: Return the number of positions at which `a` and `b` differ.
#:
#: Raises `ValueError` when the lengths differ.
#:
#: ```
#: hamming("karolin", "kathrin")   # 3
#: ```
pub fn hamming(a: str, b: str) -> int {
    if a.len() != b.len() {
        throw ValueError(f"hamming distance needs equal lengths, got {a.len()} and {b.len()}")
    }
    let source = _chars(a)
    let target = _chars(b)
    var distance = 0
    for index in 0..source.len() {
        if source[index] != target[index] { distance += 1 }
    }
    return distance
}

#: Return the Jaro similarity of `a` and `b`, from 0.0 to 1.0.
#:
#: Two characters match when they are equal and no further apart than
#: `max(len(a), len(b)) / 2 - 1` positions.
#:
#: ```
#: jaro("abc", "abc")   # 1.0
#: jaro("abc", "xyz")   # 0.0
#: ```
pub fn jaro(a: str, b: str) -> float {
    let source = _chars(a)
    let target = _chars(b)
    let rows = source.len()
    let columns = target.len()
    if rows == 0 and columns == 0 { return 1.0 }
    if rows == 0 or columns == 0 { return 0.0 }
    var window = max(rows, columns) // 2 - 1
    if window < 0 { window = 0 }
    let source_hit = [false for _ in 0..rows]
    let target_hit = [false for _ in 0..columns]
    var matches = 0
    for index in 0..rows {
        let high = min(index + window + 1, columns)
        var probe = max(0, index - window)
        while probe < high {
            if not target_hit[probe] and source[index] == target[probe] {
                source_hit[index] = true
                target_hit[probe] = true
                matches += 1
                break
            }
            probe += 1
        }
    }
    if matches == 0 { return 0.0 }
    var transpositions = 0
    var cursor = 0
    for index in 0..rows {
        if not source_hit[index] { continue }
        while not target_hit[cursor] { cursor += 1 }
        if source[index] != target[cursor] { transpositions += 1 }
        cursor += 1
    }
    let matched = float(matches)
    let half = float(transpositions // 2)
    return (matched / rows + matched / columns + (matched - half) / matched) / 3.0
}

#: Return the Jaro-Winkler similarity of `a` and `b`, from 0.0 to 1.0.
#:
#: Boosts the Jaro score by a shared prefix of up to four characters, which
#: makes it the better choice for names and typo suggestions. The boost applies
#: only above the customary 0.7 threshold, and `scaling` is capped at 0.25 so
#: the result cannot exceed 1.0.
#:
#: ```
#: jaro_winkler("abc", "abc")   # 1.0
#: ```
pub fn jaro_winkler(a: str, b: str, scaling: float = 0.1) -> float {
    if scaling < 0.0 or scaling > 0.25 {
        throw ValueError(f"scaling must be between 0.0 and 0.25, got {scaling}")
    }
    let base = jaro(a, b)
    if base <= 0.7 { return base }
    let shared = min(common_prefix(a, b).len(), 4)
    return base + shared * scaling * (1.0 - base)
}

#: Return the longest sequence of characters appearing in both `a` and `b`.
#:
#: The characters need not be adjacent, only in order. O(n*m) time and space;
#: use `longest_common_substring` when the run must be contiguous.
#:
#: ```
#: longest_common_subsequence("abcde", "ace")   # "ace"
#: ```
pub fn longest_common_subsequence(a: str, b: str) -> str {
    let source = _chars(a)
    let target = _chars(b)
    let rows = source.len()
    let columns = target.len()
    let table = [[0 for _ in 0..=columns] for _ in 0..=rows]
    for row in 1..=rows {
        for column in 1..=columns {
            if source[row - 1] == target[column - 1] {
                table[row][column] = table[row - 1][column - 1] + 1
            } elif table[row - 1][column] >= table[row][column - 1] {
                table[row][column] = table[row - 1][column]
            } else {
                table[row][column] = table[row][column - 1]
            }
        }
    }
    let backwards: list[str] = []
    var row = rows
    var column = columns
    while row > 0 and column > 0 {
        if source[row - 1] == target[column - 1] {
            backwards.push(source[row - 1])
            row -= 1
            column -= 1
        } elif table[row - 1][column] >= table[row][column - 1] {
            row -= 1
        } else {
            column -= 1
        }
    }
    let out: list[str] = []
    var index = backwards.len() - 1
    while index >= 0 {
        out.push(backwards[index])
        index -= 1
    }
    return "".join(out)
}

#: Return the longest run of adjacent characters appearing in both `a` and `b`.
#:
#: Ties resolve to the earliest run in `a`. O(n*m) time, O(m) space.
#:
#: ```
#: longest_common_substring("badminton", "administer")   # "admin"
#: ```
pub fn longest_common_substring(a: str, b: str) -> str {
    let source = _chars(a)
    let target = _chars(b)
    let columns = target.len()
    var previous = [0 for _ in 0..=columns]
    var current = [0 for _ in 0..=columns]
    var best = 0
    var best_end = 0
    for row in 1..=source.len() {
        for column in 1..=columns {
            if source[row - 1] == target[column - 1] {
                current[column] = previous[column - 1] + 1
                if current[column] > best {
                    best = current[column]
                    best_end = row
                }
            } else {
                current[column] = 0
            }
        }
        let spare = previous
        previous = current
        current = spare
    }
    if best == 0 { return "" }
    return a[best_end - best:best_end]
}

#: Whether `text` reads the same forwards and backwards.
#:
#: With `ignore_punctuation` only letters and digits are compared.
#:
#: ```
#: is_palindrome("racecar")                                      # true
#: is_palindrome("A man, a plan, a canal: Panama", true, true)   # true
#: ```
pub fn is_palindrome(
    text: str,
    ignore_case: bool = false,
    ignore_punctuation: bool = false
) -> bool {
    let kept: list[str] = []
    for ch in text {
        if ignore_punctuation and not _is_word_char(ch) { continue }
        kept.push(ignore_case ? ch.lower() : ch)
    }
    var left = 0
    var right = kept.len() - 1
    while left < right {
        if kept[left] != kept[right] { return false }
        left += 1
        right -= 1
    }
    return true
}

#: Rotate `text` left by `count` characters; a negative `count` rotates right.
#:
#: ```
#: rotate("abcdef", 2)    # "cdefab"
#: rotate("abcdef", -1)   # "fabcde"
#: ```
pub fn rotate(text: str, count: int) -> str {
    let length = text.len()
    if length == 0 { return text }
    var shift = count % length
    if shift < 0 { shift += length }
    if shift == 0 { return text }
    return text[shift:length] + text[0:shift]
}

#: Shuffle the characters of `text` reproducibly for a given `seed`.
#:
#: A Fisher-Yates shuffle driven by an internal linear congruential generator,
#: so one seed always yields one permutation regardless of platform or of any
#: state in `std.random`. Do not use it where unpredictability matters.
#:
#: ```
#: shuffle_deterministic("abc", 1) == shuffle_deterministic("abc", 1)   # true
#: ```
pub fn shuffle_deterministic(text: str, seed: int) -> str {
    let letters = _chars(text)
    var state = seed
    var index = letters.len() - 1
    while index > 0 {
        state = state *% 6364136223846793005 +% 1442695040888963407
        let pick = (state >> 33 & 0x7FFFFFFF) % (index + 1)
        let spare = letters[index]
        letters[index] = letters[pick]
        letters[pick] = spare
        index -= 1
    }
    return "".join(letters)
}

#: Substitute `{name}` placeholders in `template` from `vars`.
#:
#: `{{` and `}}` produce literal braces. An unknown name raises `KeyError`
#: unless `default` is given, which is substituted instead. This is
#: deliberately not the `std.fmt` format language: with no format specs, a
#: template supplied by a user cannot reach anything but the values in `vars`.
#:
#: ```
#: expand("Hello, {name}!", {"name": "Ada"})   # "Hello, Ada!"
#: expand("{a}-{b}", {"a": "1"}, "?")          # "1-?"
#: ```
pub fn expand(template: str, vars: dict[str, str], default: str? = null) -> str {
    let has_default = default != null
    let fallback = default ?? ""
    let out = StringBuilder()
    let length = template.len()
    var index = 0
    while index < length {
        let ch = template[index]
        if ch == "{" and index + 1 < length and template[index + 1] == "{" {
            out.push("{")
            index += 2
        } elif ch == "{" {
            var close = index + 1
            while close < length and template[close] != "}" { close += 1 }
            if close >= length {
                throw ValueError(f"unterminated '{{' in template at index {index}")
            }
            let name = template[index + 1:close]
            if name in vars {
                out.push(vars[name])
            } elif has_default {
                out.push(fallback)
            } else {
                throw KeyError(f"template variable '{name}' has no value")
            }
            index = close + 1
        } elif ch == "}" {
            if index + 1 >= length or template[index + 1] != "}" {
                throw ValueError(f"unmatched '}}' in template at index {index}")
            }
            out.push("}")
            index += 2
        } else {
            out.push(ch)
            index += 1
        }
    }
    return out.build()
}

#: Escape `text` for pasting inside a double-quoted Jaithon literal.
#:
#: Backslash, double quote and the control characters become escapes; every
#: other character, non-ASCII included, is left alone.
#:
#: ```
#: escape("tab\there")   # "tab\\there"
#: ```
pub fn escape(text: str) -> str {
    let out = StringBuilder()
    for ch in text {
        let mapped = _ESCAPES.get(ch, "")
        let code = ord(ch)
        if mapped.len() > 0 {
            out.push(mapped)
        } elif code < 32 or code == 127 {
            out.push("\\x" + _hex_byte(code))
        } else {
            out.push(ch)
        }
    }
    return out.build()
}

#: Interpret the Jaithon escape sequences in `text`.
#:
#: Understands `\n \t \r \\ \" \' \0 \xNN \u{...}`. Raises `ValueError` on an
#: unknown or truncated escape rather than passing it through, because silently
#: keeping a bad escape hides the typo that produced it.
#:
#: ```
#: unescape("a\\nb")        # "a\nb"
#: unescape("\\u{1F600}")   # "😀"
#: ```
pub fn unescape(text: str) -> str {
    let out = StringBuilder()
    let length = text.len()
    var index = 0
    while index < length {
        let ch = text[index]
        if ch != "\\" {
            out.push(ch)
            index += 1
            continue
        }
        if index + 1 >= length { throw ValueError("text ends with a lone backslash") }
        let marker = text[index + 1]
        index += 2
        if marker in _UNESCAPES {
            out.push(_UNESCAPES[marker])
        } elif marker == "x" {
            if index + 2 > length { throw ValueError("truncated '\\xNN' escape") }
            let digits = text[index:index + 2]
            let code = to_int(digits, 16) ?? -1
            if code < 0 { throw ValueError(f"invalid hex escape '\\x{digits}'") }
            out.push(chr(code))
            index += 2
        } elif marker == "u" {
            if index >= length or text[index] != "{" {
                throw ValueError("expected '{' after '\\u'")
            }
            var close = index + 1
            while close < length and text[close] != "}" { close += 1 }
            if close >= length { throw ValueError("unterminated '\\u{...}' escape") }
            let digits = text[index + 1:close]
            let code = to_int(digits, 16) ?? -1
            if code < 0 { throw ValueError(f"invalid unicode escape '\\u{{{digits}}}'") }
            out.push(chr(code))
            index = close + 1
        } else {
            throw ValueError(f"unknown escape '\\{marker}'")
        }
    }
    return out.build()
}

#: Return `text` escaped and wrapped in double quotes.
#:
#: ```
#: quote("a\"b")   # "\"a\\\"b\""
#: ```
pub fn quote(text: str) -> str {
    return "\"" + escape(text) + "\""
}

#: Escape the five characters that are unsafe in HTML text and attributes.
#:
#: ```
#: escape_html("<a href='x'>")   # "&lt;a href=&#39;x&#39;&gt;"
#: ```
pub fn escape_html(text: str) -> str {
    let out = StringBuilder()
    for ch in text { out.push(_HTML_ESCAPES.get(ch, ch)) }
    return out.build()
}

#: Quote `text` so a POSIX shell passes it through as one argument.
#:
#: Text built only from characters no shell treats specially comes back as is;
#: anything else is wrapped in single quotes, with embedded single quotes
#: spliced out and back in. The result carries its own quotes, so never add
#: more around it.
#:
#: ```
#: escape_shell("simple")   # "simple"
#: escape_shell("a b")      # "'a b'"
#: escape_shell("it's")     # "'it'\\''s'"
#: ```
pub fn escape_shell(text: str) -> str {
    if text.len() == 0 { return "''" }
    for ch in text {
        if not _is_ascii_alnum(ch) and ch not in "@%+=:,./-_" {
            return "'" + text.replace("'", "'\\''") + "'"
        }
    }
    return text
}

#: Escape the regular-expression metacharacters in `text`.
#:
#: The result matches `text` literally when compiled by `std.re`.
#:
#: ```
#: escape_regex("a.b*c")   # "a\\.b\\*c"
#: ```
pub fn escape_regex(text: str) -> str {
    let out = StringBuilder()
    for ch in text {
        if ch in "\\^$.|?*+()[]{}" { out.push("\\") }
        out.push(ch)
    }
    return out.build()
}

#: Parse `text` as an integer, or return null when it is not one.
#:
#: Surrounding whitespace is ignored and an optional sign may precede the
#: digits. Digit-group separators are rejected, because input from a user is not
#: source code, and out-of-range values return null rather than wrapping.
#: Failure is ordinary here, so this returns an optional instead of raising and
#: `??` supplies the fallback.
#:
#: ```
#: to_int("42")       # 42
#: to_int(" -7 ")     # -7
#: to_int("ff", 16)   # 255
#: to_int("12.5")     # null
#: ```
pub fn to_int(text: str, base: int = 10) -> int? {
    if base < 2 or base > 36 { throw ValueError(f"base must be 2 to 36, got {base}") }
    let trimmed = text.strip()
    if trimmed.len() == 0 { return null }
    var index = 0
    var negative = false
    if trimmed[0] == "-" or trimmed[0] == "+" {
        negative = trimmed[0] == "-"
        index = 1
    }
    if index >= trimmed.len() { return null }
    # Accumulate towards the negative end: it holds one more value than the
    # positive end, so "-9223372036854775808" parses like any other input.
    var value = 0
    try {
        while index < trimmed.len() {
            let digit = _digit_value(trimmed[index])
            if digit < 0 or digit >= base { return null }
            value = value * base - digit
            index += 1
        }
    } catch _overflow: OverflowError {
        return null
    }
    if negative { return value }
    if value == _INT_MIN { return null }
    return -value
}

#: Parse `text` as a float, or return null when it is not one.
#:
#: Accepts an optional sign, decimal digits with an optional point, and an
#: optional `e` exponent. Named values such as `inf` and `nan` are rejected.
#:
#: ```
#: to_float("3.5")     # 3.5
#: to_float("1e-3")    # 0.001
#: to_float("hello")   # null
#: ```
pub fn to_float(text: str) -> float? {
    let trimmed = text.strip()
    if not _looks_numeric(trimmed) { return null }
    return float(trimmed)
}

#: Whether every character of `text` is in the ASCII range.
#:
#: ```
#: is_ascii("café")   # false
#: ```
pub fn is_ascii(text: str) -> bool {
    for ch in text {
        if ord(ch) > 127 { return false }
    }
    return true
}

#: Fold `text` down to ASCII, transliterating what it can.
#:
#: Accented Latin letters lose their accents, ligatures expand and typographic
#: punctuation becomes its ASCII equivalent. Anything with no sensible ASCII
#: spelling becomes `replacement`. This is for filenames, slugs and identifiers,
#: never for display.
#:
#: ```
#: to_ascii("Crème brûlée")   # "Creme brulee"
#: to_ascii("naïve — ok")     # "naive - ok"
#: ```
pub fn to_ascii(text: str, replacement: str = "?") -> str {
    let out = StringBuilder()
    for ch in text {
        if ord(ch) < 128 {
            out.push(ch)
        } elif ch in _ACCENT_MAP {
            out.push(_ACCENT_MAP[ch])
        } elif ch in _SYMBOL_MAP {
            out.push(_SYMBOL_MAP[ch])
        } else {
            out.push(replacement)
        }
    }
    return out.build()
}

#: Collapse every run of whitespace in `text` to one space and trim the ends.
#:
#: ```
#: normalize_whitespace("  a \n\t b  ")   # "a b"
#: ```
pub fn normalize_whitespace(text: str) -> str {
    return " ".join(_split_whitespace(text))
}

fn _chars(text: str) -> list[str] {
    let out: list[str] = []
    for ch in text { out.push(ch) }
    return out
}

fn _split_whitespace(text: str) -> list[str] {
    let out: list[str] = []
    let current = StringBuilder()
    for ch in text {
        if not _is_space(ch) {
            current.push(ch)
        } elif current.len() > 0 {
            out.push(current.build())
            current.clear()
        }
    }
    if current.len() > 0 { out.push(current.build()) }
    return out
}

fn _leading_whitespace(line: str) -> str {
    var index = 0
    while index < line.len() and _is_space(line[index]) { index += 1 }
    return line[0:index]
}

fn _is_space(ch: str) -> bool { return ch in " \t\n\r\x0B\x0C" }

fn _is_digit(ch: str) -> bool {
    let code = ord(ch)
    return code >= 48 and code <= 57
}

fn _is_ascii_alnum(ch: str) -> bool {
    let code = ord(ch)
    if code >= 48 and code <= 57 { return true }
    if code >= 65 and code <= 90 { return true }
    return code >= 97 and code <= 122
}

fn _is_upper(ch: str) -> bool { return ch != ch.lower() }

fn _is_lower(ch: str) -> bool { return ch != ch.upper() }

# A character belongs to a word when it is a digit, a cased letter, or a CJK
# scalar. CJK has no case distinction, so testing for one would discard it.
fn _is_word_char(ch: str) -> bool {
    if _is_digit(ch) { return true }
    if ch.lower() != ch.upper() { return true }
    return ord(ch) >= 0x3041
}

fn _word_boundary(letters: list[str], index: int) -> bool {
    if not _is_upper(letters[index]) { return false }
    if not _is_upper(letters[index - 1]) { return true }
    return index + 1 < letters.len() and _is_lower(letters[index + 1])
}

fn _capitalize(word: str) -> str {
    if word.len() == 0 { return word }
    return word[0:1].upper() + word[1:word.len()].lower()
}

fn _hex_byte(code: int) -> str {
    let high = code // 16
    let low = code % 16
    return _HEX_DIGITS[high:high + 1] + _HEX_DIGITS[low:low + 1]
}

fn _digit_value(ch: str) -> int {
    let code = ord(ch)
    if code >= 48 and code <= 57 { return code - 48 }
    if code >= 97 and code <= 122 { return code - 87 }
    if code >= 65 and code <= 90 { return code - 55 }
    return -1
}

fn _looks_numeric(text: str) -> bool {
    let length = text.len()
    var index = 0
    var mantissa_digits = 0
    if index < length and (text[index] == "-" or text[index] == "+") { index += 1 }
    while index < length and _is_digit(text[index]) {
        index += 1
        mantissa_digits += 1
    }
    if index < length and text[index] == "." {
        index += 1
        while index < length and _is_digit(text[index]) {
            index += 1
            mantissa_digits += 1
        }
    }
    if mantissa_digits == 0 { return false }
    if index < length and (text[index] == "e" or text[index] == "E") {
        index += 1
        if index < length and (text[index] == "-" or text[index] == "+") { index += 1 }
        var exponent_digits = 0
        while index < length and _is_digit(text[index]) {
            index += 1
            exponent_digits += 1
        }
        if exponent_digits == 0 { return false }
    }
    return index == length
}
