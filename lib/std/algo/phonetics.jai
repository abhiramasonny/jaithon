#: Phonetic keys: strings spelled differently that sound alike collapse to the
#: same key. Soundex (1918) is the coarse, English-only original; Metaphone
#: (1990) is the finer-grained successor that keeps sound order instead of
#: packing everything into four characters.
let _SOUNDEX_DIGITS: dict[str, str] = {
    "B": "1",
    "F": "1",
    "P": "1",
    "V": "1",
    "C": "2",
    "G": "2",
    "J": "2",
    "K": "2",
    "Q": "2",
    "S": "2",
    "X": "2",
    "Z": "2",
    "D": "3",
    "T": "3",
    "L": "4",
    "M": "5",
    "N": "5",
    "R": "6",
}

#: Return the four-character Soundex key of `name`.
#:
#: An initial letter and three digits standing for the consonant sounds after
#: it, so that names spelled differently but said alike collide. Vowels, `H`
#: and `W` drop out, but only a vowel resets the rule that collapses a repeated
#: digit — which is why the `2` and `1` of `Ashcraft` survive the `H` between
#: them. Non-letters are ignored, and a name with no letters returns "".
#:
#: Soundex is coarse and English-specific. `metaphone` is the better key for
#: anything beyond the surname matching this was designed for in 1918.
#:
#: ```
#: soundex("Robert")     # "R163"
#: soundex("Rupert")     # "R163"
#: soundex("Ashcraft")   # "A261"
#: ```
pub fn soundex(name: str) -> str {
    let letters = _letters(name)
    if letters.len() == 0 { return "" }
    let out: list[str] = [letters[0]]
    var previous = _SOUNDEX_DIGITS.get(letters[0], "0")
    var index = 1
    while index < letters.len() and out.len() < 4 {
        let letter = letters[index]
        let digit = _SOUNDEX_DIGITS.get(letter, "0")
        if digit != "0" {
            if digit != previous { out.push(digit) }
            previous = digit
        } elif letter != "H" and letter != "W" {
            previous = "0"
        }
        index += 1
    }
    while out.len() < 4 { out.push("0") }
    return "".join(out)
}

#: Return the Metaphone key of `word`.
#:
#: Lawrence Philips' original 1990 rules: the word is rewritten as the
#: consonant sounds of English, with `0` standing for the "th" sound and `X`
#: for "sh". Unlike Soundex the key has no fixed length and keeps the order of
#: the sounds, so it discriminates far better on ordinary words. Vowels survive
#: only in first position, and silent letters — the `K` of `knight`, the `GH`
#: of `night`, the `B` of `lamb` — go by rule rather than by table.
#:
#: Non-letters are ignored. This is the original algorithm, not Double
#: Metaphone: one key per word, and tuned for English.
#:
#: ```
#: metaphone("phone")                          # "FN"
#: metaphone("Smith")                          # "SM0"
#: metaphone("knight") == metaphone("night")   # true, both "NT"
#: ```
pub fn metaphone(word: str) -> str {
    let letters = _letters(word)
    let n = letters.len()
    if n == 0 { return "" }
    let out: list[str] = []
    var index = 0
    # Where the word's first sound begins. Dropping a silent first letter moves
    # it, so that the `E` of "aero" is still an initial vowel and survives.
    var first_sound = 0
    # Silent first letters, which no later rule can catch because they are
    # silent only at the very start.
    let opening = n >= 2 ? letters[0] + letters[1] : ""
    if opening == "AE" or opening == "GN" or opening == "KN" or opening == "PN" or opening == "WR" {
        index = 1
        first_sound = 1
    } elif letters[0] == "X" {
        out.push("S")
        index = 1
    } elif opening == "WH" {
        out.push("W")
        index = 2
    }
    while index < n {
        let ch = letters[index]
        let next = _at(letters, index + 1)
        let after = _at(letters, index + 2)
        let before = _at(letters, index - 1)
        # A doubled letter says nothing twice, except CC as in "accident".
        if ch != "C" and before == ch {
            index += 1
            continue
        }
        if _is_vowel(ch) {
            if index == first_sound { out.push(ch) }
        } elif ch == "B" {
            if index != n - 1 or before != "M" { out.push("B") }
        } elif ch == "C" {
            if next == "I" and after == "A" {
                out.push("X")
            } elif next == "H" {
                out.push("X")
            } elif next == "I" or next == "E" or next == "Y" {
                # SC before a front vowel is a single S, as in "science".
                if before != "S" { out.push("S") }
            } else {
                out.push("K")
            }
        } elif ch == "D" {
            if next == "G" and (after == "E" or after == "Y" or after == "I") {
                out.push("J")
                index += 2
            } else {
                out.push("T")
            }
        } elif ch == "G" {
            if next == "H" {
                # GH speaks only at the end of a word or before a vowel.
                if index + 1 == n - 1 or _is_vowel(after) { out.push("K") } else { index += 1 }
            } elif next == "I" or next == "E" or next == "Y" {
                out.push("J")
            } elif next != "N" {
                out.push("K")
            }
        } elif ch == "H" {
            let after_vowel_only = _is_vowel(before) and not _is_vowel(next)
            let part_of_digraph = before == "C" or
                before == "S" or
                before == "P" or
                before == "T" or
                before == "G"
            if not after_vowel_only and not part_of_digraph { out.push("H") }
        } elif ch == "K" {
            if before != "C" { out.push("K") }
        } elif ch == "P" {
            if next == "H" {
                out.push("F")
                index += 1
            } else {
                out.push("P")
            }
        } elif ch == "Q" {
            out.push("K")
        } elif ch == "S" {
            if next == "H" {
                out.push("X")
                index += 1
            } elif next == "I" and (after == "O" or after == "A") {
                out.push("X")
            } else {
                out.push("S")
            }
        } elif ch == "T" {
            if next == "I" and (after == "O" or after == "A") {
                out.push("X")
            } elif next == "H" {
                out.push("0")
                index += 1
            } elif not (next == "C" and after == "H") {
                out.push("T")
            }
        } elif ch == "V" {
            out.push("F")
        } elif ch == "W" or ch == "Y" {
            # A semivowel is only heard when a vowel follows it.
            if _is_vowel(next) { out.push(ch) }
        } elif ch == "X" {
            out.push("KS")
        } elif ch == "Z" {
            out.push("S")
        } else {
            out.push(ch)
        }
        index += 1
    }
    return "".join(out)
}

fn _letters(text: str) -> list[str] {
    let out: list[str] = []
    for ch in text.upper() {
        let code = ord(ch)
        if code >= 65 and code <= 90 { out.push(ch) }
    }
    return out
}

fn _at(letters: list[str], index: int) -> str {
    if index < 0 or index >= letters.len() { return "" }
    return letters[index]
}

# Guards against the empty string that `_at` returns past either end: "" is a
# substring of every string, so `"" in "AEIOU"` would be true.
fn _is_vowel(ch: str) -> bool {
    return ch.len() == 1 and ch in "AEIOU"
}
