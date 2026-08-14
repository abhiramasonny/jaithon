#: Bytes in and out of text: UTF-8, base64, hex, and the two checks that say
#: whether the bytes arrived intact.
#:
#: `utf8_encode`/`utf8_decode` convert between `str` and the `list[int]` of
#: 0..=255 that the rest of this file works on. `base64_encode` and
#: `hex_encode` then move those bytes through a text-only channel, and
#: `crc32` and `fnv1a` say whether they arrived intact — a checksum for
#: accidental corruption and a hash for keying, not the same job.
const BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
const HEX_LOWER = "0123456789abcdef"
const HEX_UPPER = "0123456789ABCDEF"
const CRC32_POLYNOMIAL = 0xEDB88320
const FNV_OFFSET_BASIS = 2166136261
const FNV_PRIME = 16777619

let _BASE64_VALUES: dict[str, int] = {digit: value for (digit, value) in zip(BASE64_ALPHABET, range(64))}

#: Encode `text` as UTF-8 bytes.
#:
#: ```
#: utf8_encode("hé")   # [104, 195, 169]
#: ```
pub fn utf8_encode(text: str) -> list[int] {
    let out: list[int] = []
    for ch in text {
        let code = ord(ch)
        if code < 0x80 {
            out.push(code)
        } elif code < 0x800 {
            out.push(0xC0 | code >> 6)
            out.push(0x80 | code & 0x3F)
        } elif code < 0x10000 {
            out.push(0xE0 | code >> 12)
            out.push(0x80 | code >> 6 & 0x3F)
            out.push(0x80 | code & 0x3F)
        } else {
            out.push(0xF0 | code >> 18)
            out.push(0x80 | code >> 12 & 0x3F)
            out.push(0x80 | code >> 6 & 0x3F)
            out.push(0x80 | code & 0x3F)
        }
    }
    return out
}

#: Decode UTF-8 bytes into a string.
#:
#: Raises `ValueError` on a truncated sequence or on a byte that can neither
#: start nor continue one, rather than substituting a replacement character: a
#: decoder that silently repairs its input hides the corruption that produced
#: it.
#:
#: ```
#: utf8_decode([104, 195, 169])   # "hé"
#: ```
pub fn utf8_decode(data: list[int]) -> str {
    _check_bytes(data)
    let out: list[str] = []
    let n = data.len()
    var index = 0
    while index < n {
        let lead = data[index]
        var code = 0
        var extra = 0
        if lead < 0x80 {
            code = lead
        } elif lead >= 0xF0 {
            code = lead & 0x07
            extra = 3
        } elif lead >= 0xE0 {
            code = lead & 0x0F
            extra = 2
        } elif lead >= 0xC0 {
            code = lead & 0x1F
            extra = 1
        } else {
            throw ValueError(f"byte {lead} at index {index} cannot start a sequence")
        }
        if index + extra >= n { throw ValueError(f"sequence at index {index} is truncated") }
        for step in 1..=extra {
            let continuation = data[index + step]
            if continuation < 0x80 or continuation > 0xBF {
                throw ValueError(
                    f"byte {continuation} at index {index + step} is not a continuation"
                )
            }
            code = code << 6 | continuation & 0x3F
        }
        out.push(chr(code))
        index += extra + 1
    }
    return "".join(out)
}

#: Encode bytes as base64, standard alphabet, padded with `=`.
#:
#: ```
#: base64_encode(utf8_encode("hello"))   # "aGVsbG8="
#: ```
pub fn base64_encode(data: list[int]) -> str {
    _check_bytes(data)
    let out: list[str] = []
    let n = data.len()
    var index = 0
    while index + 2 < n {
        let chunk = data[index] << 16 | data[index + 1] << 8 | data[index + 2]
        out.push(_base64_digit(chunk >> 18))
        out.push(_base64_digit(chunk >> 12))
        out.push(_base64_digit(chunk >> 6))
        out.push(_base64_digit(chunk))
        index += 3
    }
    let rest = n - index
    if rest == 1 {
        let chunk = data[index] << 16
        out.push(_base64_digit(chunk >> 18))
        out.push(_base64_digit(chunk >> 12))
        out.push("==")
    } elif rest == 2 {
        let chunk = data[index] << 16 | data[index + 1] << 8
        out.push(_base64_digit(chunk >> 18))
        out.push(_base64_digit(chunk >> 12))
        out.push(_base64_digit(chunk >> 6))
        out.push("=")
    }
    return "".join(out)
}

#: Decode base64 back into bytes.
#:
#: ASCII whitespace is skipped, so text wrapped at 76 columns decodes as it
#: stands, and everything from the first `=` on is ignored. Raises `ValueError`
#: on a character outside the alphabet or on a final group of one digit, which
#: encodes no whole byte.
#:
#: ```
#: utf8_decode(base64_decode("aGVsbG8="))   # "hello"
#: ```
pub fn base64_decode(text: str) -> list[int] {
    let values: list[int] = []
    for ch in text {
        if ch in " \t\n\r" { continue }
        if ch == "=" { break }
        if ch not in _BASE64_VALUES { throw ValueError(f"'{ch}' is not a base64 character") }
        values.push(_BASE64_VALUES[ch])
    }
    let out: list[int] = []
    var index = 0
    while index + 3 < values.len() {
        let chunk = values[index] << 18 |
            values[index + 1] << 12 |
            values[index + 2] << 6 |
            values[index + 3]
        out.push(chunk >> 16 & 0xFF)
        out.push(chunk >> 8 & 0xFF)
        out.push(chunk & 0xFF)
        index += 4
    }
    let rest = values.len() - index
    if rest == 1 { throw ValueError("base64 input ends with one leftover character") }
    if rest == 2 {
        let chunk = values[index] << 18 | values[index + 1] << 12
        out.push(chunk >> 16 & 0xFF)
    } elif rest == 3 {
        let chunk = values[index] << 18 | values[index + 1] << 12 | values[index + 2] << 6
        out.push(chunk >> 16 & 0xFF)
        out.push(chunk >> 8 & 0xFF)
    }
    return out
}

#: Encode bytes as hexadecimal, two characters per byte.
#:
#: ```
#: hex_encode([255, 0, 16])   # "ff0010"
#: hex_encode([255], true)    # "FF"
#: ```
pub fn hex_encode(data: list[int], upper: bool = false) -> str {
    _check_bytes(data)
    let digits = upper ? HEX_UPPER : HEX_LOWER
    let out: list[str] = []
    for value in data {
        let high = value >> 4
        let low = value & 0xF
        out.push(digits[high:high + 1])
        out.push(digits[low:low + 1])
    }
    return "".join(out)
}

#: Decode hexadecimal, in either case, back into bytes.
#:
#: Raises `ValueError` on an odd number of digits or on a non-hex character.
#:
#: ```
#: hex_decode("ff0010")   # [255, 0, 16]
#: ```
pub fn hex_decode(text: str) -> list[int] {
    if text.len() % 2 != 0 {
        throw ValueError(f"hex input must have an even length, got {text.len()}")
    }
    let chars = _chars(text)
    let out: list[int] = []
    var index = 0
    while index < chars.len() {
        let high = _hex_value(chars[index])
        let low = _hex_value(chars[index + 1])
        if high < 0 or low < 0 {
            throw ValueError(f"'{chars[index]}{chars[index + 1]}' is not a hex byte")
        }
        out.push(high << 4 | low)
        index += 2
    }
    return out
}

#: Return the CRC-32 checksum of `data`, as used by zip, gzip and PNG.
#:
#: The reflected IEEE 802.3 polynomial, computed bit by bit: eight shifts per
#: byte and no 256-entry table to build or carry. A CRC detects accidental
#: corruption, burst errors especially; it is not a hash, and forging a
#: collision for one is arithmetic, not work.
#:
#: ```
#: crc32(utf8_encode("hello"))   # 907060870
#: ```
pub fn crc32(data: list[int]) -> int {
    _check_bytes(data)
    var crc = 0xFFFFFFFF
    for byte in data {
        crc = crc ^ byte
        for _ in 0..8 {
            let feedback = crc & 1
            crc = crc >> 1
            if feedback == 1 { crc = crc ^ CRC32_POLYNOMIAL }
        }
    }
    return crc ^ 0xFFFFFFFF
}

#: Return the 32-bit FNV-1a hash of `data`.
#:
#: One xor and one multiply per byte, which makes it the cheapest hash worth
#: using for short keys: dict keys, string interning, bloom filters. Like every
#: non-cryptographic hash it is easy to find collisions for, so never use it on
#: input an adversary chooses.
#:
#: ```
#: fnv1a(utf8_encode("hello"))   # 1335831723
#: fnv1a([])                     # 2166136261
#: ```
pub fn fnv1a(data: list[int]) -> int {
    _check_bytes(data)
    var hash = FNV_OFFSET_BASIS
    for byte in data {
        hash = (hash ^ byte) * FNV_PRIME & 0xFFFFFFFF
    }
    return hash
}

fn _chars(text: str) -> list[str] {
    let out: list[str] = []
    for ch in text { out.push(ch) }
    return out
}

fn _hex_value(ch: str) -> int {
    let code = ord(ch)
    if code >= 48 and code <= 57 { return code - 48 }
    if code >= 97 and code <= 102 { return code - 87 }
    if code >= 65 and code <= 70 { return code - 55 }
    return -1
}

fn _base64_digit(value: int) -> str {
    let index = value & 0x3F
    return BASE64_ALPHABET[index:index + 1]
}

fn _check_bytes(data: list[int]) -> void {
    for (index, value) in data.enumerate() {
        if value < 0 or value > 255 {
            throw ValueError(f"byte at index {index} must be in 0..=255, got {value}")
        }
    }
}
