#: `for c in <string>` in JIT-warmed bodies, across the byte ranges the inline
#: step treats differently. The two tiers must agree exactly.
#:
#: Fifth of the tier-agreement set. emitForIterBind grew a shape-5 arm that
#: steps an ITER_STRING inline: `index` is a BYTE offset, the ASCII case is one
#: ldrb plus a load from the shared one-byte string table, and a byte >= 0x80
#: DEOPTIMISES because the interpreter allocates a fresh ObjString for a
#: multi-byte scalar. Before it, every `for c in text` bailed at the loop head
#: on every call and ran interpreted -- 290ms against 30ms for the same work as
#: an indexed `while`.
#:
#: The edges chosen are the ones whose two paths could plausibly disagree: pure
#: ASCII (the inline step), two-, three- and four-byte UTF-8 (the deopt, which
#: must resume with NOTHING advanced or a scalar is skipped), a string that
#: mixes them so the loop bails and re-enters mid-walk, the empty string, a
#: one-character string, a string whose LAST scalar is multi-byte, and a
#: character compared and concatenated afterwards so the yielded object is
#: proved to be a real interned string and not a raw byte.
#:
#: gc-stress-every: 2000
#:
#: The three `osr_scan` strings are large on purpose -- a per-character loop has
#: to be long enough to reach the OSR tier at all, and the density census only
#: runs there. That makes this golden the slowest in the set under --gc-stress,
#: hence the interval above.
#:
#: Every body is called 400 times before the edge value arrives, so the edge is
#: taken in compiled code. Output must be byte-identical under JAITHON_NO_JIT=1;
#: the golden runner also re-runs it at JAITHON_JIT_THRESHOLD=1 and under
#: DEOPT_STRESS.

fn count_chars(text: str) -> int {
    var n = 0
    for _c in text { n += 1 }
    return n
}

fn count_newlines(text: str) -> int {
    var n = 0
    for c in text { if c == "\n" { n += 1 } }
    return n
}

fn join_back(text: str) -> str {
    var out = ""
    for c in text { out = out + c }
    return out
}

fn first_upper(text: str) -> str {
    for c in text {
        if c >= "A" and c <= "Z" { return c }
    }
    return "-"
}

#: Long enough to reach the OSR tier, which has its own head (iterKind 6) with
#: its own density census -- the bodies above are all function-tier.
fn osr_scan(text: str) -> int {
    var n = 0
    var runs = 0
    for c in text {
        n += 1
        if c == "z" { runs += 1 }
    }
    return n * 2 + runs
}

fn nested_count(rows: list[str]) -> int {
    var n = 0
    for r in rows {
        for _c in r { n += 1 }
    }
    return n
}

fn main() {
    let ascii = "abc def\nghi\n"
    let two = "café ñandú"
    let three = "日本語テキスト"
    let four = "emoji 😀🎉 end"
    let mixed = "a€b\nc😀d"
    let empty = ""
    let one = "x"
    let tail = "abc€"
    # A return from INSIDE the loop, leaving the iterator live.
    let upper = "ab€Cd"

    # Warm every body on pure ASCII so the inline step is what compiles.
    var warm = 0
    for _i in 0..400 {
        warm += count_chars(ascii)
        warm += count_newlines(ascii)
        warm += nested_count([ascii, one, empty])
    }
    var warms = ""
    for _i in 0..400 { warms = join_back(ascii) }
    var warmu = ""
    for _i in 0..400 { warmu = first_upper("no upper here") }

    print("warm " + str(warm) + " " + warms + " " + warmu)

    let names = ["ascii", "two", "three", "four", "mixed", "empty", "one",
                 "tail", "upper"]
    let texts = [ascii, two, three, four, mixed, empty, one, tail, upper]
    for i in 0..names.len() {
        let name = names[i]
        let text = texts[i]
        print(name
              + " len=" + str(count_chars(text))
              + " nl=" + str(count_newlines(text))
              + " same=" + str(join_back(text) == text)
              + " up=" + first_upper(text))
    }

    print("nested " + str(nested_count([ascii, two, three, four, mixed,
                                        empty, one, tail, upper])))

    # The OSR head: pure ASCII (compiles), a light sprinkle of multi-byte
    # scalars (compiles, and bails on each one), and mostly non-ASCII (the
    # density census refuses it and the interpreter walks it).
    let big_ascii = "abcdefghijklmnopqrstuvwxyz\n" * 8000
    let big_light = ("abcdefghij" * 20 + "é\n") * 400
    let big_wide = "日本語テキスト\n" * 8000
    print("osr ascii " + str(osr_scan(big_ascii)))
    print("osr light " + str(osr_scan(big_light)))
    print("osr wide " + str(osr_scan(big_wide)))

    # The loop variable escaping the loop: it must be a real interned string.
    var last = "-"
    for c in tail { last = c }
    print("last " + last + " eq=" + str(last == "€") + " len=" + str(last.len()))
}
