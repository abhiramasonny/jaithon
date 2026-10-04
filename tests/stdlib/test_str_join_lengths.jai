#: `sep.join(items)` for every short item length and separator length.
#:
#: The list join copies items of sixteen bytes or fewer with overlapping word
#: moves rather than a memcpy call each (builtins_str_split.c, copyShort), and
#: each length from zero to sixteen takes a slightly different path through
#: it. Checked against concatenation one piece at a time, with NULs and
#: multi-byte text in the items so a byte copied from the wrong place shows.
from std.test import assert_eq

fn by_hand(sep: str, items: list[str]) -> str {
    var out = ""
    var i = 0
    while i < items.len() {
        if i > 0 { out = out + sep }
        out = out + items[i]
        i += 1
    }
    return out
}

fn piece(n: int, seed: int) -> str {
    let alphabet = "abcdefghijklmnopqrstuvwxyz\0ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    var out = ""
    var i = 0
    while i < n {
        out = out + alphabet[(seed * 7 + i * 13) % alphabet.len()]
        i += 1
    }
    return out
}

pub fn test_every_short_length() -> void {
    for sepLen in [0, 1, 2, 3, 5, 8, 9, 16, 17, 20] {
        let sep = piece(sepLen, 99)
        var items: list[str] = []
        for n in 0..24 { items.push(piece(n, n)) }
        assert_eq(sep.join(items), by_hand(sep, items), f"separator of {sepLen}")
    }
}

pub fn test_multibyte_and_many() -> void {
    var items: list[str] = []
    for i in 0..120 { items.push(if i % 3 == 0 { "été" } elif i % 3 == 1 { "中" } else { "" }) }
    assert_eq(",".join(items), by_hand(",", items), "multi-byte items")
    assert_eq("".join(items).len(), by_hand("", items).len(), "length in characters")
}

pub fn test_tuple_and_empty() -> void {
    assert_eq("-".join(("a", "bc", "")), "a-bc-", "a tuple")
    assert_eq("-".join([]), "", "nothing")
    assert_eq("-".join(["only"]), "only", "one item")
}
