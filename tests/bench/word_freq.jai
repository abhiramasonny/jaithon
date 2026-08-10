#: Word counting by scanning one long string.
#:
#: `string_build` measures concatenation and `dict_ops` measures hashing of
#: keys that already exist. This measures the third thing: cutting substrings
#: out of a buffer and hashing each one once, which is what a scanner does.
fn main() -> int {
    var text = ""
    var seed = 7
    for _i in 0..40_000 {
        seed = (seed * 1103515245 + 12345) % 2147483648
        text = text + f"w{seed % 500} "
    }

    var counts = {}
    var start = 0
    var at = 0
    let n = text.len()
    while at < n {
        if text[at] == " " {
            if at > start {
                let word = text[start:at]
                counts[word] = counts.get(word, 0) + 1
            }
            start = at + 1
        }
        at += 1
    }

    var total = 0
    for (_key, value) in counts.items() { total += value }
    print(counts.len())
    print(total)
    return 0
}
