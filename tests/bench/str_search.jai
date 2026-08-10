#: Naive substring search over one buffer that is read many times.
#:
#: `string_build` and `word_freq` both spend most of their time making strings.
#: This one makes its text once, out of eight fixed chunks so the build stays a
#: rounding error, and after that only reads it. What is left is indexing a
#: string and comparing a character at a time — the reading half of string
#: work, which nothing else in the suite measures on its own.
fn count_occurrences(text: str, needle: str) -> int {
    let n = text.len()
    let m = needle.len()
    let first = needle[0]
    var hits = 0
    var i = 0
    while i + m <= n {
        if text[i] == first {
            var k = 1
            while k < m and text[i + k] == needle[k] { k += 1 }
            if k == m { hits += 1 }
        }
        i += 1
    }
    return hits
}

fn main() -> int {
    let chunks = ["abcdbadc", "bcadcbda", "cdabacbd", "dacbdabc",
                  "abdcadbc", "bdacbcad", "cabdbdca", "dbcaabcd"]
    var parts = []
    var seed = 7
    for _i in 0..25_000 {
        seed = (seed * 1103515245 + 12345) % 2147483648
        parts.push(chunks[(seed // 65536) % 8])
    }
    let text = "".join(parts)

    var hits = 0
    for _rep in 0..5 { hits = count_occurrences(text, "abcd") }
    print(text.len())
    print(hits)
    return 0
}
