#: Count word frequencies in a file and print the most common.
#:
#:     jaithon run examples/wordcount.jai README.md 10
import std.io
from std.collections import Counter

fn tokenize(text: str) -> list[str] {
    var words = []
    var current = ""
    for ch in text.lower().chars() {
        if ch.is_alnum() or ch == "'" {
            current += ch
        } elif current.len() > 0 {
            words.push(current)
            current = ""
        }
    }
    if current.len() > 0 { words.push(current) }
    return words
}

fn main(args: list[str]) -> int {
    if args.len() < 2 {
        print("usage: wordcount <file> [top-n]")
        return 1
    }

    let path = args[1]
    let top = if args.len() > 2 { args[2].to_int() ?? 10 } else { 10 }

    var text = ""
    try {
        text = io.read_file(path)
    } catch e: IOError {
        print(f"cannot read {path}: {e.message}")
        return 1
    }

    let words = tokenize(text)
    let counts = Counter[str].from_list(words)

    print(f"{words.len()} words, {counts.len()} distinct")
    print("")

    # Ranked once: `most_common` sorts the whole table on every call, and the
    # bar needs the largest count to scale against.
    let ranked = counts.most_common(top)
    if ranked.len() == 0 { return 0 }
    let width = ranked.map(|pair| pair[0].len()).max() ?? 0
    let busiest = ranked[0][1]

    for (word, count) in ranked {
        let bar = "█" * (count * 40 // busiest)
        print(f"{word.pad_right(width)}  {count.to_str().pad_left(5)}  {bar}")
    }
    return 0
}
