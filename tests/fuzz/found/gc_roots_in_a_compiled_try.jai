fn cfprobe(mode: int) -> str {
    var text = ""
    for _i0 in 0..36 {
        text = text + "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
        if _i0 % 5 == 1 {
            let _den = 1
        } else {
            try {
                if mode == 0 { throw IndexError("x") }
            } catch _e: IndexError {
            } finally {
            }
        }
    }
    return text
}

fn main() -> int {
    var w = ""
    for _r in 0..40000 { w = cfprobe(0) }
    print(w.len())
    return 0
}
