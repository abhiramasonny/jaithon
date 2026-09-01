#: The same segfault as its sibling, small enough to run under a debugger, but
#: it needs the collector pushed rather than waiting for one:
#:
#:   ./jaithon run --gc-stress=64 tests/fuzz/found/gc_roots_in_a_compiled_try_small.jai
fn cfprobe(mode: int) -> str {
    var text = ""
    for _i0 in 0..36 {
        text = text + "z"
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
    for _r in 0..70 { w = cfprobe(0) }
    print(w.len())
    return 0
}
