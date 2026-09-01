#: Segfault, JIT only. `jaithon run` crashes in jaiGCCollect; JAITHON_NO_JIT=1
#: is clean. Reduced from tests/fuzz/control_flow.py --seed 20184.
#:
#:   ./jaithon run tests/fuzz/found/gc_roots_in_a_compiled_try.jai
#:   JAITHON_NO_JIT=1 ./jaithon run tests/fuzz/found/gc_roots_in_a_compiled_try.jai
#:
#: Every ingredient is load-bearing: drop the string append, the finally, the
#: throw, the catch, or the local declared in the sibling branch and it stops.
#: Shrink the loop below 36 or the call count below the tier's 64 and it stops.
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
