#: The same dynamic-local bug seen through `is null` rather than a print, and
#: with no allocation anywhere: `slot` is null on every iteration, so the
#: interpreter counts 12 of them and answers 108. The compiled loop sees it on
#: the first iteration only and answers 9.
#:
#:   JAITHON_NO_JIT=1 ./jaithon run tests/fuzz/found/osr_dynamic_local_is_null.jai   # 108
#:   JAITHON_JIT_TICK_US=50 ./jaithon run tests/fuzz/found/osr_dynamic_local_is_null.jai   # 9
#:
#: Reduced from tests/fuzz/control_flow.py --seed 800597. Writing `slot ==
#: null` instead of `slot is null` makes it agree; `slot ?? 9` does not.
fn cfprobe(mode: int) -> int {
    var acc = 0
    var slot: any = 0
    for _i0 in 0..12 {
        if mode == 2 { slot = null } else { slot = 7 }
        if slot is null { acc = acc + 9 }
        slot = 5
    }
    return acc
}

fn main() -> int {
    print(cfprobe(2))
    return 0
}
