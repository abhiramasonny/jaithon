#: Silent miscompile. `slot` is an `any` local written with two kinds in one
#: loop body -- a dynamic local -- and the OSR loop tier reads every iteration
#: after the first through the kind it recorded at loop entry. The string's
#: HEAP ADDRESS is printed as a decimal integer, and the number changes run to
#: run with ASLR.
#:
#:   JAITHON_NO_JIT=1 ./jaithon run tests/fuzz/found/osr_dynamic_local_loses_its_tag.jai
#:       s5|s5|s5|...
#:   JAITHON_JIT_TICK_US=50 ./jaithon run tests/fuzz/found/osr_dynamic_local_loses_its_tag.jai
#:       s5|42825467824|42825467824|...
#:
#: Reduced from tests/fuzz/control_flow.py --seed 800558. TICK_US makes it
#: every run; see the _hot sibling for a stock `jaithon run`.
#: `[jit] osr cfprobe at 24 stopped: local 3 was given two kinds, int and
#: instance-or-null`, then it retries and compiles the loop anyway.
fn cfprobe(mode: int) -> str {
    var text = ""
    var slot: any = 0
    for _i0 in 0..21 {
        if mode == 1 { slot = "s5" } else { slot = 1 }
        text = text + f"{slot}|"
        slot = 0
    }
    return text
}

fn main() -> int {
    print(cfprobe(1))
    return 0
}
