#: The same miscompile as its sibling, reached with no environment variable at
#: all -- a plain `./jaithon run` prints the pointer roughly one run in six,
#: because whether the sampler picks this loop is a race. Under
#: JAITHON_JIT_TICK_US=50 or JAITHON_JIT_DEOPT_STRESS=1 it is every run.
#:
#:   ./jaithon run tests/fuzz/found/osr_dynamic_local_loses_its_tag_hot.jai
fn cfprobe(mode: int) -> str {
    var text = ""
    var slot: any = 0
    for _i0 in 0..64 {
        if mode == 1 { slot = "s5" } else { slot = 1 }
        text = text + f"{slot}|"
        slot = 0
    }
    return text
}

fn main() -> int {
    var w = ""
    for _r in 0..300 { w = cfprobe(0) }
    print(w.len())
    print(cfprobe(1))
    return 0
}
