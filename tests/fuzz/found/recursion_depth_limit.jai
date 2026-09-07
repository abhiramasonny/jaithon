#: differential-exempt: yes
#: The compiled tier does not enforce the interpreter's frame limit, so this
#: prints 1100 with the JIT on and RuntimeError without it. All six compiled
#: configurations disagree with the reference. DELIBERATE, and now written
#: down: see "The one divergence that is deliberate" in
#: src/vm/jit/README.md. Kept as the record of it, not as a bug -- and as
#: the reason a generated program must not recurse past JAI_FRAMES_MAX,
#: since the oracle cannot tell a designed divergence from a miscompile.
#:
#:   ./jaithon run tests/fuzz/found/recursion_depth_limit.jai            # 1100
#:   JAITHON_NO_JIT=1 ./jaithon run tests/fuzz/found/recursion_depth_limit.jai
#:
#: The interpreter's last good depth is 1021 (JAI_FRAMES_MAX 1024); compiled,
#: deep(12000) still returns and deep(16000) raises.
fn deep(n: int) -> int {
    if n <= 0 { return 0 }
    return 1 + deep(n - 1)
}

fn main() -> int {
    for _r in 0..64 { deep(1) }
    try {
        print(deep(1100))
    } catch _e: RuntimeError {
        print("RuntimeError")
    }
    return 0
}
