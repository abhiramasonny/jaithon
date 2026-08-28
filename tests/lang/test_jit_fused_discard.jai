#: A `-> void` method called as the LAST STATEMENT of a branch.
#:
#: The invoke arm has always had an escape hatch for a result nothing observes:
#: if the next instruction is `OP_POP`, no return kind has to be predicted or
#: guarded. But `lib/jaithon/compile/opt/peephole.jai` fuses `Pop; ReturnNull`
#: into `OP_POP_RETURN_NULL` at the default -O2 -- so the commonest place such a
#: call appears never matched, and the whole enclosing function declined with
#: "callee's return kind not usable". The hatch was written against
#: un-optimised bytecode.
#:
#: It was the single most expensive miss in the tier. `_scan_token`, the lexer's
#: entry point, was blocked by `self._line_continuation()` -- a `-> void` method
#: on a backslash branch that no source file in this tree even takes. Five
#: compile attempts, then the retry budget ran out and the body was refused for
#: the life of the process.
#:
#: Measured through `JAITHON_JIT_FUSED_DISCARD`, interpreted instructions on
#: `check --no-cache` (load-independent, and this arm predicts nothing so there
#: is no deopt round trip for a count to be wrong about):
#:
#:   lexer.jai     29,588,746 -> 27,353,718   -7.6%
#:   parser.jai    63,315,753 -> 59,418,638   -6.2%
#:   resolve.jai   29,175,238 -> 27,144,751   -7.0%
#:   emit.jai      62,252,754 -> 59,315,241   -4.7%
#:
#: `_scan_token` alone goes from 2,607,687 attributed interpreted instructions
#: to 377,936, an 85% cut. The wall clock moves 1-6% across the four files, all
#: in the same direction, but at load 7 that is near this machine's resolution
#: and the counts are the honest evidence.
#:
#: The fused form carries the function's RETURN with it, so it cannot simply be
#: stepped over the way a bare `OP_POP` is -- the arm emits that half itself and
#: sets `afterUncond`, which the invoke's own flag gets wrong because an invoke
#: falls through and a return does not.
#: The marker names `run_steps`, which compiles either way -- the body the fix
#: frees is `Machine.step`, and a `# jit-compiles:` marker can only name a
#: top-level `fn`. So this file guards the ANSWERS across both switch states,
#: the interpreter and the stress modes; the tier evidence is the interpreted
#: counts above, which are load-independent.
# jit-compiles: run_steps
from std.test import assert_eq, assert_true

#: Above JAI_JIT_THRESHOLD entries, so the body reaches the compiled tier by a
#: count rather than by the OSR sampler.
let CALLS = 200

class Machine {
    pub var seen: int
    pub var skipped: int

    pub fn init(self) -> void {
        self.seen = 0
        self.skipped = 0
    }

    #: `-> void`, and called below as the last statement of a branch, which is
    #: exactly what the peephole fuses.
    pub fn note_skip(self) -> void {
        self.skipped += 1
    }

    pub fn step(self, c: int) -> void {
        if c == 0 {
            self.note_skip()
        }
        self.seen += 1
    }
}

fn run_steps(m: Machine, rounds: int) -> int {
    var i = 0
    while i < rounds {
        m.step(i & 3)
        i += 1
    }
    return m.seen
}

pub fn test_a_void_method_at_the_end_of_a_branch() -> void {
    let m = Machine()
    var call = 0
    while call < CALLS {
        let _ = run_steps(m, 4)
        call += 1
    }
    assert_eq(m.seen, 4 * CALLS, "every step counted")
    assert_eq(m.skipped, CALLS, "one of four took the branch")
}

pub fn test_the_answers_do_not_move() -> void {
    let m = Machine()
    assert_eq(run_steps(m, 0), 0, "no rounds")
    assert_eq(run_steps(m, 1), 1, "one round, and it is the skip")
    assert_eq(m.skipped, 1, "the void method ran")
    assert_eq(run_steps(m, 8), 9, "eight more")
    assert_eq(m.skipped, 3, "two more skips in eight")
}
