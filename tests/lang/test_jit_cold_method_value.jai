#: A method name read as a VALUE, on a cold branch.
#:
#: `self._error(code, start, end, msg, help: "...")` has a keyword argument, so
#: it cannot be an OP_INVOKE: codegen has to materialise the callee, and reading
#: a method name lands in OP_GET_FIELD_LOCAL, where `jaiClassFieldInfo` knows
#: only fields and says no. That one refusal was **88 of resolve.jai's 104**
#: OP_GET_FIELD_LOCAL declines, and it declined the whole enclosing function.
#:
#: Every one of them is on an error path -- the lexer reaching a malformed
#: token -- so the refusal now takes the unarmed path and the body keeps its hot
#: prefix. Measured on `./jaithon check --no-cache`, six samples each through
#: `JAITHON_JIT_SOFT_FIELD`:
#:
#:   lexer.jai     0.290s -> 0.283s
#:   resolve.jai   0.281s -> 0.272s
#:   parser.jai    0.577s -> 0.563s
#:
#: and resolve.jai's total declines from 1854 to 1790.
#:
#: **Cold is the whole condition, and it is not a guess.** The identical change
#: on OP_GET_INDEX's shape refusals -- `d[k]` on a `dict[str, any]`, which is
#: the AST-record idiom and sits on the hot path -- measured **18% SLOWER on
#: resolve.jai and 8% slower on parser.jai**. A body that deopts every iteration
#: costs more than one that is simply interpreted. The unarmed path pays when
#: the instruction it starts at rarely runs, and not otherwise.
#:
#: The name is resolved before the receiver guard is emitted, because the
#: unarmed path takes over at the START of an instruction and half an
#: instruction cannot be handed back.
#:
#: No `# jit-compiles:` marker: the read has to be inside a METHOD to be an
#: OP_GET_FIELD_LOCAL at all (a keyword call on any other receiver is
#: OP_GET_FIELD, a separate arm), and the marker only names top-level
#: functions. What this file guards is that the answers do not move -- across
#: both switch states, the interpreter, and the stress modes. The tier evidence
#: is the decline count and the clock above.
from std.test import assert_eq, assert_true

#: Above JAI_JIT_THRESHOLD entries, so the body below reaches the compiled tier.
let CALLS = 200

class Counter {
    pub var n: int
    pub var complaints: int

    pub fn init(self) -> void {
        self.n = 0
        self.complaints = 0
    }

    #: Private, and takes a keyword argument -- the two things that force the
    #: callee to be read as a value rather than invoked.
    fn _complain(self, about: int, loudly: bool = false) -> int {
        self.complaints += 1
        if loudly { return about * 2 }
        return about
    }

    pub fn complain(self, about: int, loudly: bool) -> int {
        return self._complain(about, loudly: loudly)
    }

    #: The shape the change is about: a hot loop whose cold arm reads a private
    #: method of `self` as a value, because the keyword argument rules out an
    #: invoke.
    pub fn tally(self, rounds: int) -> int {
        var t = 0
        var i = 0
        while i < rounds {
            t += self.n + i
            if i < 0 {
                t += self._complain(i, loudly: true)
            }
            i += 1
        }
        return t
    }
}

pub fn test_the_hot_loop_still_compiles() -> void {
    let c = Counter()
    var total = 0
    var call = 0
    while call < CALLS {
        total += c.tally(4)
        call += 1
    }
    #: 0+1+2+3 per call, and the cold arm never runs.
    assert_eq(total, 6 * CALLS, "sum")
    assert_eq(c.complaints, 0, "the cold branch stayed cold")
}

pub fn test_the_cold_branch_is_still_correct_when_taken() -> void {
    let c = Counter()
    assert_eq(c.complain(5, false), 5, "default keyword")
    assert_eq(c.complain(5, true), 10, "given keyword")
    assert_eq(c.complaints, 2, "both counted")
}

pub fn test_a_real_field_read_is_unaffected() -> void {
    let c = Counter()
    c.n = 9
    var total = 0
    var call = 0
    while call < CALLS {
        total += c.tally(1)
        call += 1
    }
    assert_eq(total, 9 * CALLS, "the field arm still emits")
    assert_true(c.n == 9, "and does not disturb it")
}
