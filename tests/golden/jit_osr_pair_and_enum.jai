#: gc-stress-every: 1000
#:
#: This golden exists to hold TWO OSR pair heads hot at once, which means
#: 200,000-element loops run several times over -- so the suite's default
#: cadence of one collection per 20 allocations does not finish. It ran 27
#: minutes at 86% CPU before this line existed. The cost is close to quadratic:
#: every=200 takes 19.2s, 500 takes 7.1s, 1000 takes 3.9s.
#:
#: Coverage is not gutted by the larger number. run_tests.sh picked its default
#: by requiring that no golden collect fewer than 57 times; this one collects
#: 1,856 times at 1000.
#: The two pair-loop OSR heads in ONE program, hot at the same time.
#:
#: `for (a, b) in ps` over a list of 2-tuples is iterKind 4 (JAITHON_JIT_PAIR_LIST);
#: `for (i, x) in xs.enumerate()` over the ITER_LIST_ENUM snapshot is iterKind 5
#: (JAITHON_LAZY_ENUMERATE). They share the opcode -- OP_FOR_ITER_PAIR -- and
#: nothing else: 4 takes the DICT head's prologue and reads its index out of the
#: ObjIter every iteration, while 5 takes a plain LIST head's prologue, reserved
#: registers and index write-back. A single wrong digit in either ladder makes
#: one head enter the other's form, so this file exists to run both heads in one
#: process, in both orders, interleaved, and nested one inside the other.
#:
#: Every number here must match JAITHON_NO_JIT=1, JAITHON_JIT_PAIR_LIST=0 and
#: JAITHON_LAZY_ENUMERATE=0. The tiers are accelerators that may always decline
#: and must never answer differently.
class Node {
    pub var k: int

    fn init(self, k: int) -> void { self.k = k }

    pub fn weight(self) -> int { return self.k * 2 }
}

#: iterKind 4. Long enough on its own that the SIGPROF tick lands inside it.
fn pair_walk(ps: list[tuple[str, int]]) -> int {
    var acc = 0
    for (a, b) in ps {
        acc = acc + b + a.len()
    }
    return acc
}

#: iterKind 5, over a list[int].
fn enum_walk(xs: list[int]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() {
        acc = acc + i + x
    }
    return acc
}

#: iterKind 5, over a list of instances -- the element is an object, so the
#: head's guard is a tag check plus a class check and the density scan is live.
fn enum_walk_inst(ns: list[Node]) -> int {
    var acc = 0
    for (i, n) in ns.enumerate() {
        acc = acc + i + n.weight()
    }
    return acc
}

#: Both heads inside one body, so the two forms are compiled against the same
#: frame and the same slot numbering.
fn both(ps: list[tuple[str, int]], xs: list[int]) -> int {
    var acc = 0
    for (a, b) in ps {
        acc = acc + b - a.len()
    }
    for (i, x) in xs.enumerate() {
        acc = acc + x - i
    }
    return acc
}

#: An enumerate loop wrapped around a tuple-pair loop. The inner head is
#: reached from inside a body the outer head compiled, which is the one place
#: the reserved registers of 5 and the ObjIter of 4 are live at once.
fn nested(ps: list[tuple[str, int]], xs: list[int]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() {
        if i % 4096 == 0 {
            for (a, b) in ps {
                acc = acc + b + a.len()
            }
        }
        acc = acc + x + i
    }
    return acc
}

#: And the other way round.
fn nested_flip(ps: list[tuple[str, int]], xs: list[int]) -> int {
    var acc = 0
    for (a, b) in ps {
        if b % 512 == 0 {
            for (i, x) in xs.enumerate() {
                acc = acc + x - i
            }
        }
        acc = acc + b - a.len()
    }
    return acc
}

#: Both heads in one body, in the other order. `both` above runs 4 then 5;
#: this runs 5 then 4, so neither ordering is the only one the slot numbering
#: and the reserved registers have been seen to survive.
fn both_flipped(ps: list[tuple[str, int]], xs: list[int]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() {
        acc = acc + x - i
    }
    for (a, b) in ps {
        acc = acc + b - a.len()
    }
    return acc
}

#: An enumerate head with ANOTHER enumerate loop in its body: the outer takes
#: the head arm off JIT_IDX_REG / JIT_LIM_REG / JIT_START_REG, the inner takes
#: the nested arm and calls out to jitMakeEnumIter -- so the outer's reserved
#: registers have to survive a C call made by a second copy of the same arm.
fn enum_in_enum(xs: list[int], ys: list[int]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() {
        for (j, y) in ys.enumerate() {
            acc = acc + i * 7 + x - j * 3 - y
        }
        acc = acc + x
    }
    return acc
}

#: Three enumerate loops deep.
fn enum3(xs: list[int], ys: list[int], zs: list[int]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() {
        for (j, y) in ys.enumerate() {
            for (k, z) in zs.enumerate() {
                acc = acc + i - j + k + x - y + z
            }
        }
    }
    return acc
}

#: An enumerate head carrying BOTH an inner tuple-pair loop and an inner
#: enumerate loop, so iterKind 4's ObjIter and iterKind 5's registers and a
#: nested shape-5 iterator are all live inside one compiled body.
fn e_head_both(ps: list[tuple[str, int]], xs: list[int], ys: list[int]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() {
        if i % 8192 == 0 {
            for (a, b) in ps {
                acc = acc + b - a.len()
            }
            for (j, y) in ys.enumerate() {
                acc = acc + y * 2 - j
            }
        }
        acc = acc + x + i
    }
    return acc
}

#: The same three, with the tuple-pair loop as the head instead.
fn p_head_both(ps: list[tuple[str, int]], xs: list[int], ys: list[int]) -> int {
    var acc = 0
    for (a, b) in ps {
        if b % 4096 == 0 {
            for (i, x) in xs.enumerate() {
                acc = acc + x - i
            }
            for (j, y) in ys.enumerate() {
                acc = acc + y + j
            }
        }
        acc = acc + b + a.len()
    }
    return acc
}

#: Four loops over ONE list of 2-tuples, alternating the two heads inside a
#: single body. Head 4 binds (a, b) out of the tuple; head 5 binds
#: (index, tuple). Nothing but the digit keeps them apart here.
fn same_list(ps: list[tuple[str, int]]) -> int {
    var acc = 0
    for (a, b) in ps {
        acc = acc + b - a.len()
    }
    for (i, p) in ps.enumerate() {
        acc = acc + p[1] - i + p[0].len()
    }
    for (a, b) in ps {
        acc = acc + b * 2 - a.len()
    }
    for (i, p) in ps.enumerate() {
        acc = acc - p[1] + i
    }
    return acc
}

#: Wide enough to spill, so the loop variables are frame-resident and localOut
#: spends JIT_SCRATCH_C and JIT_SCRATCH_D on their tags -- which is what the
#: nested arm's saved index in JIT_SCRATCH_B has to survive.
fn wide(xs: list[int], ys: list[int]) -> int {
    var a0 = 1
    var a1 = 2
    var a2 = 3
    var a3 = 4
    var a4 = 5
    var a5 = 6
    var a6 = 7
    var a7 = 8
    var a8 = 9
    var a9 = 10
    var b0 = 11
    var b1 = 12
    var b2 = 13
    var b3 = 14
    var b4 = 15
    var b5 = 16
    var b6 = 17
    var b7 = 18
    var acc = 0
    for (i, x) in xs.enumerate() {
        for (j, y) in ys.enumerate() {
            acc = acc + i + j + x + y + a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
            acc = acc + a8 + a9 + b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7
        }
    }
    return acc
}

#: Mutual recursion across the two heads: each calls the other from inside its
#: own loop body, so a frame holding one head's state is always below a frame
#: holding the other's.
fn ping(ps: list[tuple[str, int]], xs: list[int], depth: int) -> int {
    var acc = 0
    for (a, b) in ps {
        acc = acc + b + a.len()
        if depth > 0 and b % 97 == 0 {
            acc = acc + pong(ps, xs, depth - 1)
        }
    }
    return acc
}

fn pong(ps: list[tuple[str, int]], xs: list[int], depth: int) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() {
        acc = acc + x - i
        if depth > 0 and x % 89 == 0 {
            acc = acc + ping(ps, xs, depth - 1)
        }
    }
    return acc
}

fn build_pairs(n: int) -> list[tuple[str, int]] {
    var out: list[tuple[str, int]] = []
    var i = 0
    while i < n {
        out.push((f"k{i % 7}", i))
        i = i + 1
    }
    return out
}

fn build_ints(n: int) -> list[int] {
    var out: list[int] = []
    var i = 0
    while i < n {
        out.push(i * 3)
        i = i + 1
    }
    return out
}

fn build_nodes(n: int) -> list[Node] {
    var out: list[Node] = []
    var i = 0
    while i < n {
        out.push(Node(i))
        i = i + 1
    }
    return out
}

fn main() -> void {
    var ps = build_pairs(200000)
    var xs = build_ints(200000)
    var ns = build_nodes(200000)

    #: iterKind 4 first, so its form is the one already in the cache when the
    #: enumerate head asks for one at a different `top`.
    print(pair_walk(ps))
    print(enum_walk(xs))
    print(enum_walk_inst(ns))

    #: And the other order, in the same process.
    print(enum_walk(xs))
    print(pair_walk(ps))

    #: Alternating, so neither form is ever cold when the other runs.
    var r = 0
    var acc = 0
    while r < 6 {
        acc = acc + pair_walk(ps) - enum_walk(xs)
        r = r + 1
    }
    print(acc)

    print(both(ps, xs))
    print(nested(build_pairs(64), xs))
    print(nested_flip(build_pairs(4096), build_ints(64)))

    #: A tuple-pair loop over a list the enumerate snapshot could be confused
    #: for: the elements ARE 2-tuples, and `enumerate` on it yields
    #: (index, tuple) -- so the two heads see the same container and must bind
    #: different things.
    var t = 0
    for (a, b) in ps {
        t = t + b
    }
    for (i, p) in ps.enumerate() {
        t = t - i + p[1] % 5
    }
    print(t)

    #: The harder coexistence cases: both orders in one body, each head nested
    #: inside the other and inside itself, a spilled frame, and mutual
    #: recursion between the two.
    print(both_flipped(ps, xs))
    print(enum_in_enum(build_ints(120000), build_ints(6)))
    print(enum3(build_ints(400), build_ints(30), build_ints(20)))
    print(e_head_both(build_pairs(24), xs, build_ints(12)))
    print(p_head_both(ps, build_ints(24), build_ints(12)))
    print(same_list(ps))
    print(wide(build_ints(40000), build_ints(9)))
    print(ping(build_pairs(900), build_ints(900), 2))
    print(pong(build_pairs(900), build_ints(900), 2))
}
