#!/usr/bin/env python3
"""Generator of small random Jaithon programs for the JIT differential fuzzer.

The compiled tier is an accelerator that may always decline (src/vm/jit/jit.h),
so for ANY program every configuration of it has to print the same thing as the
interpreter. That makes a random program a test case on its own, with no
expected output to write down: differential.py runs one program five ways and
diffs. This file is the half that invents the programs.

What a generated program looks like, and why it is shaped that way:

    class ... / fn helper ...            a few of each, shared by the probes
    fn probe0(n: int) -> str { ...the random part... }
    fn probe1(n: int) -> str { ...more of it... }
    fn main() -> void {
        var h = 0
        var i = 0
        while i < WARM {
            let r0 = probe0(i)
            h = h *% 1000003 +% sfold(r0)
            ...
        }
        print(f"{h}|{tail}")
    }

`main` is called for us -- naming a function `main` runs it, so a generated
program must NOT also call it or every line prints twice. The while loop is
what makes anything compile at all: JAI_JIT_THRESHOLD is 64 CALLS and there is
no environment variable that lowers it, so warming has to happen inside the
program. WARM defaults to 1500, comfortably past 64 for every probe and every
helper it calls, and long enough that the 1kHz sampler gets its ticks into the
OSR loop tier as well.

Two to five probes rather than one, because one unsupported construct stops the
whole-function tier for the whole function holding it -- see Probe below, where
splitting took that tier's reach from 19% of generated functions to 42%.

Each probe returns a string and `main` folds every one of them into a single
integer digest, so a divergence on ANY iteration is caught, not just the last
one -- a deopt-stress run can differ for one call and recover. The last three
raw strings are printed alongside the digest so a hit says roughly where it
went wrong before the shrinker is even started.

Everything is biased toward what the tier actually compiles and toward the
shapes that have broken it before:

  - a `fn init(self) {}` that must still evaluate to the new object, not null
  - a local assigned on only one branch and read afterwards
  - integer arithmetic at the int64 edges, both wrapping and checked
  - loops of every form, which is the only way into the OSR tier
  - an early `return` out of a loop, which OSR handles separately
  - field reads and writes, methods, `self.m()` calls, and overrides
  - lists, dicts, strings, indexing, recursion, match, lambdas
  - the container family: strings built and indexed and sliced, dict and set
    literals with lookups that hit and lookups that miss, tuples constructed
    and destructured, lists of lists, and containers whose element KIND is
    decided by which branch ran

Each container is declared per PROBE rather than per program -- `Probe.decls`
-- so whatever one costs the tier is confined to the one function holding it,
and a probe takes at most ONE of them beyond `d`. Every container declared is
folded into that probe's return value, because one the digest cannot see is
one the oracle cannot check. The folds RENDER each element rather than adding
it, so a slot holding 1 on one path and "1" on another reaches the digest as
two different strings.

Then the same container statements are generated a second time, into functions
that take the container as a PARAMETER -- `gen_container_helpers`. That is not
decoration. A container built in the body that reads it has no live sample, so
the dict index, the list element kind and the set length all decline; received
as a parameter the call site's inline cache has the sample and the same work
compiles. Measured on probes of nothing but `d[k]` reads: 1 of 72 as a local,
clean as a parameter.

Two spellings matter more than they look, and both were measured rather than
guessed. `sfold(str(x))` refuses the enclosing body on `OP_TYPE_GUARD: a str
guard on a object`; `sfold(f"{x}")` compiles and renders the same text. And a
string produced by a METHOD -- `.upper()`, `.replace()`, `"-".join(...)`, even
`str(n)` -- is an object to the tier, so `text = text + <that>` refuses for the
same reason and `text = text + f"{<that>}"` does not. Both are harness
plumbing: the method call is still made and still reaches the digest.

The container invariants, which are what keep a generated program from dying
the same way five times over:

  - `ys` never empties and no inner list of it ever empties, so `ys[i][j]` with
    both indices taken modulo a length is always in range;
  - `d` always carries its seeded key, and that key is only ever written an
    int, so the `d[seed]` reads that feed integer arithmetic cannot start
    raising once some branch has put a string somewhere else in the dict. The
    seed is one literal for the WHOLE program, because a container helper is
    called with whichever probe's dict;
  - a set is narrowed with `discard`, never `remove`, which raises on a member
    that is not there;
  - a string is only indexed behind a length test, since `text` starts empty --
    but SLICING needs no guard, because it clamps, on strings and lists alike;
  - a container of unknown element kind is read through a fold, never straight
    into a `+%`, which would raise a TypeError alike in all five
    configurations and throw the seed away;
  - nothing is mutated while it is being walked, and the "being walked" flag
    nests, so an inner `for x in xs` cannot clear the outer one's.

Cost is bounded on purpose, because a program that is merely SLOW is
indistinguishable from one that hangs, and one that times out under one
configuration and not another is a false positive. A fold is O(size) with a
string built per element, so: containers stop growing at 8, a loop's trip count
falls when it is already inside one and again when a container is in scope, and
a container helper is only ever called from the top of a body. Without those
three a single seed ran 65 seconds and another did not finish in five minutes.

Two invariants keep every generated program legal and terminating, since a
program that dies the same way five times proves nothing:

  - `xs` never empties (pops are guarded), so `xs[k % xs.len()]` cannot raise,
    and `d["k"]` is only ever read for a key seeded into the literal;
  - every loop carries its own counter increment as the FIRST statement of its
    body, outside anything the generator or the shrinker can delete, so no
    body and no `continue` can stop it terminating.

Integer expressions combine with the wrapping operators `+% -% *%` so an edge
literal can never raise; the checked `+ - *` arms are still covered, on
subexpressions bounded by a `%` so they provably cannot overflow. Float
division is only ever by a nonzero literal (Jaithon raises rather than giving
IEEE infinity), but overflow to `inf` is left in on purpose -- it prints, and
both tiers have to print it the same.

    python3 tests/fuzz/progen.py --seed 7            # print program 7
    python3 tests/fuzz/progen.py --seed 7 --warm 200

Same seed, same program, always.
"""

import argparse
import copy
import random
import sys

WARM_DEFAULT = 1500

# Combined with `+% -% *%` only, so any of these is safe to reach. Held as
# SOURCE TEXT rather than as numbers because of the last one: `-92233...808`
# is a unary minus on a literal that does not fit, and the checker only folds
# the two together in some positions -- `acc +% -9223372036854775808` is
# accepted and `(1 - -9223372036854775808)` is E0005. Written as a subtraction
# it reaches int64's floor from anywhere.
INT_MIN_SRC = "(-9223372036854775807 - 1)"
EDGE_INTS = [
    "0", "1", "-1", "2", "3", "7", "63", "64", "65", "255", "256", "1023",
    "65535", "65536", "2147483647", "-2147483648", "2147483648", "4294967295",
    "4611686018427387904", "9223372036854775807", INT_MIN_SRC,
]

SMALL_INTS = [0, 1, 2, 3, 5, 7, 11, 16, 31, 100, 999, -1, -2, -7, -100]

FLOAT_LITS = ["0.0", "1.0", "0.5", "2.0", "-1.5", "3.25", "1e10", "-0.125",
              "1e-8", "100.0", "0.1", "-2.75"]

STR_LITS = ['""', '"a"', '"bc"', '"xyz"', '"jai"', '"0"', '"-"', '" "',
            # Not ASCII, so `s[i]` deopts out of the compiled character path
            # rather than taking it -- the guard is `scalars == length`.
            '"é"', '"αβ"']

CMPS = ["==", "!=", "<", "<=", ">", ">="]

# Key literals a generated dict may carry. The kinds are mixed on purpose: a
# dict holding both "k" and 3 is one whose lookup cannot be told from the key's
# compile-time type alone.
DICT_KEYS = ['"k"', '"a"', '"b"', '"zz"', '"0"', '0', '1', '7', '-3', '256']

# A value expression of each kind, for the places where the point is that two
# branches put DIFFERENT kinds in the same slot. Rendered by Gen.kind_value.
VALUE_KINDS = ["int", "float", "str", "bool", "null", "list", "tuple", "inst"]


class Node:
    """One statement, possibly with nested blocks.

    `parts` interleaves literal lines at this node's own indent with lists one
    level deeper, which is enough for if/elif/else and for any loop. A list may
    hold Nodes or plain strings, and the difference is the whole safety story:
    THE SHRINKER DELETES NODES AND NEVER STRINGS, so anything a reduction must
    not be able to remove goes in as a string.

    A loop's counter increment is the case that matters. Written as a child
    Node it is deletable, and deleting it turns

        while w < 40 {
            w = w + 1          <- gone
            ...
        }

    into a program that never terminates -- which the runner then sees as a
    timeout under some configurations and not others, reads as a divergence,
    and adopts. That happened: a shrink ran twenty-five minutes on an infinite
    loop it had just created. As a string the line is part of the loop and no
    reduction can separate them.
    """

    __slots__ = ("parts",)

    def __init__(self, *parts):
        self.parts = list(parts)

    def blocks(self):
        return [p for p in self.parts if isinstance(p, list)]

    def render(self, out, depth):
        pad = "    " * depth
        for part in self.parts:
            if isinstance(part, str):
                out.append(pad + part)
            else:
                for kid in part:
                    if isinstance(kid, str):
                        out.append(pad + "    " + kid)
                    else:
                        kid.render(out, depth + 1)


def line(text):
    return Node(text)


class Probe:
    """One function full of generated statements.

    A program carries several rather than one big one, because a single
    unsupported construct stops the whole-function tier for the WHOLE function
    it appears in. With one 9-statement `probe` per program that tier reached
    it 6 times in 60; split into four short ones, whatever blocks the third
    leaves the other three compiling, and a divergence names its function
    before the shrinker starts.
    """

    def __init__(self, name):
        self.name = name
        self.body = []
        # Each of these declares one more container in the preamble, and each
        # is decided PER PROBE rather than per program: whatever a container
        # costs the tier is then confined to the one function that has it.
        self.use_dict = False
        # The one key `d`'s literal always carries, so an unguarded `d[K]` read
        # can never miss. Every other key is only ever read behind a test.
        self.dict_seed = '"k"'
        self.use_nested = False    # ys, a list of lists
        self.use_set = False       # st
        self.use_tuple = False     # tp
        self.use_mixed = False     # ms, whose elements are of several kinds
        self.use_empty = False     # ez / ed, which start out empty

    def decls(self):
        """The container preamble, and the digest terms that read it back.

        A container that is never folded into the return value is a container
        the oracle cannot see, so the two are declared together.
        """
        out, folds = [], []
        if self.use_dict:
            out.append(f"    var d: dict = {{{self.dict_seed}: 1}}")
            folds.append("dfold(d)")
        if self.use_nested:
            # Never emptied and no inner list ever emptied, so ys[i][j] with
            # both indices taken modulo a length can never raise.
            out.append("    var ys: list = [[1, 2], [3], [4, 5, 6]]")
            folds.append("vfold(ys)")
        if self.use_set:
            out.append("    var st: set = {1, 2, 3}")
            folds.append("qfold(st)")
        if self.use_tuple:
            out.append('    let tp = (1, "t", 2)')
            folds.append("tfold(tp)")
        if self.use_mixed:
            out.append('    var ms: list = [1, "a", 2]')
            folds.append("vfold(ms)")
        if self.use_empty:
            out.append("    var ez: list = []")
            out.append("    var ed: dict = {}")
            folds.append("vfold(ez)")
            folds.append("dfold(ed)")
        return out, folds

    def render(self, out):
        out.append(f"fn {self.name}(n: int) -> str {{")
        out.append("    var acc = 0")
        out.append("    var a0 = 0")
        out.append("    var a1 = 0")
        out.append("    var f = 0.5")
        out.append('    var text = ""')
        out.append("    var xs = [1, 2, 3]")
        decls, folds = self.decls()
        out.extend(decls)
        for node in self.body:
            node.render(out, 1)
        # Folded to ONE integer before it is formatted. An f-string is refused
        # outright past JIT_MAX_ARGS_OUT (12) parts, counting the literal
        # chunks between the holes, and the eight-hole version of this line was
        # the single largest reason probes failed to compile -- 41 of 203, all
        # of it self-inflicted by the harness rather than by the program.
        #
        # Every fold is a call to a generated Jaithon function, never to `str`
        # or `.len()` directly, so the native call that renders a container
        # sits inside the fold and not in the body the tier is walking.
        extra = "".join(f" +% {call}" for call in folds)
        out.append("    let z = acc *% 31 +% (a0 *% 7) +% a1 +% text.len()"
                   " +% sfold(text) +% lfold(xs)" + extra)
        out.append('    return f"{z}|{f}"')
        out.append("}")


# Emitted next to sfold/lfold when something asks for them. Each renders every
# element with `str` and folds the characters, so the digest distinguishes the
# KIND of what a container holds and not only its numeric value.
FOLDS = [
    ("vfold",
     "fn vfold(v: list) -> int {\n"
     "    var h = 0\n"
     '    for x in v { h = h *% 131 +% sfold(f"{x}") }\n'
     "    return h\n"
     "}"),
    ("dfold",
     "fn dfold(m: dict) -> int {\n"
     "    var h = 0\n"
     "    for (k, v) in m.items() {\n"
     '        h = (h *% 1009 +% sfold(f"{k}")) *% 31 +% sfold(f"{v}")\n'
     "    }\n"
     "    return h\n"
     "}"),
    ("qfold",
     "fn qfold(q: set) -> int {\n"
     "    var h = 0\n"
     '    for x in q { h = h *% 8191 +% sfold(f"{x}") }\n'
     "    return h\n"
     "}"),
    ("tfold",
     "fn tfold(t: tuple) -> int {\n"
     "    var h = 0\n"
     '    for x in t { h = h *% 257 +% sfold(f"{x}") }\n'
     "    return h\n"
     "}"),
    # One element out of a list of lists, with both indices normalised HERE so
    # a caller may pass anything. |i % n| < n whichever sign convention `%`
    # follows, and a negative index counts from the end, so both are in range.
    ("gpick",
     "fn gpick(v: list, i: int, j: int) -> int {\n"
     "    if v.len() == 0 { return -1 }\n"
     "    let row = v[i % v.len()]\n"
     "    if row.len() == 0 { return -2 }\n"
     '    return sfold(f"{row[j % row.len()]}")\n'
     "}"),
]


class Program:
    """A generated program, kept as parts the shrinker can drop one at a time."""

    def __init__(self, seed, warm):
        self.seed = seed
        self.warm = warm
        self.classes = []      # list of source blocks, each a str
        self.helpers = []      # list of source blocks, each a str
        self.probes = []       # list of Probe

    def clone(self):
        dup = Program(self.seed, self.warm)
        dup.classes = list(self.classes)
        dup.helpers = list(self.helpers)
        dup.probes = copy.deepcopy(self.probes)
        return dup

    def render(self):
        out = []
        out.append(f"#: Generated by tests/fuzz/progen.py --seed {self.seed}"
                   f" --warm {self.warm}. Do not edit.")
        out.append("")
        for block in self.classes:
            out.append(block.rstrip())
            out.append("")
        # Folds a string to an integer so `probe` can return full fidelity in a
        # bounded amount of text: every character reaches the digest.
        out.append("fn sfold(s: str) -> int {")
        out.append("    var h = 0")
        out.append("    var j = 0")
        out.append("    while j < s.len() {")
        out.append("        h = h *% 131 +% ord(s[j])")
        out.append("        j = j + 1")
        out.append("    }")
        out.append("    return h")
        out.append("}")
        out.append("")
        out.append("fn lfold(v: list) -> int {")
        out.append("    var h = 0")
        out.append("    for x in v { h = h *% 31 +% x }")
        out.append("    return h")
        out.append("}")
        out.append("")
        # The kind-blind folds. `lfold` above adds its element, which needs an
        # int; these render it instead, so a slot that holds 1 on one path and
        # "1" on another reaches the digest as two different strings -- which
        # is the whole point, since a tag rebuilt from a compile-time kind is
        # exactly what has been wrong here before.
        #
        # Which of them to emit is read back off the text that was just
        # rendered rather than tracked while generating. That way a reduction
        # which deletes the last caller of one also deletes the fold, and a new
        # statement that calls one cannot forget to ask for it -- both of which
        # went wrong when this was a set maintained by hand.
        rest = []
        for block in self.helpers:
            rest.append(block.rstrip())
            rest.append("")
        for probe in self.probes:
            probe.render(rest)
            rest.append("")
        body = "\n".join(rest)
        for want, block in FOLDS:
            if want + "(" in body:
                out.append(block)
                out.append("")
        out.extend(rest)
        out.append("fn main() -> void {")
        out.append("    var h = 0")
        out.append('    var tail = ""')
        out.append("    var i = 0")
        out.append(f"    while i < {self.warm} {{")
        for k, probe in enumerate(self.probes):
            out.append(f"        let r{k} = {probe.name}(i)")
            out.append(f"        h = h *% 1000003 +% sfold(r{k})")
        joined = ' + "/" + '.join(f"r{k}" for k in range(len(self.probes)))
        if joined:
            out.append(f"        if i >= {max(0, self.warm - 3)} "
                       '{ tail = tail + ' + joined + ' + ";" }')
        out.append("        i = i + 1")
        out.append("    }")
        out.append('    print(f"{h}|{tail}")')
        out.append("}")
        return "\n".join(out) + "\n"


class Gen:
    """Everything random about one program, driven by one seeded Random."""

    def __init__(self, seed, warm):
        self.rng = random.Random(seed)
        self.prog = Program(seed, warm)
        self.ints = ["n", "acc", "a0", "a1"]   # int-typed names readable here
        # A loop variable is an immutable binding, so what may be READ and what
        # may be ASSIGNED are two different lists.
        self.assignable = ["acc", "a0", "a1"]
        self.floats = ["f"]
        self.strs = ["text"]
        # Containers being walked right now. Pushing to a list mid-`for` is a
        # RuntimeError, and every config would raise it, so it tests nothing.
        self.iterating = set()
        # Two features the whole-function tier refuses outright. Both are worth
        # generating -- they are the OSR tier's problem and the decline path's
        # -- but a program carrying one never reaches the compiled tier at all,
        # so they are rationed rather than mixed into everything.
        self.use_dict = False
        self.use_pow = False
        # The rest of the container family, decided per probe in build().
        self.dict_seed = '"k"'
        self.use_nested = False
        self.use_set = False
        self.use_tuple = False
        self.use_mixed = False
        self.use_empty = False
        self.insts = []                 # (varname, classname) in scope
        self.classes = {}               # name -> dict describing its API
        self.helpers = {}               # name -> (arity, return kind)
        self.uid = 0
        self.loop_depth = 0
        # What `return` produces here. A probe returns a string; a container
        # helper returns an int, and an early return out of one has to agree.
        self.early_ret = 'f"E{acc}|{f}"'
        # True only while a container helper's body is generated.
        self.focus = False
        # container kind -> [helper names]. Filled by gen_container_helpers.
        self.cont_helpers = {}
        # `xs`, `d` and `text` are locals of `probe`. A helper body is
        # generated with the same expression machinery but must not reach
        # them, so every atom that names one is gated on this.
        self.containers = True

    def fresh(self, stem):
        self.uid += 1
        return f"{stem}{self.uid}"

    @staticmethod
    def render_fold(expr):
        """An int digest of a value of ANY kind, in the one spelling that
        does not stop the tier.

        `sfold(str(x))` refuses the whole enclosing body -- `OP_TYPE_GUARD: a
        str guard on a object` -- and `sfold(f"{x}")` compiles. The two render
        identically, so this is the only spelling used anywhere here.
        """
        return 'sfold(f"{' + expr + '}")'

    def enable(self, flag):
        """Make one container visible to the statement machinery."""
        setattr(self, flag, True)

    def trips(self, outer, inner):
        """How many times a loop runs, smaller when it is already inside one.

        The bound has to fall with depth or the cost multiplies: a 200-trip
        loop nested in a 35-trip one is 7,000 iterations, and with a container
        fold in the body that was a program taking two minutes under the
        interpreter -- which the runner then reads as a timeout in one
        configuration and a slow pass in another, and reports as a divergence.
        """
        if self.focus:
            return self.rng.choice([3, 5] if self.loop_depth else [3, 5, 8])
        if self.loop_depth > 0:
            return self.rng.choice(inner)
        if self.focus or any(getattr(self, f) for _n, _t, f
                             in self.CONTAINERS.values()) or self.use_empty:
            # A container in scope means a fold may land in the body, and a
            # fold is O(size) with a string built per element. `for i in
            # 0..200` around one of those was a probe taking 65 seconds.
            return self.rng.choice([t for t in outer if t <= 64])
        return self.rng.choice(outer)

    def index_expr(self, name):
        """An index that is always inside `name`, including both boundaries.

        The last valid position and the first from the far end are where a
        bounds check is wrong if it is wrong at all, so they get a third of the
        weight between them rather than turning up by accident.
        """
        r = self.rng
        pick = r.randrange(6)
        if pick == 0:
            return f"{name}.len() - 1"
        if pick == 1:
            return f"-{name}.len()"
        if pick == 2:
            return "0"
        return f"{r.randint(0, 40)} % {name}.len()"

    def kind_value(self, kind=None):
        """One expression, of a kind chosen at random.

        The point of this is the recurring bug shape: a slot that holds an int
        on one path and an object on another cannot have its tag rebuilt from
        a compile-time kind, so anywhere two branches write the SAME slot, they
        are fed from here.
        """
        r = self.rng
        kind = kind or r.choice(VALUE_KINDS)
        if kind == "int":
            return self.int_expr(1)
        if kind == "float":
            return self.float_expr(1)
        if kind == "str":
            return self.str_expr(1)
        if kind == "bool":
            return self.bool_expr(0)
        if kind == "null":
            return "null"
        if kind == "list":
            return f"[{self.int_expr(0)}, {self.int_expr(0)}]"
        if kind == "tuple":
            return f"({self.int_expr(0)}, {r.choice(STR_LITS)})"
        names = sorted(self.classes)
        if not names:
            return self.int_expr(1)
        name = r.choice(names)
        return self.classes[name]["ctor"].replace(
            "ARG2", self.int_expr(0)).replace("ARG", self.int_expr(0))

    def scoped(self, ints, floats, strs):
        """Swap the visible names while a helper body is generated."""
        saved = (self.ints, self.assignable, self.floats, self.strs,
                 self.insts, self.containers, self.loop_depth)
        self.ints, self.floats, self.strs = ints, floats, strs
        self.assignable = []
        self.insts, self.containers, self.loop_depth = [], False, 0
        return saved

    def unscope(self, saved):
        (self.ints, self.assignable, self.floats, self.strs, self.insts,
         self.containers, self.loop_depth) = saved

    # -- expressions ----------------------------------------------------

    def int_expr(self, depth=2):
        r = self.rng
        if depth <= 0 or r.random() < 0.35:
            return self.int_atom()
        pick = r.randrange(10)
        if pick < 3:
            op = r.choice(["+%", "-%", "*%"])
            return f"({self.int_expr(depth - 1)} {op} {self.int_expr(depth - 1)})"
        if pick == 3:
            return f"({self.int_expr(depth - 1)} // {r.randint(1, 64)})"
        if pick == 4:
            return f"({self.int_expr(depth - 1)} % {r.randint(1, 64)})"
        if pick == 5:
            op = r.choice(["&", "|", "^"])
            return f"({self.int_expr(depth - 1)} {op} {self.int_expr(depth - 1)})"
        if pick == 6:
            return f"({self.int_expr(depth - 1)} >> {r.randint(0, 16)})"
        if pick == 7:
            # Bounded by the `%` so the checked shift arm cannot overflow.
            return (f"(({self.int_expr(depth - 1)} % 256) << "
                    f"{r.randint(0, 8)})")
        if pick == 8:
            # The checked `+ - *` arms, on operands a `%` has made small.
            a = f"({self.int_expr(depth - 1)} % 100)"
            return f"({a} * {r.randint(1, 50)} + {r.randint(-100, 100)})"
        fn = r.choice(["min", "max"])
        return f"{fn}({self.int_expr(depth - 1)}, {self.int_expr(depth - 1)})"

    def int_atom(self):
        r = self.rng
        options = [
            lambda: str(r.choice(SMALL_INTS)),
            lambda: r.choice(EDGE_INTS),
            lambda: r.choice(self.ints),
        ]
        # `n` is the only thing that differs between one call and the next, so
        # it is what makes a late iteration take a different path from the
        # early ones the tier specialised on. Worth more than a 1-in-4 share of
        # the variable slot.
        if "n" in self.ints:
            options += [lambda: "n", lambda: f"(n % {r.randint(2, 97)})"]
        if self.use_pow:
            options.append(
                lambda: f"({r.choice(self.ints)} % 5) ** {r.randint(0, 3)}")
        if self.containers:
            options += [
                lambda: "xs.len()",
                lambda: "text.len()",
                lambda: f"xs[{abs(r.randint(0, 40))} % xs.len()]",
                # Exactly the last valid position, and exactly the first from
                # the other end -- the two off-by-ones a bounds check gets
                # wrong. Neither can raise: xs never empties.
                lambda: "xs[xs.len() - 1]",
                lambda: "xs[-xs.len()]",
            ]
        # Every read of a container whose elements may be of ANY kind goes
        # through `sfold(f"{...}")` or a fold helper, never straight into the
        # `+%` that consumes an int atom -- a slot holding "a" would raise a
        # TypeError alike in all five configurations, which tests nothing and
        # throws the seed away.
        if self.use_dict:
            options += [lambda: "d.len()", lambda: f"d[{self.dict_seed}]",
                        lambda: self.render_fold(
                            f"d.get({r.choice(DICT_KEYS)}, "
                            f"{r.choice(SMALL_INTS)})"),
                        ]
        if self.use_nested:
            options += [
                lambda: "ys.len()",
                lambda: f"ys[{r.randint(0, 30)} % ys.len()].len()",
                lambda: f"gpick(ys, {self.index_expr('ys')}, "
                        f"{r.randint(-4, 30)})",
            ]
        if self.use_set:
            options.append(lambda: "st.len()")
        if self.use_tuple:
            # tp is (int, str, int), so 0 and 2 are the int members and a
            # literal index is the only kind a tuple accepts -- the checker
            # rejects one it can prove out of range.
            options += [lambda: "tp.len()",
                        lambda: f"tp[{r.choice([0, 2])}]"]
        if self.use_mixed:
            options += [lambda: "ms.len()",
                        lambda: self.render_fold(
                            f"ms[{r.randint(0, 30)} % ms.len()]"),
                        lambda: self.render_fold("ms[ms.len() - 1]")]
        if self.use_empty:
            # A length, a lookup that misses, and a first/last of nothing.
            # All four are legal on an empty container and none can raise.
            options += [lambda: "ez.len()", lambda: "ed.len()",
                        lambda: self.render_fold(
                            f"ed.get({r.choice(DICT_KEYS)}, "
                            f"{r.choice(SMALL_INTS)})"),
                        lambda: self.render_fold(
                            f"ez.first({r.choice(SMALL_INTS)})")]
        if self.helpers:
            options.append(self.helper_call)
        if self.insts:
            options.append(self.method_call)
        return r.choice(options)()

    def helper_call(self):
        r = self.rng
        name = r.choice(sorted(self.helpers))
        arity, kind = self.helpers[name]
        if kind == "list":
            if not self.containers:
                return r.choice(self.ints)
            return f"{name}(xs)"
        if kind == "rec":
            # Every argument is bounded to 0..11 by a positive modulus, so the
            # depth is bounded too -- an int_expr straight in blows the stack.
            return f"{name}(({self.int_expr(0)}) % 12)"
        args = ", ".join(self.int_expr(0) for _ in range(arity))
        return f"{name}({args})"

    def method_call(self):
        r = self.rng
        var, cls = r.choice(self.insts)
        api = self.classes[cls]
        name = r.choice(api["int_methods"])
        if name == "bump":
            return f"{var}.bump({self.int_expr(0)})"
        return f"{var}.{name}()"

    def float_expr(self, depth=2):
        r = self.rng
        if depth <= 0 or r.random() < 0.4:
            options = [
                lambda: r.choice(FLOAT_LITS),
                lambda: f"float({self.int_expr(0)} % 1000)",
            ]
            if self.floats:
                options.append(lambda: r.choice(self.floats))
            if self.insts:
                cands = [v for v, c in self.insts
                         if self.classes[c]["float_field"]]
                if cands:
                    options.append(lambda: f"{r.choice(cands)}.b")
            return r.choice(options)()
        pick = r.randrange(5)
        if pick < 3:
            op = r.choice(["+", "-", "*"])
            return (f"({self.float_expr(depth - 1)} {op} "
                    f"{self.float_expr(depth - 1)})")
        if pick == 3:
            # Never by zero: Jaithon raises rather than producing infinity.
            div = r.choice(["2.0", "3.5", "-4.0", "0.25", "10.0", "-1.5"])
            return f"({self.float_expr(depth - 1)} / {div})"
        if self.use_pow:
            return f"({self.float_expr(depth - 1)} ** 2.0)"
        return f"({self.float_expr(depth - 1)} * {r.choice(FLOAT_LITS)})"

    def bool_expr(self, depth=1):
        r = self.rng
        pick = r.randrange(10)
        if depth > 0 and pick == 0:
            op = r.choice(["and", "or"])
            return (f"({self.bool_expr(depth - 1)} {op} "
                    f"{self.bool_expr(depth - 1)})")
        if depth > 0 and pick == 1:
            return f"(not {self.bool_expr(depth - 1)})"
        if pick == 2:
            return (f"({self.float_expr(1)} {r.choice(CMPS)} "
                    f"{self.float_expr(1)})")
        if pick == 3 and self.containers:
            return f"({self.int_expr(1)} % 8 in xs)"
        if pick == 4 and self.containers:
            return self.membership()
        if pick == 5 and self.strs:
            return (f"({r.choice(self.strs)} {r.choice(['==', '!='])} "
                    f"{r.choice(STR_LITS)})")
        if pick == 6 and self.insts:
            var, _ = r.choice(self.insts)
            return f"({var} {r.choice(['==', '!='])} null)"
        return f"({self.int_expr(1)} {r.choice(CMPS)} {self.int_expr(1)})"

    def membership(self):
        """`in` / `not in` against whichever containers this probe declares.

        Half of these are seeded so they hit and half are chosen to miss --
        the miss is the half a lookup gets wrong.
        """
        r = self.rng
        options = []
        if self.containers:
            options += [lambda: f"({self.int_expr(1)} % 8 in xs)",
                        lambda: f"({r.randint(50, 99)} not in xs)"]
        if self.use_dict:
            options += [lambda: f"({self.dict_seed} in d)",
                        lambda: f"({r.choice(DICT_KEYS)} in d)",
                        lambda: '("missing" not in d)']
        if self.use_set:
            options += [lambda: f"({r.randint(0, 6)} in st)",
                        lambda: f"({r.randint(40, 99)} not in st)"]
        if self.use_tuple:
            options += [lambda: '("t" in tp)', lambda: "(9 not in tp)"]
        if self.use_mixed:
            options += [lambda: '("a" in ms)',
                        lambda: f"({r.choice(SMALL_INTS)} in ms)"]
        if self.use_empty:
            # Nothing is ever in an empty container, which is exactly the
            # answer a lookup that skips its length check gets wrong.
            options += [lambda: f"({r.choice(SMALL_INTS)} not in ez)",
                        lambda: f"({r.choice(DICT_KEYS)} not in ed)"]
        if self.strs:
            options.append(
                lambda: f"({r.choice(STR_LITS)} in {r.choice(self.strs)})")
        if not options:
            return f"({self.int_expr(1)} {r.choice(CMPS)} {self.int_expr(1)})"
        return r.choice(options)()

    def str_expr(self, depth=1):
        r = self.rng
        if depth <= 0 or r.random() < 0.45:
            options = [
                lambda: r.choice(STR_LITS),
                lambda: f'f"{{{self.int_expr(1)}}}"',
                lambda: f'f"{{{self.float_expr(1)}}}"',
                lambda: f"str({self.int_expr(1)})",
                # A literal is never empty, so these cannot raise however the
                # rest of the program went. Both boundaries are covered.
                lambda: f'"jaithon"[{r.randint(0, 6)}]',
                lambda: '"jaithon"[-1]',
                lambda: f'"jaithon"[{r.randint(0, 3)}:{r.randint(3, 12)}]',
            ]
            if self.strs:
                name = r.choice(self.strs)
                options += [
                    lambda: name,
                    # Slicing CLAMPS, so this is safe on the empty string.
                    lambda: f"{name}[{r.randint(0, 4)}:{r.randint(1, 9)}]",
                    lambda: f"{name}.slice({r.randint(0, 3)}, "
                            f"{r.randint(0, 12)})",
                ]
            if self.containers:
                # Sliced: a container that a loop grew renders to a long
                # string, and `text` is folded character by character on every
                # one of 1500 calls. The slice is what keeps that bounded.
                options += [
                    lambda: 'f"{xs}".slice(0, 24)',
                    lambda: f'f"{{xs[{r.randint(0, 30)} % xs.len()]}}"',
                ]
            if self.use_mixed:
                options.append(
                    lambda: f'f"{{ms[{r.randint(0, 30)} % ms.len()]}}"')
            if self.use_tuple:
                options += [lambda: 'f"{tp}"', lambda: "tp[1]"]
            if self.use_dict:
                options.append(lambda: 'f"{d}".slice(0, 24)')
            if self.use_set:
                options.append(lambda: 'f"{st}".slice(0, 24)')
            return r.choice(options)()
        pick = r.randrange(6)
        if pick == 0:
            return f"({self.str_expr(depth - 1)} + {self.str_expr(depth - 1)})"
        if pick == 1:
            return f"{self.str_expr(depth - 1)}.upper()"
        if pick == 2:
            meth = r.choice(["lower", "strip", "title", "capitalize"])
            return f"{self.str_expr(depth - 1)}.{meth}()"
        if pick == 3:
            # Never the empty needle: replacing nothing has no useful meaning
            # and every implementation disagrees about how many times to do it.
            needle = r.choice(['"a"', '"bc"', '"j"', '"-"', '" "', '"0"'])
            return (f"{self.str_expr(depth - 1)}.replace("
                    f"{needle}, {r.choice(STR_LITS)})")
        if pick == 4:
            sep = r.choice(['"-"', '","', '""'])
            return f"{sep}.join([{self.str_expr(0)}, {self.str_expr(0)}])"
        return f'f"{{{self.str_expr(depth - 1)}}}|{{{self.int_expr(1)}}}"'

    # -- declarations ---------------------------------------------------

    def gen_classes(self):
        r = self.rng
        # An empty initialiser that must still evaluate to the new object.
        # A one-instruction OP_RETURN_NULL body once made this return null.
        if r.random() < 0.75:
            name = self.fresh("Empty")
            tag = r.randint(1, 99)
            self.prog.classes.append(
                f"class {name} {{\n"
                f"    fn init(self) {{}}\n"
                f"    pub fn tag(self) -> int {{ return {tag} }}\n"
                f"}}")
            self.classes[name] = {"ctor": f"{name}()", "float_field": False,
                                  "int_methods": ["tag"], "fields": []}

        if r.random() < 0.85:
            name = self.fresh("Box")
            lit = r.choice(FLOAT_LITS)
            self.prog.classes.append(
                f"class {name} {{\n"
                f"    pub var a: int\n"
                f"    pub var b: float\n"
                f"    pub var s: str\n"
                f"    fn init(self, a: int) {{\n"
                f"        self.a = a\n"
                f"        self.b = {lit}\n"
                f'        self.s = "{name.lower()}"\n'
                f"    }}\n"
                f"    pub fn geta(self) -> int {{ return self.a }}\n"
                f"    pub fn bump(self, k: int) -> int {{\n"
                f"        self.a = self.a +% k\n"
                f"        return self.geta()\n"
                f"    }}\n"
                f"    pub fn twice(self) -> int {{"
                f" return self.geta() +% self.geta() }}\n"
                f"}}")
            self.classes[name] = {"ctor": f"{name}(ARG)", "float_field": True,
                                  "int_methods": ["geta", "bump", "twice"],
                                  "fields": ["a"]}
            # An override reached through the parent's own `self.geta()`.
            if r.random() < 0.6:
                sub = self.fresh("Sub")
                self.prog.classes.append(
                    f"class {sub} extends {name} {{\n"
                    f"    pub var c: int\n"
                    f"    fn init(self, a: int, c: int) {{\n"
                    f"        super(a)\n"
                    f"        self.c = c\n"
                    f"    }}\n"
                    f"    pub fn geta(self) -> int {{"
                    f" return self.a +% self.c }}\n"
                    f"}}")
                self.classes[sub] = {"ctor": f"{sub}(ARG, ARG2)",
                                     "float_field": True,
                                     "int_methods": ["geta", "bump", "twice"],
                                     "fields": ["a", "c"]}

    def gen_helpers(self):
        r = self.rng
        if r.random() < 0.8:
            name = self.fresh("h")
            saved = self.scoped(["a", "b"], [], [])
            body = [f"fn {name}(a: int, b: int) -> int {{",
                    f"    if {self.bool_expr()} {{",
                    f"        return {self.int_expr(2)}",
                    "    }",
                    f"    var t = {self.int_expr(1)}",
                    f"    for q in 0..{r.randint(2, 12)} {{",
                    f"        t = t +% (q *% {r.randint(1, 9)})",
                    f"        if t % {r.randint(2, 17)} == 0 {{ return t }}",
                    "    }",
                    "    return t +% a +% b",
                    "}"]
            self.unscope(saved)
            self.prog.helpers.append("\n".join(body))
            self.helpers[name] = (2, "int")

        if r.random() < 0.8:
            name = self.fresh("rec")
            body = [f"fn {name}(k: int) -> int {{",
                    "    if k <= 0 { return 1 }",
                    f"    return (k *% {r.randint(2, 9)}) +% {name}(k - 1)",
                    "}"]
            self.prog.helpers.append("\n".join(body))
            self.helpers[name] = (1, "rec")

        if r.random() < 0.4:
            even = self.fresh("mutA")
            odd = self.fresh("mutB")
            self.prog.helpers.append(
                f"fn {even}(k: int) -> int {{\n"
                f"    if k <= 0 {{ return 2 }}\n"
                f"    return 1 +% {odd}(k - 1)\n"
                f"}}\n\n"
                f"fn {odd}(k: int) -> int {{\n"
                f"    if k <= 0 {{ return 3 }}\n"
                f"    return {r.randint(2, 5)} *% {even}(k - 1)\n"
                f"}}")
            self.helpers[even] = (1, "rec")

        if r.random() < 0.5:
            name = self.fresh("lst")
            self.prog.helpers.append(
                f"fn {name}(v: list) -> int {{\n"
                f"    var t = 0\n"
                f"    for x in v {{\n"
                f"        if x % {r.randint(2, 9)} == 0 {{ continue }}\n"
                f"        t = t +% x\n"
                f"    }}\n"
                f"    return t\n"
                f"}}")
            self.helpers[name] = (1, "list")

        # JIT_MAX_ARITY is 4; a 4-argument function is the widest the
        # whole-function tier will take.
        if r.random() < 0.4:
            name = self.fresh("quad")
            self.prog.helpers.append(
                f"fn {name}(a: int, b: int, c: int, e: int) -> int {{\n"
                f"    return ((a +% b) *% 3) +% (c -% e)\n"
                f"}}")
            self.helpers[name] = (4, "int")

    # A container helper's parameter name, its declared type, and the flag
    # that makes the statement machinery emit code naming it.
    CONTAINERS = {
        "dict":   ("d",  "dict",  "use_dict"),
        "set":    ("st", "set",   "use_set"),
        "tuple":  ("tp", "tuple", "use_tuple"),
        "nested": ("ys", "list",  "use_nested"),
        "mixed":  ("ms", "list",  "use_mixed"),
    }

    def gen_container_helpers(self):
        """Functions that take a container as a PARAMETER.

        This is the difference between fuzzing the tier and fuzzing nothing.
        A container BUILT in the body it is read from has no live sample, so
        every arm that needs one -- the dict index, the list element kind, the
        set length -- declines, and a probe full of dict work compiled 1 time
        in 72. Received as a parameter the call site's inline cache has the
        sample, and the same work compiles: measured on a probe of nothing but
        `d[k]` reads, 1/72 as a local against a clean compile as a parameter.

        So the container statements are generated TWICE over: once inside the
        probe, where they exercise the decline path and the OSR loop tier, and
        once here, where they reach the whole-function tier.
        """
        r = self.rng
        for kind, (pname, ptype, flag) in self.CONTAINERS.items():
            if r.random() < 0.45:
                continue
            name = self.fresh("c" + kind[0])
            saved = self.container_scope(flag)
            body = [self.stmt(0) for _ in range(r.randint(2, 4))]
            out = [f"fn {name}({pname}: {ptype}, n: int) -> int {{",
                   "    var acc = 0"]
            for node in body:
                node.render(out, 1)
            # No fold of the container here: the helper may mutate it, and the
            # PROBE folds it after the call. One local, one parameter and a
            # plain `return acc` is the leanest body that still carries the
            # shape, and lean is what compiles -- the first version copied the
            # probe's whole xs/text/f preamble and its fold tail in here as
            # well, and every one of those was one more thing for the tier to
            # stop at.
            out.append("    return acc")
            out.append("}")
            self.unscope_container(saved)
            self.prog.helpers.append("\n".join(out))
            self.cont_helpers.setdefault(kind, []).append(name)

    def container_scope(self, flag):
        """Make exactly one container flag true while a helper body is built."""
        saved = (self.use_dict, self.use_set, self.use_tuple, self.use_nested,
                 self.use_mixed, self.use_empty, self.use_pow, self.insts,
                 self.loop_depth, self.iterating, self.early_ret,
                 self.ints, self.assignable, self.floats, self.strs,
                 self.containers, self.focus)
        self.use_dict = self.use_set = self.use_tuple = False
        self.use_nested = self.use_mixed = self.use_empty = False
        self.use_pow = False
        self.enable(flag)
        self.insts = []
        self.loop_depth = 0
        self.iterating = set()
        self.early_ret = "acc"
        self.ints = ["n", "acc"]
        self.assignable = ["acc"]
        self.floats = []
        self.strs = []
        self.containers = False
        self.focus = True
        return saved

    def unscope_container(self, saved):
        (self.use_dict, self.use_set, self.use_tuple, self.use_nested,
         self.use_mixed, self.use_empty, self.use_pow, self.insts,
         self.loop_depth, self.iterating, self.early_ret,
         self.ints, self.assignable, self.floats, self.strs,
         self.containers, self.focus) = saved

    def walking(self, name):
        """Mark `name` as being iterated right now; returns a restore token.

        A plain `iterating.add` / `iterating.discard` pair is wrong once the
        loops nest: `for a in xs { for b in xs { ... } }` has the inner loop
        clear the flag on the way out, and the outer loop's remaining body is
        then free to push to a list it is still walking. That is a
        RuntimeError in every configuration alike, so the seed tests nothing.
        """
        had = name in self.iterating
        self.iterating.add(name)
        return (name, had)

    def done_walking(self, token):
        name, had = token
        if not had:
            self.iterating.discard(name)

    def visible_containers(self):
        """Which of the helper-callable containers this scope can name."""
        out = []
        for kind, (pname, _t, flag) in self.CONTAINERS.items():
            if getattr(self, flag) and pname not in self.iterating:
                out += [(kind, pname)] * len(self.cont_helpers.get(kind, ()))
        return out

    # -- statements -----------------------------------------------------

    def block(self, count, depth):
        """A list of Nodes. May legally be empty -- Jaithon accepts `if c {}`.

        Instances declared inside the block leave scope with it, so a later
        sibling of the PARENT cannot name one.
        """
        mark = len(self.insts)
        nodes = [self.stmt(depth) for _ in range(count)]
        del self.insts[mark:]
        return nodes

    def stmt(self, depth=0):
        r = self.rng
        # Everything here reads only `acc` and its own fresh locals, so it is
        # legal in a probe and in a container helper alike.
        kinds = [
            (self.st_int, 16),
            (self.st_branch, 12),
            (self.st_for_range, 11),
            (self.st_while, 6),
            (self.st_loop, 4),
            (self.st_cond_local, 9),
            (self.st_class_use, 10),
            (self.st_match, 5),
            (self.st_try, 4),
            (self.st_labelled, 4),
            (self.st_optional, 4),
            (self.st_str_build, 6),
            (self.st_build_in_loop, 7),
            (self.st_comprehension, 5),
        ]
        # `xs` and `text` are the probe's own locals. A container helper has
        # neither -- its whole body is about the container it was handed, and
        # every extra local was one more thing for the tier to refuse.
        if self.containers:
            kinds += [(self.st_str, 7), (self.st_for_list, 6),
                      (self.st_list_ops, 8), (self.st_lambda, 4),
                      (self.st_str_index, 8)]
        if self.floats:
            kinds.append((self.st_float, 8))
        # The container family. Each arm exists only where the container it
        # reads was declared, so nothing ever names something undeclared.
        if self.use_dict:
            kinds += [(self.st_dict_ops, 5), (self.st_for_dict, 4),
                      (self.st_dict_kinds, 7), (self.st_dict_miss, 5)]
        if self.use_set:
            kinds.append((self.st_set_ops, 8))
        if self.use_tuple:
            kinds.append((self.st_tuple_ops, 7))
        if self.use_nested:
            kinds.append((self.st_nested, 8))
        if self.use_mixed:
            kinds.append((self.st_mixed, 8))
        if self.use_empty:
            kinds.append((self.st_empty, 6))
        # Only at the top of a body. Inside two nested loops a helper that
        # itself loops over a fold is multiplied twice over: 64 x 13 calls
        # to one doing 80 dict folds was a single program the interpreter
        # could not finish in five minutes.
        if self.loop_depth == 0 and self.visible_containers():
            kinds.append((self.st_call_helper, 10))
        if self.loop_depth > 0:
            kinds.append((self.st_early_return, 7))
            kinds.append((self.st_break_continue, 6))
        if self.focus:
            # A container helper's body. Everything that is not about the
            # container is dropped: a class instantiation, a `try`, a lambda
            # or a call to an uncompiled helper each stop the walk, and the
            # first statement that does ends the function. The probes carry
            # the general grammar; this carries the shape under test.
            # The scaffolding is kept at a low weight so the container arms
            # are the majority of what lands: a helper whose two statements
            # both came out `acc = acc | (n % 51)` is a helper about nothing.
            scaffold = {self.st_int: 4, self.st_branch: 4,
                        self.st_for_range: 4, self.st_while: 3}
            family = (self.st_dict_ops, self.st_dict_kinds, self.st_dict_miss,
                      self.st_for_dict, self.st_set_ops, self.st_tuple_ops,
                      self.st_nested, self.st_mixed, self.st_empty)
            kinds = ([(k, scaffold[k]) for k, _w in kinds if k in scaffold]
                     + [(k, w) for k, w in kinds if k in family])
        if depth >= 2:
            # Stop nesting: only the flat kinds, so a program stays small --
            # and only the CHEAP container arms, since depth 2 is the body of
            # two nested loops and everything that folds a whole container is
            # O(size) with a string built per element.
            kinds = [(k, w) for k, w in kinds
                     if k in (self.st_int, self.st_float, self.st_str,
                              self.st_list_ops, self.st_dict_ops,
                              self.st_class_use, self.st_cond_local,
                              self.st_str_index, self.st_tuple_ops)]
        picks = [k for k, w in kinds for _ in range(w)]
        return r.choice(picks)(depth)

    def st_int(self, depth):
        r = self.rng
        target = r.choice(self.assignable) if self.assignable else "acc"
        op = r.choice(["+%", "-%", "*%", "|", "&", "^"])
        return line(f"{target} = {target} {op} {self.int_expr(2)}")

    def st_float(self, depth):
        return line(f"f = {self.float_expr(2)}")

    def st_str(self, depth):
        """Append to the probe's digest string.

        The appended value goes through an f-string, and that is harness
        plumbing rather than a choice about what to fuzz: `text` is declared
        `str`, and a string produced by a METHOD -- `.upper()`, `.replace()`,
        `"-".join(...)`, even `str(n)` -- is an object as far as the tier can
        tell, so assigning one straight in refuses the whole body on
        `OP_TYPE_GUARD: a str guard on a object`. Wrapped, all four compile,
        and the method call is still made and still reaches the digest.
        """
        # Guarded so text stays bounded however deeply this nests.
        return Node("if text.len() < 200 {",
                    [line('text = text + f"{' + self.str_expr(1) + '}"')],
                    "}")

    def st_branch(self, depth):
        r = self.rng
        parts = [f"if {self.bool_expr()} {{",
                 self.block(r.randint(1, 2), depth + 1)]
        if r.random() < 0.4:
            parts += [f"}} elif {self.bool_expr()} {{",
                      self.block(r.randint(1, 2), depth + 1)]
        if r.random() < 0.6:
            parts += ["} else {", self.block(r.randint(1, 2), depth + 1)]
        parts.append("}")
        return Node(*parts)

    def st_for_range(self, depth):
        r = self.rng
        var = self.fresh("i")
        hi = self.trips([3, 5, 8, 16, 32, 64, 200], [3, 5, 8, 12])
        lo = r.choice([0, 0, 0, 1, -3])
        rng = f"{lo}..{hi}" if r.random() < 0.7 else f"{lo}..={hi}"
        self.ints.append(var)
        self.loop_depth += 1
        body = self.block(r.randint(1, 3), depth + 1)
        self.loop_depth -= 1
        self.ints.remove(var)
        # The loop variable is consumed by a plain string leading the body, so
        # a body that never happened to name it -- or a reduction that deleted
        # the statement that did -- cannot leave an unused binding, which the
        # checker reports on stderr and the oracle then has to read past.
        return Node(f"for {var} in {rng} {{",
                    [f"acc = acc +% ({var} % 97)"] + body, "}")

    def st_while(self, depth):
        r = self.rng
        var = self.fresh("w")
        trips = self.trips([3, 6, 12, 40, 100], [3, 6, 10])
        self.ints.append(var)
        self.loop_depth += 1
        body = self.block(r.randint(1, 3), depth + 1)
        self.loop_depth -= 1
        self.ints.remove(var)
        # The increment leads the body and is a plain string, not a Node, so
        # neither a generated `continue` nor a shrink step can strand the loop.
        return Node(f"var {var} = 0",
                    f"while {var} < {trips} {{",
                    [f"{var} = {var} + 1"] + body,
                    "}")

    def st_loop(self, depth):
        r = self.rng
        var = self.fresh("l")
        trips = self.trips([2, 5, 10, 30], [2, 5, 8])
        self.ints.append(var)
        self.loop_depth += 1
        body = self.block(r.randint(1, 2), depth + 1)
        self.loop_depth -= 1
        self.ints.remove(var)
        # Both the increment and the break that consumes it are strings: this
        # loop has no other exit, so a reduction that took either would not
        # terminate.
        return Node(f"var {var} = 0",
                    "loop {",
                    [f"{var} = {var} + 1",
                     f"if {var} > {trips} {{ break }}"] + body,
                    "}")

    def st_for_list(self, depth):
        r = self.rng
        var = self.fresh("e")
        self.ints.append(var)
        self.loop_depth += 1
        mark = self.walking("xs")
        body = self.block(r.randint(1, 2), depth + 1)
        self.done_walking(mark)
        self.loop_depth -= 1
        self.ints.remove(var)
        if r.random() < 0.3:
            idx = self.fresh("p")
            return Node(f"for ({idx}, {var}) in xs.enumerate() {{",
                        [f"acc = acc +% {idx} +% {var}"] + body, "}")
        return Node(f"for {var} in xs {{",
                    [f"acc = acc +% {var}"] + body, "}")

    def st_for_dict(self, depth):
        """Walk a dict's pairs.

        Neither binding joins the visible int or str names: a key may be an
        int and a value may be anything a branch decided, so the only safe way
        to read either is through `str`. That read leads the body as a plain
        string, which also stops the checker warning about an unused binding
        when a reduction empties the rest of it.
        """
        r = self.rng
        key = self.fresh("dk")
        val = self.fresh("dv")
        self.loop_depth += 1
        mark = self.walking("d")
        body = self.block(r.randint(1, 2), depth + 1)
        self.done_walking(mark)
        self.loop_depth -= 1
        lead = (f"acc = acc +% {self.render_fold(key)}"
                f" +% {self.render_fold(val)}")
        return Node(f"for ({key}, {val}) in d.items() {{",
                    [lead] + body,
                    "}")

    def st_cond_local(self, depth):
        """A local written on one branch only, then read.

        This is the shape of the SIGSEGV: a local's register home carries no
        tag, so a deopt stub tagged the never-assigned slot VAL_OBJ over a zero
        payload and OSR dereferenced {VAL_OBJ, obj=NULL}.
        """
        r = self.rng
        var = self.fresh("m")
        cond = self.bool_expr()
        style = r.randrange(3)
        if style == 0:
            return Node(f"var {var}: any = null",
                        f"if {cond} {{",
                        [line(f"{var} = {self.int_expr(1)}")],
                        "}",
                        f"if {var} != null {{",
                        [line("acc = acc +% 1")],
                        "}")
        if style == 1:
            return Node(f"var {var}: int? = null",
                        f"if {cond} {{",
                        [line(f"{var} = {self.int_expr(1)}")],
                        "}",
                        f"acc = acc +% ({var} ?? {r.choice(SMALL_INTS)})")
        cls = sorted(self.classes)
        if not cls:
            return Node(f"var {var}: any = null",
                        f"if {cond} {{",
                        [line(f'{var} = "{var}"')],
                        "}",
                        f"if {var} == null {{",
                        [line("acc = acc -% 1")],
                        "}")
        name = r.choice(cls)
        ctor = self.classes[name]["ctor"].replace(
            "ARG2", self.int_expr(0)).replace("ARG", self.int_expr(0))
        return Node(f"var {var}: any = null",
                    f"if {cond} {{",
                    [line(f"{var} = {ctor}")],
                    "}",
                    f"if {var} != null {{",
                    [line(f"acc = acc +% {var}.geta()"
                          if "geta" in self.classes[name]["int_methods"]
                          else f"acc = acc +% {var}.tag()")],
                    "}")

    def st_early_return(self, depth):
        return Node(f"if {self.bool_expr()} {{",
                    [line(f"return {self.early_ret}")],
                    "}")

    def st_break_continue(self, depth):
        kw = self.rng.choice(["break", "continue"])
        return Node(f"if {self.bool_expr()} {{", [line(kw)], "}")

    def st_list_ops(self, depth):
        r = self.rng
        if "xs" in self.iterating:
            # Mutating mid-walk raises in every configuration alike, which
            # tests nothing. iter_mutation.py owns that axis deliberately.
            return line("acc = acc +% lfold(xs)")
        pick = r.randrange(6)
        if pick == 0:
            return line(f"xs.push({self.int_expr(1)})")
        if pick == 1:
            # Guarded: xs must never empty, or `k % xs.len()` divides by zero.
            return Node("if xs.len() > 1 {", [line("xs.pop()")], "}")
        if pick == 2:
            return line(f"xs[{r.randint(0, 30)} % xs.len()] = "
                        f"{self.int_expr(1)}")
        if pick == 3:
            return line(f"xs.insert(0, {self.int_expr(1)})")
        if pick == 4:
            return line("acc = acc +% xs[-1] +% xs[0]")
        return line("acc = acc +% lfold(xs)")

    def st_dict_ops(self, depth):
        r = self.rng
        # Only the SEEDED key may be read without a test -- it is the one the
        # literal always carries and the one nothing ever removes. Every other
        # key is read behind an `in`, or through `get` with a default.
        key = r.choice(DICT_KEYS)
        seed = self.dict_seed
        if "d" in self.iterating:
            return line(f"acc = acc +% d[{seed}]")
        pick = r.randrange(5)
        if pick == 0:
            return line(f"d[{key}] = {self.int_expr(1)}")
        if pick == 1:
            return line(f"acc = acc +% d[{seed}]")
        if pick == 2:
            return Node(f"if {key} in d {{",
                        [line("acc = acc +% "
                              + self.render_fold(f"d[{key}]"))], "}")
        if pick == 3:
            return line("acc = acc +% " + self.render_fold(
                f"d.get({key}, {r.choice(SMALL_INTS)})"))
        return line("acc = acc +% d.len()")

    def st_class_use(self, depth):
        r = self.rng
        names = sorted(self.classes)
        if not names:
            return self.st_int(depth)
        name = r.choice(names)
        api = self.classes[name]
        var = self.fresh("o")
        ctor = api["ctor"].replace("ARG2", self.int_expr(0)).replace(
            "ARG", self.int_expr(0))
        parts = [f"let {var} = {ctor}"]
        # The empty-init shape: the constructor must evaluate to the object.
        parts.append(f"if {var} == null {{")
        parts.append([line("acc = acc -% 424242")])
        parts.append("}")
        meth = r.choice(api["int_methods"])
        call = (f"{var}.bump({self.int_expr(0)})" if meth == "bump"
                else f"{var}.{meth}()")
        parts.append(f"acc = acc +% {call}")
        if api["fields"] and r.random() < 0.6:
            fld = r.choice(api["fields"])
            parts.append(f"{var}.{fld} = {var}.{fld} +% {self.int_expr(0)}")
            parts.append(f"acc = acc ^ {var}.{fld}")
        if api["float_field"] and self.floats and r.random() < 0.4:
            parts.append(f"f = f + {var}.b")
        # Stays visible to later siblings; block() drops it at the brace.
        self.insts.append((var, name))
        return Node(*parts)

    def st_match(self, depth):
        r = self.rng
        a, b, c, e = (r.randint(-99, 99) for _ in range(4))
        return line(f"acc = acc +% match ({self.int_expr(1)} % 5) {{\n"
                    f"    0 => {a},\n"
                    f"    1 | 2 => {b},\n"
                    f"    3..=4 => {c},\n"
                    f"    _ => {e},\n"
                    f"}}")

    def st_lambda(self, depth):
        r = self.rng
        pick = r.randrange(3)
        if pick == 0:
            return line(f"acc = acc +% lfold(xs.map(|v| v *% "
                        f"{r.randint(2, 9)}))")
        if pick == 1:
            return line(f"acc = acc +% lfold(xs.filter(|v| v % "
                        f"{r.randint(2, 7)} == 0))")
        return line("acc = acc +% xs.reduce(0, |t, v| t +% v)")

    def st_try(self, depth):
        """A checked overflow that is caught, so the program survives it.

        The uncaught arm is covered too, rarely, by st_late_raise.
        """
        r = self.rng
        big = r.choice(["9223372036854775807", INT_MIN_SRC,
                        "4611686018427387904"])
        op = r.choice(["+", "-", "*"])
        return Node("try {",
                    [line(f"acc = acc +% ({big} {op} {self.int_expr(0)})")],
                    "} catch _e: OverflowError {",
                    [line(f"acc = acc -% {r.randint(1, 9999)}")],
                    "}")

    def st_labelled(self, depth):
        r = self.rng
        outer = self.fresh("r")
        inner = self.fresh("c")
        label = f"'L{self.uid}"
        self.ints += [outer, inner]
        self.loop_depth += 1
        body = self.block(1, depth + 1)
        self.loop_depth -= 1
        self.ints.remove(outer)
        self.ints.remove(inner)
        kw = r.choice(["break", "continue"])
        return Node(f"{label}: for {outer} in 0..{r.randint(2, 6)} {{",
                    [Node(f"for {inner} in 0..{r.randint(2, 6)} {{",
                          body + [line(f"if {outer} *% {inner} > "
                                       f"{r.randint(1, 20)} {{ {kw} {label} }}")],
                          "}")],
                    "}")

    def st_optional(self, depth):
        r = self.rng
        var = self.fresh("q")
        return Node(f"var {var}: int? = {self.int_expr(0)}",
                    f"if {self.bool_expr()} {{",
                    [line(f"{var} = null")],
                    "}",
                    f"acc = acc +% ({var} ?? {r.choice(SMALL_INTS)})")

    # -- the container family --------------------------------------------

    def st_str_index(self, depth):
        """Index and slice a string that may legitimately be empty.

        `text` starts `""` and only some paths append to it, so every index is
        behind a length test -- and the position tested is the LAST valid one,
        which is where an off-by-one lives. Slicing needs no guard: it clamps.
        """
        r = self.rng
        pick = r.randrange(4)
        if pick == 0:
            return Node("if text.len() > 0 {",
                        [line("acc = acc +% ord(text[text.len() - 1])"),
                         line("acc = acc ^ ord(text[0])"),
                         line("acc = acc +% ord(text[-1])")],
                        "}")
        if pick == 1:
            return Node("if text.len() > 2 {",
                        [line(f"acc = acc +% ord(text[{r.randint(0, 40)} "
                              "% text.len()])")],
                        "}")
        if pick == 2:
            return Node("if text.len() < 200 {",
                        [line(f"text = text + text.slice({r.randint(0, 4)}, "
                              f"{r.randint(1, 9)})")],
                        "}")
        return line(f"a0 = a0 +% text[{r.randint(0, 6)}:"
                    f"{r.randint(1, 12)}].len()")

    def st_str_build(self, depth):
        """A string built one piece at a time inside a loop, read afterwards.

        The read is outside the loop on purpose: the OSR tier compiles the
        loop and has to hand the finished string back to the interpreter.
        """
        r = self.rng
        var = self.fresh("sb")
        cnt = self.fresh("w")
        trips = self.trips([3, 6, 12, 40], [3, 6, 10])
        piece = r.choice(['"ab"', '"-"', 'f"{str(n % 10)}"',
                          'f"{n % 7}"', 'f"{n % 3}-"'])
        return Node(f'var {var} = ""',
                    f"var {cnt} = 0",
                    f"while {cnt} < {trips} {{",
                    [f"{cnt} = {cnt} + 1",
                     line(f"{var} = {var} + {piece}")],
                    "}",
                    f"acc = acc +% {var}.len() +% sfold({var})",
                    f"if {var}.len() > 0 {{",
                    [line(f"acc = acc ^ ord({var}[{var}.len() - 1])")],
                    "}")

    def st_dict_kinds(self, depth):
        """One dict key given a different KIND on each branch, then read back.

        The seeded key is never written here, so `d[seed]` stays an int and the
        int-typed reads elsewhere in the probe cannot start raising.
        """
        r = self.rng
        if "d" in self.iterating:
            # A write mid-walk raises in every configuration alike, which
            # tests nothing. iter_mutation.py owns that axis deliberately.
            return line("acc = acc +% dfold(d)")
        keys = [k for k in DICT_KEYS if k != self.dict_seed]
        key = r.choice(keys)
        a, b = r.sample(VALUE_KINDS, 2)
        return Node(f"if {self.bool_expr()} {{",
                    [line(f"d[{key}] = {self.kind_value(a)}")],
                    "} else {",
                    [line(f"d[{key}] = {self.kind_value(b)}")],
                    "}",
                    f"if {key} in d {{",
                    [line("acc = acc +% " + self.render_fold(f"d[{key}]"))],
                    "}",
                    f"acc = acc +% dfold(d) +% d.len()")

    def st_dict_miss(self, depth):
        """A lookup that misses, three ways, none of which may raise."""
        r = self.rng
        key = r.choice(['"absent"', '"nope"', '9999', '-9999'])
        return Node(f"if {key} in d {{",
                    [line("acc = acc +% " + self.render_fold(f"d[{key}]"))],
                    "} else {",
                    [line(f"acc = acc -% {r.randint(1, 999)}")],
                    "}",
                    "acc = acc +% " + self.render_fold(
                        f"d.get({key}, {r.choice(SMALL_INTS)})"),
                    f"if d.has({key}) {{", [line("acc = acc +% 1")], "}")

    def st_set_ops(self, depth):
        r = self.rng
        pick = r.randrange(4)
        if pick == 0 and "st" not in self.iterating:
            # `discard` and not `remove`: removing an absent member raises,
            # and a raise every configuration shares tests nothing. The size
            # cap keeps the fold at the end of the probe cheap.
            return Node("if st.len() < 8 {",
                        [line(f"st.add({self.int_expr(1)} % 32)")],
                        "}",
                        f"st.discard({r.randint(0, 40)})",
                        "acc = acc +% st.len() +% qfold(st)")
        if pick == 1:
            return Node(f"if {r.randint(0, 8)} in st {{",
                        [line(f"acc = acc +% {r.randint(1, 99)}")],
                        "} else {",
                        [line("acc = acc -% 3")],
                        "}")
        if pick == 2 and "st" not in self.iterating:
            # Iterating a set: insertion-ordered, so this is deterministic.
            var = self.fresh("sv")
            self.ints.append(var)
            self.loop_depth += 1
            mark = self.walking("st")
            body = self.block(1, depth + 1)
            self.done_walking(mark)
            self.loop_depth -= 1
            self.ints.remove(var)
            return Node(f"for {var} in st {{",
                        [f"acc = acc +% {var}"] + body, "}")
        other = ", ".join(str(r.randint(0, 9)) for _ in range(r.randint(1, 3)))
        meth = r.choice(["union", "intersection", "difference"])
        return line(f"acc = acc +% qfold(st.{meth}({{{other}}}))")

    def st_tuple_ops(self, depth):
        """Construct, destructure, index and compare tuples.

        Every name a destructuring binds is read, or the checker warns about
        an unused binding and the warning reaches stderr.
        """
        r = self.rng
        pick = r.randrange(3)
        if pick == 0:
            x, y = self.fresh("u"), self.fresh("v")
            return Node(f"let ({x}, {y}) = ({self.int_expr(1)}, "
                        f"{self.str_expr(0)})",
                        f"acc = acc +% {x} +% {y}.len()")
        if pick == 1:
            x, y, z = self.fresh("u"), self.fresh("v"), self.fresh("w")
            return Node(f"let ({x}, {y}, {z}) = tp",
                        f"acc = acc +% {x} +% {y}.len() +% {z}",
                        f"acc = acc +% tfold(tp) +% tp.len()")
        made = self.fresh("tu")
        return Node(f"let {made} = ({self.int_expr(0)}, "
                    f"{self.kind_value()})",
                    f"if {made} == ({self.int_expr(0)}, "
                    f"{self.kind_value('int')}) {{",
                    [line("acc = acc +% 7")],
                    "}",
                    f"acc = acc +% tfold({made})")

    def st_nested(self, depth):
        """A list of lists: indexed, written through, and grown.

        No inner list is ever emptied and `ys` is never emptied, so both
        indices stay in range under every path.
        """
        r = self.rng
        pick = r.randrange(4)
        outer = f"{r.randint(0, 30)} % ys.len()"
        if pick == 0:
            return line(f"acc = acc +% gpick(ys, {self.index_expr('ys')}, "
                        f"{r.randint(-6, 30)})")
        if pick == 1:
            return Node("if ys.len() < 8 {",
                        [line(f"ys.push([{self.int_expr(0)}, "
                              f"{self.int_expr(0)}])")],
                        "}",
                        "acc = acc +% vfold(ys) +% ys.len()")
        if pick == 2:
            row = self.fresh("rw")
            return Node(f"let {row} = ys[{outer}]",
                        f"if {row}.len() < 8 {{",
                        [line(f"{row}.push({self.kind_value()})")],
                        "}",
                        f"acc = acc +% vfold({row})")
        row = self.fresh("rw")
        return Node(f"let {row} = ys[{outer}]",
                    f"{row}[{self.index_expr(row)}] = {self.int_expr(1)}",
                    f"acc = acc +% vfold({row}) +% {row}.len()")

    def st_mixed(self, depth):
        """A list whose element kind is decided at run time, not at compile.

        Two branches push different kinds into the same list, and everything
        that reads it back has to work out what it actually got.
        """
        r = self.rng
        a, b = r.sample(VALUE_KINDS, 2)
        pick = r.randrange(3)
        if pick == 0:
            return Node("if ms.len() < 8 {",
                        [Node(f"if {self.bool_expr()} {{",
                              [line(f"ms.push({self.kind_value(a)})")],
                              "} else {",
                              [line(f"ms.push({self.kind_value(b)})")],
                              "}")],
                        "}",
                        "acc = acc +% vfold(ms) +% ms.len()")
        if pick == 1:
            var = self.fresh("mv")
            return Node(f"var {var}: any = {self.kind_value(a)}",
                        f"if {self.bool_expr()} {{",
                        [line(f"{var} = {self.kind_value(b)}")],
                        "}",
                        f"ms[{self.index_expr('ms')}] = {var}",
                        "acc = acc +% " + self.render_fold(var)
                        + " +% vfold(ms)")
        var = self.fresh("me")
        self.strs.append(var)
        node = Node(f'let {var} = f"{{ms[{self.index_expr("ms")}]}}"',
                    f"acc = acc +% {var}.len() +% sfold({var})")
        self.strs.remove(var)
        return node

    def st_empty(self, depth):
        """An empty container, read before anything is put in it.

        A length, a membership test and a first/last are all legal on nothing,
        and the conditional fill means the SAME slot is empty on some calls and
        not on others -- which the tier compiles once, for whichever it saw.
        """
        r = self.rng
        pick = r.randrange(3)
        if pick == 0:
            return Node("acc = acc +% ez.len() +% ed.len()",
                        f"if {self.bool_expr()} {{",
                        [line(f"ez.push({self.kind_value()})"),
                         line(f"ed[{r.choice(DICT_KEYS)}] = "
                              f"{self.kind_value()}")],
                        "}",
                        "if ez.len() > 0 {",
                        [line("acc = acc +% "
                              + self.render_fold("ez[ez.len() - 1]"))],
                        "}",
                        "acc = acc +% vfold(ez) +% dfold(ed)")
        if pick == 1:
            var = self.fresh("ec")
            # `{}` is the empty DICT -- there is no empty set literal, so a set
            # has to be asked for by name.
            decl, fold = r.choice([(f"var {var}: list = []", "vfold"),
                                   (f"var {var}: dict = {{}}", "dfold"),
                                   (f"var {var} = set()", "qfold")])
            return Node(decl, f"acc = acc +% {var}.len() +% {fold}({var})")
        return Node(f"if {r.choice(SMALL_INTS)} not in ez {{",
                    [line("acc = acc +% 5")],
                    "}",
                    "acc = acc +% " + self.render_fold(
                        f"ez.first({r.choice(SMALL_INTS)})"))

    def st_build_in_loop(self, depth):
        """A container filled by a loop and read only after it ends.

        The read is what makes the loop's work visible: an OSR tier that lost
        an element, or wrote the wrong kind into one, is caught here and
        nowhere else.
        """
        r = self.rng
        var = self.fresh("bl")
        cnt = self.fresh("w")
        trips = self.trips([3, 6, 12], [3, 6])
        style = r.randrange(3)
        if style == 0:
            return Node(f"var {var}: list = []",
                        f"var {cnt} = 0",
                        f"while {cnt} < {trips} {{",
                        [f"{cnt} = {cnt} + 1",
                         line(f"{var}.push({self.kind_value()})")],
                        "}",
                        f"acc = acc +% {var}.len() +% vfold({var})",
                        f"if {var}.len() > 0 {{",
                        [line("acc = acc +% " + self.render_fold(
                            f"{var}[{var}.len() - 1]"))],
                        "}")
        if style == 1:
            return Node(f"var {var}: dict = {{}}",
                        f"var {cnt} = 0",
                        f"while {cnt} < {trips} {{",
                        [f"{cnt} = {cnt} + 1",
                         line(f'{var}[f"k{{{cnt}}}"] = {self.kind_value()}')],
                        "}",
                        f"acc = acc +% {var}.len() +% dfold({var})")
        return Node(f"var {var} = set()",
                    f"var {cnt} = 0",
                    f"while {cnt} < {trips} {{",
                    [f"{cnt} = {cnt} + 1",
                     line(f"{var}.add({cnt} % {r.randint(2, 9)})")],
                    "}",
                    f"acc = acc +% {var}.len() +% qfold({var})")

    def st_call_helper(self, depth):
        """Hand one of this scope's containers to a function that takes one.

        Never while that container is being walked: the helper is free to
        mutate it, and a mutation mid-iteration raises alike everywhere.
        """
        r = self.rng
        vis = self.visible_containers()
        if not vis:
            return self.st_int(depth)
        kind, pname = r.choice(vis)
        name = r.choice(self.cont_helpers[kind])
        return line(f"acc = acc +% {name}({pname}, {self.int_expr(1)})")

    def st_comprehension(self, depth):
        r = self.rng
        var = self.fresh("cp")
        hi = r.choice([3, 5, 8, 16])
        pick = r.randrange(4)
        if pick == 0:
            return Node(f"let {var} = [q *% {r.randint(2, 9)} for q in 0..{hi}]",
                        f"acc = acc +% lfold({var}) +% vfold({var})")
        if pick == 1:
            return Node(f"let {var} = [q for q in 0..{hi} "
                        f"if q % {r.randint(2, 5)} == 0]",
                        f"acc = acc +% {var}.len() +% vfold({var})")
        if pick == 2:
            return Node(f"let {var} = {{q: q *% q for q in 1..{hi}}}",
                        f"acc = acc +% {var}.len() +% dfold({var})")
        return Node(f"let {var} = {{q % {r.randint(2, 6)} for q in 0..{hi}}}",
                    f"acc = acc +% {var}.len() +% qfold({var})")

    def st_late_raise(self, depth):
        """An uncaught raise, thrown only once the tier has compiled.

        The traceback has to match the interpreter's exactly, and it fires late
        enough that everything before it still ran warm. Exit status is part of
        what the oracle compares, so a program that dies here still has to die
        the same way five times.

        The three shapes are the checked-overflow one this started as, and the
        two the container family adds: an index one past the end, and a key
        that is not there.
        """
        r = self.rng
        at = max(1, self.prog.warm - r.randint(2, 20))
        pick = r.randrange(3)
        if pick == 0:
            hurt = "acc = acc +% (9223372036854775807 + 1)"
        elif pick == 1:
            # Exactly one past the last valid position, computed rather than
            # written, so the checker cannot fold it away.
            hurt = "acc = acc +% xs[xs.len()]"
        else:
            hurt = 'acc = acc +% sfold(f"{xs[0 - xs.len() - 1]}")'
        return Node(f"if n == {at} {{", [line(hurt)], "}")

    def build(self):
        r = self.rng
        # One seeded key for the whole program: a container helper reads
        # `d[seed]` without a test, and it is called with whichever probe's
        # dict, so the key every dict is guaranteed to carry has to be the
        # same one everywhere.
        self.dict_seed = r.choice(DICT_KEYS)
        self.gen_classes()
        self.gen_helpers()
        self.gen_container_helpers()
        for k in range(r.randint(3, 5)):
            probe = Probe(f"probe{k}")
            # Per PROBE, not per program: a blocker confined to one function
            # leaves the others compiling. At most ONE of the container
            # families beyond `d`, for the same reason -- a probe carrying all
            # five declines for whichever of them the tier likes least, and
            # measuring said 44% of probes reached a compiled tier against 62%
            # before the family was added.
            self.use_dict = False
            self.use_nested = self.use_set = self.use_tuple = False
            self.use_mixed = self.use_empty = False
            self.use_pow = r.random() < 0.15
            if r.random() < 0.30:
                self.enable("use_dict")
            flavour = r.choice([None, None, None, "use_nested", "use_set",
                                "use_tuple", "use_mixed", "use_empty"])
            if flavour:
                self.enable(flavour)
            probe.use_dict = self.use_dict
            probe.dict_seed = self.dict_seed
            probe.use_nested = self.use_nested
            probe.use_set = self.use_set
            probe.use_tuple = self.use_tuple
            probe.use_mixed = self.use_mixed
            probe.use_empty = self.use_empty
            self.ints = ["n", "acc", "a0", "a1"]
            self.assignable = ["acc", "a0", "a1"]
            self.floats = ["f"]
            self.strs = ["text"]
            self.insts = []
            self.iterating = set()
            self.containers = True
            self.loop_depth = 0
            probe.body = [self.stmt(0) for _ in range(r.randint(1, 4))]
            # Every container helper this probe can reach is called once, at
            # the end, rather than left to turn up by chance in `stmt`. It did
            # not turn up: over 80 programs the helpers were generated 201
            # times and CALLED 0 times, so none of them was ever hot enough to
            # be considered and the whole point of them was lost. A probe runs
            # `warm` times, so a call here is a call 1500 times.
            for kind, (pname, _ty, flag) in self.CONTAINERS.items():
                if not getattr(self, flag):
                    continue
                for name in self.cont_helpers.get(kind, ()):
                    probe.body.append(line(f"acc = acc +% {name}({pname}, n)"))
            self.prog.probes.append(probe)
        if r.random() < 0.12:
            self.prog.probes[0].body.append(self.st_late_raise(0))
        return self.prog


def generate(seed, warm=WARM_DEFAULT):
    """The whole contract of this module: seed in, identical program out."""
    return Gen(seed, warm).build()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--warm", type=int, default=WARM_DEFAULT)
    args = ap.parse_args()
    sys.stdout.write(generate(args.seed, args.warm).render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
