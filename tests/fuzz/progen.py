#!/usr/bin/env python3
"""Generator of small random Jaithon programs for the JIT differential fuzzer.

The compiled tier is an accelerator that may always decline (src/vm/jit/jit.h),
so for ANY program every configuration of it has to print the same thing as the
interpreter. That makes a random program a test case on its own, with no
expected output to write down: differential.py runs one program six ways and
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
  - a variable holding one class on one branch and another on the next, which
    is the inline cache's own path; every generated class answers kind(),
    geta(), twice(), bump(k) and step(x) precisely so that any two of them are
    legal at the same call site and only the run-time value says which arrived
  - a `for` over a list holding two or more classes, which is the ONLY source
    shape that reaches the polymorphic inline cache in src/vm/jit (see
    st_inst_list, and src/vm/jit/README.md for why every other candidate
    misses). Before it was generated deliberately the arm was reached by 0 of
    100 generated programs; it is now reached by 76. tests/fuzz/pic_rate.py
    is the census that says so
  - a nullable instance conditionally assigned, and an `any` that is an int on
    one path and an object on the other
  - lists, dicts, strings, indexing, recursion, match, lambdas

Two invariants keep every generated program legal and terminating, since a
program that dies the same way five times proves nothing:

  - `xs` never empties (pops are guarded), so `xs[k % xs.len()]` cannot raise,
    and never passes 64 elements (pushes are capped); `d["k"]` is only ever
    read for a key seeded into the literal;
  - every loop carries its own counter increment as the FIRST statement of its
    body, outside anything the generator or the shrinker can delete, so no
    body and no `continue` can stop it terminating;
  - trip counts multiply, so they are drawn against TRIP_BUDGET, an iteration
    allowance one probe CALL may spend and nested loops share out. A program
    the runner cannot finish inside its timeout is reported as a divergence,
    and the two hits in a 3,000-program run were both only that: seed 300 grew
    xs to 25,200 elements, seed 1111 nested two 200-trip ranges. 66s and 69s
    respectively, now 1.3s and 1.1s, and neither was ever a miscompile.

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

# Loop iterations one probe CALL may run, shared out between nested loops.
TRIP_BUDGET = 2000

# Combined with `+% -% *%` only, so any of these is safe to reach.
EDGE_INTS = [
    0, 1, -1, 2, 3, 7, 63, 64, 65, 255, 256, 1023, 65535, 65536,
    2147483647, -2147483648, 2147483648, 4294967295,
    4611686018427387904, 9223372036854775807, -9223372036854775808,
]

SMALL_INTS = [0, 1, 2, 3, 5, 7, 11, 16, 31, 100, 999, -1, -2, -7, -100]

FLOAT_LITS = ["0.0", "1.0", "0.5", "2.0", "-1.5", "3.25", "1e10", "-0.125",
              "1e-8", "100.0", "0.1", "-2.75"]

STR_LITS = ['""', '"a"', '"bc"', '"xyz"', '"jai"', '"0"', '"-"', '" "']

CMPS = ["==", "!=", "<", "<=", ">", ">="]

# EVERY generated class answers all four, so any two of them are legal at the
# same call site whatever the run-time value turns out to be. That uniformity
# is the whole point of the class family: a variable holding one class on one
# branch and a different one on the next is the polymorphic-inline-cache path
# (src/vm/jit/jit_call_pic.c), and a cache keyed on a compile-time kind is a
# lie exactly there.
CLASS_METHODS = ["kind", "geta", "twice", "bump", "step"]

# `step` is the one method written for the polymorphic inline cache, and every
# line of it is a constraint out of src/vm/jit/README.md rather than a taste:
#
#   - it is `pub`, because a way whose InlineCache::payload is set is dropped;
#   - it returns `int`, because the arm admits only a result that carries no
#     class shape;
#   - its body is CHECKED arithmetic bounded by `%`, because the compiled tier
#     walks no wrapping operator at all -- a callee whose walk stops at a `+%`
#     never records a return kind, and the arm drops that way too;
#   - and the `%` is what makes the checked arithmetic safe, so no generated
#     program can raise on overflow reaching it.
#
# Both operands land in (-STEP_MOD, STEP_MOD) before they are added, so the sum
# provably cannot overflow whatever the field holds.
STEP_MOD = 1000003


def step_method(term):
    """`step`'s source, given the per-class term it folds in."""
    return (f"    pub fn step(self, x: int) -> int {{"
            f" return (x % {STEP_MOD} + {term}) % {STEP_MOD} }}")

# What is known about a receiver whose class is NOT known statically -- an
# `any` that two branches filled differently. Only the shared contract.
COMMON_API = {"ctor": None, "int_methods": list(CLASS_METHODS), "fields": [],
              "float_field": False, "opt_int": None, "inst_field": None,
              "ops": False, "static": None, "maybe": False, "me": False,
              "mix": False, "walk": False, "base": None, "subs": [],
              "inst_type": None}


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


def int_lit(v):
    """Source for an integer constant.

    `-9223372036854775808` is not a literal the front end takes in every
    position: the magnitude is read first, and 9223372036854775808 does not
    fit, so `x - -9223372036854775808` is rejected outright (E0005) while
    `-9223372036854775808 - x` is fine. A program that fails to compile fails
    the same way under all six configurations, so it is not a false positive
    -- it is worse, a seed that tests nothing. The most negative int is
    therefore written as the expression that reaches it.
    """
    if v == -9223372036854775808:
        return "(-9223372036854775807 -% 1)"
    return str(v)


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
        # A dict stops that tier dead -- OP_GET_INDEX cannot index one and
        # `.len()` on a predicted dict has no sample to probe -- so `d` exists
        # only in probes that were chosen to use it.
        self.use_dict = False

    def render(self, out):
        out.append(f"fn {self.name}(n: int) -> str {{")
        out.append("    var acc = 0")
        out.append("    var a0 = 0")
        out.append("    var a1 = 0")
        out.append("    var f = 0.5")
        out.append('    var text = ""')
        out.append("    var xs = [1, 2, 3]")
        if self.use_dict:
            out.append('    var d = {"k": 1}')
        for node in self.body:
            node.render(out, 1)
        # Folded to ONE integer before it is formatted. An f-string is refused
        # outright past JIT_MAX_ARGS_OUT (12) parts, counting the literal
        # chunks between the holes, and the eight-hole version of this line was
        # the single largest reason probes failed to compile -- 41 of 203, all
        # of it self-inflicted by the harness rather than by the program.
        extra = " +% d.len()" if self.use_dict else ""
        out.append("    let z = acc *% 31 +% (a0 *% 7) +% a1 +% text.len()"
                   " +% sfold(text) +% lfold(xs)" + extra)
        out.append('    return f"{z}|{f}"')
        out.append("}")


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
        for block in self.helpers:
            out.append(block.rstrip())
            out.append("")
        for probe in self.probes:
            probe.render(out)
            out.append("")
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
        self.insts = []                 # (varname, classname) in scope
        self.classes = {}               # name -> dict describing its API
        self.helpers = {}               # name -> (arity, return kind)
        self.uid = 0
        # How many mixed-class dispatch loops the grammar happened to draw.
        # Counted because a program with none reaches jit_call_pic.c nowhere;
        # see build().
        self.dispatch_loops = 0
        self.loop_depth = 0
        # Iterations still affordable at this point in the probe, per CALL.
        # Trip counts multiply, and the probe is called `warm` times on top of
        # that, so an unbudgeted `for .. 0..200` inside another one is 60
        # million iterations and a program that takes a minute. A minute under
        # the runner's parallel load is a timeout, and the runner reads a
        # timeout as a divergence -- seeds 300 and 1111 were both exactly that
        # and neither was a miscompile. 2000 x 1500 calls is a second or so.
        self.trip_budget = TRIP_BUDGET
        # `xs`, `d` and `text` are locals of `probe`. A helper body is
        # generated with the same expression machinery but must not reach
        # them, so every atom that names one is gated on this.
        self.containers = True

    def fresh(self, stem):
        self.uid += 1
        return f"{stem}{self.uid}"

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
            lambda: int_lit(r.choice(SMALL_INTS)),
            lambda: int_lit(r.choice(EDGE_INTS)),
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
            ]
        if self.containers and self.use_dict:
            options += [lambda: "d.len()", lambda: 'd["k"]']
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
        if kind == "inst":
            names = self.constructible()
            if not names:
                return r.choice(self.ints)
            return f"{name}({self.ctor_of(r.choice(names))}, {self.int_expr(0)})"
        if kind == "rec":
            # Every argument is bounded to 0..11 by a positive modulus, so the
            # depth is bounded too -- an int_expr straight in blows the stack.
            return f"{name}(({self.int_expr(0)}) % 12)"
        args = ", ".join(self.int_expr(0) for _ in range(arity))
        return f"{name}({args})"

    def method_call(self):
        r = self.rng
        var, cls = r.choice(self.insts)
        api = self.api_of(cls)
        if api["maybe"] and r.random() < 0.2:
            return (f"({var}.maybe({self.int_expr(0)}) ?? "
                    f"{r.choice(SMALL_INTS)})")
        if api["opt_int"] and r.random() < 0.2:
            return f"({var}.{api['opt_int']} ?? {r.choice(SMALL_INTS)})"
        if api["fields"] and r.random() < 0.25:
            return f"{var}.{r.choice(api['fields'])}"
        # Deliberately NOT `o.me().geta()`. A call on a call's result has no
        # receiver kind for the tier to read -- SLOT_INT is the zero of
        # SlotKind, so the census calls it "a receiver of kind int" -- and this
        # atom reaches every expression in the program, so one chain here
        # refused a quarter of all probes. st_class_use emits the chain
        # instead, where a refusal costs one statement.
        return self.method_atom(var, r.choice(api["int_methods"]))

    def method_atom(self, var, meth):
        """`var.meth(...)` with the arity that method actually has.

        Two of the shared five take an int and three do not, and three call
        sites need to know. One place, so adding a sixth is one edit and not
        a hunt for the site that still thinks every method is nullary.
        """
        if meth in ("bump", "step"):
            return f"{var}.{meth}({self.int_expr(0)})"
        return f"{var}.{meth}()"

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
                         if self.api_of(c)["float_field"]]
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
        if pick == 4 and self.containers and self.use_dict:
            return '("k" in d)'
        if pick == 5 and self.strs:
            return (f"({r.choice(self.strs)} {r.choice(['==', '!='])} "
                    f"{r.choice(STR_LITS)})")
        if pick == 6 and self.insts:
            var, _ = r.choice(self.insts)
            return f"({var} {r.choice(['==', '!='])} null)"
        return f"({self.int_expr(1)} {r.choice(CMPS)} {self.int_expr(1)})"

    def str_expr(self, depth=1):
        r = self.rng
        if depth <= 0 or r.random() < 0.45:
            options = [
                lambda: r.choice(STR_LITS),
                lambda: f'f"{{{self.int_expr(1)}}}"',
                lambda: f'f"{{{self.float_expr(1)}}}"',
                lambda: f"str({self.int_expr(1)})",
            ]
            if self.strs:
                options.append(lambda: r.choice(self.strs))
            return r.choice(options)()
        pick = r.randrange(3)
        if pick == 0:
            return f"({self.str_expr(depth - 1)} + {self.str_expr(depth - 1)})"
        if pick == 1:
            return f"{self.str_expr(depth - 1)}.upper()"
        return f'f"{{{self.str_expr(depth - 1)}}}|{{{self.int_expr(1)}}}"'

    # -- declarations ---------------------------------------------------

    def add_class(self, name, src, ctor, **api):
        """Record one class and what may legally be written about it.

        Everything the generator needs to keep a program legal lives here:
        which fields exist (a write to an absent one is a checker error), which
        of the optional members were rolled, and which class is whose base.
        """
        self.prog.classes.append(src)
        info = dict(COMMON_API)
        info["ctor"] = ctor
        info["int_methods"] = list(CLASS_METHODS)
        info["subs"] = []
        info.update(api)
        self.classes[name] = info
        return info

    def api_of(self, cls):
        """The API of a receiver, or the shared contract when None."""
        return self.classes[cls] if cls is not None else COMMON_API

    def constructible(self):
        return sorted(self.classes)

    def plain_classes(self):
        """Classes whose constructor needs no instance -- what a holder holds.

        Kept separate so a holder can never be asked to hold a holder, which
        would recurse without bound in ctor_of.
        """
        return [n for n, a in sorted(self.classes.items())
                if "INST" not in a["ctor"]]

    def holdable_by(self, cls):
        """What may legally be stored in `cls`'s instance field.

        A field write is checked at run time (LANGUAGE.md), so a declared
        `Box` field takes a Box or any subclass of one and nothing else.
        """
        want = self.classes[cls]["inst_type"]
        if want is None:
            return self.plain_classes()
        return [want] + list(self.classes[want]["subs"])

    def ctor_of(self, name):
        """A construction expression, every placeholder filled in.

        ARG2 before ARG: the other order rewrites the `ARG` inside `ARG2`.
        """
        ctor = self.classes[name]["ctor"]
        if "INST" in ctor:
            held = self.holdable_by(name)
            inner = self.ctor_of(self.rng.choice(held)) if held else "0"
            ctor = ctor.replace("INST", inner)
        return ctor.replace("ARG2", self.int_expr(0)).replace(
            "ARG", self.int_expr(0))

    def gen_classes(self):
        r = self.rng
        if r.random() < 0.7:
            self.gen_empty()
        base = self.gen_box() if r.random() < 0.9 else None
        if base is not None:
            # At least one, so a base-typed variable always exists: a variable
            # declared `any` gives the whole-function tier no kind for its
            # receiver at all (SLOT_INT is the zero of SlotKind, which is why
            # the census reads "a receiver of kind int"), and that tier then
            # declines the body. A variable typed as the BASE is polymorphic
            # in the way that matters and still compiles.
            for _ in range(r.choice([1, 1, 2, 2])):
                self.gen_sub(base)
        if r.random() < 0.7:
            # Inheritance is not what makes a receiver interchangeable. A cache
            # keyed on shape has to cope with two classes that share nothing
            # but their method names.
            self.gen_peer()
        if self.classes and r.random() < 0.5:
            self.gen_holder()
        if r.random() < 0.4:
            self.gen_node()

    def gen_empty(self):
        """An empty initialiser that must still evaluate to the new object.

        A one-instruction OP_RETURN_NULL body once made this return null.
        """
        r = self.rng
        name = self.fresh("Empty")
        tag = r.randint(1, 99)
        self.add_class(
            name,
            f"class {name} {{\n"
            f"    fn init(self) {{}}\n"
            f"    pub fn kind(self) -> int {{ return {tag} }}\n"
            f"    pub fn geta(self) -> int {{ return {tag} *% 3 }}\n"
            f"    pub fn twice(self) -> int {{"
            f" return self.geta() +% self.geta() }}\n"
            f"    pub fn bump(self, k: int) -> int {{"
            f" return self.kind() +% k }}\n"
            f"{step_method(str(tag))}\n"
            f"}}",
            f"{name}()")
        return name

    def gen_box(self):
        """The field-storing class, with the optional members rolled per run."""
        r = self.rng
        name = self.fresh("Box")
        lit = r.choice(FLOAT_LITS)
        tag = r.randint(1, 99)
        ops = r.random() < 0.45
        static = r.random() < 0.4
        maybe = r.random() < 0.45
        me = r.random() < 0.35
        opt = r.random() < 0.45
        src = [f"class {name} {{",
               "    pub var a: int",
               "    pub var b: float",
               "    pub var s: str"]
        if opt:
            src.append("    pub var o: int?")
        src += ["    fn init(self, a: int) {",
                "        self.a = a",
                f"        self.b = {lit}",
                f'        self.s = "{name.lower()}"']
        if opt:
            src.append("        self.o = null")
        src += ["    }",
                "    pub fn geta(self) -> int { return self.a }",
                f"    pub fn kind(self) -> int {{ return {tag} }}",
                "    pub fn twice(self) -> int {"
                " return self.geta() +% self.geta() }",
                "    pub fn bump(self, k: int) -> int {",
                "        self.a = self.a +% k",
                "        return self.geta()",
                "    }",
                step_method(f"self.a % {STEP_MOD}")]
        if ops:
            src += [f"    pub fn __add__(self, o: {name}) -> int {{"
                    " return self.a +% o.a }",
                    f"    pub fn __eq__(self, o: {name}) -> bool {{"
                    " return self.a == o.a }",
                    '    pub fn __str__(self) -> str'
                    ' { return f"<{self.a}>" }']
        if static:
            src.append(f"    pub static fn make(k: int) -> {name} {{"
                       f" return {name}(k % 1000) }}")
        if maybe:
            src += ["    pub fn maybe(self, k: int) -> int? {",
                    f"        if k % {r.randint(2, 5)} == 0 {{ return null }}",
                    "        return self.a +% k",
                    "    }"]
        # A return whose KIND only the run-time value decides: an instance on
        # one path and an int on the other, out of one signature.
        src += ["    pub fn mix(self, k: int) -> any {",
                f"        if k % {r.randint(2, 4)} == 0 {{ return self }}",
                "        return self.a +% k",
                "    }"]
        if me:
            src.append(f"    pub fn me(self) -> {name} {{ return self }}")
        src.append("}")
        self.add_class(name, "\n".join(src), f"{name}(ARG)",
                       fields=["a"], float_field=True,
                       opt_int="o" if opt else None, ops=ops,
                       static="make" if static else None, maybe=maybe,
                       me=me, mix=True)
        return name

    def gen_sub(self, base):
        """A subclass whose override is reached through the base's own self."""
        r = self.rng
        name = self.fresh("Sub")
        tag = r.randint(1, 99)
        style = r.randrange(3)
        src = [f"class {name} extends {base} {{",
               "    pub var c: int",
               "    fn init(self, a: int, c: int) {",
               "        super(a)",
               "        self.c = c",
               "    }",
               f"    pub fn kind(self) -> int {{ return {tag} }}"]
        if style == 0:
            src.append("    pub fn geta(self) -> int {"
                       " return self.a +% self.c }")
        elif style == 1:
            src.append("    pub fn geta(self) -> int {"
                       " return super.geta() *% 2 +% self.c }")
        # style 2 inherits geta untouched, so one call site can see two classes
        # sharing a single method implementation.
        # `step` is overridden on two styles of three, so a base and its
        # subclass in one list are sometimes two ways of the site and sometimes
        # one -- both are cases the arm has to get right.
        if style != 2:
            src.append(step_method(f"self.c % {STEP_MOD}"))
        src.append("}")
        parent = self.classes[base]
        self.add_class(name, "\n".join(src), f"{name}(ARG, ARG2)",
                       fields=["a", "c"], float_field=True, base=base,
                       opt_int=parent["opt_int"], ops=parent["ops"],
                       maybe=parent["maybe"], me=parent["me"],
                       mix=parent["mix"])
        parent["subs"].append(name)
        return name

    def gen_peer(self):
        """Same method names, no relation. The duck-typed half of the cache."""
        r = self.rng
        name = self.fresh("Peer")
        tag = r.randint(1, 99)
        mult = r.randint(2, 9)
        self.add_class(
            name,
            f"class {name} {{\n"
            f"    pub var a: int\n"
            f"    fn init(self, a: int) {{ self.a = a }}\n"
            f"    pub fn kind(self) -> int {{ return {tag} }}\n"
            f"    pub fn geta(self) -> int {{ return self.a *% {mult} }}\n"
            f"    pub fn twice(self) -> int {{"
            f" return self.geta() +% self.geta() }}\n"
            f"    pub fn bump(self, k: int) -> int {{\n"
            f"        self.a = self.a -% k\n"
            f"        return self.geta()\n"
            f"    }}\n"
            f"{step_method(f'self.a % {STEP_MOD}')}\n"
            f"}}",
            f"{name}(ARG)", fields=["a"])
        return name

    def gen_holder(self):
        """A class whose field is an instance, so a call goes through a FIELD.

        `geta` forwards to whatever is in the field -- a polymorphic call
        inside a method. The field is declared as a class where one exists
        rather than always `any`: an `any` field has no declared kind for the
        tier to read a receiver off, so it refuses the whole enclosing body,
        and a base class is polymorphic enough (every subclass fits it).
        """
        r = self.rng
        name = self.fresh("Hold")
        tag = r.randint(1, 99)
        holdable = [n for n in self.plain_classes()
                    if self.classes[n]["base"] is None]
        held = r.choice(holdable) if holdable and r.random() < 0.7 else None
        decl = held if held else "any"
        self.add_class(
            name,
            f"class {name} {{\n"
            f"    pub var inner: {decl}\n"
            f"    pub var a: int\n"
            f"    fn init(self, inner: {decl}) {{\n"
            f"        self.inner = inner\n"
            f"        self.a = {r.randint(0, 99)}\n"
            f"    }}\n"
            f"    pub fn kind(self) -> int {{ return {tag} }}\n"
            f"    pub fn geta(self) -> int {{"
            f" return self.a +% self.inner.geta() }}\n"
            f"    pub fn twice(self) -> int {{"
            f" return self.geta() +% self.geta() }}\n"
            f"    pub fn bump(self, k: int) -> int {{\n"
            f"        self.a = self.a +% k\n"
            f"        return self.a +% self.inner.kind()\n"
            f"    }}\n"
            f"{step_method(f'self.a % {STEP_MOD}')}\n"
            f"}}",
            f"{name}(INST)", fields=["a"], inst_field="inner",
            inst_type=held)
        return name

    def gen_node(self):
        """A nullable instance FIELD, walked to a literal bound."""
        r = self.rng
        name = self.fresh("Node")
        tag = r.randint(1, 99)
        cap = r.randint(2, 8)
        self.add_class(
            name,
            f"class {name} {{\n"
            f"    pub var a: int\n"
            f"    pub var next: {name}?\n"
            f"    fn init(self, a: int) {{\n"
            f"        self.a = a\n"
            f"        self.next = null\n"
            f"    }}\n"
            f"    pub fn kind(self) -> int {{ return {tag} }}\n"
            f"    pub fn geta(self) -> int {{ return self.a }}\n"
            f"    pub fn twice(self) -> int {{"
            f" return self.geta() +% self.geta() }}\n"
            f"    pub fn bump(self, k: int) -> int {{\n"
            f"        self.a = self.a +% k\n"
            f"        return self.geta()\n"
            f"    }}\n"
            f"    pub fn walk(self) -> int {{\n"
            f"        var t = 0\n"
            f"        var cur: {name}? = self\n"
            f"        var g = 0\n"
            f"        while g < {cap} {{\n"
            f"            g = g + 1\n"
            f"            if cur == null {{ break }}\n"
            f"            t = t *% 31 +% (cur?.a ?? 0)\n"
            f"            cur = cur?.next\n"
            f"        }}\n"
            f"        return t\n"
            f"    }}\n"
            f"{step_method(f'self.a % {STEP_MOD}')}\n"
            f"}}",
            f"{name}(ARG)", fields=["a"], walk=True,
            int_methods=CLASS_METHODS + ["walk"])
        return name

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

        # An instance crossing a call boundary. The callee's parameter is
        # specialised on whatever class arrived first, and every later call
        # site hands it a different one.
        if self.classes and r.random() < 0.6:
            name = self.fresh("oh")
            guard = r.choice(["k % 2 == 0", "k < 0", "k > 40"])
            self.prog.helpers.append(
                f"fn {name}(o: any, k: int) -> int {{\n"
                f"    if {guard} {{ return o.kind() }}\n"
                f"    return o.geta() +% (o.kind() *% (k % 8))\n"
                f"}}")
            self.helpers[name] = (2, "inst")

        # JIT_MAX_ARITY is 4; a 4-argument function is the widest the
        # whole-function tier will take.
        if r.random() < 0.4:
            name = self.fresh("quad")
            self.prog.helpers.append(
                f"fn {name}(a: int, b: int, c: int, e: int) -> int {{\n"
                f"    return ((a +% b) *% 3) +% (c -% e)\n"
                f"}}")
            self.helpers[name] = (4, "int")

    # -- statements -----------------------------------------------------

    def trips(self, options):
        """A trip count this loop can still afford. Never empty: the smallest
        option is taken when even that is over budget."""
        ok = [t for t in options if t <= self.trip_budget]
        return self.rng.choice(ok) if ok else min(options)

    def enter_loop(self, trips):
        """Charge `trips` to the budget for the body about to be generated."""
        saved = self.trip_budget
        self.loop_depth += 1
        self.trip_budget = max(2, self.trip_budget // max(1, trips))
        return saved

    def leave_loop(self, saved):
        self.loop_depth -= 1
        self.trip_budget = saved

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
        kinds = [
            (self.st_int, 16),
            (self.st_float, 8),
            (self.st_str, 7),
            (self.st_branch, 12),
            (self.st_for_range, 11),
            (self.st_while, 6),
            (self.st_loop, 4),
            (self.st_for_list, 6),
            (self.st_cond_local, 9),
            (self.st_list_ops, 8),
            (self.st_class_use, 10),
            (self.st_match, 5),
            (self.st_lambda, 4),
            (self.st_try, 4),
            (self.st_labelled, 4),
            (self.st_optional, 4),
            # The class family. A run-time value decides the receiver's class
            # in every one of these, which is the shape that keeps breaking.
            (self.st_poly, 9),
            (self.st_base_typed, 8),
            (self.st_inst_null, 8),
            (self.st_inst_swap, 10),
            # Weighted with st_poly and st_inst_swap rather than below them:
            # this is the only statement here that reaches jit_call_pic.c at
            # all, and at weight 6 it appeared in 35 of 100 programs.
            (self.st_inst_list, 9),
            (self.st_inst_field, 4),
            (self.st_inst_mix, 4),
            (self.st_static_call, 4),
            # An overloaded `+` is an OP_ADD the tier declines outright, so
            # this one is weighted like `**` and a dict: worth generating,
            # not worth putting in every probe.
            (self.st_operator, 2),
        ]
        if self.use_dict:
            kinds.append((self.st_dict_ops, 5))
            kinds.append((self.st_for_dict, 4))
        if self.loop_depth > 0:
            kinds.append((self.st_early_return, 7))
            kinds.append((self.st_break_continue, 6))
        if depth >= 2:
            # Stop nesting: only the flat kinds, so a program stays small.
            kinds = [(k, w) for k, w in kinds
                     if k in (self.st_int, self.st_float, self.st_str,
                              self.st_list_ops, self.st_dict_ops,
                              self.st_class_use, self.st_cond_local,
                              self.st_inst_null, self.st_static_call,
                              self.st_inst_field)]
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
        # Guarded so text stays bounded however deeply this nests.
        return Node("if text.len() < 200 {",
                    [line(f"text = text + {self.str_expr(1)}")],
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
        hi = self.trips([3, 5, 8, 16, 32, 64, 200])
        lo = r.choice([0, 0, 0, 1, -3])
        rng = f"{lo}..{hi}" if r.random() < 0.7 else f"{lo}..={hi}"
        self.ints.append(var)
        saved = self.enter_loop(hi - lo)
        body = self.block(r.randint(1, 3), depth + 1)
        self.leave_loop(saved)
        self.ints.remove(var)
        return Node(f"for {var} in {rng} {{", body, "}")

    def st_while(self, depth):
        r = self.rng
        var = self.fresh("w")
        trips = self.trips([3, 6, 12, 40, 100])
        self.ints.append(var)
        saved = self.enter_loop(trips)
        body = self.block(r.randint(1, 3), depth + 1)
        self.leave_loop(saved)
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
        trips = self.trips([2, 5, 10, 30])
        self.ints.append(var)
        saved = self.enter_loop(trips)
        body = self.block(r.randint(1, 2), depth + 1)
        self.leave_loop(saved)
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
        # xs is capped at 64 by st_list_ops, so that is its worst case.
        saved = self.enter_loop(64)
        self.iterating.add("xs")
        body = self.block(r.randint(1, 2), depth + 1)
        self.iterating.discard("xs")
        self.leave_loop(saved)
        self.ints.remove(var)
        if r.random() < 0.3:
            idx = self.fresh("p")
            return Node(f"for ({idx}, {var}) in xs.enumerate() {{", body, "}")
        return Node(f"for {var} in xs {{", body, "}")

    def st_for_dict(self, depth):
        r = self.rng
        key = self.fresh("dk")
        val = self.fresh("dv")
        self.ints.append(val)
        self.strs.append(key)
        saved = self.enter_loop(4)
        self.iterating.add("d")
        body = self.block(r.randint(1, 2), depth + 1)
        self.iterating.discard("d")
        self.leave_loop(saved)
        self.ints.remove(val)
        self.strs.remove(key)
        return Node(f"for ({key}, {val}) in d.items() {{", body, "}")

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
        return Node(f"var {var}: any = null",
                    f"if {cond} {{",
                    [line(f"{var} = {self.ctor_of(name)}")],
                    "}",
                    f"if {var} != null {{",
                    [line(f"acc = acc +% {var}.geta()")],
                    "}")

    def st_early_return(self, depth):
        return Node(f"if {self.bool_expr()} {{",
                    [line('return f"E{acc}|{f}"')],
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
            # Capped exactly as `text` is, and for the same reason. Uncapped,
            # a push nested two loops deep grew xs by 25,200 elements per call
            # -- `insert(0, ..)` is linear and every probe ends in lfold(xs)
            # -- and seed 300 took 66s under the interpreter and 120s+ under
            # the runner's parallel load. The runner reads that timeout as a
            # divergence, which is the one false positive this fuzzer must
            # never produce.
            return Node("if xs.len() < 64 {",
                        [line(f"xs.push({self.int_expr(1)})")], "}")
        if pick == 1:
            # Guarded: xs must never empty, or `k % xs.len()` divides by zero.
            return Node("if xs.len() > 1 {", [line("xs.pop()")], "}")
        if pick == 2:
            return line(f"xs[{r.randint(0, 30)} % xs.len()] = "
                        f"{self.int_expr(1)}")
        if pick == 3:
            return Node("if xs.len() < 64 {",
                        [line(f"xs.insert(0, {self.int_expr(1)})")], "}")
        if pick == 4:
            return line("acc = acc +% xs[-1] +% xs[0]")
        return line("acc = acc +% lfold(xs)")

    def st_dict_ops(self, depth):
        r = self.rng
        key = r.choice(['"k"', '"a"', '"b"', '"zz"'])
        if "d" in self.iterating:
            return line('acc = acc +% d["k"]')
        pick = r.randrange(4)
        if pick == 0:
            return line(f"d[{key}] = {self.int_expr(1)}")
        if pick == 1:
            return line('acc = acc +% d["k"]')
        if pick == 2:
            return Node(f"if {key} in d {{",
                        [line(f"acc = acc +% d[{key}]")], "}")
        return line("acc = acc +% d.len()")

    def st_class_use(self, depth):
        r = self.rng
        names = sorted(self.classes)
        if not names:
            return self.st_int(depth)
        name = r.choice(names)
        api = self.classes[name]
        var = self.fresh("o")
        parts = [f"let {var} = {self.ctor_of(name)}"]
        # The empty-init shape: the constructor must evaluate to the object.
        parts.append(f"if {var} == null {{")
        parts.append([line("acc = acc -% 424242")])
        parts.append("}")
        meth = r.choice(api["int_methods"])
        parts.append(f"acc = acc +% {self.method_atom(var, meth)}")
        if api["fields"] and r.random() < 0.6:
            fld = r.choice(api["fields"])
            parts.append(f"{var}.{fld} = {var}.{fld} +% {self.int_expr(0)}")
            parts.append(f"acc = acc ^ {var}.{fld}")
        if api["float_field"] and r.random() < 0.4:
            parts.append(f"f = f + {var}.b")
        if api["me"] and r.random() < 0.4:
            # A call on a call's RESULT, which is what a returned instance is
            # for. Confined to this statement on purpose -- see method_call.
            parts.append(f"acc = acc +% {var}.me().geta()")
        if api["opt_int"] and r.random() < 0.5:
            # A FIELD that is null on one path and an int on the other.
            fld = api["opt_int"]
            parts.append(f"if {self.bool_expr()} {{")
            parts.append([line(f"{var}.{fld} = {self.int_expr(0)}")])
            parts.append("}")
            parts.append(f"acc = acc +% ({var}.{fld} ?? "
                         f"{r.choice(SMALL_INTS)})")
        # Stays visible to later siblings; block() drops it at the brace.
        self.insts.append((var, name))
        return Node(*parts)

    def st_poly(self, depth):
        """One variable, a different class down each branch.

        The receiver's class is settled by the run-time value and by nothing
        else, so a call site that reconstructed it from a compile-time kind is
        wrong here. This is the inline cache's own path.
        """
        r = self.rng
        names = self.constructible()
        if len(names) < 2:
            return self.st_class_use(depth)
        r.shuffle(names)
        var = self.fresh("v")
        parts = [f"var {var}: any = {self.ctor_of(names[0])}",
                 f"if {self.bool_expr()} {{",
                 [line(f"{var} = {self.ctor_of(names[1])}")]]
        if len(names) > 2 and r.random() < 0.45:
            parts += [f"}} elif {self.bool_expr()} {{",
                      [line(f"{var} = {self.ctor_of(names[2])}")]]
        parts.append("}")
        meth = r.choice(CLASS_METHODS)
        parts.append(f"acc = acc +% {self.method_atom(var, meth)}")
        if r.random() < 0.5:
            parts.append(f"acc = acc ^ {var}.kind()")
        self.insts.append((var, None))
        return Node(*parts)

    def st_base_typed(self, depth):
        """An override reached through a variable typed as the BASE."""
        r = self.rng
        bases = [n for n, a in sorted(self.classes.items()) if a["subs"]]
        if not bases:
            return self.st_poly(depth)
        base = r.choice(bases)
        subs = self.classes[base]["subs"]
        var = self.fresh("bv")
        parts = [f"var {var}: {base} = {self.ctor_of(base)}",
                 f"if {self.bool_expr()} {{",
                 [line(f"{var} = {self.ctor_of(subs[0])}")]]
        if len(subs) > 1:
            parts += [f"}} elif {self.bool_expr()} {{",
                      [line(f"{var} = {self.ctor_of(subs[1])}")]]
        parts.append("}")
        parts.append(f"acc = acc +% {var}.geta() +% {var}.kind()")
        if r.random() < 0.5:
            parts.append(f"acc = acc -% {var}.twice()")
        self.insts.append((var, base))
        return Node(*parts)

    def st_inst_null(self, depth):
        """A nullable instance that only a run-time branch fills in.

        Exactly the kind whose tag the tier reconstructs, and the third style
        goes further: the slot holds an int on one path and an object on the
        other, and an f-string reads whichever arrived.
        """
        r = self.rng
        names = self.constructible()
        if not names:
            return self.st_int(depth)
        cls = r.choice(names)
        var = self.fresh("p")
        style = r.randrange(3)
        if style == 0:
            return Node(f"var {var}: {cls}? = null",
                        f"if {self.bool_expr()} {{",
                        [line(f"{var} = {self.ctor_of(cls)}")],
                        "}",
                        f"if {var} != null {{",
                        [line(f"acc = acc +% {var}.geta()")],
                        "}",
                        f"acc = acc +% ({var}?.kind() ?? "
                        f"{r.choice(SMALL_INTS)})")
        if style == 1:
            return Node(f"var {var}: {cls}? = {self.ctor_of(cls)}",
                        f"if {self.bool_expr()} {{",
                        [line(f"{var} = null")],
                        "}",
                        f"acc = acc +% ({var}?.twice() ?? "
                        f"{r.choice(SMALL_INTS)})",
                        f"if {var} == null {{",
                        [line("acc = acc -% 3")],
                        "}")
        return Node(f"var {var}: any = {self.int_expr(0)}",
                    f"if {self.bool_expr()} {{",
                    [line(f"{var} = {self.ctor_of(cls)}")],
                    "}",
                    "if text.len() < 200 {",
                    [line(f'text = text + f"{{{var}}}"')],
                    "}")

    def st_inst_field(self, depth):
        """A method call that goes through a FIELD rather than a local."""
        r = self.rng
        holders = [n for n, a in sorted(self.classes.items())
                   if a["inst_field"]]
        if not holders:
            return self.st_class_use(depth)
        cls = r.choice(holders)
        fld = self.classes[cls]["inst_field"]
        var = self.fresh("hd")
        parts = [f"let {var} = {self.ctor_of(cls)}",
                 f"acc = acc +% {var}.{fld}.geta()",
                 f"acc = acc ^ {var}.{fld}.kind()"]
        held = self.holdable_by(cls)
        if held and r.random() < 0.6:
            # The field's class changes under the same read site.
            parts.append(f"{var}.{fld} = {self.ctor_of(r.choice(held))}")
            parts.append(f"acc = acc +% {var}.{fld}.twice()")
        parts.append(f"acc = acc +% {var}.geta()")
        self.insts.append((var, cls))
        return Node(*parts)

    def st_inst_list(self, depth):
        """A list of mixed classes walked by one loop.

        One call site, a new receiver class every trip, inside the OSR tier --
        and the ONLY shape in this file that reaches the polymorphic inline
        cache. `emitInvokePic1` wants a receiver that is SLOT_INST with no
        class pinned, and src/vm/jit/README.md enumerates the single producer
        of one: the head of a list loop the OSR tier entered at, over a list
        holding more than one class. Four things here are therefore load-
        bearing rather than taste, and the census in that section is what says
        so -- the earlier form of this statement reached the arm 0 times in 100
        programs:

          - TWO DISTINCT classes at least, sampled without replacement. Drawing
            each element from one pool independently left a real fraction of
            lists monomorphic, and a monomorphic list pins the head by design.
          - SIX elements or more. The head has to collect a SIGPROF tick of its
            own to become an osrTop; a two-trip loop rides its caller's instead
            and takes emitForIterBind's ordinary arm, which pins.
          - `step` and not `kind`/`geta`, because `step` is the one method
            whose body is checked arithmetic. A callee whose walk stops at a
            wrapping operator records no return kind and the arm drops that
            way.
          - the call FIRST in the body, and its own line. The tier walks no
            wrapping operator, so the `acc +% ...` fold below stops the loop's
            walk where it stands -- which is fine after the invoke and fatal
            before it.
        """
        r = self.rng
        names = self.constructible()
        if len(names) < 2:
            return self.st_class_use(depth)
        picked = r.sample(names, 2)
        if len(names) > 2 and r.random() < 0.5:
            picked.append(r.choice([n for n in names if n not in picked]))
        # Drawn against the budget like any other trip count: a nine-element
        # list inside a 200-trip range is 1,800 iterations of a probe CALL.
        nelem = self.trips([6, 7, 8, 9])
        order = [picked[i % len(picked)] for i in range(nelem)]
        r.shuffle(order)
        var = self.fresh("ol")
        elems = ", ".join(self.ctor_of(n) for n in order)
        e = self.fresh("q")
        self.insts.append((e, None))
        saved = self.enter_loop(nelem)
        body = self.block(r.randint(0, 1), depth + 1)
        self.leave_loop(saved)
        self.insts = [p for p in self.insts if p[0] != e]
        self.dispatch_loops += 1
        return Node(f"var {var} = [{elems}]",
                    f"for {e} in {var} {{",
                    [f"acc = {e}.step(acc)",
                     f"acc = acc +% {e}.kind() +% {e}.geta()"] + body,
                    "}")

    def st_inst_swap(self, depth):
        """A receiver whose class changes on every trip of a loop.

        The variable survives the back-edge, so whatever the first trip made
        the site speculate on is wrong by the second.
        """
        r = self.rng
        names = self.constructible()
        if len(names) < 2:
            return self.st_poly(depth)
        r.shuffle(names)
        a, b = names[0], names[1]
        var = self.fresh("sw")
        idx = self.fresh("t")
        # Nested inside a 200-trip range this is 200 x trips
        # constructions per call, so the top of the range is kept
        # modest -- a program nobody can wait for is a program the
        # runner times out and misreads as a divergence.
        trips = self.trips([4, 8, 16])
        self.ints.append(idx)
        saved = self.enter_loop(trips)
        body = self.block(r.randint(0, 1), depth + 1)
        self.leave_loop(saved)
        self.ints.remove(idx)
        # Both lines are strings: the swap and the call it feeds are the whole
        # point of this statement and a reduction may not separate them.
        return Node(f"var {var}: any = {self.ctor_of(a)}",
                    f"for {idx} in 0..{trips} {{",
                    [f"if {idx} % 2 == 0 {{ {var} = {self.ctor_of(b)} }}"
                     f" else {{ {var} = {self.ctor_of(a)} }}",
                     f"acc = acc +% {var}.kind()"] + body,
                    "}",
                    f"acc = acc +% {var}.geta()")

    def st_inst_mix(self, depth):
        """A method whose RETURN kind only the argument decides."""
        r = self.rng
        mixers = [n for n, a in sorted(self.classes.items()) if a["mix"]]
        if not mixers:
            return self.st_class_use(depth)
        cls = r.choice(mixers)
        var = self.fresh("mx")
        parts = [f"let {var} = {self.ctor_of(cls)}",
                 "if text.len() < 200 {",
                 [line(f'text = text + f"{{{var}.mix({self.int_expr(0)})}}"')],
                 "}"]
        if self.classes[cls]["maybe"]:
            parts.append(f"acc = acc +% ({var}.maybe({self.int_expr(0)}) ?? "
                         f"{r.choice(SMALL_INTS)})")
        self.insts.append((var, cls))
        return Node(*parts)

    def st_static_call(self, depth):
        """A static factory: a call on the CLASS, not on an instance."""
        r = self.rng
        cands = [n for n, a in sorted(self.classes.items()) if a["static"]]
        if not cands:
            return self.st_class_use(depth)
        cls = r.choice(cands)
        name = self.classes[cls]["static"]
        return line(f"acc = acc +% {cls}.{name}({self.int_expr(0)}).geta()")

    def st_operator(self, depth):
        """`+`, `==` and an f-string routed through a class's own methods."""
        r = self.rng
        cands = [n for n, a in sorted(self.classes.items()) if a["ops"]]
        if not cands:
            return self.st_class_use(depth)
        cls = r.choice(cands)
        x, y = self.fresh("x"), self.fresh("y")
        parts = [f"let {x} = {self.ctor_of(cls)}",
                 f"let {y} = {self.ctor_of(cls)}",
                 f"acc = acc +% ({x} + {y})",
                 f"if {x} == {y} {{",
                 [line(f"acc = acc -% {r.randint(1, 999)}")],
                 "}"]
        if r.random() < 0.5:
            parts += ["if text.len() < 200 {",
                      [line(f'text = text + f"{{{x}}}"')],
                      "}"]
        self.insts.append((x, cls))
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
        big = r.choice(["9223372036854775807", "-9223372036854775808",
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
        hi_out, hi_in = r.randint(2, 8), r.randint(2, 8)
        saved = self.enter_loop(hi_out * hi_in)
        body = self.block(1, depth + 1)
        self.leave_loop(saved)
        self.ints.remove(outer)
        self.ints.remove(inner)
        kw = r.choice(["break", "continue"])
        return Node(f"{label}: for {outer} in 0..{hi_out} {{",
                    [Node(f"for {inner} in 0..{hi_in} {{",
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

    def st_late_raise(self, depth):
        """An uncaught overflow, thrown only once the tier has compiled.

        The traceback has to match the interpreter's exactly, and it fires late
        enough that everything before it still ran warm.
        """
        r = self.rng
        at = max(1, self.prog.warm - r.randint(2, 20))
        return Node(f"if n == {at} {{",
                    [line("acc = acc +% (9223372036854775807 + 1)")],
                    "}")

    def build(self):
        r = self.rng
        self.gen_classes()
        self.gen_helpers()
        for k in range(r.randint(2, 5)):
            probe = Probe(f"probe{k}")
            # Per PROBE, not per program: a blocker confined to one function
            # leaves the others compiling.
            self.use_dict = r.random() < 0.25
            self.use_pow = r.random() < 0.20
            probe.use_dict = self.use_dict
            self.ints = ["n", "acc", "a0", "a1"]
            self.assignable = ["acc", "a0", "a1"]
            self.floats = ["f"]
            self.strs = ["text"]
            self.insts = []
            self.iterating = set()
            self.containers = True
            self.loop_depth = 0
            self.trip_budget = TRIP_BUDGET
            probe.body = [self.stmt(0) for _ in range(r.randint(1, 4))]
            self.prog.probes.append(probe)
        # The polymorphic inline cache is the most intricate arm in
        # src/vm/jit and the weighted grammar alone left three programs in
        # five with no site that reaches it at all -- a program is 2-5 probes
        # of 1-4 statements, so a weight-9 statement is simply absent more
        # often than not. One is appended when none was drawn, outside the
        # grammar for the same reason st_late_raise is: some shapes are worth
        # having in EVERY program rather than in a third of them. The scope
        # state still belongs to the last probe, which is the one that gets it.
        if self.dispatch_loops == 0:
            self.prog.probes[-1].body.append(self.st_inst_list(0))
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
