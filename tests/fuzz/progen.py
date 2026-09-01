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
        self.loop_depth = 0
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
            lambda: str(r.choice(SMALL_INTS)),
            lambda: str(r.choice(EDGE_INTS)),
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
                              self.st_class_use, self.st_cond_local)]
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
        hi = r.choice([3, 5, 8, 16, 32, 64, 200])
        lo = r.choice([0, 0, 0, 1, -3])
        rng = f"{lo}..{hi}" if r.random() < 0.7 else f"{lo}..={hi}"
        self.ints.append(var)
        self.loop_depth += 1
        body = self.block(r.randint(1, 3), depth + 1)
        self.loop_depth -= 1
        self.ints.remove(var)
        return Node(f"for {var} in {rng} {{", body, "}")

    def st_while(self, depth):
        r = self.rng
        var = self.fresh("w")
        trips = r.choice([3, 6, 12, 40, 100])
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
        trips = r.choice([2, 5, 10, 30])
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
        self.iterating.add("xs")
        body = self.block(r.randint(1, 2), depth + 1)
        self.iterating.discard("xs")
        self.loop_depth -= 1
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
        self.loop_depth += 1
        self.iterating.add("d")
        body = self.block(r.randint(1, 2), depth + 1)
        self.iterating.discard("d")
        self.loop_depth -= 1
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
        if api["float_field"] and r.random() < 0.4:
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
        self.loop_depth += 1
        body = self.block(1, depth + 1)
        self.loop_depth -= 1
        self.ints.remove(outer)
        self.ints.remove(inner)
        kw = r.choice(["break", "continue"])
        return Node(f"{label}: for {outer} in 0..{r.randint(2, 8)} {{",
                    [Node(f"for {inner} in 0..{r.randint(2, 8)} {{",
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
            probe.body = [self.stmt(0) for _ in range(r.randint(1, 4))]
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
