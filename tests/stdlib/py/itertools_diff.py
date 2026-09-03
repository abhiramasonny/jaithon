#!/usr/bin/env python3
"""Differential test: std.itertools against CPython 3.13's itertools.

Generates a corpus of itertools expressions, renders each one twice -- once as
Python and once as Jaithon -- runs the Python through `uv run --python 3.13
python` and the Jaithon through `./jaithon run`, and diffs the two transcripts
byte for byte. Any difference, including a difference in which exception is
raised, is a failure.

Both drivers print one line per case, `<index>\t<value>`, where the value is
rendered by a formatter written twice to the same specification (Python's own
repr for ints, floats, strings, lists and tuples). A raising case prints
`<index>\t!<ExceptionName>` instead, so error parity is diffed too rather than
being silently skipped.

The corpus is deliberately not a sample. Iteration order and the edge cases the
port is most likely to get wrong -- an empty input, `r > n`, `r == 0`, a
`repeat` of zero, a `groupby` group read after the walk moved past it, a `tee`
branch already exhausted, `islice` with every combination of start/stop/step --
are enumerated exhaustively over small inputs; the seeded random inputs on top
of that are there to catch what the enumeration did not think of.

    python3 tests/stdlib/py/itertools_diff.py [--keep]

Exits 0 when the two transcripts are identical, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
JAITHON = os.path.join(ROOT, "jaithon")
CPYTHON = ["uv", "run", "--python", "3.13", "python"]

SEED = 20260902
MINIMUM_CASES = 300

# ------------------------------------------------------------------ rendering


def jai(value):
    """Render a Python value as the Jaithon literal that denotes it."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return '"%s"' % value
    if isinstance(value, tuple):
        return "(%s)" % ", ".join(jai(item) for item in value)
    if isinstance(value, list):
        return "[%s]" % ", ".join(jai(item) for item in value)
    raise TypeError("no Jaithon literal for %r" % (value,))


def py(value):
    """Render a Python value as the Python literal that denotes it."""
    return repr(value)


class Fn:
    """One function literal, spelled for both languages."""

    def __init__(self, python: str, jaithon: str):
        self.python = python
        self.jaithon = jaithon


ADD = Fn("None", "null")
MUL = Fn("(lambda a, b: a * b)", "|a, b| a * b")
SUB = Fn("(lambda a, b: a - b)", "|a, b| a - b")
BIGGER = Fn("max", "|a, b| a > b ? a : b")
SMALLER = Fn("min", "|a, b| a < b ? a : b")
BINARY = [ADD, MUL, SUB, BIGGER, SMALLER]

UNDER_FIVE = Fn("(lambda n: n < 5)", "|n| n < 5")
EVEN = Fn("(lambda n: n % 2 == 0)", "|n| n % 2 == 0")
POSITIVE = Fn("(lambda n: n > 0)", "|n| n > 0")
ALWAYS = Fn("(lambda n: True)", "|n| true")
NEVER = Fn("(lambda n: False)", "|n| false")
TRUTHY = Fn("(lambda n: n)", "|n| n")
PREDICATES = [UNDER_FIVE, EVEN, POSITIVE, ALWAYS, NEVER, TRUTHY]

IDENTITY_KEY = Fn("None", "null")
MOD_THREE = Fn("(lambda n: n % 3)", "|n| n % 3")
HALVED = Fn("(lambda n: n // 2)", "|n| int(n / 2)")
KEYS = [IDENTITY_KEY, MOD_THREE, HALVED]

POWER = Fn("(lambda a, b: a ** b)", "|a, b| a ** b")
JOINED = Fn("(lambda a, b: a + b)", "|a, b| a + b")

# ------------------------------------------------------------------- corpus

CASES: list[tuple[str, str]] = []


def case(python: str, jaithon: str) -> None:
    """Record one case: a Python block and a Jaithon block, each yielding `out`."""
    CASES.append((python, jaithon))


def expr(python: str, jaithon: str) -> None:
    """Record a case whose whole body is one expression."""
    case("out = %s" % python, "out = %s" % jaithon)


def pools(rng: random.Random) -> list:
    """A spread of small inputs: empty, singleton, repeated, random."""
    return [
        [],
        [0],
        [1, 2],
        [1, 1],
        [1, 1, 2, 2, 1],
        [3, 1, 2],
        list(range(5)),
        "",
        "a",
        "ab",
        "abc",
        "aab",
        [rng.randrange(0, 9) for _ in range(rng.randrange(0, 7))],
        [rng.randrange(-4, 5) for _ in range(rng.randrange(0, 7))],
    ]


def build(rng: random.Random) -> None:
    small = pools(rng)

    # --- count -------------------------------------------------------------
    for start in [0, 1, -3, 2.5]:
        for step in [1, 2, -1, 0, 0.5]:
            expr(
                "list(I.islice(I.count(%s, %s), 5))" % (py(start), py(step)),
                "it.islice(it.count(%s, %s), 5).collect()" % (jai(start), jai(step)),
            )
    expr("list(I.islice(I.count(), 4))", "it.islice(it.count(), 4).collect()")
    expr("list(I.islice(I.count(7), 4))", "it.islice(it.count(7), 4).collect()")
    expr("I.count('a')", "it.count(\"a\")")

    # --- cycle -------------------------------------------------------------
    for source in small:
        for take in [0, 1, 5]:
            expr(
                "list(I.islice(I.cycle(%s), %d))" % (py(source), take),
                "it.islice(it.cycle(%s), %d).collect()" % (jai(source), take),
            )

    # --- repeat ------------------------------------------------------------
    for value in [0, 7, "x", [1, 2]]:
        for times in [-2, -1, 0, 1, 3]:
            expr(
                "list(I.repeat(%s, %d))" % (py(value), times),
                "it.repeat(%s, %d).collect()" % (jai(value), times),
            )
    expr("list(I.islice(I.repeat(4), 3))", "it.islice(it.repeat(4), 3).collect()")

    # --- accumulate --------------------------------------------------------
    numeric = [source for source in small if not isinstance(source, str)]
    for source in numeric:
        for func in BINARY:
            expr(
                "list(I.accumulate(%s, %s))" % (py(source), func.python),
                "it.accumulate(%s, %s).collect()" % (jai(source), func.jaithon),
            )
    for source in numeric[:6]:
        for initial in [0, 100, -5]:
            expr(
                "list(I.accumulate(%s, None, initial=%d))" % (py(source), initial),
                "it.accumulate(%s, null, %d).collect()" % (jai(source), initial),
            )

    # --- chain -------------------------------------------------------------
    for left in small[:8]:
        for right in small[:8]:
            expr(
                "list(I.chain(%s, %s))" % (py(left), py(right)),
                "it.chain(%s, %s).collect()" % (jai(left), jai(right)),
            )
    expr("list(I.chain())", "it.chain().collect()")
    expr("list(I.chain([1], [2], [3], []))", "it.chain([1], [2], [3], []).collect()")
    for outer in [[], [[]], [[1], []], [[1, 2], [3]], ["ab", "cd"], [[1], [2], [3]]]:
        expr(
            "list(I.chain.from_iterable(%s))" % py(outer),
            "it.chain.from_iterable(%s).collect()" % jai(outer),
        )
    expr(
        "list(I.islice(I.chain.from_iterable(I.repeat([1, 2])), 5))",
        "it.islice(it.chain.from_iterable(it.repeat([1, 2])), 5).collect()",
    )

    # --- compress ----------------------------------------------------------
    selectors = [[], [1], [0], [1, 0, 1], [0, 0, 0], [1, 1, 1, 1, 1], ["", "x", []]]
    for data in small[:8]:
        for picks in selectors:
            expr(
                "list(I.compress(%s, %s))" % (py(data), py(picks)),
                "it.compress(%s, %s).collect()" % (jai(data), jai(picks)),
            )

    # --- dropwhile / takewhile / filterfalse -------------------------------
    for source in numeric:
        for pred in PREDICATES:
            expr(
                "list(I.dropwhile(%s, %s))" % (pred.python, py(source)),
                "it.dropwhile(%s, %s).collect()" % (pred.jaithon, jai(source)),
            )
            expr(
                "list(I.takewhile(%s, %s))" % (pred.python, py(source)),
                "it.takewhile(%s, %s).collect()" % (pred.jaithon, jai(source)),
            )
    for source in numeric:
        for pred in [None, EVEN, POSITIVE, TRUTHY]:
            spelling = ("None", "null") if pred is None else (pred.python, pred.jaithon)
            expr(
                "list(I.filterfalse(%s, %s))" % (spelling[0], py(source)),
                "it.filterfalse(%s, %s).collect()" % (spelling[1], jai(source)),
            )

    # --- groupby -----------------------------------------------------------
    runs = [
        [],
        [1],
        [1, 1],
        [1, 2, 1],
        [1, 1, 2, 2, 1],
        [0, 0, 0, 1, 1, 2],
        [4, 4, 5, 6, 6, 6, 0],
        "AAAABBBCCDAABBB",
        "aabbccdd",
    ]
    for source in runs:
        for key in KEYS:
            if isinstance(source, str) and key is not IDENTITY_KEY:
                continue
            case(
                "out = [(k, list(g)) for k, g in I.groupby(%s, %s)]"
                % (py(source), key.python),
                "let rows: list[any] = []\n"
                "    for pair in it.groupby(%s, %s) { rows.push((pair[0], pair[1].collect())) }\n"
                "    out = rows" % (jai(source), key.jaithon),
            )
            case(
                "out = [k for k, g in I.groupby(%s, %s)]" % (py(source), key.python),
                "let keys: list[any] = []\n"
                "    for pair in it.groupby(%s, %s) { keys.push(pair[0]) }\n"
                "    out = keys" % (jai(source), key.jaithon),
            )
            case(
                "held = [g for k, g in I.groupby(%s, %s)]\n"
                "out = [list(g) for g in held]" % (py(source), key.python),
                "let held: list[any] = []\n"
                "    for pair in it.groupby(%s, %s) { held.push(pair[1]) }\n"
                "    let rows: list[any] = []\n"
                "    for group in held { rows.push(group.collect()) }\n"
                "    out = rows" % (jai(source), key.jaithon),
            )
            case(
                "out = []\n"
                "for i, (k, g) in enumerate(I.groupby(%s, %s)):\n"
                "    out.append((k, list(g)) if i %% 2 == 0 else (k, []))"
                % (py(source), key.python),
                "let rows: list[any] = []\n"
                "    var index = 0\n"
                "    for pair in it.groupby(%s, %s) {\n"
                "        if index %% 2 == 0 { rows.push((pair[0], pair[1].collect())) } "
                "else { rows.push((pair[0], [])) }\n"
                "        index += 1\n"
                "    }\n"
                "    out = rows" % (jai(source), key.jaithon),
            )

    # --- islice ------------------------------------------------------------
    letters = "abcdefg"
    for stop in [None, 0, 1, 3, 7, 20]:
        expr(
            "list(I.islice(%s, %s))" % (py(letters), py(stop)),
            "it.islice(%s, %s).collect()" % (jai(letters), jai(stop)),
        )
    for start in [0, 1, 2, 6, 9]:
        for stop in [None, 0, 2, 5, 7, 20]:
            for step in [1, 2, 3]:
                expr(
                    "list(I.islice(%s, %s, %s, %d))"
                    % (py(letters), py(start), py(stop), step),
                    "it.islice(%s, %s, %s, %d).collect()"
                    % (jai(letters), jai(start), jai(stop), step),
                )
    expr("list(I.islice([], 0, 5, 2))", "it.islice([], 0, 5, 2).collect()")
    expr("list(I.islice(I.count(), 3, 9, 2))", "it.islice(it.count(), 3, 9, 2).collect()")
    expr("I.islice('abc', -1)", "it.islice(\"abc\", -1)")
    expr("I.islice('abc', 0, -1)", "it.islice(\"abc\", 0, -1)")
    expr("I.islice('abc', 0, 2, 0)", "it.islice(\"abc\", 0, 2, 0)")
    expr("I.islice('abc', 0, 2, -1)", "it.islice(\"abc\", 0, 2, -1)")
    expr("I.islice('abc')", "it.islice(\"abc\")")
    expr("I.islice('abc', 0, 1, 1, 1)", "it.islice(\"abc\", 0, 1, 1, 1)")
    case(
        "src = I.count()\nout = [list(I.islice(src, 2)), list(I.islice(src, 2))]",
        "let source = it.count()\n"
        "    let first = it.islice(source, 2).collect()\n"
        "    let second = it.islice(source, 2).collect()\n"
        "    out = [first, second]",
    )

    # --- pairwise ----------------------------------------------------------
    for source in small:
        expr(
            "list(I.pairwise(%s))" % py(source),
            "it.pairwise(%s).collect()" % jai(source),
        )

    # --- batched -----------------------------------------------------------
    for source in small[:9]:
        for size in [1, 2, 3, 5]:
            expr(
                "list(I.batched(%s, %d))" % (py(source), size),
                "it.batched(%s, %d).collect()" % (jai(source), size),
            )
            expr(
                "list(I.batched(%s, %d, strict=True))" % (py(source), size),
                "it.batched(%s, %d, true).collect()" % (jai(source), size),
            )
    expr("I.batched('abc', 0)", "it.batched(\"abc\", 0)")
    expr("I.batched('abc', -1)", "it.batched(\"abc\", -1)")

    # --- starmap -----------------------------------------------------------
    argument_lists = [
        [],
        [(2, 5)],
        [(2, 5), (3, 2), (10, 3)],
        [[1, 2], [3, 4]],
        [(0, 0), (1, 1)],
    ]
    for arguments in argument_lists:
        for func in [POWER, JOINED, MUL, SUB]:
            expr(
                "list(I.starmap(%s, %s))" % (func.python, py(arguments)),
                "it.starmap(%s, %s).collect()" % (func.jaithon, jai(arguments)),
            )

    # --- tee ---------------------------------------------------------------
    for source in small[:9]:
        for branches in [0, 1, 2, 3]:
            case(
                "out = [list(t) for t in I.tee(%s, %d)]" % (py(source), branches),
                "let branches = it.tee(%s, %d)\n"
                "    let rows: list[any] = []\n"
                "    for i in 0..branches.len() { rows.push(branches[i].collect()) }\n"
                "    out = rows" % (jai(source), branches),
            )
    for source in [[1, 2, 3, 4], "abcd", [], [9]]:
        for ahead in [0, 1, 2, 5]:
            case(
                "a, b = I.tee(%s)\n"
                "out = [list(I.islice(a, %d)), list(b), list(a), list(b)]"
                % (py(source), ahead),
                "let pair = it.tee(%s)\n"
                "    let left = pair[0]\n"
                "    let right = pair[1]\n"
                "    let head = it.islice(left, %d).collect()\n"
                "    let whole = right.collect()\n"
                "    let rest = left.collect()\n"
                "    let again = right.collect()\n"
                "    out = [head, whole, rest, again]" % (jai(source), ahead),
            )
    expr("I.tee([1], -1)", "it.tee([1], -1)")

    # --- zip_longest -------------------------------------------------------
    for left in small[:8]:
        for right in small[:8]:
            expr(
                "list(I.zip_longest(%s, %s))" % (py(left), py(right)),
                "it.zip_longest(%s, %s).collect()" % (jai(left), jai(right)),
            )
    for fill in [0, "-", -1]:
        for left in small[:6]:
            for right in small[:6]:
                expr(
                    "list(I.zip_longest(%s, %s, fillvalue=%s))"
                    % (py(left), py(right), py(fill)),
                    "it.zip_longest.with_fill(%s, %s, %s).collect()"
                    % (jai(fill), jai(left), jai(right)),
                )
    expr("list(I.zip_longest())", "it.zip_longest().collect()")
    expr("list(I.zip_longest([1, 2]))", "it.zip_longest([1, 2]).collect()")
    expr(
        "list(I.zip_longest([1, 2], 'ab', [7, 8, 9]))",
        "it.zip_longest([1, 2], \"ab\", [7, 8, 9]).collect()",
    )

    # --- product -----------------------------------------------------------
    for left in small[:8]:
        for right in small[:8]:
            expr(
                "list(I.product(%s, %s))" % (py(left), py(right)),
                "it.product(%s, %s).collect()" % (jai(left), jai(right)),
            )
    expr("list(I.product())", "it.product().collect()")
    expr("list(I.product([1, 2]))", "it.product([1, 2]).collect()")
    expr(
        "list(I.product([1, 2], 'ab', [7]))",
        "it.product([1, 2], \"ab\", [7]).collect()",
    )
    for source in ["", "a", "ab", "abc"]:
        for times in [0, 1, 2, 3]:
            expr(
                "list(I.product(%s, repeat=%d))" % (py(source), times),
                "it.product.repeated(%d, %s).collect()" % (times, jai(source)),
            )
    expr("list(I.product(repeat=0))", "it.product.repeated(0).collect()")
    expr("list(I.product(repeat=2))", "it.product.repeated(2).collect()")
    expr("I.product('ab', repeat=-1)", "it.product.repeated(-1, \"ab\")")

    # --- permutations / combinations / with replacement --------------------
    combinatoric = ["", "a", "ab", "abc", "abcd", [1, 1], [1, 1, 2], [0, 1, 2, 3]]
    for source in combinatoric:
        expr(
            "list(I.permutations(%s))" % py(source),
            "it.permutations(%s).collect()" % jai(source),
        )
        for r in [0, 1, 2, 3, 5]:
            expr(
                "list(I.permutations(%s, %d))" % (py(source), r),
                "it.permutations(%s, %d).collect()" % (jai(source), r),
            )
            expr(
                "list(I.combinations(%s, %d))" % (py(source), r),
                "it.combinations(%s, %d).collect()" % (jai(source), r),
            )
            expr(
                "list(I.combinations_with_replacement(%s, %d))" % (py(source), r),
                "it.combinations_with_replacement(%s, %d).collect()"
                % (jai(source), r),
            )

    # --- composed pipelines, on seeded inputs ------------------------------
    for _ in range(40):
        data = [rng.randrange(0, 9) for _ in range(rng.randrange(0, 9))]
        other = [rng.randrange(0, 9) for _ in range(rng.randrange(0, 9))]
        take = rng.randrange(0, 8)
        size = rng.randrange(1, 4)
        pred = rng.choice(PREDICATES)
        func = rng.choice(BINARY)
        shape = rng.randrange(0, 6)
        if shape == 0:
            expr(
                "list(I.islice(I.accumulate(I.cycle(%s), %s), %d))"
                % (py(data), func.python, take),
                "it.islice(it.accumulate(it.cycle(%s), %s), %d).collect()"
                % (jai(data), func.jaithon, take),
            )
        elif shape == 1:
            expr(
                "list(I.batched(I.chain(%s, %s), %d))" % (py(data), py(other), size),
                "it.batched(it.chain(%s, %s), %d).collect()"
                % (jai(data), jai(other), size),
            )
        elif shape == 2:
            expr(
                "list(I.compress(%s, I.cycle([1, 0, 1])))" % py(data),
                "it.compress(%s, it.cycle([1, 0, 1])).collect()" % jai(data),
            )
        elif shape == 3:
            expr(
                "list(I.pairwise(I.takewhile(%s, %s)))" % (pred.python, py(data)),
                "it.pairwise(it.takewhile(%s, %s)).collect()"
                % (pred.jaithon, jai(data)),
            )
        elif shape == 4:
            expr(
                "list(I.zip_longest(%s, %s, fillvalue=-1))" % (py(data), py(other)),
                "it.zip_longest.with_fill(-1, %s, %s).collect()"
                % (jai(data), jai(other)),
            )
        else:
            expr(
                "list(I.islice(I.starmap(%s, I.product(%s, %s)), %d))"
                % (func.python if func is not ADD else "(lambda a, b: a + b)",
                   py(data), py(other), take),
                "it.islice(it.starmap(%s, it.product(%s, %s)), %d).collect()"
                % (func.jaithon if func is not ADD else "|a, b| a + b",
                   jai(data), jai(other), take),
            )


# ------------------------------------------------------------------- drivers

PY_DRIVER_HEAD = '''\
import itertools as I


def fmt(value):
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return "'" + value + "'"
    if isinstance(value, tuple):
        if len(value) == 1:
            return "(" + fmt(value[0]) + ",)"
        return "(" + ", ".join(fmt(item) for item in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(item) for item in value) + "]"
    raise TypeError("cannot format " + type(value).__name__)


CASES = []
'''

JAI_DRIVER_HEAD = '''\
import std.itertools as it

fn fmt(value: any) -> str {
    if value is null { return "None" }
    let kind = type_of(value)
    if kind == "bool" { return value ? "True" : "False" }
    if kind == "int" or kind == "float" { return str(value) }
    if kind == "str" { return "'" + value + "'" }
    let parts: list[str] = []
    for element in value { parts.push(fmt(element)) }
    if kind == "tuple" {
        if parts.len() == 1 { return "(" + parts[0] + ",)" }
        return "(" + ", ".join(parts) + ")"
    }
    if kind == "list" { return "[" + ", ".join(parts) + "]" }
    throw ValueError(f"cannot format {kind}")
}

fn run(index: int, body: any) -> void {
    try {
        print(f"{index}\\t{fmt(body())}")
    } catch error: Error {
        print(f"{index}\\t!{type_of(error)}")
    }
}

let cases: list[any] = [
'''


def python_driver() -> str:
    parts = [PY_DRIVER_HEAD]
    for index, (block, _) in enumerate(CASES):
        body = "\n".join("    " + line for line in block.splitlines())
        parts.append("\n\ndef case_%d():\n%s\n    return out\n\n\nCASES.append(case_%d)\n"
                     % (index, body, index))
    parts.append('''

for index, body in enumerate(CASES):
    try:
        print("%d\\t%s" % (index, fmt(body())))
    except Exception as error:
        print("%d\\t!%s" % (index, type(error).__name__))
''')
    return "".join(parts)


def jaithon_driver() -> str:
    parts = [JAI_DRIVER_HEAD]
    for index, (_, block) in enumerate(CASES):
        parts.append("    fn() -> any {\n        var out: any = null\n        %s\n        return out\n    },\n"
                     % block)
    parts.append("]\n\nfor index in 0..cases.len() { run(index, cases[index]) }\n")
    return "".join(parts)


# -------------------------------------------------------------------- driver


def transcript(command: list[str], label: str) -> list[str]:
    finished = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if finished.returncode != 0:
        sys.stderr.write("%s exited %d\n%s\n%s\n"
                         % (label, finished.returncode, finished.stdout[-4000:],
                            finished.stderr[-4000:]))
        sys.exit(1)
    return finished.stdout.splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true",
                        help="leave the generated drivers on disk and name them")
    options = parser.parse_args()

    build(random.Random(SEED))
    if len(CASES) < MINIMUM_CASES:
        sys.stderr.write("only %d cases generated, wanted at least %d\n"
                         % (len(CASES), MINIMUM_CASES))
        return 1

    if not os.access(JAITHON, os.X_OK):
        sys.stderr.write("%s is not built; run make first\n" % JAITHON)
        return 1

    workspace = tempfile.mkdtemp(prefix="itertools_diff.")
    python_path = os.path.join(workspace, "driver.py")
    jaithon_path = os.path.join(workspace, "driver.jai")
    with open(python_path, "w") as handle:
        handle.write(python_driver())
    with open(jaithon_path, "w") as handle:
        handle.write(jaithon_driver())

    version = subprocess.run(
        CPYTHON + ["-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        cwd=ROOT, capture_output=True, text=True).stdout.strip()
    expected = transcript(CPYTHON + [python_path], "CPython")
    actual = transcript([JAITHON, "run", jaithon_path], "Jaithon")

    failures = 0
    for index in range(max(len(expected), len(actual))):
        want = expected[index] if index < len(expected) else "<missing>"
        got = actual[index] if index < len(actual) else "<missing>"
        if want != got:
            failures += 1
            if failures <= 20:
                source = CASES[index] if index < len(CASES) else ("?", "?")
                print("case %d differs" % index)
                print("  python : %s" % source[0].replace("\n", "\n           "))
                print("  jaithon: %s" % source[1].replace("\n", "\n           "))
                print("  cpython -> %s" % want)
                print("  jaithon -> %s" % got)

    if options.keep or failures:
        print("drivers: %s" % workspace)
    else:
        shutil.rmtree(workspace, ignore_errors=True)

    if failures:
        print("FAIL: %d of %d cases differ" % (failures, len(CASES)))
        return 1
    print("ok: %d cases identical to CPython %s" % (len(CASES), version))
    return 0


if __name__ == "__main__":
    sys.exit(main())
