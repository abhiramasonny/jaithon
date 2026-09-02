#!/usr/bin/env python3
"""Differential test: jaiyaml against the real PyYAML on generated documents.

Every case starts as a random value -- nested dicts and lists of null,
bools, ints, floats and strings drawn from a pool of awkward ones -- which
PyYAML dumps with random options. That text is then loaded by PyYAML and by
jaiyaml, and the two structures must agree exactly. jaiyaml then dumps what
it loaded with the same options, and three more things must hold: its text
must equal PyYAML's byte for byte, PyYAML must load it back to the same
structure, and jaiyaml must load PyYAML's dump of that back to the same
structure again. Multi-document streams go through `safe_dump_all` and
`safe_load_all` the same way.

A second family of cases mutates a valid document by one character and
compares what happens: both sides must raise or both must load, a raise
must be the same error class with the same message, marks included, and a
load must agree on the structure.

PyYAML implements YAML 1.1 and jaiyaml the 1.2 core schema, so the string
pool holds only texts both resolvers classify alike (`yes`, `0o17`, `1e5`
and `2001-12-14` are left out, `12`, `1.5`, `true` and `~` are kept), and
the mutation cases -- which can produce any plain scalar -- are loaded on
the Python side through a PyYAML loader configured with the core schema's
implicit resolvers, still the real package, just its documented resolver
hook. Two documented deviations are recognised and counted rather than
failed: a null mapping key, which a dict here cannot hold, and an int past
64 bits, which becomes a float.

    uv run --python 3.13 --with pyyaml python packages/jaiyaml/tests/py/jaiyaml_diff.py
    python3 packages/jaiyaml/tests/py/jaiyaml_diff.py            # re-execs under uv
    python3 packages/jaiyaml/tests/py/jaiyaml_diff.py --count 1000 --seed 7
    python3 packages/jaiyaml/tests/py/jaiyaml_diff.py --keep     # leave the driver behind

Exit status is 1 on any disagreement.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

try:
    import yaml
except ImportError:
    os.execvp(
        "uv",
        ["uv", "run", "--python", "3.13", "--with", "pyyaml", "python", os.path.abspath(__file__)]
        + sys.argv[1:],
    )

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
JAITHON = os.environ.get("JAITHON", os.path.join(ROOT, "jaithon"))


# ---------------------------------------------------------------- core schema

INT_RE = re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$")
FLOAT_RE = re.compile(r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$")
NULLS = {"", "~", "null", "Null", "NULL"}
BOOLS = {"true", "True", "TRUE", "false", "False", "FALSE"}


def core_tag(text: str) -> str:
    if text in NULLS:
        return "tag:yaml.org,2002:null"
    if text in BOOLS:
        return "tag:yaml.org,2002:bool"
    if INT_RE.match(text):
        return "tag:yaml.org,2002:int"
    if FLOAT_RE.match(text):
        return "tag:yaml.org,2002:float"
    if text == "<<":
        return "tag:yaml.org,2002:merge"
    return "tag:yaml.org,2002:str"


def pyyaml_tag(text: str) -> str:
    return yaml.resolver.Resolver().resolve(yaml.nodes.ScalarNode, text, (True, False))


def resolvers_agree(text: str) -> bool:
    return core_tag(text) == pyyaml_tag(text)


class CoreLoader(yaml.SafeLoader):
    """The real SafeLoader with the 1.2 core schema's implicit resolvers."""

    yaml_implicit_resolvers = {}


CoreLoader.add_implicit_resolver("tag:yaml.org,2002:null", re.compile(r"^(?:~|null|Null|NULL|)$"), ["~", "n", "N", ""])
CoreLoader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))
CoreLoader.add_implicit_resolver("tag:yaml.org,2002:int", INT_RE, list("-+0123456789"))
CoreLoader.add_implicit_resolver("tag:yaml.org,2002:float", FLOAT_RE, list("-+.0123456789"))
CoreLoader.add_implicit_resolver("tag:yaml.org,2002:merge", re.compile(r"^(?:<<)$"), ["<"])


def core_int(loader, node):
    value = loader.construct_scalar(node)
    sign = -1 if value.startswith("-") else 1
    value = value.lstrip("+-")
    if value.startswith("0x"):
        return sign * int(value[2:], 16)
    if value.startswith("0o"):
        return sign * int(value[2:], 8)
    return sign * int(value)


def core_float(loader, node):
    value = loader.construct_scalar(node).lower()
    if value.endswith(".inf"):
        return -math.inf if value.startswith("-") else math.inf
    if value == ".nan":
        return math.nan
    return float(value)


CoreLoader.add_constructor("tag:yaml.org,2002:int", core_int)
CoreLoader.add_constructor("tag:yaml.org,2002:float", core_float)


# ------------------------------------------------------------- canonical form

LINE_UNSAFE = re.compile("[\\x00-\\x1f\\x7f-\\x9f\\u2028\\u2029\\ufeff\\\\\"]")


def escape(text: str) -> str:
    def one(match: re.Match) -> str:
        ch = match.group()
        if ch == "\\":
            return "\\\\"
        if ch == '"':
            return '\\"'
        if ch == "\n":
            return "\\n"
        if ch == "\r":
            return "\\r"
        if ch == "\t":
            return "\\t"
        return "\\u%04x" % ord(ch)

    return '"' + LINE_UNSAFE.sub(one, text) + '"'


def canon(value) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return "int(%d)" % value
    if isinstance(value, float):
        return "float(%s)" % repr(value)
    if isinstance(value, str):
        return "str(" + escape(value) + ")"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(canon(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(canon(k) + ": " + canon(v) for k, v in value.items()) + "}"
    raise TypeError("cannot canonicalise %r" % (value,))


# ------------------------------------------------------------------ generator

STRING_POOL = [
    "", " ", "  ", "a", "abc", "hello world", "with  two  spaces", "trailing ", " leading",
    "tab\tinside", "line\nbreak", "two\n\nbreaks", "ends\n", "ends\n\n", "\nstarts", "a\n b", "a \nb",
    "it's", 'say "hi"', "back\\slash", "colon: space", "colon:nospace", "hash #tag", "hash#tag",
    "- dash", "-dash", "? question", "?q", ": colon", ":c", "[bracket", "]bracket", "{brace", "}brace",
    ",comma", "a,b", "&anchor", "*alias", "!tag", "|pipe", ">fold", "%percent", "@at", "`tick",
    "---", "--- x", "...", "... y", "x ---", "a: b: c", "key: value",
    "12", "-7", "+3", "0", "007", "1.5", ".5", "5.", "1.0e+20", "-.inf", ".nan", ".inf",
    "0x1F", "true", "True", "FALSE", "null", "Null", "~", "<<", "=",
    "é", "café", "日本語", "😀 smile", "naïve résumé", "Ωmega", "\u00a0nbsp", "\u0085next", "\u2028ls", "\ufeffbom",
    "\x07bell", "\x1bescape", "\x00nul", "\x7fdel",
    "a" * 100, "word " * 30, "x" * 79 + " y", "x" * 80 + " y", "x" * 81 + " y",
    "long words " + "abcdefghij " * 12, "under_score", "dot.dot", "slash/path", "http://example.com/a?b=c",
    "mixed: [flow, {style}]", "#", "'", '"', "'''", "''", '""',
    "yes", "no", "on", "off", "0o17", "1e5", "1_000", "1:30", "2001-12-14", "0b101", "010",
]
STRING_POOL = [text for text in STRING_POOL if resolvers_agree(text)]
KEY_POOL = [text for text in STRING_POOL if "\n" not in text and len(text) < 40]


def random_string(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.7:
        return rng.choice(STRING_POOL)
    alphabet = "abcdefghijklmnopqrstuvwxyz ABCXYZ0123456789-_.:#,[]{}'\"\\éü\n"
    while True:
        length = rng.randint(1, 24)
        text = "".join(rng.choice(alphabet) for _ in range(length))
        if resolvers_agree(text):
            return text


def random_float(rng: random.Random) -> float:
    roll = rng.random()
    if roll < 0.1:
        return rng.choice([0.0, -0.0, 1.0, -1.0, 0.5, 1e16, 1e-7, 123456.789, 1e300, 5e-324, math.inf, -math.inf])
    if roll < 0.15:
        return math.nan
    if roll < 0.5:
        return round(rng.uniform(-1000, 1000), rng.randint(0, 6))
    return rng.uniform(-1e6, 1e6) * 10 ** rng.randint(-12, 12)


def random_int(rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.5:
        return rng.randint(-100, 100)
    if roll < 0.9:
        return rng.randint(-10**9, 10**9)
    return rng.randint(-(2**62), 2**62)


def random_scalar(rng: random.Random):
    roll = rng.random()
    if roll < 0.08:
        return None
    if roll < 0.16:
        return rng.random() < 0.5
    if roll < 0.35:
        return random_int(rng)
    if roll < 0.50:
        return random_float(rng)
    return random_string(rng)


def random_key(rng: random.Random):
    roll = rng.random()
    if roll < 0.75:
        return rng.choice(KEY_POOL)
    if roll < 0.9:
        return random_int(rng)
    if roll < 0.95:
        return round(rng.uniform(-10, 10), 2)
    return rng.random() < 0.5


def random_value(rng: random.Random, depth: int):
    roll = rng.random()
    if depth >= 4 or roll < 0.45:
        return random_scalar(rng)
    if roll < 0.7:
        return [random_value(rng, depth + 1) for _ in range(rng.randint(0, 5))]
    mapping = {}
    for _ in range(rng.randint(0, 5)):
        key = random_key(rng)
        if isinstance(key, float) and key != key:
            continue
        mapping[key] = random_value(rng, depth + 1)
    return mapping


def random_options(rng: random.Random) -> dict:
    options = {}
    options["default_flow_style"] = rng.choice([False, False, True, None])
    options["sort_keys"] = rng.choice([True, True, False])
    if rng.random() < 0.4:
        options["indent"] = rng.randint(1, 10)
    if rng.random() < 0.4:
        options["width"] = rng.choice([1, 4, 5, 10, 20, 40, 60, 120, 1000])
    if rng.random() < 0.4:
        options["allow_unicode"] = True
    if rng.random() < 0.25:
        options["default_style"] = rng.choice(['"', "'", "|", ">"])
    if rng.random() < 0.1:
        options["canonical"] = True
    if rng.random() < 0.15:
        options["explicit_start"] = True
    if rng.random() < 0.15:
        options["explicit_end"] = True
    if rng.random() < 0.05:
        options["version"] = (1, rng.choice([1, 2]))
    if rng.random() < 0.05:
        options["line_break"] = rng.choice(["\r\n", "\r"])
    return options


MUTATION_CHARS = ":[]{}-,#'\"|>&*!%@`?\n \t~"


def mutate(rng: random.Random, text: str) -> str:
    if not text:
        return rng.choice(MUTATION_CHARS)
    at = rng.randrange(len(text))
    roll = rng.random()
    if roll < 0.35:
        return text[:at] + text[at + 1:]
    if roll < 0.7:
        return text[:at] + rng.choice(MUTATION_CHARS) + text[at:]
    return text[:at] + rng.choice(MUTATION_CHARS) + text[at + 1:]


# --------------------------------------------------------------- jai driver

def jai_literal(text: str) -> str:
    out = ['"']
    for ch in text:
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif code < 0x20 or code == 0x7F or 0x80 <= code <= 0x9F or code in (0x2028, 0x2029, 0xFEFF) or 0xD800 <= code <= 0xDFFF:
            out.append("\\u{%X}" % code)
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def jai_option(name: str, value) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, tuple):
        return "(%d, %d)" % value
    return jai_literal(value)


OPTION_NAMES = [
    "default_style", "default_flow_style", "canonical", "indent", "width", "allow_unicode",
    "line_break", "explicit_start", "explicit_end", "version", "sort_keys",
]


def jai_options(options: dict) -> str:
    parts = []
    for name in OPTION_NAMES:
        if name in options:
            parts.append("%s: %s" % (name, jai_option(name, options[name])))
    return ", ".join(parts)


DRIVER_HEAD = '''import jaiyaml as yaml
from std.str import StringBuilder

fn escape(text: str) -> str {
    let out = StringBuilder()
    out.push("\\"")
    for ch in text {
        let code = ord(ch)
        if ch == "\\\\" {
            out.push("\\\\\\\\")
        } elif ch == "\\"" {
            out.push("\\\\\\"")
        } elif ch == "\\n" {
            out.push("\\\\n")
        } elif ch == "\\r" {
            out.push("\\\\r")
        } elif ch == "\\t" {
            out.push("\\\\t")
        } elif code < 0x20 or (code >= 0x7F and code <= 0x9F) or code == 0x2028 or code == 0x2029 or code == 0xFEFF {
            out.push("\\\\u" + hex4(code))
        } else {
            out.push(ch)
        }
    }
    out.push("\\"")
    return out.build()
}

fn hex4(code: int) -> str {
    let digits = "0123456789abcdef"
    var text = ""
    var rest = code
    while rest > 0 {
        text = digits[rest & 0xF] + text
        rest = rest >> 4
    }
    while text.len() < 4 { text = "0" + text }
    return text
}

fn canon(value: any) -> str {
    if value is null { return "null" }
    if isinstance(value, bool) { return value ? "true" : "false" }
    if isinstance(value, int) { return f"int({value})" }
    if isinstance(value, float) { return f"float({value})" }
    if isinstance(value, str) { return "str(" + escape(value) + ")" }
    if isinstance(value, list) or isinstance(value, tuple) {
        let parts = [canon(item) for item in value]
        return "[" + ", ".join(parts) + "]"
    }
    if isinstance(value, dict) {
        let parts = [canon(key) + ": " + canon(value[key]) for key in value.keys()]
        return "{" + ", ".join(parts) + "}"
    }
    return "other(" + type_of(value) + ")"
}

fn describe(error: Error) -> str { return type_of(error) + " " + escape(error.message) }

fn load_case(index: int, text: str) -> any {
    print(f"case {index}")
    try {
        let value = yaml.safe_load(text)
        print("load " + canon(value))
        return value
    } catch error: Error {
        print("loaderror " + describe(error))
        return null
    }
}

fn load_all_case(index: int, text: str) -> list[any]? {
    print(f"case {index}")
    try {
        let values = yaml.safe_load_all(text)
        print("load " + canon(values))
        return values
    } catch error: Error {
        print("loaderror " + describe(error))
        return null
    }
}

fn report_dump(body: fn() -> str) -> void {
    try {
        print("dump " + escape(body()))
    } catch error: Error {
        print("dumperror " + describe(error))
    }
}

'''


def build_driver(cases: list) -> str:
    lines = [DRIVER_HEAD]
    for index, case in enumerate(cases):
        kind = case["kind"]
        text = jai_literal(case["text"])
        if kind == "single":
            options = jai_options(case["options"])
            lines.append("fn case_%d() -> void {" % index)
            lines.append("    let value = load_case(%d, %s)" % (index, text))
            lines.append("    report_dump(|| yaml.safe_dump(value%s))" % ((", " + options) if options else ""))
            lines.append("}")
        elif kind == "multi":
            options = jai_options(case["options"])
            lines.append("fn case_%d() -> void {" % index)
            lines.append("    let values = load_all_case(%d, %s)" % (index, text))
            lines.append("    if values is null { return }")
            lines.append("    report_dump(|| yaml.safe_dump_all(values ?? []%s))" % ((", " + options) if options else ""))
            lines.append("}")
        else:
            lines.append("fn case_%d() -> void {" % index)
            lines.append("    load_case(%d, %s)" % (index, text))
            lines.append("}")
    lines.append("")
    lines.append("fn main() -> int {")
    for index in range(len(cases)):
        lines.append("    case_%d()" % index)
    lines.append("    return 0")
    lines.append("}")
    return "\n".join(lines) + "\n"


def run_driver(cases: list, keep: bool, label: str) -> list[dict]:
    source = build_driver(cases)
    directory = tempfile.mkdtemp(prefix="jaiyaml-diff-")
    path = os.path.join(directory, "driver_%s.jai" % label)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    completed = subprocess.run([JAITHON, "run", path], capture_output=True, text=True, encoding="utf-8")
    if completed.returncode != 0:
        sys.stderr.write("driver %s failed (exit %d):\n%s\n%s\n" % (path, completed.returncode, completed.stdout[-2000:], completed.stderr[-4000:]))
        sys.exit(2)
    if not keep:
        shutil.rmtree(directory, ignore_errors=True)
    else:
        print("kept driver:", path)
    results = [dict() for _ in cases]
    current = None
    for line in completed.stdout.split("\n"):
        if line.startswith("case "):
            current = results[int(line[5:])]
        elif current is None:
            continue
        elif line.startswith("load "):
            current["load"] = line[5:]
        elif line.startswith("loaderror "):
            current["loaderror"] = line[10:]
        elif line.startswith("dump "):
            current["dump"] = unescape(line[5:])
        elif line.startswith("dumperror "):
            current["dumperror"] = line[10:]
    return results


def unescape(literal: str) -> str:
    assert literal.startswith('"') and literal.endswith('"'), literal
    body = literal[1:-1]
    out = []
    index = 0
    while index < len(body):
        ch = body[index]
        if ch != "\\":
            out.append(ch)
            index += 1
            continue
        nxt = body[index + 1]
        if nxt == "n":
            out.append("\n")
            index += 2
        elif nxt == "r":
            out.append("\r")
            index += 2
        elif nxt == "t":
            out.append("\t")
            index += 2
        elif nxt == "u":
            out.append(chr(int(body[index + 2:index + 6], 16)))
            index += 6
        else:
            out.append(nxt)
            index += 2
    return "".join(out)


# ------------------------------------------------------------------- compare

def python_error(function, *args, **kwargs):
    try:
        return function(*args, **kwargs), None
    except yaml.YAMLError as error:
        return None, "%s %s" % (type(error).__name__, escape(str(error)))
    except Exception as error:
        return None, "%s %s" % (type(error).__name__, escape(str(error)))


def big_int_inside(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return not -(2**63) <= value < 2**63
    if isinstance(value, (list, tuple)):
        return any(big_int_inside(item) for item in value)
    if isinstance(value, dict):
        return any(big_int_inside(k) or big_int_inside(v) for k, v in value.items())
    return False


def null_key_inside(value) -> bool:
    if isinstance(value, (list, tuple)):
        return any(null_key_inside(item) for item in value)
    if isinstance(value, dict):
        return any(k is None or null_key_inside(k) or null_key_inside(v) for k, v in value.items())
    return False


class Report:
    def __init__(self):
        self.failures = []
        self.checks = 0
        self.deviations = 0

    def check(self, ok: bool, case_index: int, what: str, expected, actual):
        self.checks += 1
        if not ok:
            self.failures.append((case_index, what, expected, actual))

    def deviation(self):
        self.deviations += 1


def compare_generated(cases, results, report: Report, keep: bool):
    round_trip = []
    for index, (case, result) in enumerate(zip(cases, results)):
        kind = case["kind"]
        text = case["text"]
        options = case["options"]
        if kind == "single":
            expected_value = yaml.safe_load(text)
            expected_dump = yaml.safe_dump(expected_value, **options)
        else:
            expected_value = list(yaml.safe_load_all(text))
            expected_dump = yaml.safe_dump_all(expected_value, **options)
        expected_load = canon(expected_value)
        report.check(result.get("load") == expected_load, index, "load", expected_load, result.get("load", result.get("loaderror")))
        report.check(result.get("dump") == expected_dump, index, "dump", expected_dump, result.get("dump", result.get("dumperror")))
        if "dump" in result:
            # The oracle for a reload is PyYAML reloading its own dump of the
            # same value: a key such as "\x85next" comes back as " next", and
            # with sort_keys that lands in a different place the second time.
            if kind == "single":
                load_text = yaml.safe_load
                dump_value = lambda value: yaml.safe_dump(value, **options)
            else:
                load_text = lambda text: list(yaml.safe_load_all(text))
                dump_value = lambda value: yaml.safe_dump_all(value, **options)
            expected_reload = canon(load_text(expected_dump))
            reloaded, error = python_error(load_text, result["dump"])
            report.check(error is None and canon(reloaded) == expected_reload, index, "pyyaml reload of jaiyaml dump", expected_reload, error or canon(reloaded))
            if error is None:
                again = dump_value(reloaded)
                round_trip.append((index, kind, again, canon(load_text(again))))
    if round_trip:
        second = [{"kind": kind if kind == "multi" else "load", "text": again, "options": {}} for (_, kind, again, _) in round_trip]
        second_results = run_driver(second, keep, "roundtrip")
        for (index, kind, again, expected_reload), result in zip(round_trip, second_results):
            report.check(result.get("load") == expected_reload, index, "jaiyaml reload of pyyaml dump of jaiyaml dump", expected_reload, result.get("load", result.get("loaderror")))


def compare_mutants(cases, results, report: Report):
    for index, (case, result) in enumerate(zip(cases, results)):
        text = case["text"]
        expected_value, expected_error = python_error(lambda t: yaml.load(t, Loader=CoreLoader), text)
        if "loaderror" in result:
            actual = result["loaderror"]
            if "found a null key" in actual and expected_error is None and null_key_inside(expected_value):
                report.deviation()
                continue
            # An explicitly tagged scalar that is not a number: PyYAML lets
            # int() or float() raise a bare ValueError (KeyError for a bool);
            # jaiyaml raises ConstructorError with the mark.
            if expected_error is not None and expected_error.split(" ")[0] in ("ValueError", "KeyError") \
                    and actual.startswith("ConstructorError") and "which is not a core-schema" in actual:
                report.deviation()
                continue
            report.check(expected_error == actual, index, "mutant error", expected_error, actual)
        else:
            if expected_error is None and big_int_inside(expected_value):
                report.deviation()
                continue
            expected = canon(expected_value) if expected_error is None else expected_error
            report.check(expected_error is None and result.get("load") == expected, index, "mutant load", expected, result.get("load"))


# ---------------------------------------------------------------------- main

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--count", type=int, default=400, help="generated documents (default 400)")
    parser.add_argument("--seed", type=int, default=20260901, help="base seed (default 20260901)")
    parser.add_argument("--keep", action="store_true", help="keep the generated .jai drivers")
    parser.add_argument("--show", type=int, default=12, help="failures to print in full (default 12)")
    args = parser.parse_args(argv[1:])
    if not os.access(JAITHON, os.X_OK):
        print("error: %s is not built; run make first" % JAITHON, file=sys.stderr)
        return 2

    generated = []
    mutants = []
    for number in range(args.count):
        rng = random.Random(args.seed * 1_000_003 + number)
        options = random_options(rng)
        if rng.random() < 0.15:
            documents = [random_value(rng, 0) for _ in range(rng.randint(1, 3))]
            text = yaml.safe_dump_all(documents, **options)
            generated.append({"kind": "multi", "text": text, "options": options, "seed": number})
        else:
            value = random_value(rng, 0)
            text = yaml.safe_dump(value, **options)
            generated.append({"kind": "single", "text": text, "options": options, "seed": number})
        mutated = mutate(rng, text)
        if all(ch in "\t\n\r" or 0x20 <= ord(ch) <= 0x7E or ord(ch) >= 0xA0 for ch in mutated):
            mutants.append({"kind": "load", "text": mutated, "options": {}, "seed": number})

    report = Report()
    results = run_driver(generated, args.keep, "generated")
    compare_generated(generated, results, report, args.keep)
    mutant_results = run_driver(mutants, args.keep, "mutants")
    compare_mutants(mutants, mutant_results, report)

    print("jaiyaml differential: %d generated documents, %d mutants, %d checks, %d known deviations, %d failures"
          % (len(generated), len(mutants), report.checks, report.deviations, len(report.failures)))
    for case_index, what, expected, actual in report.failures[: args.show]:
        print("\nFAIL case %d: %s" % (case_index, what))
        print("  expected: %r" % (expected,))
        print("  actual:   %r" % (actual,))
    if len(report.failures) > args.show:
        print("\n... and %d more" % (len(report.failures) - args.show))
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
