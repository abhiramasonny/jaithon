#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["tomli-w"]
# ///
"""Differential test: jaitoml against Python's tomllib and tomli_w.

Three sources of input, all diffed exactly:

1. Generated documents. Random Python values are dumped by tomli_w, loaded by
   tomllib and by jaitoml, and the two results are compared through one
   canonical text form; then both loaders' results are dumped again and the
   two TOML texts are compared byte for byte.
2. Mutations. Each generated document is damaged at a random position. When
   tomllib rejects it, jaitoml must raise TOMLDecodeError with the same
   message and position; when tomllib accepts it, both must agree as above.
3. The official toml-test corpus, when it is at --corpus (default
   /tmp/toml-test): every valid/ file must load to what tomllib loads, and
   every invalid/ file must raise. Pass counts are reported.

Exit status is 1 on any mismatch in 1 or 2, or any corpus file jaitoml gets
wrong other than the documented deviation (integers beyond 64 bits).

    uv run --python 3.13 --with tomli-w python3 packages/jaitoml/tests/py/jaitoml_diff.py
    uv run --python 3.13 --with tomli-w python3 packages/jaitoml/tests/py/jaitoml_diff.py --cases 1000 --seed 7
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import tomli_w

ROOT = Path(__file__).resolve().parents[4]
JAITHON = ROOT / "jaithon"

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
OUT_OF_RANGE = "Integer is outside the 64-bit range this runtime can hold"

BARE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
ODD_CHARS = ' "\'\\\t\n\r\x00\x01\x1f\x7f#=[]{}.,é漢😀' + " "


# ----------------------------------------------------------------- generation


class Gen:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def key(self) -> str:
        r = self.rng
        if r.random() < 0.7:
            return "".join(r.choice(BARE) for _ in range(r.randint(1, 8)))
        if r.random() < 0.1:
            return ""
        return "".join(r.choice(BARE + ODD_CHARS) for _ in range(r.randint(1, 6)))

    def string(self) -> str:
        r = self.rng
        pool = BARE + " " if r.random() < 0.5 else BARE + ODD_CHARS
        text = "".join(r.choice(pool) for _ in range(r.randint(0, 20)))
        if r.random() < 0.1:
            text = r.choice(['"""', "'''", '""', "\\", "\r\n", "\n\n", '"'])
        return text

    def integer(self) -> int:
        r = self.rng
        pick = r.random()
        if pick < 0.5:
            return r.randint(-1000, 1000)
        if pick < 0.8:
            return r.randint(-(10**12), 10**12)
        return r.choice([0, -0, 1, -1, INT64_MIN, INT64_MAX, INT64_MIN + 1, INT64_MAX - 1])

    def floating(self) -> float:
        r = self.rng
        pick = r.random()
        if pick < 0.3:
            return round(r.uniform(-1000.0, 1000.0), r.randint(0, 6))
        if pick < 0.5:
            return r.uniform(-1e300, 1e300)
        if pick < 0.7:
            return r.uniform(-1e-300, 1e-300)
        if pick < 0.85:
            return float(r.randint(-10**16, 10**16))
        return r.choice(
            [0.0, -0.0, 1.0, -1.0, 1e16, 1e15, 1e-5, 1e-4, 5e-324, 1.7976931348623157e308,
             float("inf"), float("-inf"), float("nan"), 0.1, 1 / 3]
        )

    def date(self) -> dt.date:
        r = self.rng
        year = r.choice([1, 100, 1900, 1970, 2000, 2024, 9999, r.randint(1, 9999)])
        month = r.randint(1, 12)
        while True:
            try:
                return dt.date(year, month, r.randint(1, 31))
            except ValueError:
                continue

    def time(self) -> dt.time:
        r = self.rng
        micro = r.choice([0, 0, 1, 999999, 500000, r.randint(0, 999999)])
        return dt.time(r.randint(0, 23), r.randint(0, 59), r.randint(0, 59), micro)

    def datetime(self) -> dt.datetime:
        r = self.rng
        d, t = self.date(), self.time()
        tz = None
        pick = r.random()
        if pick < 0.3:
            tz = dt.timezone.utc
        elif pick < 0.6:
            minutes = r.randint(-23 * 60 - 59, 23 * 60 + 59)
            if r.random() < 0.5:
                minutes = r.choice([60, -60, 330, -420, 90, 1439, -1439, 0])
            tz = dt.timezone(dt.timedelta(minutes=minutes))
        return dt.datetime.combine(d, t, tzinfo=tz)

    def scalar(self):
        r = self.rng
        return r.choice(
            [self.string, self.integer, self.floating, lambda: r.random() < 0.5,
             self.date, self.time, self.datetime]
        )()

    def value(self, depth: int):
        r = self.rng
        if depth <= 0 or r.random() < 0.55:
            return self.scalar()
        if r.random() < 0.5:
            return self.array(depth - 1)
        return self.table(depth - 1)

    def array(self, depth: int) -> list:
        r = self.rng
        count = r.randint(0, 5)
        if r.random() < 0.3:
            return [self.table(depth) for _ in range(max(count, 1))]
        if r.random() < 0.5:
            maker = r.choice([self.scalar, lambda: self.value(depth)])
            return [maker() for _ in range(count)]
        return [self.value(depth) for _ in range(count)]

    def table(self, depth: int) -> dict:
        r = self.rng
        out: dict = {}
        for _ in range(r.randint(0, 6)):
            out[self.key()] = self.value(depth)
        return out

    def document(self) -> dict:
        return self.table(self.rng.randint(1, 4))


# -------------------------------------------------------------- canonical form


def quote(text: str) -> str:
    out = ['"']
    for ch in text:
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif code < 0x20 or code == 0x7F:
            out.append("\\u%04x" % code)
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def canon(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return "i:" + str(value)
    if isinstance(value, float):
        return "f:" + ("nan" if value != value else repr(value))
    if isinstance(value, str):
        return "s:" + quote(value)
    if isinstance(value, dt.datetime):
        return "dt:" + value.isoformat()
    if isinstance(value, dt.date):
        return "d:" + value.isoformat()
    if isinstance(value, dt.time):
        return "t:" + value.isoformat()
    if isinstance(value, list):
        return "[" + ",".join(canon(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(quote(k) + ":" + canon(v) for k, v in value.items()) + "}"
    raise TypeError(type(value))


DRIVER = r'''
import jaitoml
from jaitoml import Date, DateTime, Time, TOMLDecodeError
from std.io import read_file
from std.math import is_nan
from std.str import StringBuilder

fn quote(text: str) -> str {
    let out = StringBuilder()
    out.push("\"")
    for ch in text {
        let code = ord(ch)
        if ch == "\\" {
            out.push("\\\\")
        } elif ch == "\"" {
            out.push("\\\"")
        } elif code < 0x20 or code == 0x7F {
            out.push("\\u" + hex4(code))
        } else {
            out.push(ch)
        }
    }
    out.push("\"")
    return out.build()
}

fn hex4(code: int) -> str {
    let digits = "0123456789abcdef"
    var out = ""
    var shift = 12
    while shift >= 0 {
        out = out + digits[code >> shift & 0xF]
        shift -= 4
    }
    return out
}

fn canon(value: any) -> str {
    if isinstance(value, bool) { return value ? "true" : "false" }
    if isinstance(value, int) { return "i:" + str(value) }
    if isinstance(value, float) { return "f:" + (is_nan(value) ? "nan" : str(value)) }
    if isinstance(value, str) { return "s:" + quote(value) }
    if isinstance(value, DateTime) { return "dt:" + value.to_iso() }
    if isinstance(value, Date) { return "d:" + value.to_iso() }
    if isinstance(value, Time) { return "t:" + value.to_iso() }
    if isinstance(value, list) {
        return "[" + ",".join([canon(item) for item in value]) + "]"
    }
    if isinstance(value, dict) {
        let parts = [quote(key) + ":" + canon(item) for (key, item) in value.items()]
        return "{" + ",".join(parts) + "}"
    }
    throw TypeError(f"no canonical form for {type_of(value)}")
}

fn main() -> int {
    let cases = read_file("__CASES__").split("\n")
    for line in cases {
        if line.len() == 0 { continue }
        let parts = line.split("\t")
        let path = parts[0]
        let multiline = parts[1] == "1"
        let indent = int(parts[2])
        let text = read_file(path)
        print("=== " + path)
        try {
            let value = jaitoml.loads(text)
            print(canon(value))
            print("---")
            print(jaitoml.dumps(value, multiline_strings: multiline, indent: indent))
        } catch e: TOMLDecodeError {
            print("!!! " + e.message)
        } catch e: Error {
            print("??? " + type_of(e) + ": " + e.message)
        }
        print("=== end")
    }
    return 0
}
'''


# ----------------------------------------------------------------- the driver


class Result:
    def __init__(self, error: str | None, canon: str | None, dump: str | None):
        self.error = error
        self.canon = canon
        self.dump = dump


def run_jaithon(work: Path, cases: list[tuple[Path, bool, int]]) -> dict[str, Result]:
    case_list = work / "cases.tsv"
    case_list.write_text(
        "".join(f"{path}\t{int(multi)}\t{indent}\n" for path, multi, indent in cases),
        encoding="utf-8",
    )
    driver = work / "jaitoml_diff_driver.jai"
    driver.write_text(DRIVER.replace("__CASES__", str(case_list)), encoding="utf-8")
    proc = subprocess.run(
        [str(JAITHON), "run", str(driver)],
        cwd=ROOT,
        capture_output=True,
        env={**os.environ, "JAITHON_NO_COLOR": "1"},
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        sys.stderr.write(proc.stdout.decode("utf-8", "replace")[-2000:])
        raise SystemExit(f"jaithon exited {proc.returncode}")
    text = proc.stdout.decode("utf-8")
    results: dict[str, Result] = {}
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("=== ") or line == "=== end":
            index += 1
            continue
        name = line[4:]
        index += 1
        body: list[str] = []
        while lines[index] != "=== end":
            body.append(lines[index])
            index += 1
        index += 1
        if body and body[0].startswith("!!! "):
            results[name] = Result(body[0][4:], None, None)
        elif body and body[0].startswith("??? "):
            results[name] = Result(body[0], None, None)
        else:
            split = body.index("---")
            results[name] = Result(None, body[0], "\n".join(body[split + 1 :]))
    return results


def python_result(text: str, multi: bool, indent: int) -> Result:
    try:
        value = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        return Result(str(error), None, None)
    except (RecursionError, ValueError, TypeError) as error:
        return Result(f"??? {type(error).__name__}: {error}", None, None)
    return Result(None, canon(value), tomli_w.dumps(value, multiline_strings=multi, indent=indent))


def mutate(rng: random.Random, text: str) -> str:
    if not text:
        return rng.choice(["=", "[", '"', "a"])
    pos = rng.randrange(len(text))
    pick = rng.random()
    if pick < 0.4:
        return text[:pos] + text[pos + 1 :]
    if pick < 0.8:
        return text[:pos] + rng.choice(BARE + ODD_CHARS + "+-:.0123456789tfnaiZ") + text[pos:]
    return text[:pos] + text[pos] + text[pos:]


def has_big_int(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return not INT64_MIN <= value <= INT64_MAX
    if isinstance(value, list):
        return any(has_big_int(v) for v in value)
    if isinstance(value, dict):
        return any(has_big_int(v) for v in value.values())
    return False


def compare(name: str, expected: Result, actual: Result, failures: list[str]) -> str:
    if expected.error is not None:
        if actual.error == expected.error:
            return "error"
        if actual.error is not None and actual.error.startswith(OUT_OF_RANGE):
            return "deviation"
        failures.append(f"{name}\n  python : {expected.error!r}\n  jaithon: {actual.error!r}")
        return "fail"
    if actual.error is not None:
        if actual.error.startswith(OUT_OF_RANGE):
            return "deviation"
        failures.append(f"{name}\n  python : ok\n  jaithon: {actual.error!r}")
        return "fail"
    if actual.canon != expected.canon:
        failures.append(f"{name}\n  canon differs\n  python : {expected.canon}\n  jaithon: {actual.canon}")
        return "fail"
    if actual.dump != expected.dump:
        failures.append(f"{name}\n  dump differs\n  python :\n{expected.dump}\n  jaithon:\n{actual.dump}")
        return "fail"
    return "ok"


def corpus_files(corpus: Path) -> tuple[list[Path], list[Path]]:
    listing = corpus / "tests" / "files-toml-1.0.0"
    if listing.is_file():
        names = [line.strip() for line in listing.read_text().splitlines() if line.strip()]
        valid = [corpus / "tests" / n for n in names if n.startswith("valid/") and n.endswith(".toml")]
        invalid = [corpus / "tests" / n for n in names if n.startswith("invalid/") and n.endswith(".toml")]
        return valid, invalid
    valid = sorted((corpus / "tests" / "valid").rglob("*.toml"))
    invalid = sorted((corpus / "tests" / "invalid").rglob("*.toml"))
    return valid, invalid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=int, default=400, help="generated documents (default 400)")
    parser.add_argument("--mutations", type=int, default=400, help="mutated documents (default 400)")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--corpus", type=Path, default=Path("/tmp/toml-test"))
    parser.add_argument("--show", type=int, default=5, help="failures to print in full")
    args = parser.parse_args()

    if not JAITHON.is_file():
        print(f"missing {JAITHON}; run make first", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    gen = Gen(rng)
    work = Path(tempfile.mkdtemp(prefix="jaitoml-diff-"))
    cases: list[tuple[Path, bool, int]] = []
    expected: dict[str, Result] = {}
    kinds: dict[str, str] = {}

    docs: list[str] = []
    for index in range(args.cases):
        multi = rng.random() < 0.4
        indent = rng.choice([0, 1, 2, 4, 4, 4])
        obj = gen.document()
        try:
            text = tomli_w.dumps(obj, multiline_strings=multi, indent=indent)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"generator produced an undumpable object: {error}")
        path = work / f"gen_{index:04d}.toml"
        path.write_text(text, encoding="utf-8", newline="")
        cases.append((path, multi, indent))
        expected[str(path)] = python_result(text, multi, indent)
        kinds[str(path)] = "generated"
        docs.append(text)

    for index in range(args.mutations):
        base = docs[rng.randrange(len(docs))] if docs else ""
        text = mutate(rng, base)
        for _ in range(rng.randint(0, 2)):
            text = mutate(rng, text)
        path = work / f"mut_{index:04d}.toml"
        path.write_text(text, encoding="utf-8", newline="")
        cases.append((path, False, 4))
        expected[str(path)] = python_result(text, False, 4)
        kinds[str(path)] = "mutation"

    valid_files: list[Path] = []
    invalid_files: list[Path] = []
    not_utf8 = 0
    if args.corpus.is_dir():
        valid_files, invalid_files = corpus_files(args.corpus)
        for path in valid_files + invalid_files:
            try:
                text = path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                not_utf8 += 1
                continue
            cases.append((path, False, 4))
            expected[str(path)] = python_result(text, False, 4)
            kinds[str(path)] = "valid" if path in valid_files else "invalid"

    actual = run_jaithon(work, cases)

    failures: list[str] = []
    tally: dict[str, dict[str, int]] = {}
    corpus_ok = {"valid": 0, "invalid": 0}
    corpus_total = {"valid": 0, "invalid": 0}
    corpus_same_message = 0
    corpus_tomllib_wrong = {"valid": 0, "invalid": 0}
    corpus_failures: list[str] = []
    for path, _multi, _indent in cases:
        name = str(path)
        kind = kinds[name]
        got = actual.get(name)
        if got is None:
            failures.append(f"{name}\n  no output from jaithon")
            continue
        if kind in ("generated", "mutation"):
            verdict = compare(name, expected[name], got, failures)
            tally.setdefault(kind, {}).setdefault(verdict, 0)
            tally[kind][verdict] += 1
            continue
        corpus_total[kind] += 1
        want = expected[name]
        if kind == "valid":
            if want.error is not None:
                corpus_tomllib_wrong["valid"] += 1
            if got.error is None and (want.error is not None or got.canon == want.canon):
                corpus_ok["valid"] += 1
            elif got.error is not None and got.error == want.error:
                corpus_failures.append(f"{name}: shared with tomllib, {got.error}")
            elif got.error is not None and got.error.startswith(OUT_OF_RANGE):
                corpus_ok["valid"] += 1
                corpus_failures.append(f"{name}: deviation, {got.error}")
            else:
                corpus_failures.append(
                    f"{name}\n  python : {want.error or want.canon}\n  jaithon: {got.error or got.canon}"
                )
        else:
            if want.error is None:
                corpus_tomllib_wrong["invalid"] += 1
            if got.error is not None and not got.error.startswith("???"):
                corpus_ok["invalid"] += 1
                if got.error == want.error:
                    corpus_same_message += 1
            else:
                corpus_failures.append(f"{name}: jaithon accepted it; python: {want.error or 'accepted too'}")

    for kind, counts in tally.items():
        summary = ", ".join(f"{count} {verdict}" for verdict, count in sorted(counts.items()))
        print(f"{kind}: {summary}")
    if corpus_total["valid"] or corpus_total["invalid"]:
        print(
            f"toml-test valid/: {corpus_ok['valid']}/{corpus_total['valid']} pass "
            f"(tomllib itself fails {corpus_tomllib_wrong['valid']})"
        )
        print(
            f"toml-test invalid/: {corpus_ok['invalid']}/{corpus_total['invalid']} pass, "
            f"{corpus_same_message} with tomllib's exact message "
            f"(tomllib itself accepts {corpus_tomllib_wrong['invalid']})"
        )
        if not_utf8:
            print(f"toml-test: {not_utf8} files are not UTF-8 and were skipped; both loaders take text")
        for line in corpus_failures:
            print("  " + line.replace("\n", "\n  "))
    else:
        print(f"toml-test corpus not found at {args.corpus}; skipped")

    hard_corpus_failures = [
        f for f in corpus_failures if "deviation" not in f and "shared with tomllib" not in f
    ]
    if failures or hard_corpus_failures:
        print(f"\n{len(failures)} mismatches, {len(hard_corpus_failures)} corpus failures")
        for failure in failures[: args.show]:
            print("\n" + failure)
        if len(failures) > args.show:
            print(f"\n... {len(failures) - args.show} more; work dir kept at {work}")
        return 1
    print(f"all {len(cases)} cases agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
