#!/usr/bin/env python3
"""A test named `test_jit_*` can pass with its own fix reverted, because
`jaithon test` never makes the body it means to exercise hot enough to
compile. Measured across the six JIT tests in tests/lang, counting functions
that actually reach `[jit] compiled` or an OSR form (docs/roadmap.md §7):

    test_jit_field_read                    9/18
    test_jit_list_iter_kind                6/14
    test_jit_dict_items_iter               4/24
    test_jit_invoke_result_kind            4/13
    test_jit_string_local_const_compare    2/9
    test_string_char_compare               1/12

Every author verified teeth by reverting their fix and watching the test
still pass -- so the fixes are real. But nothing enforced it: an edit that
drops a loop below JAI_JIT_THRESHOLD (64) hollows the gate with no signal,
silently, forever.

This closes that hole with a declaration next to the test rather than a
separate list that drifts:

    # jit-compiles: helper_one, helper_two

one line, naming the functions THIS FILE claims reach the compiled tier
(the helper under test, never the `test_*` wrapper -- assertions are not
expected to compile). For each name this:

  1. checks it names a real top-level `fn` in the same file -- a renamed or
     deleted function rots the marker, and that must fail too;
  2. runs the file (`jaithon test` under tests/lang, `jaithon run` under
     tests/golden) with `JAI_JIT_WHY=1` and checks the name reaches
     `[jit] compiled NAME ...` (the deterministic, call-count-driven
     whole-function tier) or `[jit] osr NAME at ...` (the sampler-driven
     loop tier).

`JAI_JIT_WHY=1` prints once per compile ATTEMPT and the tier retries hot
sites, so 114 raw lines can collapse to 9 real ones (docs/roadmap.md §7).
Collapsing here is automatic: hits are gathered into sets, so repeats do not
inflate anything and only presence/absence is asked.

Known blind spot: an `[jit] osr NAME at ...` success line carries no arity,
so a NAME that also happens to be a function inside the self-hosted
compiler or test-tool front end (both loaded by every `jaithon test` run,
and both hot) could in principle be credited instead of the real one. A
`[jit] compiled NAME arity=N ...` line does carry an arity and is
cross-checked against the marked function's own parameter count when this
script can parse it, which is why `sum_labels`, `count_matches` and friends
were chosen deliberately distinctive rather than `count` or `helper`. Pick
names accordingly.

Some of the marked functions in these six files compile only through the
sampler (OSR): `JIT_MAX_ARITY` is 4, so a 5-argument function can never
reach the whole-function tier at all, and a couple of others exist
specifically to test OSR's own entry mechanics (a stale interpreter stack
slot, a nested iterator's operand-stack bookkeeping) -- something the
whole-function tier does not share and forcing the call-count route on them
would test the wrong mechanism entirely. For those, reliability comes from
giving the sampler enough on-function CPU time to be virtually certain of a
tick (see the warm-up loops added alongside the markers), not from making
OSR itself deterministic -- it is not, and no amount of iterating changes
that.

    python3 scripts/jit_compile_check.py             # auto-discovered files
    python3 scripts/jit_compile_check.py FILE...      # just these

Exit status is non-zero if any marker rotted or any named function failed
to reach a compiled form.
"""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAITHON = os.environ.get("JAITHON", os.path.join(ROOT, "jaithon"))
JAITHON_PATH = os.environ.get("JAITHON_PATH", os.path.join(ROOT, "lib"))

MARKER_RE = re.compile(r"^#\s*jit-compiles:\s*(.*)$")
FN_RE = re.compile(r"^fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
COMPILED_RE = re.compile(r"^\[jit\] compiled (\S+)\s+arity=(\d+)")
OSR_RE = re.compile(r"^\[jit\] osr (\S+) at \d+:\s+\d+ instructions")


def default_targets():
    files = []
    for base in ("tests/lang", "tests/golden"):
        for path in sorted(glob.glob(os.path.join(ROOT, base, "**", "*.jai"),
                                      recursive=True)):
            with open(path, encoding="utf-8", errors="replace") as f:
                if any(MARKER_RE.match(line.rstrip("\n")) for line in f):
                    files.append(path)
    return files


def parse_marker(text, path, problems):
    """Returns the declared name list, or None if there is no marker at all.

    A marker line that parses to zero names, or more than one marker line in
    one file, is the marker itself rotting -- both are reported as problems
    rather than silently ignored.
    """
    lines = [l for l in text.splitlines() if MARKER_RE.match(l)]
    if not lines:
        return None
    if len(lines) > 1:
        problems.append(f"{rel(path)}: {len(lines)} 'jit-compiles:' marker "
                         f"lines -- exactly one is expected, so the extras "
                         f"are dead or the file is trying to say two "
                         f"different things")
        return None
    names = [n.strip() for n in MARKER_RE.match(lines[0]).group(1).split(",")]
    names = [n for n in names if n]
    if not names:
        problems.append(f"{rel(path)}: 'jit-compiles:' marker names nothing "
                         f"-- rotted to an empty list")
        return None
    return names


def parse_top_level_fns(text):
    """name -> arity for every `fn name(...)` at column 0.

    Arity counts commas at bracket depth 0 in the parameter list, so a
    parameter typed `dict[str, int]` does not look like two parameters.
    Returns arity None for a signature this cannot balance (a `{` without a
    matching close before EOF, or similar) rather than guess.
    """
    fns = {}
    for m in FN_RE.finditer(text):
        name = m.group(1)
        i = m.end() - 1          # index of the opening '('
        depth = 0
        commas = 0
        j = i
        n = len(text)
        while j < n:
            c = text[j]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
                if depth == 0:
                    break
            elif c == "," and depth == 1:
                commas += 1
            j += 1
        if j >= n or depth != 0:
            fns[name] = None
            continue
        inner = text[i + 1:j].strip()
        arity = 0 if inner == "" else commas + 1
        fns[name] = arity
    return fns


def rel(path):
    return os.path.relpath(path, ROOT)


def run_target(path):
    """(stdout, stderr, returncode) for the file under JAI_JIT_WHY=1.

    Run once, unmeasured, first. On a cold __jaicache__ the self-hosted front
    end has to compile ITSELF (and this file) from source, which is enough
    CPU time that its own hot loops soak up most of the sampler's ticks --
    measured: on a fresh cache one_call_sum/nested in
    test_jit_dict_items_iter.jai missed OSR entirely, 2/2 runs, and passed
    every time once warm. scripts/jit_declines.sh hits the same thing
    ("warm every cache first, or the front end's own compilation
    dominates") and fixes it the same way.
    """
    env = dict(os.environ)
    env["JAITHON_PATH"] = JAITHON_PATH
    parts = os.path.normpath(path).split(os.sep)
    mode = "test" if "lang" in parts and "tests" in parts else "run"
    cmd = [JAITHON, mode, path]
    subprocess.run(cmd, cwd=ROOT, env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    env["JAI_JIT_WHY"] = "1"
    proc = subprocess.run(cmd, cwd=ROOT, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True)
    return proc.stdout, proc.stderr, proc.returncode


def collect_reached(stderr):
    """name -> set of arities seen compiled; name -> reached via osr."""
    compiled = {}
    osr = set()
    for line in stderr.splitlines():
        m = COMPILED_RE.match(line)
        if m:
            compiled.setdefault(m.group(1), set()).add(int(m.group(2)))
            continue
        m = OSR_RE.match(line)
        if m:
            osr.add(m.group(1))
    return compiled, osr


def check_file(path, problems, oks):
    text = open(path, encoding="utf-8", errors="replace").read()
    names = parse_marker(text, path, problems)
    if names is None:
        return

    fns = parse_top_level_fns(text)
    missing = [n for n in names if n not in fns]
    for n in missing:
        problems.append(f"{rel(path)}: 'jit-compiles: {n}' names no "
                         f"top-level `fn {n}(...)` in this file -- renamed, "
                         f"deleted, or never existed. The marker rotted.")
    if missing:
        return   # running the file cannot answer for a function it lacks

    stdout, stderr, code = run_target(path)
    if code != 0:
        problems.append(f"{rel(path)}: exited {code} running under "
                         f"JAI_JIT_WHY=1 -- fix the test before its "
                         f"compile coverage can be trusted\n"
                         f"{indent(stdout)}{indent(stderr)}")
        return

    compiled, osr = collect_reached(stderr)
    for n in names:
        want_arity = fns[n]
        if n in osr:
            oks.append(f"{rel(path)}: {n} -- osr")
            continue
        got_arities = compiled.get(n)
        if got_arities is not None and (want_arity is None or
                                         want_arity in got_arities):
            oks.append(f"{rel(path)}: {n} -- compiled")
            continue
        if got_arities is not None:
            problems.append(
                f"{rel(path)}: {n} declared with {want_arity} parameter(s) "
                f"but the only '[jit] compiled {n}' seen had arity(ies) "
                f"{sorted(got_arities)} -- that is a different function of "
                f"the same name (likely the self-hosted front end), not "
                f"this one")
            continue
        problems.append(
            f"{rel(path)}: {n} is declared to reach the compiled tier "
            f"('jit-compiles:') but neither '[jit] compiled {n}' nor "
            f"'[jit] osr {n} at ...' appeared under JAI_JIT_WHY=1 -- it "
            f"stopped compiling")


def indent(s, prefix="    "):
    if not s.strip():
        return ""
    return "\n".join(prefix + l for l in s.splitlines()) + "\n"


def main(argv):
    targets = argv[1:] if len(argv) > 1 else default_targets()
    if not targets:
        print("jit-compiles check: no file declares a 'jit-compiles:' "
              "marker -- nothing to check")
        return 0

    problems = []
    oks = []
    for path in targets:
        check_file(os.path.abspath(path), problems, oks)

    if problems:
        print(f"jit-compiles check FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  {p}")
        return 1

    by_file = {}
    for line in oks:
        f = line.split(":", 1)[0]
        by_file[f] = by_file.get(f, 0) + 1
    print(f"jit-compiles check ok: {len(oks)} marked function(s) reached "
          f"the compiled tier across {len(by_file)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
