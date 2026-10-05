#!/usr/bin/env python3
"""Differential fuzz for the f-string memo and the inline dict paths.

progen.py writes almost none of what these paths are for: no counting idiom
(`d[k] = d.get(k, n) + c`, `d[k] += c`), no dict store through a computed
key, and one f-string shape (`f"{z}|{f}"`) that is not the memo's. This
writes little else: loops over one-int-hole f-strings -- literal runs, runs
held in a variable and changing call to call, long runs past the short
limit, unicode, `f"{n}"` and `str(n)` -- used as keys of untyped, typed and
mixed-value dicts, counted, read, stored, removed (tombstones), compared with
`is`, and int keys at the extremes, with collections in the middle. Every
program runs under the configurations below and any difference in output,
exit status or exception is a miscompile. The memo is checked by the `is`
counts: a hit must be the very object the leaf would have returned.

The last three configurations switch one mechanism off each, which says
whether a disagreement is that mechanism or something older.

Its teeth are established rather than assumed: with the collector's
jaiFmtMemoClear removed (gc.c), so that an entry outlives the string it
names, 46 of 120 programs disagree. `make dictfmt-fuzz` runs 400.

    python3 tests/fuzz/dictfmt_differential.py               # 100 programs
    python3 tests/fuzz/dictfmt_differential.py --count 300 --gc
    python3 tests/fuzz/dictfmt_differential.py --seed 17 --count 1 --keep
"""
import argparse
import concurrent.futures
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JAITHON = os.environ.get("JAITHON", os.path.join(ROOT, "jaithon"))

CONFIGS = [
    ("interp", {"JAITHON_NO_JIT": "1"}, []),
    ("default", {}, []),
    ("thresh", {"JAITHON_JIT_THRESHOLD": "1"}, []),
    ("tick", {"JAITHON_JIT_TICK_US": "50"}, []),
    ("deopt", {"JAITHON_JIT_DEOPT_STRESS": "1"}, []),
    ("memo-off", {"JAITHON_JIT_FMT_MEMO": "0"}, []),
    ("add-off", {"JAITHON_JIT_DICT_ADD_INLINE": "0"}, []),
    ("slot-off", {"JAITHON_JIT_DICT_SLOT_INLINE": "0"}, []),
]
GC_CONFIG = ("gc", {}, ["--gc-stress=64"])

RUNS = ["", "k", "user:", "order:", "item-", "é", "-", "x" * 30, "a b"]
STEPS = [-4096, -4095, -1, 0, 1, 2, 7, 4095, 4096]
MODS = [1, 3, 7, 100, 1000, 5000]
BIG = "9223372036854775807"
SMALL = "(-9223372036854775807 - 1)"


def int_expr(r):
    k = r.choice(MODS)
    return r.choice([
        f"i % {k}",
        f"-(i % {k})",
        f"(i * 7919) % {k} - {k // 2}",
        f"i // 3 % {k}",
        f"i % {k} + 1000000",
        f"i % {k} * 1000003",
        BIG,
        SMALL,
    ])


def key_expr(r):
    n = int_expr(r)
    pick = r.randrange(7)
    if pick == 0:
        return f'f"{{{n}}}"'
    if pick == 1:
        return f"str({n})"
    if pick == 2:
        return f'f"{{p}}{{{n}}}"'
    if pick == 3:
        return f'f"{{p}}{{{n}}}{{q}}"'
    if pick == 4:
        return f'f"{{{n}}}-{{p}}-{{{int_expr(r)}}}"'
    pre = r.choice(RUNS)
    post = r.choice(RUNS) if pick == 5 else ""
    return f'f"{pre}{{{n}}}{post}"'


def stmt(r, idx):
    k = f"k{idx}"
    lines = [f"let {k} = {key_expr(r)}"]
    step = r.choice(STEPS)
    pick = r.randrange(11)
    if pick == 0:
        lines.append(f"d[{k}] = d.get({k}, {r.choice(['0', '5', BIG + ' - 9000'])}) + {step}")
    elif pick == 1:
        lines.append(f"if {k} in d {{ d[{k}] += {step} }} else {{ d[{k}] = {step} }}")
    elif pick == 2:
        lines.append(f"if {k} in d {{ d[{k}] -= {step} }} else {{ d[{k}] = 1 }}")
    elif pick == 3:
        lines.append(f"acc = acc +% d.get({k}, 3)")
    elif pick == 4:
        lines.append(f"if i % {r.choice([5, 13, 97])} == 0 and {k} in d {{ d.remove({k}) }}")
    elif pick == 5:
        lines.append(f"let {k}b = {key_expr(r) if r.random() < 0.3 else lines[0].split(' = ', 1)[1]}")
        lines.append(f"if {k} is {k}b {{ same += 1 }}")
        lines.append(f"if {k} == {k}b {{ same += 1000 }}")
    elif pick == 6:
        lines.append(f"m[{k}] = m.get({k}, 0) + {step}")
        lines.append(f"if i % 11 == 0 {{ m[{k}] = {r.choice(['1.5', '-0.25', '2'])} }}")
    elif pick == 7:
        lines.append(f"t[{k}] = t.get({k}, 0) + {step}")
    elif pick == 8:
        ik = int_expr(r)
        lines.append(f"di[{ik}] = di.get({ik}, 0) + {step}")
        lines.append(f"acc = acc +% di.get({int_expr(r)}, 1)")
    elif pick == 9:
        lines.append(f"slen += {k}.len() + str({int_expr(r)}).len()")
    else:
        lines.append(f"d[{k}] = {step}")
        lines.append(f"acc = acc +% d[{k}]")
    return lines


def program(r):
    out = []
    nfn = r.randint(2, 4)
    for f in range(nfn):
        body = []
        for s in range(r.randint(2, 5)):
            body.extend(stmt(r, s))
        if r.random() < 0.5:
            body.append(f"if i % {r.choice([501, 2477])} == 0 {{ __prim__.gc_collect() }}")
        loop = r.choice(["while", "for"])
        out.append(f"fn body{f}(d: dict, m: dict, t: dict[str, int], di: dict, "
                   f"ps: list[str], reps: int) -> list[int] {{")
        out.append("    var acc = 0")
        out.append("    var same = 0")
        out.append("    var slen = 0")
        if loop == "while":
            out.append("    var i = 0")
            out.append("    while i < reps {")
        else:
            out.append("    for i in 0..reps {")
        out.append("        let p = ps[i % ps.len()]")
        out.append("        let q = ps[(i // 2) % ps.len()]")
        for b in body:
            out.append("        " + b)
        if loop == "while":
            out.append("        i += 1")
        out.append("    }")
        out.append("    return [acc, same, slen]")
        out.append("}")
        out.append("")
    pres = [r.choice(RUNS) for _ in range(r.randint(1, 3))]
    out.append("fn main() -> int {")
    out.append("    var d = {}")
    out.append("    var m = {}")
    out.append("    var t: dict[str, int] = {}")
    out.append("    var di = {}")
    out.append("    var ps = [" + ", ".join(f'"{p}"' for p in pres) + "]")
    out.append("    var round = 0")
    out.append(f"    while round < {r.randint(2, 4)} {{")
    for f in range(nfn):
        out.append(f"        print(body{f}(d, m, t, di, ps, {r.choice([700, 3000, 6000])}))")
    out.append("        ps.push(\"r\" + str(round))")
    out.append("        round += 1")
    out.append("    }")
    for name in ("d", "t", "di"):
        out.append(f"    var s{name} = 0")
        out.append(f"    for (_k, v) in {name}.items() {{ s{name} = s{name} +% v }}")
        out.append(f"    print({name}.len(), s{name})")
    out.append("    var sm = 0.0")
    out.append("    for (_k, v) in m.items() { sm = sm + v }")
    out.append("    print(m.len(), sm)")
    out.append("    return 0")
    out.append("}")
    return "\n".join(out) + "\n"


def run(path, envx, flags, timeout):
    env = dict(os.environ)
    env.update(envx)
    env["JAITHON_NO_GPU"] = "1"
    try:
        # --no-cache in every configuration: `is` on two equal run-time
        # strings is true only while the intern table is under its soft cap,
        # and a front end loaded from __jaicache__ interns less than one
        # compiled from source, so the cache's state alone moved the count.
        p = subprocess.run([JAITHON, *flags, "run", "--no-cache", path],
                           env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ("timeout", "", "")
    err = p.stderr.strip().splitlines()
    first = err[0] if err else ""
    last = err[-1] if err else ""
    # A compiled frame is never pushed, so tracebacks list different frames;
    # the exception itself is the first and last lines.
    return (p.returncode, p.stdout, first + " | " + last)


def check(seed, configs, timeout, keep, tmp):
    r = random.Random(seed)
    src = program(r)
    path = os.path.join(tmp, f"dictfmt_{seed}.jai")
    with open(path, "w") as fh:
        fh.write(src)
    results = [(name,) + run(path, envx, flags, timeout)
               for name, envx, flags in configs]
    ref = results[0]
    bad = [res for res in results[1:] if res[1:] != ref[1:]]
    if not bad and not keep:
        os.remove(path)
    return seed, path, ref, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--gc", action="store_true",
                    help="add a --gc-stress configuration (slow)")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    configs = CONFIGS + ([GC_CONFIG] if args.gc else [])
    tmp = tempfile.mkdtemp(prefix="jaidictfmt")
    seeds = range(args.seed, args.seed + args.count)
    errors = 0
    raised = 0
    hits = 0
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for seed, path, ref, bad in pool.map(
                lambda s: check(s, configs, args.timeout, args.keep, tmp), seeds):
            if ref[1] != 0:
                raised += 1
            if bad:
                hits += 1
                print(f"MISMATCH seed {seed}: {path}")
                print(f"  {ref[0]}: exit={ref[1]} err={ref[3]}")
                for name, code, out, err in bad:
                    print(f"  {name}: exit={code} err={err}")
                    for a, b in zip(ref[2].splitlines(), out.splitlines()):
                        if a != b:
                            print(f"    first differing line: {a!r} vs {b!r}")
                            break
    print(f"dictfmt differential: {args.count} programs, {len(configs)} "
          f"configurations, {raised} raised in the reference, "
          f"{hits} disagreement(s)")
    return 1 if hits or errors else 0


if __name__ == "__main__":
    sys.exit(main())
