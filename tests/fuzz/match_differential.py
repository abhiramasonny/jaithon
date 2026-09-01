#!/usr/bin/env python3
"""Differential fuzz for the enum-variant `match`.

Generates random enums and random `match` bodies over them, calls each body
with every subject the checker lets through -- including values that are not
enums at all -- and runs the whole program under six configurations. Any
disagreement between them is a miscompile.

The sixth configuration is JAITHON_JIT_MATCH=0, which puts the four `match`
opcodes back on the unarmed path: it is the one that says whether a
disagreement is the ARMS or something the interpreter and the tier already
disagreed about.

Its teeth are established rather than assumed. With the pointer half of
emitEnumTypeGuard removed -- so a `match` on one enum accepts a value of
another with the same tag, which is exactly what that compare is for -- it
reports 5 of 10 programs mismatched. Every generated program from the second
enum onward carries a value of the PREVIOUS enum among its subjects for that
reason; without those the same break goes unnoticed.

Out of `make test` because it is a few minutes of subprocesses. Run it when
touching the tier's `match` arms:  make match-fuzz
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JAITHON = os.path.join(ROOT, "jaithon")

CONFIGS = [
    ("baseline", {}),
    ("no-jit", {"JAITHON_NO_JIT": "1"}),
    ("threshold-1", {"JAITHON_JIT_THRESHOLD": "1"}),
    ("deopt-stress", {"JAITHON_JIT_DEOPT_STRESS": "1"}),
    ("tick-50", {"JAITHON_JIT_TICK_US": "50"}),
    ("match-off", {"JAITHON_JIT_MATCH": "0"}),
]

NAMES = ["Aa", "Bb", "Cc", "Dd", "Ee", "Ff", "Gg", "Hh", "Ii", "Jj", "Kk", "Ll"]


def gen(rng, i):
    n = rng.randint(1, len(NAMES))
    variants = NAMES[:n]
    lines = [f"enum E{i} {{ {', '.join(variants)} }}"]

    # A match over a random subset of the variants, in a random order, with
    # random alternation grouping, an optional guard arm and an optional
    # default.
    arms = []
    pool = variants[:]
    rng.shuffle(pool)
    taken = pool[: rng.randint(1, n)]
    idx = 0
    result = 1
    while idx < len(taken):
        group = taken[idx: idx + rng.randint(1, 3)]
        idx += len(group)
        pat = " | ".join(f"E{i}.{v}" for v in group)
        arms.append(f"        {pat} => {result},")
        result += 1
    guard = rng.random() < 0.3
    if guard and arms:
        at = rng.randrange(len(arms))
        arms.insert(at, f"        g if k > {rng.randint(-1, 1)} => 900,")
    has_default = rng.random() < 0.8 or len(taken) < n
    if has_default:
        arms.append("        _ => 0,")

    subject_any = rng.random() < 0.35
    nullable = (not subject_any) and rng.random() < 0.25
    if nullable:
        arms.insert(rng.randrange(len(arms) + 1), "        null => -7,")
    ty = "any" if subject_any else (f"E{i}?" if nullable else f"E{i}")
    lines.append(f"fn f{i}(v: {ty}, k: int) -> int {{")
    lines.append("    return match v {")
    lines.extend(arms)
    if not arms:
        lines.append("        _ => 0,")
    lines.append("    }")
    lines.append("}")

    subjects = [f"E{i}.{v}" for v in variants]
    if subject_any:
        subjects += ['"x"', "[1, 2]", "17", "3.5", "{1: 2}", "true"]
        #: A value of a DIFFERENT enum, which is the only subject the pointer
        #: half of the type guard can be caught getting wrong: it has the same
        #: Obj::type and, often, the same tag.
        if i > 0:
            subjects += [f"E{i-1}.Aa"]
    if nullable:
        subjects += ["null"]
    return "\n".join(lines), subjects, n


def program(rng, count):
    out = []
    warm = []
    show = []
    for i in range(count):
        body, subjects, _ = gen(rng, i)
        out.append(body)
        for s in subjects:
            for k in (-1, 0, 2):
                warm.append(f"        total += f{i}({s}, {k})")
                show.append(f"    print(f{i}({s}, {k}))")
    src = "\n\n".join(out)
    # 400 repetitions puts every body far past JAI_JIT_THRESHOLD; the printing
    # happens after the loop, so the transcript stays small while the bodies
    # still get hot. `total` is printed too, so a wrong answer inside the loop
    # shows up even where the transcript would not.
    src += "\n\nfn main() {\n    var total = 0\n"
    src += "    var rep = 0\n    while rep < 400 {\n"
    src += "\n".join(warm)
    src += "\n        rep += 1\n    }\n"
    src += "\n".join(show)
    src += "\n    print(total)\n}\n"
    return src


def run(path, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    env["JAITHON_NO_GPU"] = "1"
    p = subprocess.run([JAITHON, "run", "--no-cache", path], env=env,
                       capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout, p.stderr.strip()[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--per-program", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bad = 0
    done = 0
    tmp = tempfile.mkdtemp(prefix="jaimatchfuzz")
    while done < args.count:
        k = min(args.per_program, args.count - done)
        src = program(rng, k)
        done += k
        path = os.path.join(tmp, f"case{done}.jai")
        with open(path, "w") as fh:
            fh.write(src)
        results = []
        for name, envx in CONFIGS:
            results.append((name,) + run(path, envx))
        base = results[0]
        for name, code, out, err in results[1:]:
            if (code, out) != (base[1], base[2]):
                bad += 1
                print(f"MISMATCH {name} vs baseline in {path}")
                print(f"  baseline exit={base[1]} err={base[3]}")
                print(f"  {name} exit={code} err={err}")
                for a, b in zip(base[2].splitlines(), out.splitlines()):
                    if a != b:
                        print(f"  first differing line: {a!r} vs {b!r}")
                        break
                break
    print(f"match differential: {done} bodies in "
          f"{(done + args.per_program - 1)//args.per_program} programs, "
          f"{len(CONFIGS)} configurations, {bad} disagreement(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
