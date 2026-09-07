#: Dict, string, slice, tuple and list-method edges, taken inside JIT-warmed
#: bodies. The two tiers must agree exactly.
#:
#: Third of the tier-agreement set: jit_tier_agreement.jai covers arithmetic and
#: indexing, jit_try_tier_agreement.jai covers who CATCHES a raise from compiled
#: code, and this one covers the container surface -- a different set of arms
#: (jit_body_index.c's dict and string paths, jit_list.c's storage dispatch,
#: emitGetSlice) with their own guards and their own deopt sites.
#:
#: The edges chosen are the ones whose two tiers could plausibly disagree: a
#: missing dict key (raises) against a present one (an inline-cached read), a
#: slice whose end runs past the container, a REVERSED slice that must come back
#: empty rather than negative-length, a negative slice, `find` returning -1
#: rather than raising, tuple indexing out of range, and popping an empty list.
#:
#: Every body is called 400 times before the edge value arrives, so the edge is
#: taken in compiled code. Output must be byte-identical under JAITHON_NO_JIT=1;
#: the golden runner also re-runs it at JAITHON_JIT_THRESHOLD=1 and under
#: DEOPT_STRESS.
fn d_get(d: dict[str, int], k: str) -> int { return d[k] }
fn d_has(d: dict[str, int], k: str) -> bool { return k in d }
fn s_slice(s: str, a: int, b: int) -> str { return s[a:b] }
fn l_slice(xs: list[int], a: int, b: int) -> list[int] { return xs[a:b] }
fn t_idx(t: tuple[int, str], i: int) -> any { return t[i] }
fn s_find(s: str, n: str) -> int { return s.find(n) }
fn l_pop(xs: list[int]) -> int { return xs.pop() }
fn s_upper(s: str) -> str { return s.upper() }
fn l_join(xs: list[str], sep: str) -> str { return sep.join(xs) }

fn main() {
    var d: dict[str, int] = {}
    d["a"] = 1
    d["b"] = 2
    var xs: list[int] = []
    for k in 0..6 { xs.push(k * 10) }
    var ss: list[str] = []
    for k in 0..4 { ss.push("x") }
    let t = (7, "seven")
    let abc = "abcdef"

    var w = 0
    var i = 0
    while i < 400 {
        w = w + d_get(d, "a") + s_find(abc, "cd") + l_slice(xs, 1, 3).len()
        w = w + s_slice(abc, 1, 3).len() + s_upper(abc).len()
        w = w + l_join(ss, "-").len()
        if d_has(d, "b") { w = w + 1 }
        i = i + 1
    }
    print(f"warm {w}")

    try { print(f"missing {d_get(d, "zz")}") } catch _e: KeyError { print("missing KeyError") }
    print(f"has-no {d_has(d, "zz")}")
    print(f"slice-past {l_slice(xs, 2, 99).len()}")
    print(f"slice-neg {l_slice(xs, -3, -1).len()}")
    print(f"sslice-past {s_slice(abc, 2, 99)}")
    print(f"sslice-rev {s_slice(abc, 4, 2)}")
    print(f"find-absent {s_find(abc, "zz")}")
    print(f"tup0 {t_idx(t, 0)}")
    try { print(f"tup9 {t_idx(t, 9)}") } catch _e: IndexError { print("tup9 IndexError") }
    var ys: list[int] = []
    try { print(f"pop-empty {l_pop(ys)}") } catch _e: IndexError { print("pop-empty IndexError") }
}
