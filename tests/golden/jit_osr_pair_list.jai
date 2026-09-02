#: The pair loop head over a LIST of 2-tuples (OSR iterKind 4, behind
#: JAITHON_JIT_PAIR_LIST), and everything its entry decides from one sample.
#:
#: The head samples the element at the iterator's own index and pins both
#: component tags from it. What it must then get right: a component of another
#: kind later in the same list (the tag guard, not a misread payload); a bool
#: component read one byte wide, not eight; a list that shrinks under the loop;
#: and NULL components -- a `list[tuple[str, int?]]` where a null beside the
#: sampled int is a guard failure per element, so the head refuses past 1 null
#: in 64 and compiles below it. Every number here must match JAITHON_NO_JIT=1;
#: the tier is an accelerator that may always decline and never answer
#: differently.
fn walk(ps: list[any]) -> int {
    var acc = 0
    for (a, b) in ps {
        if a == b { acc = acc + 7 }
        acc = acc + 1
    }
    return acc
}

fn hot(ps: list[any], rounds: int) -> int {
    var acc = 0
    var r = 0
    while r < rounds {
        acc = acc + walk(ps)
        r = r + 1
    }
    return acc
}

fn build(n: int, every: int) -> list[tuple[str, int?]] {
    var out: list[tuple[str, int?]] = []
    var i = 0
    while i < n {
        if every > 0 and i % every == every - 1 {
            out.push((f"k{i % 7}", null))
        } else {
            out.push((f"k{i % 7}", i * 3))
        }
        i = i + 1
    }
    return out
}

fn sum_nullable(ps: list[tuple[str, int?]], rounds: int) -> int {
    var acc = 0
    var nulls = 0
    var r = 0
    while r < rounds {
        for (name, v) in ps {
            if v is null {
                nulls = nulls + 1
            } else {
                acc = acc + v + name.len()
            }
        }
        r = r + 1
    }
    return acc * 10 + nulls
}

fn main() {
    #: str/int throughout: this is what the head samples and compiles for.
    var strint: list[any] = []
    var i = 0
    while i < 300 {
        strint.push((f"k{i}", i))
        i = i + 1
    }
    print(hot(strint, 400))

    #: int/int, same head, same offset -- the component tag guard has to fire
    #: rather than read the str payload as an int.
    var intint: list[any] = []
    i = 0
    while i < 300 {
        intint.push((i, i))
        i = i + 1
    }
    print(hot(intint, 400))

    #: bool/float. A bool is ONE byte of the Value union.
    var boolfloat: list[any] = []
    i = 0
    while i < 300 {
        boolfloat.push((i % 2 == 0, i * 0.25))
        i = i + 1
    }
    print(hot(boolfloat, 400))

    #: Back to str/int, to prove the form is still usable after the misses.
    print(hot(strint, 400))

    #: Every kind interleaved inside ONE list, so the guard fires per element
    #: rather than per entry -- and the head's kind-density scan refuses it.
    var mixed: list[any] = []
    i = 0
    while i < 300 {
        if i % 4 == 0 {
            mixed.push((f"k{i}", i))
        } elif i % 4 == 1 {
            mixed.push((i, i))
        } elif i % 4 == 2 {
            mixed.push((i % 3 == 0, i * 0.5))
        } else {
            mixed.push((i * 1.5, f"v{i}"))
        }
        i = i + 1
    }
    print(hot(mixed, 400))

    #: A list that shrinks between rounds, so the bound guard is what stops the
    #: compiled step rather than the version guard.
    var shrink: list[any] = []
    i = 0
    while i < 600 {
        shrink.push((f"k{i}", i))
        i = i + 1
    }
    var acc = 0
    var r = 0
    while r < 400 {
        acc = acc + walk(shrink)
        if r % 50 == 49 { shrink.pop() }
        r = r + 1
    }
    print(acc)

    #: Nulls at 1 in 3: too dense, the head refuses, the loop runs interpreted.
    print(sum_nullable(build(3000, 3), 40))
    #: Nulls at 1 in 1000 -- the last element among them: sparse enough to
    #: compile, and each null is one guard failure and a re-entry.
    print(sum_nullable(build(3000, 1000), 40))
    #: No nulls at all.
    print(sum_nullable(build(3000, 0), 40))
}
