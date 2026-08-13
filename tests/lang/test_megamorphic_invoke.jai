#: One call site with more receiver classes than its inline cache has ways.
#:
#: Four ways hold four classes; the fifth marks the site IC_MEGA and it stops
#: caching entirely, re-resolving on every call. A shared table now answers for
#: those sites, keyed by (class, method name) and validated against the class's
#: own method-table version, so nothing has to remember to invalidate it.
#:
#: What that table can get wrong is which method a call reaches, and the answer
#: below depends on every one of them: two classes that share a method name,
#: one class asked for two different names, and a name that only some classes
#: have. A shared cache that keyed on the class alone, or on the name alone,
#: comes out with a different number here rather than with a crash.
#:
#: Run it under `JAITHON_MEGA_STRESS=1`, which is where its teeth are: nine
#: classes in five hundred slots never collide, so a cache keyed only on the
#: name passes this file exactly as a correct one does. The stress flag
#: collapses the table to a single entry, and then the key is the only thing
#: deciding anything. Verified by removing the class from the key: green
#: plain, red under the flag.
from std.test import assert_eq, assert_throws

trait Op {
    fn tag(self) -> int

    fn twice(self) -> int
}

class A: Op {
    pub fn tag(self) -> int { return 1 }

    pub fn twice(self) -> int { return 10 }
}

class B: Op {
    pub fn tag(self) -> int { return 2 }

    pub fn twice(self) -> int { return 20 }
}

class C: Op {
    pub fn tag(self) -> int { return 3 }

    pub fn twice(self) -> int { return 30 }
}

class D: Op {
    pub fn tag(self) -> int { return 4 }

    pub fn twice(self) -> int { return 40 }
}

class E: Op {
    pub fn tag(self) -> int { return 5 }

    pub fn twice(self) -> int { return 50 }
}

class F: Op {
    pub fn tag(self) -> int { return 6 }

    pub fn twice(self) -> int { return 60 }
}

class G: Op {
    pub fn tag(self) -> int { return 7 }

    pub fn twice(self) -> int { return 70 }
}

class H: Op {
    pub fn tag(self) -> int { return 8 }

    pub fn twice(self) -> int { return 80 }
}

#: Nine classes at one site, so the cache overflows however many ways it has;
#: the ninth also has a method the other eight do not.
class I: Op {
    pub fn tag(self) -> int { return 9 }

    pub fn twice(self) -> int { return 90 }

    pub fn only_mine(self) -> int { return 900 }
}

fn build() -> list {
    var xs: list = []
    #: Interleaved rather than grouped, so no site ever settles into a run of
    #: one class and a cache that only ever holds the last one still fails.
    for _r in 0..40 {
        xs.push(A())
        xs.push(F())
        xs.push(B())
        xs.push(I())
        xs.push(G())
        xs.push(C())
        xs.push(H())
        xs.push(D())
        xs.push(E())
    }
    return xs
}

#: Two megamorphic sites in one loop, over the same receivers. The two names
#: must not answer for each other.
fn total(xs: list) -> int {
    var t = 0
    for o in xs { t = t + o.tag() * 1000 + o.twice() }
    return t
}

fn warm(xs: list) -> void {
    for _r in 0..40 { total(xs) }
}

fn test_every_class_reaches_its_own_method() -> void {
    let xs = build()
    warm(xs)
    #: 40 rounds of one of each: sum(tag) = 45, sum(twice) = 450.
    assert_eq(total(xs), 40 * (45 * 1000 + 450))
}

fn test_a_name_only_one_class_has_still_resolves() -> void {
    let xs = build()
    warm(xs)
    var t = 0
    for o in xs {
        t = t + o.tag()
        if o is I { t = t + o.only_mine() }
    }
    assert_eq(t, 40 * 45 + 40 * 900)
}

#: `only_mine` exists on exactly one of the nine, so one site both resolves and
#: fails to resolve it; a cache answering by name alone would hand class A's
#: receiver class I's method.
fn all_only_mine(xs: list) -> int {
    var t = 0
    for o in xs { t = t + o.only_mine() }
    return t
}

fn test_a_missing_method_still_raises_after_the_site_is_megamorphic() -> void {
    let xs = build()
    warm(xs)
    assert_throws(AttributeError, || all_only_mine(xs))
}
