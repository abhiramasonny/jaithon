#: A body whose return sites disagree -- an int on one edge, a list on
#: another, a float, a bool, an instance, null, a string -- compiles to
#: SLOT_DYNAMIC: each site hands its own Value tag up beside the value, and
#: the entry point rebuilds the Value from the tag rather than from one kind
#: settled at compile time. This is the shape of jaithon.ast.schema's
#: _default_field, the callee that kept ast.node.init interpreted.
#:
#: Every kind a site can carry is returned here, and each is read back the
#: way the interpreter would read it, so a wrong tag or a payload read too
#: wide shows as a diff rather than as a crash somewhere later. The float
#: sites include a NaN and a negative zero, whose bits a conversion could
#: canonicalise. The nullable sites return the same local as an instance on
#: one call and null on the next, so the tag has to come off the payload.

import std.math as math

class Pt {
    pub var x: int
    fn init(self, x: int) { self.x = x }
}

fn pick(tag: str, n: int) -> any {
    if tag == "int" { return -1 }
    if tag == "list" { return [] }
    if tag == "bool" { return n % 2 == 0 }
    if tag == "float" { return 0.5 }
    if tag == "nan" { return math.NAN }
    if tag == "negzero" { return -0.0 }
    if tag == "inst" { return Pt(n % 2) }
    if tag == "str" { return "s" }
    if tag == "zero" { return 0 }
    if tag == "false" { return false }
    return null
}

fn maybe(p: Pt?, n: int) -> any {
    if n < 0 { return -n }
    return p
}

fn bare(n: int) -> any {
    if n > 5 { return n * 2 }
    return
}

fn describe(v: any) -> str {
    if v == null { return "null" }
    if v is bool { return "bool:" + str(v) }
    if v is int { return "int:" + str(v) }
    if v is float {
        if v != v { return "float:nan" }
        if v == 0.0 and str(v) == "-0.0" { return "float:-0.0" }
        return "float:" + str(v)
    }
    if v is str { return "str:" + v }
    if v is list { return "list:" + str(len(v)) }
    if v is Pt { return "Pt:" + str(v.x) }
    return "other"
}

fn main() {
    let tags = ["int", "list", "bool", "float", "nan", "negzero", "inst",
                "str", "zero", "false", "none"]
    var seen: dict[str, str] = {}
    # Past the function-tier threshold, and long enough for the OSR tier to
    # take the loop, so both tiers see the callee.
    for i in 0..200 {
        for t in tags {
            let d = describe(pick(t, i))
            let key = t + "/" + str(i % 2)
            if key in seen {
                if seen[key] != d { print("MISMATCH " + key + " " + seen[key] + " " + d) }
            } else {
                seen[key] = d
            }
        }
        let p = Pt(i)
        let a = describe(maybe(p, i))
        let b = describe(maybe(null, i))
        let c = describe(maybe(p, -i - 1))
        let e = describe(bare(i % 10))
        if i < 3 or i > 197 {
            print(str(i) + " " + a + " " + b + " " + c + " " + e)
        }
    }
    let keys = sorted(seen.keys())
    for k in keys { print(k + " -> " + seen[k]) }
}
