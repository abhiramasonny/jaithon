#: "Did you mean" suggestions for a name that resolved to nothing: a bounded
#: edit distance, and the diagnostic helper that turns the closest field or
#: member name into a `with_help`. A port of `jaiNameDistance` and
#: `jaiNameIsCloser` in `src/common/diag.c`, plus the member-suggestion half of
#: `declFindMember`'s caller in `src/sema/check_expr.c`.
#:
#: Nothing here takes a `Ctx`: the distance table depends only on the two
#: strings being compared, and `suggest_member` only on the declaration and the
#: diagnostic it is attaching to. `check/expr.jai` also runs this same search
#: over parameter names for an unknown keyword argument, inline, rather than
#: importing `suggest_member` itself — a parameter list is not a `TypeDecl`.
from jaithon.compile.check.ty import TypeDecl
from jaithon.compile.diag import Diagnostic

#: `JAI_SUGGEST_NO_MATCH` in src/common/diag.h: worse than any accepted distance.
pub const SUGGEST_NO_MATCH: int = 4

# `JAI_SUGGEST_MAX_LEN`: past this a name is not compared at all.
const SUGGEST_MAX_LEN: int = 64

# `jaiNameDistance` in src/common/diag.c: a bounded Damerau-Levenshtein, three
# rows because a transposition looks back two. The C compares bytes and this
# compares scalars, which agree over ASCII.
pub fn name_distance(a: str, b: str, max_distance: int) -> int {
    if max_distance < 0 { return max_distance + 1 }
    let la = a.len()
    let lb = b.len()
    if la > SUGGEST_MAX_LEN or lb > SUGGEST_MAX_LEN { return max_distance + 1 }
    # The length gap alone is already a lower bound on the distance.
    let gap = if la > lb { la - lb } else { lb - la }
    if gap > max_distance { return max_distance + 1 }

    var prev2: list[int] = []
    var prev: list[int] = []
    var cur: list[int] = []
    var column = 0
    while column <= lb {
        prev2.push(0)
        prev.push(column)
        cur.push(0)
        column += 1
    }

    var row = 1
    while row <= la {
        cur[0] = row
        var best = cur[0]
        column = 1
        while column <= lb {
            let cost = if a[row - 1] == b[column - 1] { 0 } else { 1 }
            var d = prev[column] + 1
            if cur[column - 1] + 1 < d { d = cur[column - 1] + 1 }
            if prev[column - 1] + cost < d { d = prev[column - 1] + cost }
            # Adjacent transposition costs one, not two.
            if row > 1 and column > 1 {
                let swapped = a[row - 1] == b[column - 2] and a[row - 2] == b[column - 1]
                if swapped and prev2[column - 2] + 1 < d { d = prev2[column - 2] + 1 }
            }
            cur[column] = d
            if d < best { best = d }
            column += 1
        }
        # Every remaining row can only add to the running minimum.
        if best > max_distance { return max_distance + 1 }
        var at = 0
        while at <= lb {
            prev2[at] = prev[at]
            prev[at] = cur[at]
            at += 1
        }
        row += 1
    }
    return prev[lb]
}

#: `jaiNameIsCloser`: the new best distance when `candidate` beats `best`, else
#: null. The tolerance is the one in `src/common/diag.h`.
pub fn closer(name: str, candidate: str?, best: int) -> int? {
    if candidate is null or name == candidate { return null }
    let length = if name.len() > 3 { name.len() } else { 3 }
    var limit = length // 3
    if limit > 3 { limit = 3 }
    let distance = name_distance(name, candidate, limit)
    if distance > limit or distance >= best { return null }
    return distance
}

#: Attach a "did you mean" to `d` when a field or member of `decl` is close to
#: `name`. Fields are searched before members, and the closest candidate under
#: the tolerance in `src/common/diag.h` wins. Nothing is attached otherwise.
pub fn suggest_member(d: Diagnostic, decl: TypeDecl, name: str) -> void {
    var best: str? = null
    var best_distance = SUGGEST_NO_MATCH
    for field in decl.fields {
        let distance = closer(name, field.name, best_distance)
        if distance is null { continue }
        best = field.name
        best_distance = distance
    }
    for member in decl.members {
        let distance = closer(name, member.name, best_distance)
        if distance is null { continue }
        best = member.name
        best_distance = distance
    }
    if best is not null { d.with_help(f"did you mean `{best}`?") }
}
