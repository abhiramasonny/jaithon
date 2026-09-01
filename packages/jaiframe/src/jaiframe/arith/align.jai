#: Two indexes put on one set of labels, and where each side's rows land.
from std.gpu import Buffer

from ..column import Column, Index
from ..column import MISSING_POSITION
from ..column import concat_columns, positions_buffer, unify_levels
from .cells import column_cells

#: One collision-free host key per row of an index.
#:
#: Length-prefixed so that a two-level label of ("a", "bc") cannot read as
#: ("ab", "c"), and present-tagged so that a null level is not the string it
#: would print as.
fn index_keys(index: Index) -> list[str] {
    let levels = index.materialize()
    var keys: list[str] = []
    for _row in 0..index.len() { keys.push("") }
    for level in levels {
        for (row, cell) in column_cells(level).enumerate() {
            let part = if cell.is_na() { "!" } else { "=" + cell.to_str() }
            keys[row] = keys[row] + f"{part.len()}:{part}"
        }
    }
    for level in levels { level.free() }
    return keys
}

#: Where each side's rows land once two indexes are put on one set of labels.
class Alignment {
    #: The labels the result carries.
    pub var index: Index
    #: How many rows the result has.
    pub var length: int
    #: Source row per result row for the left side, `null` when it already
    #: lines up and nothing has to be gathered.
    pub var left: Buffer?
    #: The same for the right side.
    pub var right: Buffer?

    pub fn init(self, index: Index, length: int, left: Buffer?, right: Buffer?) {
        self.index = index
        self.length = length
        self.left = left
        self.right = right
    }

    #: Whether both sides already sit on the result's labels.
    pub fn is_identity(self) -> bool { return self.left is null and self.right is null }

    #: Release the position lanes. The index is handed to the result, so it
    #: outlives this and is not freed here.
    pub fn free_positions(self) -> void {
        let a = self.left
        if a is not null { a.free() }
        let b = self.right
        if b is not null { b.free() }
    }
}

#: Whether two indexes label the same rows without anything being compared.
#:
#: The same object is the common case — `frame["a"] + frame["b"]` share one —
#: and two default range indexes are the next. Anything else goes through the
#: label pass, which will discover an equality it could not have assumed.
fn indexes_agree(a: Index, b: Index) -> bool {
    if a == b { return true }
    if a.len() != b.len() { return false }
    if a.is_range() and b.is_range() { return a.start == b.start and a.step == b.step }
    return false
}

fn shared_names(a: Index, b: Index) -> list[str?] {
    var out: list[str?] = []
    for (at, name) in a.names.enumerate() {
        out.push(if at < b.names.len() and name == b.names[at] { name } else { null })
    }
    return out
}

#: Two indexes put on the union of their labels.
#:
#: The union is in first-appearance order — the left index, then the labels
#: only the right has — which is the order `unify_levels` and `concat_columns`
#: already use. pandas sorts it, and would then need the labels to be
#: orderable, which a multi-index of mixed dtypes is not.
#:
#: The matching is a host pass over the labels: one download per index level,
#: then a dictionary of keys. That is a pass over the rows, and it is why the
#: identity check above matters. The union index itself is built on the device,
#: by taking from the two label sets stacked end to end.
#:
#: This is the alignment the whole module runs on, and it belongs with the
#: index rather than with arithmetic; when `index.jai` exports `align`, this
#: function becomes a call into it.
pub fn align_indexes(a: Index, b: Index) -> Alignment {
    if indexes_agree(a, b) { return Alignment(a.copy(), a.len(), null, null) }
    if a.nlevels() != b.nlevels() {
        throw ValueError(f"an index of {a.nlevels()} levels cannot align with one of {b.nlevels()}")
    }
    let left_keys = index_keys(a)
    let right_keys = index_keys(b)
    var slot: dict[str, int] = {}
    var left_at: list[int] = []
    var right_at: list[int] = []
    for (row, key) in left_keys.enumerate() {
        if slot.get(key, -1) >= 0 {
            throw ValueError("cannot align on an index with duplicate labels")
        }
        slot[key] = left_at.len()
        left_at.push(row)
        right_at.push(MISSING_POSITION)
    }
    var placed: dict[str, int] = {}
    for (row, key) in right_keys.enumerate() {
        if placed.get(key, -1) >= 0 {
            throw ValueError("cannot align on an index with duplicate labels")
        }
        placed[key] = 1
        let at = slot.get(key, -1)
        if at >= 0 {
            right_at[at] = row
            continue
        }
        slot[key] = left_at.len()
        left_at.push(MISSING_POSITION)
        right_at.push(row)
    }
    let length = left_at.len()
    #: Same labels in the same order after all, which a stored index cannot be
    #: assumed into but is common enough to be worth not gathering for.
    if length == a.len() and length == b.len() and left_at == right_at {
        return Alignment(a.copy(), length, null, null)
    }
    let a_levels = a.materialize()
    let b_levels = b.materialize()
    var stacked: list[Column] = []
    for (at, level) in a_levels.enumerate() {
        stacked.push(concat_columns([level, b_levels[at]]))
    }
    var reach: list[int] = []
    for (at, from_left) in left_at.enumerate() {
        reach.push(if from_left >= 0 { from_left } else { a.len() + right_at[at] })
    }
    let lane = positions_buffer(reach)
    let index = Index.of(stacked.map(|level| level.take(lane, length)), shared_names(a, b))
    lane.free()
    for level in stacked { level.free() }
    for level in a_levels { level.free() }
    for level in b_levels { level.free() }
    return Alignment(index, length, positions_buffer(left_at), positions_buffer(right_at))
}

#: The column a side contributes to an aligned operation.
pub fn aligned_column(source: Column, positions: Buffer?, length: int) -> (Column, bool) {
    if positions is null { return (source, false) }
    return (source.take(positions, length, allow_missing: true), true)
}

#: pandas keeps a name only when both sides agree on it.
pub fn shared_name(a: str?, b: str?) -> str? { return if a == b { a } else { null } }
