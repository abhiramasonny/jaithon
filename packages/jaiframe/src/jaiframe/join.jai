#: Combining frames: hash joins across the rows, concatenation along both axes.
#:
#: `merge` is the centre of it. `join`, the index alignment `concat(axis: 1)`
#: needs and `compare` are that machinery pointed at a different pair of key
#: columns. The two ordered merges are the exception: `merge_asof` and
#: `merge_ordered` bisect a sorted key lane instead, because a bucket answers
#: "which rows hold this key" and an asof merge is not asking that.
#:
#: A merge is a hash join, never a nested loop. Two frames of a million rows
#: are 10^12 key comparisons that way, which is hours; hashing both sides and
#: probing is two million hashes and about as many bucket reads. The nested
#: loop wins only when one side has a handful of rows, and it loses so badly
#: past that it is not worth the second code path.
#:
#: The table is built over whichever side has fewer rows and the other side
#: probes it. The table costs a bucket per build row — three lanes of it — so
#: building the smaller side is what keeps a join against a small dimension
#: table cheap in memory, and it leaves the long side streaming through the
#: probe, one bucket read per row, which is the access pattern a GPU wants.
#: The cost is that the output then comes out in probe order, and pandas' order
#: is the left frame's; when the probe is not the side the order follows, the
#: two position lanes are put back in order by a stable radix pass before a
#: single column is materialised. Reordering two lanes is far cheaper than
#: building the wrong table.
#:
#: The whole join is two lanes: left positions and right positions, with
#: `MISSING_POSITION` where a side has no row. Every result column is one
#: `Column.take(..., allow_missing: true)` off those lanes, so the width of the
#: frame costs a dispatch per column and nothing else.
#:
#: ```
#: let orders = pd.DataFrame(["city", "total"], [cities, totals])
#: let sizes = pd.DataFrame(["city", "population"], [names, people])
#: let joined = merge(orders, sizes, how: How.Left, on: "city")
#: defer { joined.free() }
#: ```
#:
#: The work is in the sibling directory: `join.kernels` holds the Metal source,
#: `join.types` the `How` / `Validate` / `AsofDirection` vocabulary,
#: `join.pairing` the hash table and the position lanes it produces,
#: `join.merge` the merges built on them, `join.concat` the stacking and the
#: comparison, and `join.ordered` the two merges that bisect a sorted key.
from .join.types import AsofDirection, How, Validate
from .join.pairing import Pairing, key_order, pair_rows
from .join.merge import fill_forward, join, merge
from .join.concat import append, compare, concat
from .join.ordered import merge_asof, merge_ordered

export { How, Validate, AsofDirection }
export { Pairing, pair_rows, key_order }
export { merge, join, fill_forward }
export { concat, append, compare }
export { merge_asof, merge_ordered }
