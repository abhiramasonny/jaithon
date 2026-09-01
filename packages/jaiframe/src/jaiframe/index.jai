#: What an `Index` does once it has to answer questions about its labels:
#: where a label sits, which labels two indexes share, and how to put two
#: frames on one set of rows.
#:
#: Every row lookup goes through an `IndexLookup`, which picks one of three
#: plans and never builds more machinery than the labels need:
#:
#: - A range index is arithmetic. `start + i * step` inverts to a division, so
#:   the default index of every frame answers a lookup without storing a label,
#:   let alone a table.
#: - A monotonic single-level index bisects. The labels are already in the
#:   order a search wants, and standing 2n slots of hash table beside an index
#:   to answer what a bisect answers would cost more memory than the index.
#: - Anything else — unsorted, or a multi-index — gets one open-addressed table
#:   over the per-row hash `Column.hash_into` folds across the levels. A
#:   32-bit hash agreeing is not two labels agreeing, so every candidate the
#:   walk stops on is confirmed with `Column.equal_rows`, and the rare
#:   disagreement resumes the walk from where it stopped. The table is exact,
#:   not probabilistic.
#:
#: The table is built once and reused for as many probes as the caller makes,
#: which is why `IndexLookup` is a value you can hold rather than something
#: hidden inside `get_loc`. Building it costs a handful of dispatches and one
#: short download per settling round — normally one round.
#:
#: Text levels compare by dictionary code, so a text index is monotonic in the
#: order its levels were interned, not in alphabetical order. That is what
#: makes a string lookup an integer compare; `sort_index` in `keys.jai` is
#: where alphabetical order comes from, and sorting deliberately does not live
#: here. Set operations keep first-appearance order for the same reason:
#: pandas sorts its unions, and a caller who wants that sorts afterwards.
#:
#: ```
#: let joint = union(left.index, right.index)
#: defer { joint.free() }
#: let both = align(left, right, "outer")
#: let stacked = set_index(frame, ["city", "year"])
#: ```
#:
#: The work is in the sibling directory: `index.shader` and `index.kernels`
#: hold the Metal source and the plumbing that dispatches it, `index.labels`
#: the host-side label helpers, `index.lookup` the three plans and
#: `IndexLookup`, `index.setops` the set operations and the comparisons,
#: `index.align` the reindexing and the alignment, and `index.levels` the
#: multi-index constructors and the level moves.
from .index.lookup import IndexLookup, NOT_FOUND, PLAN_HASHED, PLAN_RANGE, PLAN_SORTED
from .index.lookup import SIDE_LEFT, SIDE_RIGHT, get_indexer, get_indexer_for_index
from .index.lookup import get_loc, get_loc_tuple, is_monotonic_decreasing
from .index.lookup import is_monotonic_increasing, is_unique, searchsorted
from .index.setops import append, difference, equals, identical, intersection
from .index.setops import symmetric_difference, union, unique
from .index.align import METHOD_BACKFILL, METHOD_NEAREST, METHOD_PAD, align
from .index.align import align_frame_series, align_series, joint_index, reindex
from .index.align import reindex_indexer, reindex_series
from .index.levels import droplevel, from_arrays, from_product, from_tuples
from .index.levels import get_level_values, rename_axis, reorder_levels, reset_index
from .index.levels import set_index, set_names, swaplevel, xs

export { NOT_FOUND, METHOD_PAD, METHOD_BACKFILL, METHOD_NEAREST }
export { SIDE_LEFT, SIDE_RIGHT, PLAN_RANGE, PLAN_SORTED, PLAN_HASHED }
export { IndexLookup, is_monotonic_increasing, is_monotonic_decreasing, is_unique }
export { get_loc, get_loc_tuple, get_indexer, get_indexer_for_index, searchsorted }
export { unique, append, union, intersection, difference, symmetric_difference }
export { equals, identical, joint_index }
export { reindex, reindex_series, align, align_series, align_frame_series }
export { reindex_indexer }
export { from_arrays, from_tuples, from_product }
export { droplevel, swaplevel, reorder_levels, get_level_values, set_names }
export { rename_axis, xs, set_index, reset_index }
