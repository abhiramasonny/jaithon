#: Every way of reading or writing a subset of a frame: `iloc` by position,
#: `loc` by label, boolean masks, the scalar fast paths, and item assignment.
#:
#: All five selection shapes — one row, a list, a slice with a step, a boolean
#: mask, everything — are one class, `Pick` for positions and `Labels` for
#: labels, rather than one function apiece. pandas spells them all `df.iloc[]`
#: and dispatches on the Python type of the subscript; a named constructor per
#: shape says the same thing without the guessing, and it is what lets `iloc`,
#: `loc`, `set_iloc` and `set_loc` share one resolver.
#:
#: A selection resolves to a positions lane on the device and then to one
#: `take` per column. That is one dispatch per column whatever the selection
#: was, and no step anywhere walks the rows on the host. The mask path scans
#: once for the whole frame rather than calling `Column.compress` per column,
#: which would repeat the scan `width` times over the same flags.
#:
#: The one exception is `query`, whose predicate is a Jaithon callback and so
#: has to run once per row on the host. It is here because a predicate is
#: sometimes the only way to say what you mean; `query_mask` is the same
#: selection driven by a `Bool` column, and that is the path to reach for.
#:
#: ```
#: let top = head(frame, 3)
#: let adults = query_mask(frame, between(frame.column("age"), Value.of_int(18), Value.of_int(65)))
#: let one = iat(frame, 0, 1)
#: ```
from .access.pick import Pick, SelectKind
from .access.labels import Labels
from .access.select import at, at_series, filter_rows, filter_rows_series, first
from .access.select import head, head_series, iat, iat_series, iloc, iloc_series
from .access.select import last, loc, loc_series, nth, sample, sample_series, tail
from .access.select import tail_series, take, take_series
from .access.predicate import between, filter_names, frame_isin, frame_mask
from .access.predicate import frame_where, isin, mask, where
from .access.query import Row, query, query_flags, query_mask
from .access.assign import drop_rows, insert_rows, set_column_values, set_iloc
from .access.assign import set_loc

export { SelectKind, Pick, Labels, Row }
export { iloc, iloc_series, loc, loc_series, iat, at, iat_series, at_series }
export { filter_rows, filter_rows_series, take, take_series }
export { head, tail, head_series, tail_series, nth, first, last }
export { sample, sample_series }
export { filter_names, isin, frame_isin, between, where, mask }
export { frame_where, frame_mask }
export { query_flags, query, query_mask }
export { set_iloc, set_loc, set_column_values, insert_rows, drop_rows }
