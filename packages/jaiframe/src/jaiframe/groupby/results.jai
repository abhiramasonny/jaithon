#: The vocabulary around a call: what an argument names, where it is defined,
#: and what a callback's answer becomes.
#:
#: `as_agg` and `supports` are asked before any dispatch, `as_bool` reads what
#: a `filter` predicate said, and `stack_frames` is what an `apply` hands back.
from ..column import Agg, Column, DataFrame, Dtype, Index, Value
from ..column import concat_columns

#: The aggregation a name or an `Agg` stands for.
pub fn as_agg(value: any) -> Agg {
    if value is Agg { return value }
    if value is str { return Agg.from_name(value) }
    throw TypeError("an aggregation is an Agg or its pandas name")
}

#: Whether an aggregation is defined over a dtype.
#:
#: `min` and `max` refuse text. The codes of a dictionary column are assignment
#: order, not alphabetical order, so comparing them would answer a question
#: nobody asked; sorting the level table is the business of whoever owns it.
#: `first` and `last` are positional and work over anything.
pub fn supports(dtype: Dtype, agg: Agg) -> bool {
    return match agg {
        Agg.Count | Agg.Size | Agg.First | Agg.Last | Agg.Nunique => true,
        Agg.Min | Agg.Max => dtype.is_numeric() or dtype == Dtype.Datetime,
        _ => dtype.is_numeric(),
    }
}

#: The truth value a `filter` callback answered with.
pub fn as_bool(value: any) -> bool {
    if value is bool { return value }
    if value is Value { return not value.is_na() and value.to_bool() }
    throw TypeError("a group filter must answer with a bool")
}

#: The frames an `apply` produced, stacked.
#:
#: The result gets a fresh range index. pandas builds a hierarchical one from
#: the group keys and the pieces' own labels; that is a multi-index
#: construction, which belongs to `reshape.jai` and not to a call that has
#: already spent a host loop.
pub fn stack_frames(pieces: list[DataFrame]) -> DataFrame {
    let names = [name for name in pieces[0].names]
    var columns: list[Column] = []
    for name in names {
        var parts: list[Column] = []
        for piece in pieces { parts.push(piece.column(name)) }
        columns.push(concat_columns(parts))
    }
    var rows = 0
    for piece in pieces {
        rows += piece.len()
        piece.free()
    }
    return DataFrame(names, columns, index: Index.range(rows))
}
