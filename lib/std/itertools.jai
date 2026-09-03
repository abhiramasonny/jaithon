#: CPython's `itertools`, as lazy iterators over `std.iter`'s `Iter`.
#:
#: Every name CPython 3.13 exports from `itertools` is here, spelled the same,
#: and every one of them answers an `Iter` — so the whole `std.iter` surface is
#: available on the result and an itertools pipeline and an `Iter` pipeline are
#: the same pipeline:
#:
#: ```
#: from std.itertools import islice, count, groupby
#: islice(count(1), 5).map(|n| n * n).collect()   # [1, 4, 9, 16, 25]
#: ```
#:
#: `std.iter` already had most of the *ideas* under Rust's names — `take_while`
#: for `takewhile`, `drop_while` for `dropwhile`, `chunks` for `batched`,
#: `windows(2)` for `pairwise`, `cross` for a two-way `product`, `Iter.chain`,
#: `Iter.permutations` and `Iter.combinations`. Nothing here re-implements one
#: of those; where the semantics already matched, the CPython name is a wrapper
#: a line or two long over the `std.iter` adapter. What is genuinely new is
#: `accumulate`, `chain.from_iterable`, `compress`, `filterfalse`, an
#: n-ary `product` with `repeat`, `combinations_with_replacement`, `islice`'s
#: three-argument slicing, `starmap`, `tee`, `zip_longest`, and a `groupby`
#: with CPython's shared-view groups rather than `std.iter`'s eager lists.
#:
#: ## Where this deviates from CPython, and why
#:
#: **Null ends a sequence.** `Iter.next` signals exhaustion by returning null,
#: so no lazy sequence in Jaithon can carry a null element. An input containing
#: null therefore stops there, and `repeat(null, 3)` yields nothing rather than
#: three nulls. Null is fine *inside* a yielded value — `zip_longest`'s default
#: fill really is null, and the tuples it builds carry it — it is only a null
#: element of a stream that cannot be represented.
#:
#: **Three names are classes.** CPython reaches part of `chain`, `product` and
#: `zip_longest` through spellings a Jaithon signature cannot carry:
#: `chain.from_iterable` is an attribute of the callable, and `product`'s
#: `repeat` and `zip_longest`'s `fillvalue` are keyword-only *after*
#: `*iterables`, which Jaithon rejects as E0112 — a parameter may not follow a
#: rest parameter. All three keep the CPython spelling for the ordinary call
#: and move the rest onto a static method: `chain.from_iterable(xs)`,
#: `product.repeated(n, ...)`, `zip_longest.with_fill(value, ...)`.
#:
#: **`batched`'s `strict` is positional-or-named**, not keyword-only, which
#: Jaithon has no way to declare.
#:
#: **Error messages, not error types, differ.** `ValueError` and `TypeError`
#: are raised where CPython raises them; `permutations` and `combinations`
#: report a negative `r` through `std.iter`'s wording rather than CPython's.
#:
#: Facade: imports and re-exports every sibling module in `itertools/`.
from .itertools.adapters import accumulate, batched, chain, compress, dropwhile
from .itertools.adapters import filterfalse, groupby, islice, pairwise, starmap
from .itertools.adapters import takewhile, tee, zip_longest
from .itertools.combinatoric import combinations, combinations_with_replacement
from .itertools.combinatoric import permutations, product
from .itertools.infinite import count, cycle, repeat

export { accumulate, batched, chain, combinations, combinations_with_replacement, compress, count, cycle, dropwhile, filterfalse, groupby, islice, pairwise, permutations, product, repeat, starmap, takewhile, tee, zip_longest }
