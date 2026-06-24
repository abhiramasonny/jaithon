# Jaithon C-builtin audit

Every `_`-prefixed builtin is registered in `src/lang/parser.c` (the
`setVariable("_x", makeNativeFunc(nativeX))` block) and implemented as a
`static Value nativeX(Value*, int)` in the same file (a few live in
`src/core/parallel.c` / `src/core/gpu.m`).

This document classifies each one:

- **Irreducible** — needs C: it is the hardware/OS/memory boundary, or a
  primitive that everything else is built from. Cannot be written in Jaithon
  without already having an equivalent primitive.
- **Bootstrapped** — now implemented (or fully implementable) in Jaithon on top
  of the irreducible primitives. See `lib/modules/`.
- **Perf/syntax primitive** — *could* be bootstrapped, but is kept in C on
  purpose because (a) it is fused into language syntax (`[..]`, `a[i]`,
  `d[key]`) in the parser/VM, and/or (b) it is on the hot path and a Jaithon
  reimplementation would be far slower. The bootstrapped equivalent is provided
  as a library so the technique is demonstrated.

---

## Irreducible (must stay in C)

| Builtin | Why irreducible |
|---|---|
| `_cell _car _cdr _setcar _setcdr` | The cons-cell memory primitive. The root of all heap structure. |
| `_buf _bget _bset _blen` | Fixed-size mutable memory buffer — the one new memory primitive added for this work. Dynamic arrays & hash tables are bootstrapped on it (see below). |
| `_ord _chr` | char ↔ integer code. Cannot derive a character's numeric value without a primitive that reads the byte. |
| `_fopen _fclose _fread _fwrite _input _system` | OS / syscalls (file IO, stdin, shell). |
| `_time _rand` | OS clock / entropy. |
| `_type` | Runtime type introspection — reads the `ValueType` tag, which is a C-level fact. |
| `_sin _cos _sqrt _log _exp` | libm. Could be reimplemented as Taylor/CORDIC series in Jaithon, but at a large precision and speed cost; kept as the numeric primitive. (`tan` is **not** here — see Bootstrapped.) |
| `_pmap _pfor` | The general threading core (pthreads + thread-local module isolation, `src/core/parallel.c`). Threads are an OS/runtime primitive. These replaced the old eval-specific `_peval_pop`. |
| `_gpu_matmul _gpu_matmul_batched _gpu_available` | Metal GPU (`src/core/gpu.m`). Hardware. |

---

## Bootstrapped in Jaithon

| Was / is | Now | Where |
|---|---|---|
| `_peval_pop` (was a dedicated C builtin) | **removed from C**; reimplemented as `peval_pop(pop, fn)` = `_pmap(pop, fn)` | `lib/modules/core/parallel.jai` |
| dynamic array growth (the user's `_push` objection) | `darrNew/darrPush/darrPop/darrGet/darrSet` — amortized-doubling logic in Jaithon, only `_buf`/`_bget`/`_bset` underneath | `lib/modules/ds/dynarray.jai` |
| hash map (the linear-scan `HashMap` was O(n)) | `htNew/htSet/htGet/htHas` — open-addressing O(1) hash table, djb2 via `_ord`, built on `_buf` | `lib/modules/ds/hashtable.jai` |
| `tan` | `sin(x)/cos(x)` wrapper (the `_tan` native is now unnecessary for the public `tan`) | `lib/modules/core/core.jai` |
| `split/trim/lines/...` string helpers | already pure Jaithon over `_charAt`/`_substr`/`_concat` | `lib/modules/core/string.jai` |
| public wrappers `len/str/num/concat/...` | already thin Jaithon wrappers | `lib/modules/core/core.jai` |

---

## Perf / syntax primitives (kept in C by design)

These are the ones whose *logic* is simple but which are deliberately native.

| Builtin | Reason kept | Bootstrapped equivalent |
|---|---|---|
| `_array _push _apush _pop _get _set _alen` | `[..]` literals and `a[i]` / `a[i]=x` indexing are emitted by the parser/VM directly against the native `JaiArray` (`src/vm/vm.c`). The native array is itself a growable `Value` buffer — effectively the irreducible array primitive — and it is on every hot path. Rerepresenting arrays as a Jaithon `{buf,len}` object would mean every `a[i]` becomes an interpreted function call: correct but a large slowdown, directly against the project's performance goals. | `ds/dynarray.jai` shows the full bootstrapped version on `_buf`. |
| `_dict _dict_has _dict_keys _dict_values _dict_del` | `d[key]` indexing is wired into the VM against the native `JaiObject` (`src/vm/vm.c:541`). Same syntax-fusion argument as arrays. (Note the native version is currently an O(n) linear scan.) | `ds/hashtable.jai` is a real O(1) bootstrapped replacement (library API, not `d[key]` syntax). |
| `_len` | O(1) for strings via C `strlen`; also dispatches array/object length. A pure-Jaithon string length would be an O(n) `charAt` loop on a very common call. | `len()` wrapper in `core.jai` already delegates; array/object length is just `_alen`/field count. |
| `_charAt _substr _concat` | Strings are the native `char*` primitive; these are the primitive string ops. | n/a (they are the string primitive layer). |

---

## Conversions (mixed)

`_str _num _int _float _double _bool _char _long _short _byte` read/write the
C-level `ValueType` tag and do numeric formatting/parsing.

- `_num` (string→number) and integer `_str` (number→string) are bootstrappable
  via `_ord`/`charAt` digit loops.
- Float formatting and the typed-number tags (`_float`/`_double`/`_long`/…)
  depend on the C numeric representation, so the tag-producing casts stay in C.

The public `num()/str()/toInt()/…` wrappers in `core.jai` are already Jaithon.

---

## Summary

- **One** new irreducible memory primitive was added (`_buf` family) plus
  `_ord`/`_chr`; on top of these, **dynamic arrays and a hash map are now fully
  bootstrapped in Jaithon**.
- The eval-specific `_peval_pop` C builtin was **deleted** and rebuilt in
  Jaithon on the new general `_pmap`/`_pfor` threading core.
- The native indexable `array`/`dict` and the `libm`/OS/GPU builtins remain in
  C — documented above with the specific reason (syntax fusion, hot path, or
  hardware boundary) rather than left implicit.
