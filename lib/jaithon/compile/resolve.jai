#: Scope tree, symbol table, slot and upvalue assignment over a `jaithon.ast` tree.
#:
#: Counterpart of `src/sema/resolve.c`; `make bootstrap` diffs the two front
#: ends' bytecode, so a slot numbered differently here is a divergence:
#:
#: - Two passes per block: mutually-recursive declarations (`fn`, `class`,
#:   `trait`, `enum`, `type`, `import`) hoist first, then statements walk in
#:   order, so `let`/`var`/`const` bind at their declaration point — legal
#:   `fn a() { b() } fn b() { a() }`, but `E0201` for `print(x)` above `let x = 1`.
#: - Slot 0 is the callee/receiver; declared parameters are slots 1..arity,
#:   and sibling blocks reuse slots.
#: - Captures use the Crafting Interpreters algorithm: the innermost function
#:   captures the owner's local directly and every function in between
#:   recaptures the upvalue of the one outside it.
#:
#: Results live beside the tree in a `Resolution` keyed by node identity,
#: since a `jaithon.ast` node holds only what the C AST serialises. `Symbol`,
#: `Resolution` and the rest of the output model are `jaithon.compile.symbol`;
#: this package is the `Resolver` pass that builds them.
#:
#: ## What lives where
#:
#: Facade: imports and re-exports the modules of the `resolve/` directory. The
#: surface is three names, and it was three names before the split. Import
#: from `jaithon.compile.resolve` and take no interest in the directory.
#:
#: The vocabulary comes first and depends on nothing but the tree and the
#: output model: `scope.jai` holds `ScopeKind`, `Scope` and `ClassInfo`, the
#: scaffolding the pass drops on the way out; `predicate.jai` holds the
#: questions about a symbol or a name that need no scope to answer. On top of
#: those, `resolver.jai` holds the `Resolver` class itself — scopes, slots,
#: declaration, capture and name lookup — and the three seams, since the
#: productions above it are one knot of mutual recursion and a seam exists
#: exactly where an import may not go. `decl.jai` is what a declaration binds:
#: hoisting, patterns, functions, and the three type bodies. `walk.jai` is
#: what a node uses: the statement forms and the per-kind dispatch. `wire.jai`
#: sits last, fills the seams in as it is imported, and holds `resolve`.
from jaithon.compile.resolve.scope import ScopeKind
from jaithon.compile.resolve.resolver import Resolver
from jaithon.compile.resolve.wire import resolve

export { ScopeKind, Resolver, resolve }
