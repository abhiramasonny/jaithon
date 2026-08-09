#: The type of every expression form: operators, member access, calls,
#: indexing and slicing, comprehensions and function expressions.
#:
#: A port of `src/sema/check_expr.c`. Every function here answers the same
#: question — what type does this node have — and has the same two side effects:
#: it records that type for the node, and where an `any -> T` boundary needs a
#: run-time check it wraps the node in a `Cast`. Nothing here decides control
#: flow; `if` and `match` in statement position hand back to the statement
#: checker.
#:
#: **`Ctx` is the seam, and `checker.jai` will supply most of it.** The C splits
#: the checker across five files that call each other the way the grammar nests,
#: and `check_internal.h` is the whole of that coupling. Only this file exists so
#: far, so everything the other four own arrives through `Ctx`: per-run state as
#: fields, and the operations as callbacks with conservative defaults.
#: `checker.jai` constructs a `Ctx`, fills the callbacks in, and this file is
#: unchanged by that. The defaults are what make the port testable today, not a
#: second implementation to keep in step — where one is a real port of a
#: `check.c` function it says so, and where one is a stub it says what it
#: answers.
#:
#: **Types live in a side table, not on the node.** A `jaithon.ast.Node` carries
#: only what the C AST serialises, so there is nowhere on it for `AstNode.type`.
#: `Ctx.set_type` and `Ctx.type_of` are that field, keyed by `id(node)`; the same
#: trick `Resolution` uses for symbols, and for the same reason. `Ctx` also holds
#: the symbol type table the C keeps in `Symbol.type`, because a Jaithon `Symbol`
#: has no such field either.
#:
#: **What is not implemented yet, and what it answers instead.** Each of these is
#: a `Ctx` callback whose default is stated, so nothing here answers wrongly in
#: silence:
#:
#: | form | today | supplied by |
#: |---|---|---|
#: | `match` in expression position | `any` | `Ctx.match_expr`, from the statement checker |
#: | a block in expression position | `any`, body unchecked | `Ctx.check_block` |
#: | a statement in expression position | `void`, unchecked | `Ctx.check_stmt` |
#: | a comprehension's pattern binding | unbound, element type discarded | `Ctx.check_pattern` |
#: | a lambda body | unchecked; the signature is still built | `Ctx.check_function` |
#: | `m.f` after `import m` | `any` | `Ctx.is_module` / `Ctx.module_member` |
#: | a written type annotation | `any` | `Ctx.resolve_ast_type` |
#: | a class laid out on demand | never laid out | `Ctx.layout_decl` |
#: | a declaration inside a block | unchecked | `Ctx.check_decl`, from `check_decl.c` |
#:
#: The last row is the one callback nothing in this file reaches. It is declared
#: here because `Ctx` is where the seam lives, and `stmt.jai` — which does reach
#: it, for the `class` and `fn` a block may contain — must not have to grow a
#: second context object to hold one field.
#:
#: Two further gaps are the AST's rather than a callback's, and neither can be
#: closed from here:
#:
#: - **A `Cast` node has no `widen` field.** The C's `AST_CAST` records whether a
#:   cast is §2.2's `int -> float` conversion or an `any -> T` guard, and the two
#:   emit different instructions. The Jaithon `Cast` schema has `operand` and
#:   `target` only, so the bit is kept in `Ctx.is_widen_cast` beside the type
#:   table.
#: - **An imported function is not a declaration here.** The C's `Symbol` has an
#:   `importedDecl` back-link, so a call through `from m import f` binds against
#:   `f`'s real parameter list. A Jaithon `Symbol` has none, so such a call is
#:   checked against its type alone.
#:
#: `jaiCheckExpr`, the C's public one-expression entry point, is not ported: it
#: is `jaiChkExpr` behind two null tests, and `expr` already makes both.
#:
#: ```
#: let c = Ctx(Universe())
#: render(expr(c, Node.binary(OpKind.Add, Node.int_lit(1), Node.float_lit(2.0))))
#: # "float"
#: render(expr(c, Node.str_lit("hi")))    # "str"
#: ```
from jaithon.ast import CompKind, DeclKind, Node, NodeKind, OpKind, Span, TypeExpr
from jaithon.ast import TypeKind as SynKind
from jaithon.ast import kind_name
from jaithon.ast import is_statement, op_is_comparison, op_text
from jaithon.compile.check.assign import assignable, widen_target
from jaithon.compile.check.dump import Record
from jaithon.compile.check.fold import ConstKind, const_eval, const_replace, try_fold
from jaithon.compile.check.kinds import MAX_DECL_DEPTH, TyKind
from jaithon.compile.check.nominal import find_accessor, find_field, find_member
from jaithon.compile.check.nominal import inherits_opaquely, is_subclass_of
from jaithon.compile.check.nominal import names_opaque_trait, owner_of_field, owner_of_member
from jaithon.compile.check.operator import binary_result, unary_result
from jaithon.compile.check.predicate import is_callable, is_indexable, is_iterable, is_optional
from jaithon.compile.check.relate import join, narrow
from jaithon.compile.check.render import render
from jaithon.compile.check.ty import MemberInfo, Type, TypeDecl
from jaithon.compile.check.union import optional_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, list_of, set_of, tuple_of
from jaithon.compile.lexer import Diagnostic, DiagnosticBag
from jaithon.compile.resolve import Resolution, Symbol, SymbolKind

# ------------------------------------------------------------------
# Diagnostic codes
# ------------------------------------------------------------------

#: A value does not fit the slot it is going into.
#:
#: The E04xx, E06xx and E07xx groups are spelled here rather than in
#: `jaithon.compile.lexer.Code` or `jaithon.compile.resolve.SemaCode` for the
#: reason `fold.jai` gives for E0900: those tables stop at the parser and the
#: resolver, and these are the checker's own. The strings are the C's
#: `JaiDiagCode` names in `src/common/diag.h`, and both the code and the message
#: text are a contract — Phase 4 turns `tests/errors` into goldens.
pub const E0400_TYPE_MISMATCH: str = "E0400"

#: A condition is not a `bool`. There is no truthiness.
pub const E0401_CONDITION_NOT_BOOL: str = "E0401"

#: A value that cannot appear before a `(`.
pub const E0403_NOT_CALLABLE: str = "E0403"

#: A value that cannot be subscripted or sliced.
pub const E0404_NOT_INDEXABLE: str = "E0404"

#: A value that cannot be looped over.
pub const E0405_NOT_ITERABLE: str = "E0405"

#: An operator that does not apply to the operands it was given.
pub const E0406_BAD_OPERAND_TYPES: str = "E0406"

#: A `void` expression used where a value was wanted.
pub const E0409_VOID_VALUE_USED: str = "E0409"

#: A member that the declaration does not have.
pub const E0410_UNKNOWN_MEMBER: str = "E0410"

#: A `float` reaching an `int`, the direction that loses information.
pub const E0411_INT_FLOAT_MIX: str = "E0411"

#: A call of an unnamed callable with the wrong number of arguments.
pub const E0600_ARITY_MISMATCH: str = "E0600"

#: A call of a named function with the wrong number of arguments.
pub const E0602_BAD_CALL_ARITY: str = "E0602"

#: A keyword argument naming no parameter.
pub const E0603_UNKNOWN_KEYWORD_ARG: str = "E0603"

#: One argument given twice.
pub const E0604_DUPLICATE_ARGUMENT: str = "E0604"

#: A positional argument after a keyword argument.
pub const E0605_POSITIONAL_AFTER_KEYWORD: str = "E0605"

#: A member reached from outside the visibility it was declared with.
pub const E0701_PRIVATE_ACCESS: str = "E0701"

#: A trait or an abstract class used as a constructor.
pub const E0710_ABSTRACT_INSTANTIATION: str = "E0710"

#: Code that cannot run — here, the right operand of a pointless `??`.
pub const W0102_UNREACHABLE_CODE: str = "W0102"

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

#: A member or field declared with no modifier.
#:
#: `FieldInfo.visibility` and `MemberInfo.visibility` in `ty.jai` are plain
#: `int`s, so these three name the values. They are the ordinals of
#: `jaithon.ast.Visibility`, which is the order of the C `AstVisibility`.
pub const VIS_PRIVATE: int = 0

#: Reachable from the declaring class and its subclasses.
pub const VIS_PROTECTED: int = 1

#: Reachable from anywhere: the declaration was marked `pub`.
pub const VIS_PUBLIC: int = 2

#: How many arguments or parameters one call binds, matching `JAI_MAX_ARGS`.
#:
#: The C's limit is the size of two stack arrays; nothing here needs one. It is
#: reproduced because it is observable: past it the C stops binding arguments to
#: parameters altogether, and a port that kept binding would report errors the C
#: does not.
pub const MAX_ARGS: int = 255

#: How many narrowing facts one condition contributes, matching `MAX_FACTS`.
pub const MAX_FACTS: int = 8

# `JAI_SUGGEST_NO_MATCH` in src/common/diag.h: worse than any accepted distance,
# so the first candidate under the limit always wins.
const SUGGEST_NO_MATCH: int = 4

# `JAI_SUGGEST_MAX_LEN`: past this a name is not compared at all.
const SUGGEST_MAX_LEN: int = 64

# ------------------------------------------------------------------
# The checker seam
# ------------------------------------------------------------------

#: One thing a condition proved about a binding, on the branch it names.
#:
#: `NarrowFact` in `src/sema/check_internal.h`, which the C shares between
#: `check.c` and both of its callers. It lives here because this is the only
#: file of the group that exists; `checker.jai` will import it from here.
pub class NarrowFact {
    #: The binding the fact is about.
    pub let symbol: Symbol
    #: The type the test compared it against.
    pub let by: Type
    #: True on the edge where the test succeeded.
    pub let positive: bool

    #: Record that `symbol` was tested against `by`.
    pub fn init(self, symbol: Symbol, by: Type, positive: bool) {
        self.symbol = symbol
        self.by = by
        self.positive = positive
    }
}

#: Everything one checker run holds, and every operation this file borrows.
#:
#: The C's `Checker *c` and its file-static `gJaiCheck` in one object, plus the
#: two side tables a Jaithon AST needs. `checker.jai` will construct one, install
#: its own callbacks over the defaults, and hand it to `expr`.
#:
#: The callbacks are fields rather than methods so that `checker.jai` can replace
#: them without subclassing, and so that a test can install a recorder for one of
#: them and leave the rest alone. Each default is documented where it is defined,
#: below.
#:
#: ```
#: let c = Ctx(Universe())
#: let node = Node.int_lit(7)
#: c.set_type(node, c.universe.t_int)
#: render(c.type_of(node))     # "int"
#: c.type_of(Node.int_lit(7)) is null     # true, identity is the key
#: ```
pub class Ctx {
    #: The intern table every answer is built from.
    pub let universe: Universe
    #: Where diagnostics go.
    pub let bag: DiagnosticBag
    #: What the resolver decided: symbols, and which brackets held type
    #: arguments.
    pub let resolution: Resolution

    #: The class whose body is being checked, for `self`, `super` and
    #: visibility.
    pub var current_class: TypeDecl?
    #: What the function whose body is being checked was declared to return,
    #: or null outside every body.
    #:
    #: The C's `Checker.currentReturnType`. `stmt.jai` saves it, sets it and
    #: restores it around each body, and checks every `return` against it.
    pub var current_return_type: Type?
    #: `--strict`: an unannotated parameter is an error.
    #:
    #: The C's `Checker.strict`, and false by default for the same reason — the
    #: driver turns it on, nothing here does.
    pub var strict: bool
    #: Whether a folded operator node is replaced by its literal. The C's
    #: `c->foldConstants`, and true for the same reason: the compiler folds.
    pub var fold_constants: bool
    #: `const` name to the initialiser it folded to, as `fold.jai` takes it.
    pub var consts: dict[str, Node]
    #: The sema-dump records, when one is being collected.
    pub var dump: list[Record]?
    #: How far each declaration has got through layout, keyed by `id(decl)`:
    #: absent or 0 for unstarted, 1 for in progress, 2 for laid out.
    #:
    #: `DeclEntry.status` in `src/sema/check.c`, which `check_decl.c` reads for
    #: two things: to make layout idempotent, and to recognise a cycle — a
    #: declaration reached again while it is still being laid out is `E0706`. A
    #: Jaithon `TypeDecl` has no such field and `DeclEntry` is not ported, so
    #: the status lives here beside the registry it belongs to. Empty by
    #: default, which reads as "nothing has been laid out yet"; `decl.jai` is
    #: the only writer.
    pub var layout_status: dict[int, int]

    var node_types: dict[int, Type]
    var symbol_types: dict[int, Type]
    var widen_casts: dict[int, bool]
    var decls_by_name: dict[str, TypeDecl]
    var decls_by_node: dict[int, TypeDecl]
    # id(TypeExpr) -> the semantic type it resolved to; the C's AstType.resolved.
    var resolved_types: dict[int, Type]
    #: The signature loader, so a qualified member can be resolved against the
    #: module it came from. Installed by `checker.check_program`; null when
    #: nothing set one, in which case `m.f` answers `any`.
    pub var sig_loader: any?
    #: The directory of the file being checked, which is where an import
    #: resolves from.
    pub var source_dir: str
    var narrow_symbols: list[Symbol]
    var narrow_saved: list[Type]
    var type_names: list[tuple[str, Type]]

    #: Check that `source` fits `target`, converting or guarding where it must.
    #:
    #: `jaiChkRequireAssignable` in `src/sema/check.c`. The default is a port of
    #: it; `checker.jai` may take it back.
    pub var require_assignable: fn(Ctx, Node?, Type?, Type?, str, str) -> bool
    #: Push an expected type down into a literal before it is checked.
    #:
    #: `jaiChkApplyContext`. The default is a port of it.
    pub var apply_context: fn(Ctx, Node?, Type?) -> void
    #: Rewrite an `int` expression so that it produces `to`, a `float`.
    #:
    #: `jaiChkWidenToFloat`. The default is a port of it.
    pub var widen_to_float: fn(Ctx, Node?, Type) -> void
    #: The type a written annotation denotes.
    #:
    #: `jaiChkResolveAstType`. **The default answers null**, so every annotation
    #: reads as `any` until `checker.jai` supplies one.
    pub var resolve_ast_type: fn(Ctx, TypeExpr?) -> Type?
    #: The type a bare name denotes, when it denotes one.
    #:
    #: `jaiChkLookupTypeName`. The default answers from the declaration registry
    #: below; `checker.jai` extends it with the scalars, the aliases in scope and
    #: the generic parameters of the enclosing declaration.
    pub var lookup_type_name: fn(Ctx, str?) -> Type?
    #: Lay a declaration out, so that its members can be looked up.
    #:
    #: `jaiChkLayoutDecl`. **The default does nothing**: a declaration answers
    #: with whatever members it already carries.
    pub var layout_decl: fn(Ctx, TypeDecl?) -> void
    #: Bind a pattern against the type it destructures.
    #:
    #: `jaiChkPattern`. **The default does nothing**, so a comprehension's loop
    #: variable is unbound and reads as `any` in the element expression.
    pub var check_pattern: fn(Ctx, Node?, Type) -> void
    #: Check a block in expression position.
    #:
    #: `jaiChkBlock`. **The default does nothing.**
    pub var check_block: fn(Ctx, Node?) -> void
    #: Check a statement that reached expression position.
    #:
    #: `jaiChkStmt`. **The default does nothing.**
    pub var check_stmt: fn(Ctx, Node?) -> void
    #: Check a declaration that appeared in a block.
    #:
    #: `jaiChkDecl`, which `check_decl.c` owns. **The default does nothing**, so
    #: a nested `class` or `fn` keeps the names the resolver bound and gets no
    #: layout and no body check. Nothing in this file reaches it; `stmt.jai`
    #: does, for the declarations a block may contain.
    pub var check_decl: fn(Ctx, Node?) -> void
    #: Check the body of a lambda or an anonymous function.
    #:
    #: `jaiChkFunction`. **The default does nothing**; the signature is still
    #: built by `function_type`, so a call through the lambda is still checked.
    pub var check_function: fn(Ctx, Node?) -> void
    #: The `Fn` type of a function, lambda or anonymous function node.
    #:
    #: `jaiChkFunctionType`. The default reads the parameter list and the
    #: variadic and kwrest flags, and resolves each written type through
    #: `resolve_ast_type` — which means `any` per parameter until that is
    #: supplied. The arity and the flags are right either way, and those are what
    #: a call is bound against.
    pub var function_type: fn(Ctx, Node?) -> Type
    #: The type of a `match` in expression position.
    #:
    #: `jaiChkMatchExpr`. **The default answers `any`** and checks nothing.
    pub var match_expr: fn(Ctx, Node?, bool) -> Type
    #: Whether this expression is a module object bound by `import m`.
    #:
    #: `moduleSigOfIdent`. **The default answers false.**
    pub var is_module: fn(Ctx, Node?) -> bool
    #: The type of `name` in a module, and the declaration behind it.
    #:
    #: `moduleMemberType`. **The default answers `(any, null)`**, which is also
    #: what the C answers for a name a module defines at run time.
    pub var module_member: fn(Ctx, Node?, str) -> (Type, Node?)

    #: Build a run over `universe`, with a fresh bag and resolution by default.
    pub fn init(
        self,
        universe: Universe,
        bag: DiagnosticBag? = null,
        resolution: Resolution? = null
    ) {
        self.universe = universe
        self.bag = bag ?? DiagnosticBag()
        self.resolution = resolution ?? Resolution()
        self.current_class = null
        self.current_return_type = null
        self.strict = false
        self.fold_constants = true
        self.consts = {}
        self.dump = null
        self.layout_status = {}
        self.node_types = {}
        self.symbol_types = {}
        self.widen_casts = {}
        self.decls_by_name = {}
        self.decls_by_node = {}
        self.resolved_types = {}
        self.sig_loader = null
        self.source_dir = ""
        self.narrow_symbols = []
        self.narrow_saved = []
        self.type_names = []
        self.require_assignable = _default_require_assignable
        self.apply_context = _default_apply_context
        self.widen_to_float = _default_widen_to_float
        self.resolve_ast_type = _default_resolve_ast_type
        self.lookup_type_name = _default_lookup_type_name
        self.layout_decl = _default_layout_decl
        self.check_pattern = _default_check_pattern
        self.check_block = _default_check_block
        self.check_stmt = _default_check_stmt
        self.check_decl = _default_check_decl
        self.check_function = _default_check_function
        self.function_type = _default_function_type
        self.match_expr = _default_match_expr
        self.is_module = _default_is_module
        self.module_member = _default_module_member
    }

    #: Record the type of `node`. The C writes `AstNode.type`.
    pub fn set_type(self, node: Node?, ty: Type?) -> void {
        if node is null or ty is null { return }
        self.node_types[id(node)] = ty
    }

    #: The type recorded for `node`, or null when there is none.
    pub fn type_of(self, node: Node?) -> Type? {
        if node is null { return null }
        return self.node_types.get(id(node), null)
    }

    #: Forget the type of `node`. The C writes `node->type = NULL`.
    pub fn clear_type(self, node: Node?) -> void {
        if node is null { return }
        if id(node) in self.node_types { self.node_types.remove(id(node)) }
    }

    #: The type of a binding, which is `Symbol.type` in the C.
    pub fn symbol_type(self, symbol: Symbol?) -> Type? {
        if symbol is null { return null }
        return self.symbol_types.get(id(symbol), null)
    }

    #: Give `symbol` a type.
    pub fn set_symbol_type(self, symbol: Symbol?, ty: Type?) -> void {
        if symbol is null or ty is null { return }
        self.symbol_types[id(symbol)] = ty
    }

    #: Mark a `Cast` node as §2.2's `int -> float` conversion.
    #:
    #: The C keeps this on the node, in `AST_CAST.widen`. The Jaithon `Cast`
    #: schema has no such field, so it is kept here; an unmarked cast is the
    #: `any -> T` guard.
    pub fn mark_widen_cast(self, node: Node?) -> void {
        if node is null { return }
        self.widen_casts[id(node)] = true
    }

    #: Whether `node` is a widening cast rather than a guard.
    pub fn is_widen_cast(self, node: Node?) -> bool {
        if node is null { return false }
        return self.widen_casts.get(id(node), false)
    }

    #: Register a laid-out declaration, by name and by declaring node.
    #:
    #: `gJaiCheck.decls` in the C, minus the import machinery: this holds what
    #: `jaiTypeDeclFind` and `jaiChkDeclForNode` answer from.
    pub fn register_decl(self, decl: TypeDecl) -> void {
        self.decls_by_name[decl.name] = decl
        let node = decl.node
        if node is not null { self.decls_by_node[id(node)] = decl }
    }

    #: The registered declaration called `name`.
    pub fn decl_named(self, name: str?) -> TypeDecl? {
        if name is null { return null }
        return self.decls_by_name.get(name, null)
    }

    #: Every registered declaration, in registration order.
    #:
    #: `checker.jai` lays all of them out before checking anything: half the
    #: questions the checker asks about a type — is it iterable, does it
    #: implement that trait, is it callable — are asked in code with no
    #: declaration in hand to lay out on demand, and an empty declaration
    #: silently answers "no" to all of them.
    pub fn all_decls(self) -> list[TypeDecl] {
        var out: list[TypeDecl] = []
        for name in self.decls_by_name.keys() { out.push(self.decls_by_name[name]) }
        return out
    }

    #: The type a written annotation resolved to, if it has been resolved.
    #:
    #: The C caches this on `AstType.resolved`. A Jaithon `TypeExpr` has no
    #: such field, so like every other annotation this port carries it lives
    #: beside the tree instead.
    pub fn resolved_type(self, t: TypeExpr?) -> Type? {
        if t is null { return null }
        return self.resolved_types.get(id(t), null)
    }

    #: Remember what `t` resolved to.
    pub fn set_resolved_type(self, t: TypeExpr?, ty: Type?) -> void {
        if t is null or ty is null { return }
        self.resolved_types[id(t)] = ty
    }

    #: The registered declaration `node` declares.
    pub fn decl_for_node(self, node: Node?) -> TypeDecl? {
        if node is null { return null }
        return self.decls_by_node.get(id(node), null)
    }

    #: The class, trait or enum a bound name denotes, or null.
    #:
    #: `jaiChkIdentTypeDecl`. A name the resolver called a class answers from the
    #: registry; anything else answers through its declaring node, which is how
    #: an imported binding — whose symbol kind is a slot, not a class — is still
    #: recognised.
    pub fn ident_type_decl(self, symbol: Symbol?) -> TypeDecl? {
        if symbol is null { return null }
        if _is_type_symbol(symbol.kind) { return self.decl_named(symbol.name) }
        return self.decl_for_node(symbol.decl)
    }

    #: Where the type-name scope stack stands, so a declaration can unwind to
    #: here.
    #:
    #: `jaiChkTypeNameMark`. The stack holds the names that denote a type only
    #: inside the declaration that introduced them — a generic parameter, and
    #: `Self` — and is consulted before the registry, so an innermost binding
    #: wins.
    pub fn type_name_mark(self) -> int {
        return self.type_names.len()
    }

    #: Bind `name` to `ty` until the enclosing declaration ends.
    #:
    #: `jaiChkPushTypeName`. A null name or type is dropped: the C pushes the
    #: entry and no lookup can ever match it, which is the same thing.
    pub fn push_type_name(self, name: str?, ty: Type?) -> void {
        if name is null or ty is null { return }
        self.type_names.push((name, ty))
    }

    #: Drop every binding pushed since `mark`.
    #:
    #: `jaiChkTypeNameRestore`, down to its tolerance of a mark outside the
    #: stack, which it ignores rather than reporting.
    pub fn type_name_restore(self, mark: int) -> void {
        if mark < 0 or mark > self.type_names.len() { return }
        while self.type_names.len() > mark { let _dropped = self.type_names.pop() }
    }

    #: The type `name` denotes in the innermost scope that binds it, or null.
    #:
    #: The first half of `jaiChkLookupTypeName`, which `_default_lookup_type_name`
    #: consults before the declaration registry.
    pub fn scoped_type_name(self, name: str?) -> Type? {
        if name is null { return null }
        var at = self.type_names.len() - 1
        while at >= 0 {
            let (bound, ty) = self.type_names[at]
            if bound == name { return ty }
            at -= 1
        }
        return null
    }

    #: Report an error, and hand it back so the caller can decorate it.
    pub fn error(self, code: str, span: Span, message: str) -> Diagnostic {
        return self.bag.error(code, span, message)
    }

    #: Report a warning, and hand it back.
    pub fn warn(self, code: str, span: Span, message: str) -> Diagnostic {
        return self.bag.warn(code, span, message)
    }

    #: Where the narrowing stack stands, so a branch can be unwound to here.
    pub fn narrow_mark(self) -> int {
        return self.narrow_symbols.len()
    }

    #: Apply `facts`, saving what each binding held so it can be restored.
    #:
    #: `invert` takes the other edge of the same condition, which is how an
    #: `else` arm reuses the facts the `then` arm collected.
    pub fn narrow_apply(self, facts: list[NarrowFact], invert: bool) -> void {
        for fact in facts {
            let current = self.symbol_type(fact.symbol)
            if current is null { continue }
            let positive = if invert { not fact.positive } else { fact.positive }
            let next = narrow(self.universe, current, fact.by, positive)
            if next is null or next == current { continue }
            self.narrow_symbols.push(fact.symbol)
            self.narrow_saved.push(current)
            self.set_symbol_type(fact.symbol, next)
        }
    }

    #: Undo every fact applied since `mark`.
    pub fn narrow_restore(self, mark: int) -> void {
        while self.narrow_symbols.len() > mark {
            let symbol = self.narrow_symbols.pop()
            let saved = self.narrow_saved.pop()
            self.set_symbol_type(symbol, saved)
        }
    }

    #: The type a binding was declared with, before any live fact rewrote it.
    #:
    #: A fact only ever replaces an already-saved type, so the oldest save for
    #: the symbol holds the declaration.
    pub fn declared_type(self, symbol: Symbol?) -> Type? {
        if symbol is null { return null }
        for (position, saved) in self.narrow_symbols.enumerate() {
            if saved == symbol { return self.narrow_saved[position] }
        }
        return self.symbol_type(symbol)
    }

    #: End the fact a store to `symbol` invalidated.
    #:
    #: `jaiChkNarrowInvalidate`. What the binding holds after the store is
    #: whatever its declaration admits, so the narrowed view is dropped. The
    #: save stack is deliberately left alone: the enclosing `narrow_restore`
    #: still has to unwind in order.
    pub fn narrow_invalidate(self, symbol: Symbol?) -> void {
        let declared = self.declared_type(symbol)
        if declared is not null { self.set_symbol_type(symbol, declared) }
    }

    #: What `cond` proves, on the edge `negate` names.
    #:
    #: `jaiChkCollectFacts`. `x is T`, `x == null`, `isinstance(x, T)` and their
    #: negations narrow; `a and b` gives both facts on the true edge and
    #: `not (a or b)` gives both on the false edge, and every other combination
    #: is a disjunction where no single fact is guaranteed.
    pub fn collect_facts(self, cond: Node?, negate: bool) -> list[NarrowFact] {
        var out: list[NarrowFact] = []
        self._collect_facts(cond, negate, out)
        return out
    }

    fn _collect_facts(self, cond: Node?, negate: bool, out: list[NarrowFact]) -> void {
        if cond is null or out.len() >= MAX_FACTS { return }

        match cond.kind {
            NodeKind.Unary => {
                if cond.op() == OpKind.Not {
                    self._collect_facts(cond.child("operand"), not negate, out)
                }
            }

            NodeKind.Logical => {
                let conjunction = (cond.op() == OpKind.And) != negate
                if not conjunction { return }
                self._collect_facts(cond.child("left"), negate, out)
                self._collect_facts(cond.child("right"), negate, out)
            }

            # `isinstance(x, T)` is the call form of `x is T` and the one the
            # library uses on unions; it narrows the same way.
            NodeKind.Call => {
                let callee = cond.child("callee")
                if callee is null or callee.kind != NodeKind.Ident { return }
                if callee.opt_text("name") != "isinstance" { return }
                let args = cond.records("args")
                if args.len() != 2 { return }
                let symbol = self._narrow_target(_arg_value(args[0]))
                let target = self._type_operand(_arg_value(args[1]))
                if symbol is null or target is null { return }
                out.push(NarrowFact(symbol, target, not negate))
            }

            NodeKind.Binary => {
                self._comparison_fact(cond.op(), cond.child("left"), cond.child("right"), negate, out)
            }

            NodeKind.CompareChain => {
                let ops: list[OpKind] = cond.get("ops")
                let operands = cond.child_list("operands")
                if ops.len() != 1 or operands.len() != 2 { return }
                self._comparison_fact(ops[0], operands[0], operands[1], negate, out)
            }

            _ => { }
        }
    }

    fn _comparison_fact(
        self,
        op: OpKind,
        lhs: Node?,
        rhs: Node?,
        negate: bool,
        out: list[NarrowFact]
    ) -> void {
        if op == OpKind.Is or op == OpKind.IsNot {
            # `null` is a unit type with one value (§2.1), so `x is null` is the
            # null identity test and narrows exactly as `x == null` does. It is
            # not an identifier, so `_type_operand` cannot see it.
            var target: Type? = null
            if _is_null_literal(rhs) {
                target = self.universe.t_null
            } else {
                target = self._type_operand(rhs)
            }
            let symbol = self._narrow_target(lhs)
            if target is null or symbol is null { return }
            out.push(NarrowFact(symbol, target, (op == OpKind.Is) != negate))
            return
        }

        if op == OpKind.Eq or op == OpKind.Ne {
            var symbol: Symbol? = null
            if _is_null_literal(rhs) {
                symbol = self._narrow_target(lhs)
            } elif _is_null_literal(lhs) {
                symbol = self._narrow_target(rhs)
            }
            if symbol is null { return }
            out.push(NarrowFact(symbol, self.universe.t_null, (op == OpKind.Eq) != negate))
        }
    }

    # Narrowing a `var` is still sound within a scope: an assignment to it is
    # checked against the declared type and then ends the fact.
    fn _narrow_target(self, node: Node?) -> Symbol? {
        if node is null or node.kind != NodeKind.Ident { return null }
        let symbol = self.resolution.symbol_of(node)
        if symbol is null or self.symbol_type(symbol) is null { return null }
        return symbol
    }

    # The right operand of `is` names a type when it is a bare identifier bound
    # to a class, trait or enum; anything else is an identity comparison.
    fn _type_operand(self, node: Node?) -> Type? {
        if node is null or node.kind != NodeKind.Ident { return null }
        let symbol = self.resolution.symbol_of(node)
        if symbol is not null and _is_type_symbol(symbol.kind) {
            let known = self.symbol_type(symbol)
            if known is not null { return known }
            return self.lookup_type_name(self, symbol.name) ?? self.universe.t_any
        }
        # `str`, `int`, `list`... are bound as builtin conversion functions as
        # well as naming types (spec §9). Comparing a value against a function
        # object is never what `x is str` means, so the type reading wins.
        if symbol is not null and symbol.kind != SymbolKind.Builtin { return null }
        return self.lookup_type_name(self, node.opt_text("name"))
    }
}

fn _is_type_symbol(kind: SymbolKind) -> bool {
    if kind == SymbolKind.Class or kind == SymbolKind.Trait { return true }
    return kind == SymbolKind.Enum
}

fn _is_null_literal(node: Node?) -> bool {
    return node is not null and node.kind == NodeKind.NullLit
}

# ------------------------------------------------------------------
# The default callbacks
# ------------------------------------------------------------------

# A port of `jaiChkRequireAssignable` in src/sema/check.c, which is where the
# `any -> T` guard and the `int -> float` conversion enter the tree. It is a
# default rather than a stub because nothing above it can be exercised without
# one: an argument, an element and an index all report through here.
fn _default_require_assignable(
    c: Ctx,
    node: Node?,
    source: Type?,
    target: Type?,
    code: str,
    what: str
) -> bool {
    if target is null or source is null { return true }
    if _is_never(source) { return true }

    # §2.2's widening, before the general rule, because it is the one assignment
    # whose value changes on the way in and so the one that has to put a
    # conversion into the tree.
    let widen = widen_target(c.universe, source, target)
    if widen is not null {
        if node is not null { c.widen_to_float(c, node, widen) }
        return true
    }

    let (accepted, needs_guard) = assignable(c.universe, source, target)
    if accepted {
        if needs_guard and node is not null { _wrap_in_cast(c, node, target, false) }
        return true
    }

    let want = render(target)
    let got = render(source)
    # float to int is the direction that loses information, so it is the
    # direction that stays an error, and it keeps E0411 to itself.
    let narrowing = target.kind == TyKind.Int and source.kind == TyKind.Float
    let span = _span_of(node)
    let reported = if narrowing { E0411_INT_FLOAT_MIX } else { code }
    let d = c.error(reported, span, f"{what}: expected `{want}`, found `{got}`")
    if narrowing {
        let subject = subject_of(node, "x")
        let fix = f"write `int({subject})` to truncate"
        d.with_help(f"float does not narrow to int, because it would lose the fraction; {fix}")
    } elif is_optional(target) and source.kind == TyKind.Null {
        d.with_help("`null` is only assignable to an optional type")
    }
    return false
}

# A port of `jaiChkApplyContext`. Only literal shapes are rewritten; a variable
# keeps its type and produces E0411 as it should.
fn _default_apply_context(c: Ctx, node: Node?, expected: Type?) -> void {
    if node is null or expected is null { return }

    if _union_wants_float(expected) {
        if _retype_int_literal(c, node) { return }
        # Arithmetic over literals is still a literal expression as far as the
        # conversion is concerned: `let x: float = 1 + 2`.
        if node.kind == NodeKind.Binary and _is_arithmetic_context(node.op()) {
            c.apply_context(c, node.child("left"), expected)
            c.apply_context(c, node.child("right"), expected)
        } elif node.kind == NodeKind.Ternary {
            c.apply_context(c, node.child("thenExpr"), expected)
            c.apply_context(c, node.child("elseExpr"), expected)
        } elif node.kind == NodeKind.IfExpr {
            c.apply_context(c, node.child("thenBranch"), expected)
            c.apply_context(c, node.child("elseBranch"), expected)
        }
        return
    }

    # Container literals additionally adopt the expected type itself, not just
    # the element context. Without this an empty `[]` has no items to recurse
    # into and infers `list[any]`, and a populated one infers the join of its
    # elements — so `let s: list[Shape] = [Circle(), Square()]` infers
    # `list[Circle | Square]` and is then rejected against its own annotation.
    # The literal checkers read this back and verify each element against the
    # target instead of against its siblings.
    match expected.kind {
        TyKind.List | TyKind.Set => {
            # A comprehension builds a fresh container, so it takes the
            # annotation the same way a literal does.
            if node.kind == NodeKind.Comprehension {
                c.set_type(node, expected)
                return
            }
            let matches = node.kind == NodeKind.ListLit or node.kind == NodeKind.SetLit
            if matches and expected.arg_count() == 1 {
                for item in node.child_list("items") { c.apply_context(c, item, expected.args[0]) }
                c.set_type(node, expected)
            }
        }

        TyKind.Dict => {
            if node.kind == NodeKind.Comprehension {
                c.set_type(node, expected)
                return
            }
            if node.kind == NodeKind.DictLit and expected.arg_count() == 2 {
                for key in node.child_list("keys") { c.apply_context(c, key, expected.args[0]) }
                for held in node.child_list("values") { c.apply_context(c, held, expected.args[1]) }
                c.set_type(node, expected)
            }
        }

        TyKind.Tuple => {
            if node.kind != NodeKind.TupleLit { return }
            let items = node.child_list("items")
            if items.len() != expected.arg_count() { return }
            for (position, item) in items.enumerate() {
                c.apply_context(c, item, expected.args[position])
            }
            # Stamped for the same reason a list literal is: each member is then
            # checked against the position it is going into, so an int widens
            # into a `float` member and a mismatch names the member rather than
            # failing the whole tuple against its annotation.
            c.set_type(node, expected)
        }

        _ => { }
    }
}

# A port of `jaiChkWidenToFloat`. An int literal becomes a float literal, which
# costs nothing at run time; anything else gets the conversion node §2.2 calls
# for, so the code generator emits one `OP_TO_FLOAT` rather than leaving the VM
# to discover the mismatch on every evaluation.
fn _default_widen_to_float(c: Ctx, node: Node?, to: Type) -> void {
    if node is null { return }
    if _retype_int_literal(c, node) { return }
    _wrap_in_cast(c, node, to, true)
}

# `checker.jai` supplies this. Until then every written annotation reads as
# `any`, which is what the C answers for a name it cannot resolve.
fn _default_resolve_ast_type(c: Ctx, written: TypeExpr?) -> Type? {
    return null
}

# The type-name scope stack, then the registry — the first two of the four
# things `jaiChkLookupTypeName` consults. `checker.jai` adds the remaining two:
# the builtin scalars, and a name read out of another module's namespace.
fn _default_lookup_type_name(c: Ctx, name: str?) -> Type? {
    let scoped = c.scoped_type_name(name)
    if scoped is not null { return scoped }
    let decl = c.decl_named(name)
    if decl is null { return null }
    return decl.ty
}

fn _default_layout_decl(c: Ctx, decl: TypeDecl?) -> void {
}

fn _default_check_pattern(c: Ctx, pattern: Node?, subject: Type) -> void {
}

fn _default_check_block(c: Ctx, node: Node?) -> void {
}

fn _default_check_stmt(c: Ctx, node: Node?) -> void {
}

fn _default_check_decl(c: Ctx, node: Node?) -> void {
}

fn _default_check_function(c: Ctx, node: Node?) -> void {
}

# The signature without the body: arity, the two flags, and whatever
# `resolve_ast_type` can make of each written type. That is enough for a call
# through the value to be bound and counted, which is what `checkArgs` reads a
# `TY_FN` for.
fn _default_function_type(c: Ctx, node: Node?) -> Type {
    let u = c.universe
    if node is null { return u.t_any }
    var params: list[Type] = []
    var flags = 0
    for parameter in node.records("params") {
        params.push(c.resolve_ast_type(c, _param_type(parameter)) ?? u.t_any)
        if _param_variadic(parameter) { flags = flags | 1 }
        if _param_kwrest(parameter) { flags = flags | 2 }
    }
    let returns = c.resolve_ast_type(c, node.type_field("returnType"))
    return fn_of(u, params, returns ?? u.t_any, flags)
}

fn _default_match_expr(c: Ctx, node: Node?, is_statement_position: bool) -> Type {
    return c.universe.t_any
}

fn _default_is_module(c: Ctx, node: Node?) -> bool {
    return false
}

fn _default_module_member(c: Ctx, node: Node?, name: str) -> (Type, Node?) {
    return (c.universe.t_any, null)
}

# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

# `jaiChkIsAny`. Statically unknown: `any` by construction, and an unsubstituted
# generic parameter because generics are erased (spec §6.1).
fn _is_any(t: Type?) -> bool {
    if t is null { return true }
    return t.kind == TyKind.Any or t.kind == TyKind.GenericParam
}

fn _is_never(t: Type?) -> bool {
    return t is not null and t.kind == TyKind.Never
}

# `jaiChkIsVoid`. `void` is `null` under the name a return position uses, so
# `t_void == t_null` and this is always false — in the C as well, where
# `gTypes.tVoid = gTypes.tNull`. It is ported because E0409 is ported, and
# because the day either universe gives `void` an identity of its own this is
# the line that starts reporting it.
fn _is_void(c: Ctx, t: Type?) -> bool {
    if t is null { return false }
    return t == c.universe.t_void and c.universe.t_void != c.universe.t_null
}

fn _span_of(node: Node?) -> Span {
    if node is null { return Span.none() }
    return node.span
}

# `jaiChkDeclType`: the interned type a declaration declares, or `any` when it
# has not been given one yet.
fn _decl_type(c: Ctx, decl: TypeDecl?) -> Type {
    if decl is null { return c.universe.t_any }
    return decl.ty ?? c.universe.t_any
}

# `jaiChkTypeDeclOf`: the declaration behind a class, trait or enum type. The C
# searches its registry and falls back to a lookup by name; interning already
# attached the declaration here, so the field is the answer.
fn _type_decl_of(t: Type?) -> TypeDecl? {
    if t is null { return null }
    if not t.is_named() { return null }
    return t.decl
}

# The type an already-checked element contributes to a container's join.
# `value` reports `any` for a void expression after diagnosing it but leaves the
# recorded type as `void`; re-reading the tree has to normalise the same way or
# an error cascade would change shape.
fn _element_type(c: Ctx, node: Node?) -> Type {
    if node is null { return c.universe.t_any }
    let t = c.type_of(node)
    if _is_void(c, t) { return c.universe.t_any }
    return t ?? c.universe.t_any
}

# The contextual type `apply_context` stamped on a container literal, if it is
# the right shape for this literal. Consumed, so a stale hint cannot leak into a
# later unannotated literal that happens to reuse the node.
fn _take_container_context(c: Ctx, node: Node, want: TyKind, arity: int) -> Type? {
    let hint = c.type_of(node)
    c.clear_type(node)
    if hint is null or hint.kind != want or hint.arg_count() != arity { return null }
    return hint
}

# One half of §2.2's implicit conversion: the half that costs nothing at run
# time, because an int literal can simply be respelled as a float literal.
#
# This is not the assignment rule and must not be read as one. `let n = 1; let
# x: float = n` is accepted and yields 1.0 — `require_assignable` consults
# `widen_target` and calls `widen_to_float`, which falls back to a cast for
# every shape this refuses. What this decides is only whether a conversion node
# is needed, never whether a conversion happens.
fn _retype_int_literal(c: Ctx, node: Node?) -> bool {
    if node is null { return false }

    if node.kind == NodeKind.IntLit {
        let literal = Node.float_lit(float(node.number("intLit")))
        node.kind = literal.kind
        node.fields = literal.fields
        c.set_type(node, c.universe.t_float)
        return true
    }

    if node.kind == NodeKind.Unary {
        let op = node.op()
        if op != OpKind.Neg and op != OpKind.Pos { return false }
        if not _retype_int_literal(c, node.child("operand")) { return false }
        c.set_type(node, c.universe.t_float)
        return true
    }

    return false
}

fn _union_wants_float(t: Type?) -> bool {
    if t is null { return false }
    if t.kind == TyKind.Float { return true }
    if t.kind != TyKind.Union { return false }
    var has_float = false
    for member in t.args {
        # int is a better fit, so the union takes the value as it is.
        if member.kind == TyKind.Int { return false }
        if member.kind == TyKind.Float { has_float = true }
    }
    return has_float
}

# The seven operators `jaiChkApplyContext` recurses through.
fn _is_arithmetic_context(op: OpKind) -> bool {
    match op {
        OpKind.Add | OpKind.Sub | OpKind.Mul | OpKind.Div => { return true }
        OpKind.FloorDiv | OpKind.Mod | OpKind.Pow => { return true }
        _ => { return false }
    }
}

# Does `t` mention an unsubstituted generic parameter anywhere? Such a type has
# no runtime shape after erasure, so it cannot be guarded against.
fn _mentions_generic_param(t: Type?) -> bool {
    if t is null { return false }
    if t.kind == TyKind.GenericParam { return true }
    for argument in t.args {
        if _mentions_generic_param(argument) { return true }
    }
    return t.kind == TyKind.Fn and _mentions_generic_param(t.ret)
}

# `wrapInCast` in src/sema/check.c. The node becomes the cast and the original
# is copied into a fresh node that becomes the cast's operand, so no caller has
# to hand over the slot that points at the node.
#: The name a runtime type guard checks against.
#:
#: `runtimeTypeName` in `src/codegen/codegen.c`. NOT the rendered type: a guard
#: is a runtime test, and the run time has erased every type argument, so
#: `list[int]` guards as `list`. Emitting the rendered form instead put
#: `"list[int]"` in the constant pool where the C puts `"list"`, which
#: `make bootstrap` reported as a differing constant on every guarded container
#: in the tree.
#:
#: A class, trait or enum answers its *bare* binding name rather than the
#: qualified one it is interned under, because the VM resolves this operand
#: against the module's globals and the binding there is the bare name.
pub fn runtime_type_name(t: Type?) -> str? {
    if t is null { return "any" }
    match t.kind {
        TyKind.Any => { return "any" }
        TyKind.Never => { return "any" }
        TyKind.GenericParam => { return "any" }
        TyKind.Alias => { return "any" }
        TyKind.Null => { return "null" }
        TyKind.Bool => { return "bool" }
        TyKind.Int => { return "int" }
        TyKind.Float => { return "float" }
        TyKind.Str => { return "str" }
        TyKind.Bytes => { return "bytes" }
        TyKind.List => { return "list" }
        TyKind.Dict => { return "dict" }
        TyKind.Set => { return "set" }
        TyKind.Tuple => { return "tuple" }
        TyKind.Range => { return "range" }
        TyKind.Fn => { return "fn" }
        TyKind.Module => { return "module" }
        # A union has no single runtime name. The C answers NULL here and its
        # caller emits a tuple of the members' names instead; `_guard_target`
        # below builds the shape that carries that.
        TyKind.Union => { return null }
        _ => { return _binding_name(t) }
    }
}

# The bare name a nominal type is bound to here, dropping the module
# qualification it is interned under.
fn _binding_name(t: Type) -> str {
    let name = t.name
    if name is null { return "any" }
    let cut = name.rfind(".")
    if cut < 0 { return name }
    return name[cut + 1:]
}

#: A union wider than this guards as `any`.
#:
#: Dropping members would turn the check into a wrong rejection, and no
#: hand-written union comes near it. `MAX_UNION_GUARD` in `src/codegen/codegen.c`.
pub const MAX_UNION_GUARD: int = 32

# What a cast records as the thing to check against.
#
# A plain name for anything with a single runtime type; a `Union` type
# expression holding one name per member when the target is a union, which the
# emitter turns into the tuple constant the C emits.
fn _guard_target(to: Type) -> TypeExpr {
    let name = runtime_type_name(to)
    if name is not null { return TypeExpr.named(name) }
    if to.arg_count() > MAX_UNION_GUARD { return TypeExpr.named("any") }

    var members: list[TypeExpr] = []
    for member in to.args {
        # Unions are flattened, so a member is never itself a union.
        members.push(TypeExpr.named(runtime_type_name(member) ?? "any"))
    }
    return TypeExpr(SynKind.Union, null, members)
}

fn _wrap_in_cast(c: Ctx, node: Node?, to: Type?, widen: bool) -> void {
    if node is null or to is null or node.kind == NodeKind.Cast { return }
    # Erased: a guard would look up a class named `T`.
    if not widen and _mentions_generic_param(to) { return }

    let inner = Node(node.kind, node.fields, node.span)
    c.set_type(inner, c.type_of(node))
    # The resolver keys symbols on node identity, and the operand is a *new*
    # node — the original's identity is about to become the cast. Without
    # re-binding, an identifier that gets wrapped loses its symbol and the
    # emitter reports it undefined: wrapping `total` in a guard turned
    # `return total` into `NameError: undefined name 'total'`. The C has no
    # equivalent step because its symbol lives on the node it copies.
    c.resolution.bind(inner, c.resolution.symbol_of(node))

    # Recorded against the *operand's* kind and carrying both sides, which is
    # what the C's jaiSemaDumpCast does: it is called from inside wrapInCast
    # with `inner->kind` and `inner->type`, before the outer node becomes the
    # cast. Recording `AST_CAST` and the target alone reads as a different
    # decision at the same span, and `make sema-diff` reports it as one.
    if c.dump is not null and node.span.is_known() {
        let field = if widen { "widen" } else { "guard" }
        let source = c.type_of(inner)
        let rendered = f"{render(source)}->{render(to)}"
        c.dump.push(Record(node.span.start, node.span.end, kind_name(inner.kind), field, rendered))
    }

    # `widen` goes on the node, not into a side table beside it. The emitter is
    # what reads it, and it is the whole difference between `OP_TO_FLOAT` and
    # `OP_TYPE_GUARD` — a flag the emitter cannot see is a conversion it cannot
    # emit. `Ctx.mark_widen_cast` is kept in step for anything still asking the
    # context rather than the tree.
    let cast = Node(
        NodeKind.Cast,
        {"operand": inner, "target": _guard_target(to), "widen": widen},
        node.span
    )
    node.kind = cast.kind
    node.fields = cast.fields
    c.set_type(node, to)
    if widen { c.mark_widen_cast(node) }
}

fn _arg_name(record: dict[str, any]) -> str? {
    let value: str? = record.get("name", null)
    return value
}

fn _arg_value(record: dict[str, any]) -> Node? {
    let value: Node? = record.get("value", null)
    return value
}

fn _arg_spread(record: dict[str, any]) -> bool {
    let value: bool = record.get("isSpread", false)
    return value
}

fn _arg_span(record: dict[str, any]) -> Span {
    let value: Span = record.get("span", Span.none())
    return value
}

fn _param_name(record: dict[str, any]) -> str? {
    let value: str? = record.get("name", null)
    return value
}

fn _param_type(record: dict[str, any]) -> TypeExpr? {
    let value: TypeExpr? = record.get("type", null)
    return value
}

fn _param_default(record: dict[str, any]) -> Node? {
    let value: Node? = record.get("defaultValue", null)
    return value
}

fn _param_variadic(record: dict[str, any]) -> bool {
    let value: bool = record.get("isVariadic", false)
    return value
}

fn _param_kwrest(record: dict[str, any]) -> bool {
    let value: bool = record.get("isKwRest", false)
    return value
}

fn _clause_pattern(record: dict[str, any]) -> Node? {
    let value: Node? = record.get("pattern", null)
    return value
}

fn _clause_iterable(record: dict[str, any]) -> Node? {
    let value: Node? = record.get("iterable", null)
    return value
}

fn _clause_conditions(record: dict[str, any]) -> list[Node] {
    let value: list[Node] = record.get("conditions", [])
    return value
}

# `jaiNameDistance` in src/common/diag.c: a bounded Damerau-Levenshtein, three
# rows because a transposition looks back two.
#
# The C compares bytes and this compares scalars. Every name it is asked about
# is an identifier, and the two units agree over ASCII, so no suggestion changes.
fn _name_distance(a: str, b: str, max_distance: int) -> int {
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

# `jaiNameIsCloser`: the new best distance when `candidate` beats `best`, else
# null. The tolerance is the one in `src/common/diag.h`, so a suggestion the
# resolver would refuse is not offered here under a looser rule.
fn _closer(name: str, candidate: str?, best: int) -> int? {
    if candidate is null or name == candidate { return null }
    let length = if name.len() > 3 { name.len() } else { 3 }
    var limit = length // 3
    if limit > 3 { limit = 3 }
    let distance = _name_distance(name, candidate, limit)
    if distance > limit or distance >= best { return null }
    return distance
}

# ------------------------------------------------------------------
# Conditions — there is no truthiness
# ------------------------------------------------------------------

#: The subject of a help that shows the fix.
#:
#: Only a bare name can be quoted back verbatim; for anything larger a
#: placeholder is honest, where echoing a guess at the reader's expression would
#: not be.
#:
#: ```
#: subject_of(Node.ident("total"), "x")   # "total"
#: subject_of(Node.int_lit(1), "x")       # "x"
#: subject_of(null, "xs")                 # "xs"
#: ```
pub fn subject_of(node: Node?, fallback: str) -> str {
    if node is null or node.kind != NodeKind.Ident { return fallback }
    return node.opt_text("name") ?? fallback
}

#: Check that `node` is a `bool`, naming `where` in the diagnostic.
#:
#: Every boolean position in the language funnels through here: `if`, `elif`,
#: `while`, `and`, `or`, `not`, a ternary, a match guard, a comprehension filter
#: and `assert`.
#:
#: ```
#: let c = Ctx(Universe())
#: condition(c, Node.bool_lit(true), "an `if`")
#: c.bag.items.len()                              # 0
#: condition(c, Node.ident("items"), "an `if`")
#: c.bag.items.len()                              # 0, an unresolved name is `any`
#: condition(c, Node.int_lit(1), "an `if`")
#: c.bag.items[0].code                            # "E0401"
#: c.bag.items[0].message
#: # "the condition of an `if` must be `bool`, found `int`"
#: c.bag.items[0].help                            # "compare explicitly: `n != 0`"
#: ```
pub fn condition(c: Ctx, node: Node?, where: str) -> void {
    if node is null { return }
    let t = value(c, node)
    if _is_any(t) or _is_never(t) or t.kind == TyKind.Bool { return }

    let got = render(t)
    let d = c.error(
        E0401_CONDITION_NOT_BOOL,
        node.span,
        f"the condition of {where} must be `bool`, found `{got}`"
    )
    let help = _condition_help(t, node)
    if help is not null { d.with_help(help) }
}

# Written with the condition's own spelling, so the help is a line the reader
# can paste rather than a template to translate.
fn _condition_help(t: Type?, node: Node?) -> str? {
    if t is null { return null }
    if is_optional(t) or t.kind == TyKind.Null {
        let subject = subject_of(node, "x")
        return f"test for absence explicitly: `{subject} != null`"
    }
    match t.kind {
        TyKind.List | TyKind.Dict | TyKind.Set | TyKind.Str | TyKind.Bytes | TyKind.Tuple => {
            let subject = subject_of(node, "xs")
            return f"test the length explicitly: `{subject}.len() > 0`"
        }
        TyKind.Int | TyKind.Float => {
            let subject = subject_of(node, "n")
            return f"compare explicitly: `{subject} != 0`"
        }
        _ => { return "Jaithon has no truthiness; a condition must be a `bool`" }
    }
}

# ------------------------------------------------------------------
# Iteration
# ------------------------------------------------------------------

#: What `for x in node` binds, reporting E0405 when `t` cannot be looped over.
#:
#: A user type is iterable when it implements `Iterable` (spec §5.2); the type
#: system cannot see that through a bare class type, so the declaration is asked.
#:
#: ```
#: let c = Ctx(Universe())
#: let u = c.universe
#: iterable_element(c, null, list_of(u, u.t_str)) == u.t_str   # true
#: iterable_element(c, null, u.t_bytes) == u.t_int             # true
#: iterable_element(c, null, u.t_int) == u.t_any               # true
#: c.bag.items[0].message                                      # "`int` is not iterable"
#: c.bag.items[0].help                       # "loop over a range: `for i in 0..n`"
#: ```
pub fn iterable_element(c: Ctx, node: Node?, t: Type?) -> Type {
    let u = c.universe
    if _is_any(t) or _is_never(t) { return u.t_any }

    # Whether a declared type is iterable is a question about its members and
    # its traits, and a declaration read from another module is laid out on
    # demand — so demand it before asking, or every imported iterable would
    # answer "no members, no traits" and be reported as not iterable.
    let iter_decl = _type_decl_of(t)
    if iter_decl is not null { c.layout_decl(c, iter_decl) }

    let elem = is_iterable(u, t)
    if elem is not null { return elem }

    if t is not null and (t.kind == TyKind.Class or t.kind == TyKind.Trait) {
        if _class_is_iterable(_type_decl_of(t)) { return u.t_any }
    }

    let got = render(t)
    let d = c.error(E0405_NOT_ITERABLE, _span_of(node), f"`{got}` is not iterable")
    if t is not null and t.kind == TyKind.Class {
        d.with_help(f"implement `trait Iterable` on `{got}` to loop over it")
    } elif t is not null and t.kind == TyKind.Int {
        d.with_help("loop over a range: `for i in 0..n`")
    }
    return u.t_any
}

# An unknown shape is assumed iterable: the author knows something this reading
# does not. `opaque_traits` is consulted alongside `traits` because a class that
# implements an imported `Iterable` names it there and nowhere else — the C's
# `TypeDecl` does not make that split, so its loop over `traits` covers both.
fn _class_is_iterable(decl: TypeDecl?) -> bool {
    if decl is null { return true }
    if find_member(decl, "__iter__") is not null { return true }
    if find_member(decl, "__next__") is not null { return true }
    for implemented in decl.traits {
        if implemented.name == "Iterable" { return true }
    }
    return names_opaque_trait(decl, "Iterable")
}

# ------------------------------------------------------------------
# Expressions
# ------------------------------------------------------------------

#: The type of `node` used as a value, rather than for its effect.
#:
#: `expr` with the one extra rule that a `void` expression produces no value.
#: That rule reports nothing today and cannot: `void` is `null` under the name a
#: return position uses, so the two are one interned type in both front ends and
#: the test below is always false. See `_is_void`.
#:
#: ```
#: let c = Ctx(Universe())
#: value(c, Node.str_lit("hi")) == c.universe.t_str   # true
#: value(c, null) == c.universe.t_any                 # true
#: ```
pub fn value(c: Ctx, node: Node?) -> Type {
    let t = expr(c, node)
    if _is_void(c, t) {
        c.error(
            E0409_VOID_VALUE_USED,
            _span_of(node),
            "this expression has type `void` and produces no value"
        )
        return c.universe.t_any
    }
    return t
}

#: Whether `op` is one of the ten arithmetic operators.
#:
#: ```
#: is_arithmetic(OpKind.Pow)        # true
#: is_arithmetic(OpKind.AddWrap)    # true
#: is_arithmetic(OpKind.Band)       # false
#: ```
pub fn is_arithmetic(op: OpKind) -> bool {
    match op {
        OpKind.Add | OpKind.Sub | OpKind.Mul | OpKind.Div => { return true }
        OpKind.FloorDiv | OpKind.Mod | OpKind.Pow => { return true }
        OpKind.AddWrap | OpKind.SubWrap | OpKind.MulWrap => { return true }
        _ => { return false }
    }
}

#: Whether `op` widens an `int` operand against a `float` one (spec §2.5).
#:
#: The wrapping forms are excluded along with the bitwise operators: both are
#: 64-bit integer operations with no float form, so a float operand there is
#: still E0406 rather than a conversion.
#:
#: ```
#: widens_operands(OpKind.Add)        # true
#: widens_operands(OpKind.AddWrap)    # false
#: widens_operands(OpKind.Shl)        # false
#: ```
pub fn widens_operands(op: OpKind) -> bool {
    match op {
        OpKind.AddWrap | OpKind.SubWrap | OpKind.MulWrap => { return false }
        _ => { return is_arithmetic(op) }
    }
}

fn _check_ident(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let symbol = c.resolution.symbol_of(node)
    # The resolver already reported the undefined name; do not pile on.
    if symbol is null { return u.t_any }

    # A folded `const` is inlined at its use site (spec §3).
    if c.fold_constants and symbol.mutability == DeclKind.Const and symbol.name in c.consts {
        let folded = const_eval(node, null, c.consts)
        if not folded.is_none() {
            let literal = const_replace(u, node, folded, c.dump)
            c.set_type(node, c.symbol_type(symbol) ?? literal)
            return c.type_of(node) ?? u.t_any
        }
    }

    # Classes and enums may be referenced before their layout pass ran.
    #
    # The kind test is on the *resolved type* and not only on the symbol,
    # because the symbol kind is not enough. The C reaches a builtin exception
    # through the runtime's own symbol registration — `ValueError` arrives
    # already carrying its type — and this front end has no such registration,
    # so a name the resolver called `Builtin` or `Global` would answer `any`
    # where the C answers `ValueError`. Measured against the C on lib/std,
    # that one gap was every remaining disagreement in tuple.jai and most of
    # func.jai's. Locals and parameters stay excluded, so a binding that
    # shadows a class name is still the binding.
    if c.symbol_type(symbol) is null and symbol.kind != SymbolKind.Local {
        if symbol.kind != SymbolKind.Param {
            let named = c.lookup_type_name(c, symbol.name)
            let nominal = named is not null and named.is_named()
            if named is not null and (_is_type_symbol(symbol.kind) or nominal) {
                c.set_symbol_type(symbol, named)
            }
        }
    }
    return c.symbol_type(symbol) ?? u.t_any
}

# Numeric fusion (spec §2.2). `nodes` are the already-checked elements of an
# unannotated literal position. When the position mixes an exact `int` with an
# exact `float`, every int node is converted in place — retyped if it is a
# literal, wrapped in a widening cast otherwise — so that the element really is
# a float at run time and not merely called one.
#
# Nothing is returned, and that is the point. The caller joins by re-reading the
# recorded types, so the type it computes is the type the tree actually holds: a
# node that failed to convert stays an int, the join stays a union, and the
# failure mode of a bug here is today's behaviour rather than a lie.
fn _fuse_numeric(c: Ctx, nodes: list[Node]) -> void {
    let u = c.universe
    var has_int = false
    var has_float = false
    for node in nodes {
        let t = c.type_of(node)
        # `widen_target` is the exact-int test: it resolves to null for `int?`,
        # `any`, `never` and `list[int]` alike.
        if widen_target(u, t, u.t_float) is not null { has_int = true }
        if t == u.t_float { has_float = true }
    }
    if not has_int or not has_float { return }

    for node in nodes {
        if widen_target(u, c.type_of(node), u.t_float) is null { continue }
        c.widen_to_float(c, node, u.t_float)
    }
}

fn _check_sequence_literal(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let items = node.child_list("items")

    if node.kind == NodeKind.TupleLit {
        let want_tuple = _take_container_context(c, node, TyKind.Tuple, items.len())
        if want_tuple is not null {
            for (position, item) in items.enumerate() {
                let got = value(c, item)
                let _ok = c.require_assignable(
                    c,
                    item,
                    got,
                    want_tuple.args[position],
                    E0400_TYPE_MISMATCH,
                    "mismatched member type"
                )
            }
            return want_tuple
        }
        for item in items { value(c, item) }
        var members: list[Type] = []
        for item in items { members.push(_element_type(c, item)) }
        return tuple_of(u, members)
    }

    let shape = if node.kind == NodeKind.SetLit { TyKind.Set } else { TyKind.List }
    let want = _take_container_context(c, node, shape, 1)

    # Annotated: every element is checked against the declared element type, so
    # a heterogeneous literal reports the offending element rather than the
    # whole container failing against its annotation.
    if want is not null {
        for item in items {
            let got = value(c, item)
            let _ok = c.require_assignable(
                c,
                item,
                got,
                want.args[0],
                E0400_TYPE_MISMATCH,
                "mismatched element type"
            )
        }
        return want
    }

    # Unannotated. Check every element first and discard the reported types,
    # then fuse, then join by re-reading the tree. Folding the join as we went
    # would make the result order-dependent — `[1, "a", 2.0]` would compute
    # join(join(int, str), float) and never see the int and the float as a pair
    # — as well as leaving no point at which the whole literal is known.
    #
    # A set literal is deliberately not fused. A number in a hash position
    # decides which elements are the same element, so converting one can merge
    # two members that were distinct above 2^53 and silently shrink the set. An
    # annotation is an instruction and may do that; an inference is a guess and
    # must never change how many elements a container has.
    for item in items { value(c, item) }
    if node.kind == NodeKind.ListLit { _fuse_numeric(c, items) }

    var elem: Type? = null
    for item in items {
        let t = _element_type(c, item)
        if elem is null { elem = t } else { elem = join(u, elem, t) }
    }
    let element = elem ?? u.t_any
    return if node.kind == NodeKind.SetLit { set_of(u, element) } else { list_of(u, element) }
}

fn _check_dict_literal(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let keys = node.child_list("keys")
    let values = node.child_list("values")
    let want = _take_container_context(c, node, TyKind.Dict, 2)

    if want is not null {
        for (position, key) in keys.enumerate() {
            let got = value(c, key)
            let _ok = c.require_assignable(
                c,
                key,
                got,
                want.args[0],
                E0400_TYPE_MISMATCH,
                "mismatched key type"
            )
            if position >= values.len() { continue }
            let held = values[position]
            let held_type = value(c, held)
            let _fits = c.require_assignable(
                c,
                held,
                held_type,
                want.args[1],
                E0400_TYPE_MISMATCH,
                "mismatched value type"
            )
        }
        return want
    }

    # Unannotated: check, then fuse the values, then join by re-reading the tree
    # — see `_check_sequence_literal` for why the join cannot be folded as we
    # go. Keys are left alone for the same reason a set literal is: they are a
    # hash position, and an inferred conversion there could merge two entries
    # the programmer wrote separately.
    # Key then value, entry by entry, because that is the order the C checks
    # them in and so the order two errors in one entry are reported in.
    for (position, key) in keys.enumerate() {
        value(c, key)
        if position < values.len() { value(c, values[position]) }
    }
    _fuse_numeric(c, values)

    var key_type: Type? = null
    var value_type: Type? = null
    for key in keys {
        let k = _element_type(c, key)
        if key_type is null { key_type = k } else { key_type = join(u, key_type, k) }
    }
    for held in values {
        let v = _element_type(c, held)
        if value_type is null { value_type = v } else { value_type = join(u, value_type, v) }
    }
    return dict_of(u, key_type ?? u.t_any, value_type ?? u.t_any)
}

# One int, one float: the int side widens and the expression is float. The
# conversion goes into the tree here rather than being left to the VM, so
# `n * 1.5` costs a conversion and not a type test.
fn _widen_numeric_mix(c: Ctx, node: Node, op: OpKind, left: Type, right: Type) -> (Type, Type) {
    let u = c.universe
    if not widens_operands(op) { return (left, right) }
    let left_int = left.kind == TyKind.Int and right.kind == TyKind.Float
    let right_int = right.kind == TyKind.Int and left.kind == TyKind.Float
    if not left_int and not right_int { return (left, right) }

    let int_side = if left_int { node.child("left") } else { node.child("right") }
    c.widen_to_float(c, int_side, u.t_float)
    return if left_int { (u.t_float, right) } else { (left, u.t_float) }
}

fn _check_binary(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let op = node.op()
    let checked_left = value(c, node.child("left"))
    let checked_right = value(c, node.child("right"))

    # Identity never constrains its operands: `x is Circle` compares a value
    # against a class, and `x is y` compares two references.
    if op == OpKind.Is or op == OpKind.IsNot { return u.t_bool }

    let (left, right) = _widen_numeric_mix(c, node, op, checked_left, checked_right)
    if _is_never(left) or _is_never(right) { return u.t_never }
    if _is_any(left) or _is_any(right) {
        return if op_is_comparison(op) { u.t_bool } else { u.t_any }
    }

    let result = binary_result(u, op, left, right)
    if result is not null { return result }

    let a = render(left)
    let b = render(right)
    let d = c.error(
        E0406_BAD_OPERAND_TYPES,
        node.span,
        f"operator `{op_text(op)}` does not apply to `{a}` and `{b}`"
    )
    if _type_decl_of(left) is not null {
        d.with_help(f"define the matching dunder method on `{a}`")
    }
    return u.t_any
}

fn _check_logical(c: Ctx, node: Node) -> Type {
    let op = node.op()
    let what = if op == OpKind.And { "`and`" } else { "`or`" }
    let left = node.child("left")
    condition(c, left, what)

    # The right operand of `and` sees the facts the left operand established.
    let mark = c.narrow_mark()
    let facts = c.collect_facts(left, op == OpKind.Or)
    c.narrow_apply(facts, false)
    condition(c, node.child("right"), what)
    c.narrow_restore(mark)
    return c.universe.t_bool
}

fn _check_chain(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let ops: list[OpKind] = node.get("ops")
    let operands = node.child_list("operands")
    if ops.len() <= 0 or operands.len() != ops.len() + 1 { return u.t_bool }

    for operand in operands { value(c, operand) }

    for (position, op) in ops.enumerate() {
        if op == OpKind.Is or op == OpKind.IsNot { continue }
        let left = c.type_of(operands[position]) ?? u.t_any
        let right = c.type_of(operands[position + 1]) ?? u.t_any
        if _is_any(left) or _is_any(right) or _is_never(left) or _is_never(right) { continue }
        if binary_result(u, op, left, right) is not null { continue }

        let a = render(left)
        let b = render(right)
        c.error(
            E0406_BAD_OPERAND_TYPES,
            node.span,
            f"operator `{op_text(op)}` does not apply to `{a}` and `{b}`"
        )
    }
    return u.t_bool
}

fn _check_unary(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let op = node.op()
    if op == OpKind.Not {
        condition(c, node.child("operand"), "`not`")
        return u.t_bool
    }

    let operand = value(c, node.child("operand"))
    if _is_any(operand) or _is_never(operand) { return operand }

    let result = unary_result(u, op, operand)
    if result is not null { return result }

    let got = render(operand)
    c.error(
        E0406_BAD_OPERAND_TYPES,
        node.span,
        f"operator `{op_text(op)}` does not apply to `{got}`"
    )
    return u.t_any
}

fn _check_ternary(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let cond = node.child("cond")
    condition(c, cond, "a ternary")

    let facts = c.collect_facts(cond, false)
    var mark = c.narrow_mark()
    c.narrow_apply(facts, false)
    let then_type = value(c, node.child("thenExpr"))
    c.narrow_restore(mark)

    mark = c.narrow_mark()
    c.narrow_apply(facts, true)
    let else_type = value(c, node.child("elseExpr"))
    c.narrow_restore(mark)

    return join(u, then_type, else_type)
}

fn _check_coalesce(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let left_node = node.child("left")
    let right_node = node.child("right")
    let left = value(c, left_node)
    let right = value(c, right_node)
    if _is_any(left) { return u.t_any }

    let present = narrow(u, left, u.t_null, false) ?? left
    if left.kind != TyKind.Null and not is_optional(left) {
        # Not an error — `a ?? b` on a non-optional is merely dead — but the
        # author almost certainly meant something else.
        let got = render(left)
        let d = c.warn(
            W0102_UNREACHABLE_CODE,
            _span_of(right_node),
            f"the right operand of `??` is unreachable: `{got}` is never null"
        )
        d.with_help("remove the `??`, or make the left operand optional")
    }
    return join(u, present, right)
}

fn _check_index(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let object = node.child("object")
    let subscript = node.child("index")

    # `Box[int](21)` applies type arguments to a generic declaration; erasure
    # makes the result the bare declared type. The resolver settled which
    # reading the brackets have — it is the only pass that can, since an
    # imported name carries no kind here (spec §8).
    if c.resolution.is_type_args(node) {
        let base = value(c, object)
        c.set_type(subscript, u.t_any)
        return base
    }

    if object is not null and object.kind == NodeKind.Ident {
        let symbol = c.resolution.symbol_of(object)
        if symbol is not null and _names_a_type(c, symbol) {
            var base = c.symbol_type(symbol)
            if base is null { base = c.lookup_type_name(c, symbol.name) }
            let resolved = base ?? u.t_any
            c.set_type(object, resolved)
            c.set_type(subscript, u.t_any)
            return resolved
        }
    }

    let target = value(c, object)
    let subscript_type = value(c, subscript)
    if _is_any(target) or _is_never(target) { return u.t_any }

    # `__getitem__` is a member, so the declaration has to be laid out first.
    let index_decl = _type_decl_of(target)
    if index_decl is not null { c.layout_decl(c, index_decl) }

    # A tuple's members have different types, so a constant subscript answers
    # with the one it actually selects: `pair[0]` on `(str, int)` is a `str`,
    # not `str | int`. Only a literal index can be resolved this way — a
    # computed one keeps the join.
    if target.kind == TyKind.Tuple {
        let key = const_eval(subscript, null, c.consts)
        if key.kind == ConstKind.Int {
            let written = key.as_int()
            let at = if written < 0 { written + target.arg_count() } else { written }
            if at < 0 or at >= target.arg_count() {
                c.error(
                    E0404_NOT_INDEXABLE,
                    node.span,
                    f"index {written} is out of range for a {target.arg_count()}-member tuple"
                )
                return u.t_any
            }
            return target.args[at]
        }
    }

    let indexable = is_indexable(u, target)
    if indexable is not null {
        let (want_index, result) = indexable
        if not _is_any(subscript_type) {
            let _ok = c.require_assignable(
                c,
                subscript,
                subscript_type,
                want_index,
                E0400_TYPE_MISMATCH,
                "invalid index type"
            )
        }
        return result
    }

    # `__getitem__` makes a user type indexable.
    let decl = _type_decl_of(target)
    if decl is not null and find_member(decl, "__getitem__") is not null { return u.t_any }
    if decl is null and (target.kind == TyKind.Class or target.kind == TyKind.Trait) {
        return u.t_any
    }

    let got = render(target)
    let d = c.error(E0404_NOT_INDEXABLE, node.span, f"`{got}` cannot be indexed")
    if decl is not null { d.with_help(f"define `__getitem__` on `{got}`") }
    return u.t_any
}

# The names whose brackets mean a generic instantiation rather than a subscript.
fn _names_a_type(c: Ctx, symbol: Symbol) -> bool {
    if _is_type_symbol(symbol.kind) or symbol.kind == SymbolKind.TypeAlias { return true }
    return c.ident_type_decl(symbol) is not null
}

#: The type of `object[start:stop:step]`.
#:
#: A list, a `str`, `bytes` and a tuple slice to themselves; a user type needs
#: `__getitem__`. Every bound that is written must be an `int` or `null`.
#:
#: ```
#: let c = Ctx(Universe())
#: let u = c.universe
#: let text = Node(NodeKind.Slice, {"object": Node.str_lit("abc")})
#: slice(c, text) == u.t_str    # true
#: ```
pub fn slice(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let target = value(c, node.child("object"))

    var parts: list[Node] = []
    for name in ["start", "stop", "step"] {
        let part = node.child(name)
        if part is not null { parts.push(part) }
    }
    for part in parts {
        let t = value(c, part)
        if _is_any(t) or t.kind == TyKind.Int or t.kind == TyKind.Null { continue }
        let _ok = c.require_assignable(
            c,
            part,
            t,
            u.t_int,
            E0400_TYPE_MISMATCH,
            "a slice bound must be an int"
        )
    }

    if _is_any(target) or _is_never(target) { return u.t_any }
    match target.kind {
        TyKind.List | TyKind.Str | TyKind.Bytes | TyKind.Tuple => { return target }
        _ => { }
    }

    let decl = _type_decl_of(target)
    if decl is not null and find_member(decl, "__getitem__") is not null { return u.t_any }
    if decl is null and target.kind == TyKind.Class { return u.t_any }

    let got = render(target)
    c.error(E0404_NOT_INDEXABLE, node.span, f"`{got}` cannot be sliced")
    return u.t_any
}

fn _check_range(c: Ctx, node: Node) -> Type {
    let u = c.universe
    var ends: list[Node] = []
    for name in ["start", "stop"] {
        let end = node.child(name)
        if end is not null { ends.push(end) }
    }
    for end in ends {
        let t = value(c, end)
        if _is_any(t) or t.kind == TyKind.Int { continue }
        let _ok = c.require_assignable(
            c,
            end,
            t,
            u.t_int,
            E0400_TYPE_MISMATCH,
            "a range bound must be an int"
        )
    }
    return u.t_range
}

fn _check_fstring(c: Ctx, node: Node) -> Type {
    let u = c.universe
    for part in node.child_list("parts") {
        if part.kind == NodeKind.StrLit {
            c.set_type(part, u.t_str)
            continue
        }
        value(c, part)
    }
    return u.t_str
}

# ------------------------------------------------------------------
# Member access
# ------------------------------------------------------------------

#: Whether a member declared with `vis` on `owner` is reachable from here.
#:
#: `public` is reachable everywhere and an unknown owner is not checked;
#: `protected` reaches subclasses, and `private` reaches only the declaring
#: class. The class being checked is `Ctx.current_class`, so a lookup outside a
#: class body sees only the public surface.
#:
#: ```
#: let c = Ctx(Universe())
#: visible_from(c, TypeDecl("Box", Span.none()), VIS_PUBLIC)    # true
#: visible_from(c, TypeDecl("Box", Span.none()), VIS_PRIVATE)   # false
#: visible_from(c, null, VIS_PRIVATE)                           # true
#: ```
pub fn visible_from(c: Ctx, owner: TypeDecl?, vis: int) -> bool {
    if vis == VIS_PUBLIC or owner is null { return true }
    let source = c.current_class
    if source is null { return false }
    if vis == VIS_PROTECTED { return is_subclass_of(source, owner) }
    return source == owner
}

#: How a visibility reads in prose.
#:
#: ```
#: visibility_word(VIS_PROTECTED)   # "protected"
#: visibility_word(VIS_PRIVATE)     # "private"
#: visibility_word(VIS_PUBLIC)      # "private"
#: ```
pub fn visibility_word(vis: int) -> str {
    return if vis == VIS_PROTECTED { "protected" } else { "private" }
}

#: Attach a "did you mean" to `d` when a field or member of `decl` is close to
#: `name`.
#:
#: Fields are searched before members, and the closest candidate under the
#: tolerance in `src/common/diag.h` wins. Nothing is attached when none is close
#: enough, which is what keeps a wrong suggestion from being worse than none.
#:
#: ```
#: from jaithon.compile.check.ty import FieldInfo
#: let bag = DiagnosticBag()
#: let d = bag.error("E0410", Span.none(), "boom")
#: let decl = TypeDecl("Point", Span.none())
#: decl.fields.push(FieldInfo("width"))
#: suggest_member(d, decl, "widht")
#: d.help                       # "did you mean `width`?"
#: suggest_member(d, decl, "colour")
#: d.help                       # "did you mean `width`?", nothing is close
#: ```
pub fn suggest_member(d: Diagnostic, decl: TypeDecl, name: str) -> void {
    var best: str? = null
    var best_distance = SUGGEST_NO_MATCH
    for field in decl.fields {
        let distance = _closer(name, field.name, best_distance)
        if distance is null { continue }
        best = field.name
        best_distance = distance
    }
    for member in decl.members {
        let distance = _closer(name, member.name, best_distance)
        if distance is null { continue }
        best = member.name
        best_distance = distance
    }
    if best is not null { d.with_help(f"did you mean `{best}`?") }
}

#: The declaration behind `Account` in `Account.total()`, as opposed to an
#: instance of it.
#:
#: Answers null for anything that is not a bare name bound to a class, trait or
#: enum, and stamps the name's own type on the way past so that a later pass
#: reads a class there and not an unknown.
pub fn static_target_decl(c: Ctx, object: Node?) -> TypeDecl? {
    if object is null or object.kind != NodeKind.Ident { return null }
    let decl = c.ident_type_decl(c.resolution.symbol_of(object))
    if decl is not null { c.set_type(object, _decl_type(c, decl)) }
    return decl
}

# Resolve `name` against a declared type. `static_access` distinguishes
# `Account.count` from `account.count`; enum variants live in the field list
# with their tag in `slot`.
fn _lookup_member(
    c: Ctx,
    site: Node,
    decl: TypeDecl,
    name: str,
    static_access: bool
) -> (Type, Node?) {
    let u = c.universe

    let field = find_field(decl, name)
    if field is not null {
        let owner = owner_of_field(decl, name)
        if not visible_from(c, owner, field.visibility) {
            let owner_name = if owner is null { decl.name } else { owner.name }
            let word = visibility_word(field.visibility)
            let d = c.error(
                E0701_PRIVATE_ACCESS,
                site.span,
                f"`{name}` is {word} to `{owner_name}`"
            )
            if owner is not null { d.label(owner.span, "declared here") }
            d.with_help("mark it `pub` or add an accessor method")
        }
        return (field.ty ?? u.t_any, null)
    }

    var found: MemberInfo? = find_accessor(decl, name, false)
    if found is null { found = find_member(decl, name) }
    if found is not null {
        let owner = owner_of_member(decl, name)
        if not visible_from(c, owner, found.visibility) {
            let owner_name = if owner is null { decl.name } else { owner.name }
            let word = visibility_word(found.visibility)
            let d = c.error(
                E0701_PRIVATE_ACCESS,
                site.span,
                f"`{name}` is {word} to `{owner_name}`"
            )
            if owner is not null { d.label(owner.span, "declared here") }
        }
        return (found.ty ?? u.t_any, found.node)
    }

    # A parent this unit cannot see may well declare the member. Saying "no
    # member" would be a guess, and it was a wrong one: every subclass of an
    # imported class lost the whole inherited surface. Defer to the runtime.
    #
    # An imported declaration is only read for its signatures, so the same
    # caution applies one step further out: it may carry a static extension or a
    # member its own imports contribute, none of which this reading sees. It
    # answers for what it does declare and stays quiet about the rest.
    if inherits_opaquely(decl) { return (u.t_any, null) }

    let got = render(decl.ty)
    let d = c.error(E0410_UNKNOWN_MEMBER, site.span, f"`{got}` has no member `{name}`")
    d.label(decl.span, f"`{decl.name}` is declared here")
    suggest_member(d, decl, name)
    if static_access and decl.is_enum {
        let plural = if decl.variant_count == 1 { "" } else { "s" }
        d.with_help(f"`{decl.name}` has {decl.variant_count} variant{plural}")
    }
    return (u.t_any, null)
}

#: The type of `object.name`, and the declaration node behind it when there is
#: one.
#:
#: The second answer is the C's `AstNode **outDecl`: the `fn` a method load
#: found, which is what lets a call through it be bound against a real parameter
#: list. It is null for a field, for a dynamic load and for anything the C would
#: have left the out-parameter untouched for.
#:
#: `?.` narrows `null` off the receiver before the lookup and puts it back on the
#: result, so `a?.b` on an `A?` is a `B?`; a plain `.` on an optional is E0400.
pub fn member(c: Ctx, node: Node) -> (Type, Node?) {
    let u = c.universe
    let object = node.child("object")
    let name = node.opt_text("name") ?? ""
    let optional = node.kind == NodeKind.OptMember

    let static_decl = static_target_decl(c, object)
    if static_decl is not null {
        c.layout_decl(c, static_decl)
        return _lookup_member(c, node, static_decl, name, true)
    }

    # `super.method()` resolves in the parent, not in the current class.
    if object is not null and object.kind == NodeKind.Super {
        let owner = c.current_class
        let parent = if owner is null { null } else { owner.superclass }
        c.set_type(object, _decl_type(c, parent))
        if parent is null { return (u.t_any, null) }
        return _lookup_member(c, node, parent, name, false)
    }

    var target = value(c, object)

    # `import m` binds a module object, and `m.f` is resolved by name at run
    # time. The signature loader knows what `m` declares, so a call through one
    # is checked exactly like a call through `from m import f`.
    if c.is_module(c, object) {
        let (result, decl_node) = c.module_member(c, object, name)
        return (if optional { optional_of(u, result) } else { result }, decl_node)
    }

    if optional {
        let present = narrow(u, target, u.t_null, false)
        if present is not null { target = present }
    }
    if _is_any(target) or _is_never(target) { return (u.t_any, null) }

    # Builtin containers, modules and the builtin Error classes carry their
    # methods in the runtime, not in a declaration.
    let decl = _type_decl_of(target)
    if decl is null { return (u.t_any, null) }
    c.layout_decl(c, decl)

    if is_optional(target) and not optional {
        let got = render(target)
        let d = c.error(
            E0400_TYPE_MISMATCH,
            node.span,
            f"`{got}` may be null, so `.{name}` is not safe"
        )
        d.with_help(f"use `?.{name}`, or test with `!= null` first")
    }

    let (result, decl_node) = _lookup_member(c, node, decl, name, false)
    return (if optional { optional_of(u, result) } else { result }, decl_node)
}

# ------------------------------------------------------------------
# Calls
# ------------------------------------------------------------------

# What is known about the thing being called. `CalleeInfo` in the C, filled in
# by `_check_call` and read by `_check_args`.
class CalleeInfo {
    #: The function declaration, when the callee is statically known.
    pub var decl: Node?
    #: The callee's type, when it is a function type.
    pub var ty: Type?
    #: What to call it in a diagnostic.
    pub var name: str?
    #: 1 when the declaration writes `self` first.
    pub var self_skip: int
    #: A named callee reports E0602 rather than E0600.
    pub var named: bool
    #: Where the value being called was bound.
    #:
    #: When the callee is a value rather than a function declaration there is no
    #: `fn` to point at, and an arity error that only states a number leaves the
    #: reader guessing which `name` was checked. This and `binding_what` say
    #: where the value was bound and what kind of binding it is.
    pub var binding_span: Span?
    #: What the binding is called in prose: "parameter", "local binding"...
    pub var binding_what: str?
    #: Set when that binding hides a method of the same name on the enclosing
    #: class: the arity then looks wrong only because the reader is reading the
    #: method's signature, which is the whole confusion worth naming.
    pub var shadowed_in: str?

    #: An empty record, which is the C's `memset`.
    pub fn init(self) {
        self.decl = null
        self.ty = null
        self.name = null
        self.self_skip = 0
        self.named = false
        self.binding_span = null
        self.binding_what = null
        self.shadowed_in = null
    }
}

# What to call a binding in prose.
fn _binding_word(kind: SymbolKind) -> str {
    match kind {
        SymbolKind.Param => { return "parameter" }
        SymbolKind.Local => { return "local binding" }
        SymbolKind.Upvalue => { return "captured binding" }
        SymbolKind.Global => { return "global binding" }
        SymbolKind.Field => { return "field" }
        SymbolKind.Method => { return "method" }
        SymbolKind.Class => { return "class" }
        SymbolKind.Trait => { return "trait" }
        SymbolKind.Enum => { return "enum" }
        SymbolKind.EnumVariant => { return "enum variant" }
        SymbolKind.Module => { return "module" }
        SymbolKind.TypeAlias => { return "type alias" }
        SymbolKind.GenericParam => { return "generic parameter" }
        SymbolKind.Builtin => { return "builtin" }
    }
}

# The function declaration this name *denotes*, or null. `Symbol.decl` is the
# node that DECLARES the symbol, which for a parameter, for `self` and for a
# generic parameter is the enclosing `fn` — so a `FnDecl` there alone makes
# every parameter masquerade as the function it belongs to, and a call through
# an `fn`-typed parameter gets bound to its own function's parameter list. Only
# the declaration's back-link, which the resolver writes wherever a `fn` is
# bound to a name, says which binding actually names the function.
#
# The C also follows `Symbol.importedDecl`, so that a call across a module
# boundary binds against the real signature. A Jaithon `Symbol` has no such
# field, so `from m import f` is checked against `f`'s type alone.
fn _denoted_function_decl(c: Ctx, symbol: Symbol?) -> Node? {
    if symbol is null { return null }
    let decl = symbol.decl
    if decl is null or decl.kind != NodeKind.FnDecl { return null }
    return if c.resolution.symbol_of(decl) == symbol { decl } else { null }
}

#: How many leading parameters of `fn_node` are the receiver: 1 or 0.
#:
#: ```
#: from jaithon.ast import param
#: self_skip_of(Node.fn_decl("area", [param("self")], Node.block([])))   # 1
#: self_skip_of(Node.fn_decl("area", [param("n")], Node.block([])))      # 0
#: self_skip_of(null)                                                    # 0
#: ```
pub fn self_skip_of(fn_node: Node?) -> int {
    if fn_node is null or fn_node.kind != NodeKind.FnDecl { return 0 }
    if fn_node.flag("isStatic") { return 0 }
    let params = fn_node.records("params")
    if params.len() <= 0 { return 0 }
    return if _param_name(params[0]) == "self" { 1 } else { 0 }
}

#: The `init` of `decl` or of the nearest ancestor that declares one, and the
#: declaration it was found on.
#:
#: ```
#: let base = TypeDecl("Shape", Span.none())
#: base.members.push(MemberInfo("init", null, 0, false, false, false, false, Node.block([])))
#: let derived = TypeDecl("Circle", Span.none())
#: derived.superclass = base
#: find_init_decl(derived)[1]?.name    # "Shape"
#: find_init_decl(null)[0] is null     # true
#: ```
pub fn find_init_decl(decl: TypeDecl?) -> (Node?, TypeDecl?) {
    var at = decl
    var depth = 0
    while at is not null and depth < MAX_DECL_DEPTH {
        let found = at.find_member("init")
        if found is not null and found.node is not null { return (found.node, at) }
        at = at.superclass
        depth += 1
    }
    return (null, null)
}

fn _check_arg_expr(c: Ctx, node: Node?, want: Type?, what: str) -> void {
    if want is not null { c.apply_context(c, node, want) }
    let got = value(c, node)
    if want is null { return }
    let _ok = c.require_assignable(c, node, got, want, E0400_TYPE_MISMATCH, what)
}

# Bind arguments to parameters and type-check each one. Every argument is
# checked exactly once, whether or not it found a home.
fn _check_args(c: Ctx, call: Node, info: CalleeInfo) -> void {
    let u = c.universe
    let args = call.records("args")
    let arg_count = args.len()
    let arity_code = if info.named { E0602_BAD_CALL_ARITY } else { E0600_ARITY_MISMATCH }
    let name = info.name ?? "this callable"

    # Per-argument expected type and the parameter name it bound to, so the
    # final pass can check each argument once with a precise label.
    let tracked = if arg_count < MAX_ARGS { arg_count } else { MAX_ARGS }
    var targets: list[Type?] = []
    var bound: list[str?] = []
    var slot_index = 0
    while slot_index < tracked {
        targets.push(null)
        bound.push(null)
        slot_index += 1
    }

    var saw_keyword = false
    var saw_spread = false
    for (position, given) in args.enumerate() {
        if _arg_spread(given) {
            saw_spread = true
            continue
        }
        let given_name = _arg_name(given)
        if given_name is null {
            if saw_keyword {
                c.error(
                    E0605_POSITIONAL_AFTER_KEYWORD,
                    _arg_span(given),
                    "positional argument follows a keyword argument"
                )
            }
            continue
        }
        saw_keyword = true
        var earlier = 0
        while earlier < position {
            if _arg_name(args[earlier]) == given_name {
                let d = c.error(
                    E0604_DUPLICATE_ARGUMENT,
                    _arg_span(given),
                    f"argument `{given_name}` is given twice"
                )
                d.label(_arg_span(args[earlier]), "first given here")
                break
            }
            earlier += 1
        }
    }

    let decl_node = info.decl
    var params: list[dict[str, any]] = []
    var param_count = 0
    var declared = false
    if decl_node is not null and decl_node.kind == NodeKind.FnDecl {
        declared = true
        let written = decl_node.records("params")
        param_count = written.len() - info.self_skip
        if param_count < 0 { param_count = 0 }
        if param_count > 0 { params = written[info.self_skip:] }
    }

    # Resolved once, up front, because a declaration in another module has to be
    # read in *its* namespace and the loop below reports this file's errors in
    # this file's. The C swaps the origin for that; `resolve_ast_type` does not
    # take one yet, so an imported annotation naming an imported type is `any`.
    var param_types: list[Type] = []
    if declared and param_count <= MAX_ARGS {
        for parameter in params {
            let written = _param_type(parameter)
            var resolved: Type? = u.t_any
            if written is not null { resolved = c.resolve_ast_type(c, written) }
            param_types.push(resolved ?? u.t_any)
        }
    }

    if declared and param_count <= MAX_ARGS {
        var filled: list[bool] = []
        var at = 0
        while at < param_count {
            filled.push(false)
            at += 1
        }
        var variadic = -1
        var kw_rest = -1
        for (position, parameter) in params.enumerate() {
            if _param_variadic(parameter) { variadic = position }
            if _param_kwrest(parameter) { kw_rest = position }
        }

        var positional = 0
        var reported_overflow = false
        for (position, given) in args.enumerate() {
            if _arg_spread(given) { continue }
            let given_name = _arg_name(given)

            if given_name is null {
                var slot = -1
                if variadic >= 0 and positional >= variadic {
                    slot = variadic
                } elif positional < param_count and not _param_kwrest(params[positional]) {
                    slot = positional
                }

                if slot < 0 {
                    if not saw_spread and not reported_overflow {
                        let plural = if param_count == 1 { "" } else { "s" }
                        let were = if arg_count == 1 { "was" } else { "were" }
                        let d = c.error(
                            arity_code,
                            _arg_span(given),
                            f"`{name}` takes {param_count} argument{plural} but {arg_count} {were} given"
                        )
                        if decl_node is not null {
                            d.label(decl_node.span, f"`{name}` is declared here")
                        }
                        reported_overflow = true
                    }
                } else {
                    if slot != variadic { filled[slot] = true }
                    if position < tracked {
                        targets[position] = param_types[slot]
                        bound[position] = _param_name(params[slot])
                    }
                }
                positional += 1
                continue
            }

            var slot = -1
            for (candidate, parameter) in params.enumerate() {
                if _param_variadic(parameter) or _param_kwrest(parameter) { continue }
                if _param_name(parameter) == given_name {
                    slot = candidate
                    break
                }
            }

            if slot < 0 {
                if kw_rest < 0 {
                    let d = c.error(
                        E0603_UNKNOWN_KEYWORD_ARG,
                        _arg_span(given),
                        f"`{name}` has no parameter named `{given_name}`"
                    )
                    var best: str? = null
                    var best_distance = SUGGEST_NO_MATCH
                    for parameter in params {
                        let distance = _closer(given_name, _param_name(parameter), best_distance)
                        if distance is null { continue }
                        best = _param_name(parameter)
                        best_distance = distance
                    }
                    if best is not null { d.with_help(f"did you mean `{best}`?") }
                }
                continue
            }

            if filled[slot] {
                c.error(
                    E0604_DUPLICATE_ARGUMENT,
                    _arg_span(given),
                    f"argument `{given_name}` is given twice"
                )
            }
            filled[slot] = true
            if position < tracked {
                targets[position] = param_types[slot]
                bound[position] = _param_name(params[slot])
            }
        }

        if not saw_spread {
            var missing = 0
            var first_missing: str? = null
            for (position, parameter) in params.enumerate() {
                if filled[position] { continue }
                if _param_variadic(parameter) or _param_kwrest(parameter) { continue }
                if _param_default(parameter) is not null { continue }
                if first_missing is null { first_missing = _param_name(parameter) }
                missing += 1
            }
            if missing > 0 {
                let plural = if missing == 1 { "" } else { "s" }
                let d = c.error(
                    arity_code,
                    call.span,
                    f"`{name}` is missing {missing} required argument{plural}"
                )
                if first_missing is not null { d.with_help(f"the first one is `{first_missing}`") }
                if decl_node is not null { d.label(decl_node.span, f"`{name}` is declared here") }
            }
        }
    } elif info.ty is not null and info.ty.kind == TyKind.Fn and not saw_spread and not saw_keyword {
        let signature = info.ty
        let want = signature.arg_count()
        let variadic = signature.is_variadic()
        if arg_count != want and not (variadic and arg_count >= want - 1) {
            let plural = if want == 1 { "" } else { "s" }
            let were = if arg_count == 1 { "was" } else { "were" }
            let d = c.error(
                arity_code,
                call.span,
                f"`{name}` takes {want} argument{plural} but {arg_count} {were} given"
            )
            # Show the signature that was actually checked, at its binding.
            let binding_span = info.binding_span
            let binding_what = info.binding_what ?? "binding"
            if info.binding_what is not null and binding_span is not null {
                if binding_span.is_known() {
                    let rendered = render(signature)
                    d.label(
                        binding_span,
                        f"`{name}` is the {binding_what} declared here, of type `{rendered}`"
                    )
                }
            }
            let shadowed = info.shadowed_in
            if shadowed is not null {
                let hint = f"which shadows the method `{shadowed}.{name}`"
                let fix = f"write `self.{name}(...)` to call the method"
                d.with_help(f"this `{name}` is the {binding_what}, {hint}; {fix}")
            }
        }
        var position = 0
        while position < tracked and position < want {
            targets[position] = signature.args[position]
            position += 1
        }
    }

    # One difference from the C, in a place no real program reaches: the label
    # goes into a 96-byte buffer there and a longer one is truncated, an
    # artefact of the buffer rather than a decision. Nothing here truncates.
    for (position, given) in args.enumerate() {
        let held = _arg_value(given)
        if position >= tracked {
            value(c, held)
            continue
        }
        var label = f"argument {position + 1} of `{name}`"
        let bound_name = bound[position]
        if bound_name is not null { label = f"argument `{bound_name}` of `{name}`" }
        let want = if _arg_spread(given) { null } else { targets[position] }
        _check_arg_expr(c, held, want, label)
    }
}

fn _check_construct(c: Ctx, call: Node, decl: TypeDecl) -> Type {
    c.layout_decl(c, decl)

    if decl.is_trait {
        c.error(
            E0710_ABSTRACT_INSTANTIATION,
            call.span,
            f"`{decl.name}` is a trait and cannot be instantiated"
        )
    } elif decl.is_abstract {
        let d = c.error(
            E0710_ABSTRACT_INSTANTIATION,
            call.span,
            f"`{decl.name}` is abstract and cannot be instantiated"
        )
        d.label(decl.span, "declared abstract here")
        for member_info in decl.members {
            if member_info.is_abstract {
                d.with_help(f"`{member_info.name}` has no implementation")
                break
            }
        }
    }

    let info = CalleeInfo()
    info.name = decl.name
    info.named = true
    let (init_decl, _owner) = find_init_decl(decl)
    info.decl = init_decl
    info.self_skip = self_skip_of(init_decl)
    if init_decl is null and call.records("args").len() > 0 {
        c.error(
            E0602_BAD_CALL_ARITY,
            call.span,
            f"`{decl.name}` has no `init`, so it takes no arguments"
        )
    }
    _check_args(c, call, info)
    return _decl_type(c, decl)
}

fn _check_call(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let callee = node.child("callee")
    if callee is null { return u.t_any }

    # ClassName(...) constructs; there is no `new`.
    if callee.kind == NodeKind.Ident {
        let decl = c.ident_type_decl(c.resolution.symbol_of(callee))
        if decl is not null and not decl.is_enum {
            c.set_type(callee, _decl_type(c, decl))
            return _check_construct(c, node, decl)
        }
    }

    # super(...) runs the parent constructor.
    if callee.kind == NodeKind.Super {
        let owner = c.current_class
        let parent = if owner is null { null } else { owner.superclass }
        c.set_type(callee, _decl_type(c, parent))
        let info = CalleeInfo()
        info.name = "super"
        info.named = true
        if parent is not null {
            let (init_decl, _owner) = find_init_decl(parent)
            info.decl = init_decl
            info.self_skip = self_skip_of(init_decl)
            info.name = parent.name
        }
        _check_args(c, node, info)
        return u.t_void
    }

    var callee_decl: Node? = null
    var fn_type: Type? = null
    # `a?.m(x)` short-circuits the whole call, not just the member load (spec
    # §4.2): when `a` is null nothing is called and the call yields null. So the
    # optional `member` put on the member is taken off the callee and put back
    # on the result.
    let opt_chain = callee.kind == NodeKind.OptMember

    if callee.kind == NodeKind.Member or opt_chain {
        let (loaded, loaded_decl) = member(c, callee)
        fn_type = loaded
        callee_decl = loaded_decl
        if opt_chain {
            let present = narrow(u, fn_type, u.t_null, false)
            if present is not null { fn_type = present }
        }
        c.set_type(callee, fn_type)
        # `m.Thing(...)` after `import m` names a class, not a value: the member
        # load yielded the declaration itself, so this is construction and not a
        # call of something that happens to be a class.
        if c.is_module(c, callee.child("object")) {
            let class_decl = _type_decl_of(fn_type)
            if class_decl is not null and not class_decl.is_enum {
                return _check_construct(c, node, class_decl)
            }
        }
    } else {
        fn_type = value(c, callee)
        if callee.kind == NodeKind.Ident {
            callee_decl = _denoted_function_decl(c, c.resolution.symbol_of(callee))
        }
    }

    let info = CalleeInfo()
    info.decl = callee_decl
    if fn_type is not null and fn_type.kind == TyKind.Fn { info.ty = fn_type }
    info.self_skip = self_skip_of(callee_decl)

    if callee.kind == NodeKind.Ident {
        info.name = callee.opt_text("name")
        info.named = true
        # A name that denotes no declaration is a value being called. Record its
        # binding so an arity error can show the signature it checked.
        let symbol = c.resolution.symbol_of(callee)
        if callee_decl is null and symbol is not null {
            info.binding_span = symbol.decl_span
            info.binding_what = _binding_word(symbol.kind)
            # Members are copied down during layout, so one lookup covers
            # inherited and trait methods too.
            let enclosing = c.current_class
            if enclosing is not null and _is_frame_binding(symbol.kind) {
                if find_member(enclosing, symbol.name) is not null {
                    let owner = owner_of_member(enclosing, symbol.name)
                    if owner is not null { info.shadowed_in = owner.name }
                }
            }
        }
    } elif callee.kind == NodeKind.Member or opt_chain {
        info.name = callee.opt_text("name")
        info.named = callee_decl is not null
    }

    # An enum variant with a payload is spelled as a call.
    if fn_type is not null and fn_type.kind == TyKind.Fn {
        _check_args(c, node, info)
        let result = fn_type.ret ?? u.t_any
        return if opt_chain { optional_of(u, result) } else { result }
    }

    if _is_any(fn_type) or _is_never(fn_type) {
        info.decl = null
        info.ty = null
        _check_args(c, node, info)
        return u.t_any
    }

    let decl = _type_decl_of(fn_type)
    if decl is not null { c.layout_decl(c, decl) }
    if decl is not null and find_member(decl, "__call__") is not null {
        info.decl = null
        info.ty = null
        _check_args(c, node, info)
        return u.t_any
    }
    if is_callable(fn_type) {
        info.decl = null
        info.ty = null
        _check_args(c, node, info)
        return u.t_any
    }

    let got = render(fn_type)
    let d = c.error(E0403_NOT_CALLABLE, node.span, f"`{got}` is not callable")
    if decl is not null { d.with_help(f"define `__call__` on `{got}`") }
    info.decl = null
    info.ty = null
    _check_args(c, node, info)
    return u.t_any
}

# The three symbol kinds that occupy a frame slot and so can shadow a method.
fn _is_frame_binding(kind: SymbolKind) -> bool {
    if kind == SymbolKind.Param or kind == SymbolKind.Local { return true }
    return kind == SymbolKind.Upvalue
}

# ------------------------------------------------------------------
# Comprehensions and function expressions
# ------------------------------------------------------------------
fn _check_comprehension(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let comp_kind: CompKind = node.get("compKind")
    var shape = TyKind.List
    if comp_kind == CompKind.Set {
        shape = TyKind.Set
    } elif comp_kind == CompKind.Dict {
        shape = TyKind.Dict
    }
    let arity = if shape == TyKind.Dict { 2 } else { 1 }
    let want = _take_container_context(c, node, shape, arity)

    let mark = c.narrow_mark()
    for clause in node.records("clauses") {
        let over = _clause_iterable(clause)
        let source = value(c, over)
        let elem = iterable_element(c, over, source)
        c.check_pattern(c, _clause_pattern(clause), elem)
        for filter in _clause_conditions(clause) {
            condition(c, filter, "a comprehension filter")
        }
    }

    let element = node.child("element")
    let value_type = if element is null { u.t_any } else { value(c, element) }
    let key_node = node.child("keyExpr")
    var key_type: Type? = null
    if key_node is not null { key_type = value(c, key_node) }
    c.narrow_restore(mark)

    # Annotated: report the element that does not fit rather than the whole
    # comprehension failing against its annotation.
    if want is not null {
        if key_node is not null {
            let _key_ok = c.require_assignable(
                c,
                key_node,
                key_type ?? u.t_any,
                want.args[0],
                E0400_TYPE_MISMATCH,
                "mismatched key type"
            )
        }
        let _ok = c.require_assignable(
            c,
            element,
            value_type,
            want.args[want.arg_count() - 1],
            E0400_TYPE_MISMATCH,
            "mismatched element type"
        )
        return want
    }

    match comp_kind {
        CompKind.List => { return list_of(u, value_type) }
        CompKind.Set => { return set_of(u, value_type) }
        CompKind.Dict => { return dict_of(u, key_type ?? u.t_any, value_type) }
        CompKind.Generator => { return u.t_any }
    }
}

fn _check_function_expr(c: Ctx, node: Node) -> Type {
    c.check_function(c, node)
    let t = c.function_type(c, node)
    c.set_symbol_type(c.resolution.symbol_of(node), t)
    return t
}

fn _check_if_expr(c: Ctx, node: Node) -> Type {
    let u = c.universe
    let cond = node.child("cond")
    condition(c, cond, "an `if`")

    let facts = c.collect_facts(cond, false)
    var mark = c.narrow_mark()
    c.narrow_apply(facts, false)
    let then_type = expr(c, node.child("thenBranch"))
    c.narrow_restore(mark)

    let else_branch = node.child("elseBranch")
    if else_branch is null { return optional_of(u, then_type) }

    let else_facts = c.collect_facts(cond, true)
    mark = c.narrow_mark()
    c.narrow_apply(else_facts, false)
    let else_type = expr(c, else_branch)
    c.narrow_restore(mark)
    return join(u, then_type, else_type)
}

fn _check_cast(c: Ctx, node: Node) -> Type {
    let u = c.universe
    # A cast this checker inserted carries its target in the type table, because
    # a Jaithon `TypeExpr` has no `resolved` field for `wrapInCast` to fill in
    # the way the C's does. A written cast has only the annotation.
    let want = c.resolve_ast_type(c, node.type_field("target")) ?? c.type_of(node)
    value(c, node.child("operand"))
    return want ?? u.t_any
}

#: The type of `node`, recorded for it and folded where it folds.
#:
#: The dispatcher, and the one function every other in this file eventually
#: reaches. It records the answer through `Ctx.set_type` before folding, because
#: `fold.try_fold` reads the checked type to decide whether the literal it
#: produced is the more precise answer.
#:
#: ```
#: let c = Ctx(Universe())
#: render(expr(c, Node.int_lit(1)))                        # "int"
#: let pair = Node(NodeKind.ListLit, {"items": [Node.int_lit(1), Node.float_lit(2.0)]})
#: render(expr(c, pair))                                   # "list[float]"
#: render(expr(c, Node.unary(OpKind.Neg, Node.float_lit(1.5))))   # "float"
#: expr(c, null) == c.universe.t_any                       # true
#: ```
pub fn expr(c: Ctx, node: Node?) -> Type {
    if node is null { return c.universe.t_any }
    let u = c.universe
    var t: Type? = null

    match node.kind {
        NodeKind.IntLit => { t = u.t_int }
        NodeKind.FloatLit => { t = u.t_float }
        NodeKind.StrLit => { t = u.t_str }
        NodeKind.BoolLit => { t = u.t_bool }
        NodeKind.NullLit => { t = u.t_null }
        NodeKind.FString => { t = _check_fstring(c, node) }

        NodeKind.Ident => {
            t = _check_ident(c, node)
            # `_check_ident` may have folded the node into a literal.
            if node.kind != NodeKind.Ident { return c.type_of(node) ?? u.t_any }
        }

        NodeKind.Self => {
            let owner = c.current_class
            t = if owner is null { u.t_any } else { _decl_type(c, owner) }
        }

        NodeKind.Super => {
            let owner = c.current_class
            let parent = if owner is null { null } else { owner.superclass }
            t = if parent is null { u.t_any } else { _decl_type(c, parent) }
        }

        NodeKind.ListLit | NodeKind.SetLit | NodeKind.TupleLit => {
            t = _check_sequence_literal(c, node)
        }

        NodeKind.DictLit => { t = _check_dict_literal(c, node) }
        NodeKind.Unary => { t = _check_unary(c, node) }
        NodeKind.Binary => { t = _check_binary(c, node) }
        NodeKind.Logical => { t = _check_logical(c, node) }
        NodeKind.CompareChain => { t = _check_chain(c, node) }
        NodeKind.Ternary => { t = _check_ternary(c, node) }
        NodeKind.Coalesce => { t = _check_coalesce(c, node) }
        NodeKind.Call => { t = _check_call(c, node) }
        NodeKind.Index => { t = _check_index(c, node) }
        NodeKind.Slice => { t = slice(c, node) }

        NodeKind.Member | NodeKind.OptMember => {
            let (loaded, _decl) = member(c, node)
            t = loaded
        }

        NodeKind.Lambda | NodeKind.AnonFn => { t = _check_function_expr(c, node) }
        NodeKind.Comprehension => { t = _check_comprehension(c, node) }
        NodeKind.Range => { t = _check_range(c, node) }
        NodeKind.IfExpr => { t = _check_if_expr(c, node) }
        NodeKind.MatchExpr => { t = c.match_expr(c, node, false) }
        NodeKind.Cast => { t = _check_cast(c, node) }

        NodeKind.Yield | NodeKind.Await => {
            let operand = node.child("operand")
            if operand is not null { value(c, operand) }
            t = u.t_any
        }

        # A block in expression position is the body of a lambda or of an `if`
        # expression arm; its value is the value of its last expression.
        NodeKind.Block => {
            c.check_block(c, node)
            t = u.t_any
        }

        _ => {
            if is_statement(node.kind) {
                c.check_stmt(c, node)
                t = u.t_void
            } else {
                t = u.t_any
            }
        }
    }

    c.set_type(node, t ?? u.t_any)
    if c.fold_constants {
        let (_folded, after) = try_fold(u, node, c.type_of(node), null, c.consts, c.dump)
        c.set_type(node, after)
    }
    return c.type_of(node) ?? u.t_any
}
