#: The checker's shared state, and the default callbacks it starts with.
from jaithon.ast import Node, NodeKind, OpKind, Span, TypeExpr
from jaithon.ast import TypeKind as SynKind
from jaithon.compile.check.assign import assignable, widen_target
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.predicate import is_optional
from jaithon.compile.check.relate import narrow
from jaithon.compile.check.render import render
from jaithon.compile.check.ty import Type, TypeDecl
from jaithon.compile.check.universe import Universe, fn_of
from jaithon.compile.lexer import Diagnostic, DiagnosticBag
from jaithon.compile.resolve import Resolution, Symbol, SymbolKind

#: A `float` reaching an `int`, the direction that loses information.
pub const E0411_INT_FLOAT_MIX: str = "E0411"

#: How many narrowing facts one condition contributes, matching `MAX_FACTS`.
pub const MAX_FACTS: int = 8

#: One thing a condition proved about a binding, on the branch it names.
#:
#: `NarrowFact` in `src/sema/check_internal.h`.
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
#: two side tables a Jaithon AST needs. The callbacks are fields rather than
#: methods so `checker.jai` can replace them without subclassing.
pub class Ctx {
    #: The intern table every answer is built from.
    pub let universe: Universe
    #: Where diagnostics go. A `var` because a foreign scope swaps in a scratch
    #: bag: a signature read out of another module must not raise a diagnostic
    #: against this file. `jaiChkForeignBegin` does the same to `gDiags`.
    pub var bag: DiagnosticBag
    #: What the resolver decided: symbols, and which brackets held type args.
    pub let resolution: Resolution

    #: The class whose body is being checked, for `self`, `super` and visibility.
    pub var current_class: TypeDecl?
    #: What the function whose body is being checked was declared to return, or
    #: null outside every body. The C's `Checker.currentReturnType`.
    pub var current_return_type: Type?
    #: `--strict`: an unannotated parameter is an error. The C's `Checker.strict`.
    pub var strict: bool
    #: Whether a folded operator node is replaced by its literal. The C's
    #: `c->foldConstants`.
    pub var fold_constants: bool
    #: `const` name to the initialiser it folded to, as `fold.jai` takes it.
    pub var consts: dict[str, Node]
    #: How far each declaration has got through layout, keyed by `id(decl)`:
    #: absent or 0 for unstarted, 1 for in progress, 2 for laid out. Layout has
    #: to be idempotent, and a declaration reached again while in progress is
    #: `E0706` — `DeclEntry.status` in `src/sema/check.c`. `decl.jai` writes it.
    pub var layout_status: dict[int, int]

    var node_types: dict[int, Type]
    var symbol_types: dict[int, Type]
    var decls_by_name: dict[str, TypeDecl]
    var decls_by_node: dict[int, TypeDecl]
    # Every registered declaration in registration order, this file's and the
    # imported ones. A list, not a map: layout walks it by index and may append.
    var decl_order: list[TypeDecl]
    # `id(TypeDecl)` of everything in `decl_order`, so registering twice is cheap.
    var decl_seen: set[int]
    # `id(TypeDecl)` to the name it answers to here when an import renamed it;
    # `DeclEntry.alias` in the C.
    var decl_aliases: dict[int, str]
    # id(Symbol) -> the declaration node another module holds for it. The C's
    # `Symbol.importedDecl`; a Jaithon `Symbol` has no such field.
    var imported_decls: dict[int, Node]
    # id(TypeExpr) -> the semantic type it resolved to; the C's AstType.resolved.
    var resolved_types: dict[int, Type]
    #: The signature loader, so a qualified member can be resolved against the
    #: module it came from. Null when nothing set one, and then `m.f` is `any`.
    pub var sig_loader: any?
    #: The module whose namespace names currently resolve in, or null for this
    #: file's own. `gJaiCheck.foreignOrigin`, held as `any` like `TypeDecl.origin`.
    pub var foreign_origin: any?
    #: Every name an import bound in this file. A name in here that denotes no
    #: declaration this checker could load is `any` rather than E0402.
    pub var imported_names: set[str]
    #: The directory of the file being checked, where an import resolves from.
    pub var source_dir: str
    var narrow_symbols: list[Symbol]
    var narrow_saved: list[Type]
    var type_names: list[tuple[str, Type]]

    #: Check that `source` fits `target`, converting or guarding where it must.
    #: `jaiChkRequireAssignable` in `src/sema/check.c`; the default is a port.
    pub var require_assignable: fn(Ctx, Node?, Type?, Type?, str, str) -> bool
    #: Push an expected type down into a literal before it is checked.
    #: `jaiChkApplyContext`. The default is a port of it.
    pub var apply_context: fn(Ctx, Node?, Type?) -> void
    #: Rewrite an `int` expression so that it produces `to`, a `float`.
    #: `jaiChkWidenToFloat`. The default is a port of it.
    pub var widen_to_float: fn(Ctx, Node?, Type) -> void
    #: The type a written annotation denotes. `jaiChkResolveAstType`.
    #: **The default answers null**, so every annotation reads as `any`.
    pub var resolve_ast_type: fn(Ctx, TypeExpr?) -> Type?
    #: The type a bare name denotes, when it denotes one. `jaiChkLookupTypeName`.
    #: The default answers from the declaration registry; `checker.jai` extends it.
    pub var lookup_type_name: fn(Ctx, str?) -> Type?
    #: Lay a declaration out, so that its members can be looked up.
    #: `jaiChkLayoutDecl`. **The default does nothing.**
    pub var layout_decl: fn(Ctx, TypeDecl?) -> void
    #: Bind a pattern against the type it destructures. `jaiChkPattern`.
    #: **The default does nothing**, so a comprehension's loop variable is `any`.
    pub var check_pattern: fn(Ctx, Node?, Type) -> void
    #: Check a block in expression position.
    #: `jaiChkBlock`. **The default does nothing.**
    pub var check_block: fn(Ctx, Node?) -> void
    #: Check a statement that reached expression position.
    #: `jaiChkStmt`. **The default does nothing.**
    pub var check_stmt: fn(Ctx, Node?) -> void
    #: Check a declaration that appeared in a block. `jaiChkDecl`, which
    #: `check_decl.c` owns. **The default does nothing.** `stmt.jai` reaches it.
    pub var check_decl: fn(Ctx, Node?) -> void
    #: Check the body of a lambda or an anonymous function. `jaiChkFunction`.
    #: **The default does nothing**; `function_type` still builds the signature.
    pub var check_function: fn(Ctx, Node?) -> void
    #: The `Fn` type of a function, lambda or anonymous function node.
    #: `jaiChkFunctionType`. The default gives `any` per parameter until
    #: `resolve_ast_type` is supplied; the arity and flags are right anyway.
    pub var function_type: fn(Ctx, Node?) -> Type
    #: The type of a `match` in expression position.
    #: `jaiChkMatchExpr`. **The default answers `any`** and checks nothing.
    pub var match_expr: fn(Ctx, Node?, bool) -> Type
    #: Whether this expression is a module object bound by `import m`.
    #: `moduleSigOfIdent`. **The default answers false.**
    pub var is_module: fn(Ctx, Node?) -> bool
    #: The type of `name` in a module, and the declaration behind it.
    #: `moduleMemberType`. **The default answers `(any, null)`.**
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
        self.layout_status = {}
        self.node_types = {}
        self.symbol_types = {}
        self.decls_by_name = {}
        self.decls_by_node = {}
        self.decl_order = []
        self.decl_seen = set()
        self.decl_aliases = {}
        self.imported_decls = {}
        self.resolved_types = {}
        self.sig_loader = null
        self.foreign_origin = null
        self.imported_names = set()
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

    #: Register a declaration, by name and by declaring node. `visible` is
    #: `DeclEntry.visible`: everything an imported module declares is registered
    #: so annotations can mention siblings, but only what this file imported may
    #: be reached by a bare name. `binding` is `DeclEntry.alias`.
    pub fn register_decl(self, decl: TypeDecl, visible: bool = true, binding: str? = null) -> void {
        let name = binding ?? decl.name
        # A name this file already declares wins; a genuine clash is E0302.
        if visible and name not in self.decls_by_name {
            self.decls_by_name[name] = decl
            if name != decl.name { self.decl_aliases[id(decl)] = name }
        }
        let node = decl.node
        if node is not null and id(node) not in self.decls_by_node {
            self.decls_by_node[id(node)] = decl
        }
        if id(decl) not in self.decl_seen {
            self.decl_seen.add(id(decl))
            self.decl_order.push(decl)
        }
    }

    #: The registered declaration `name` reaches in this file.
    pub fn decl_named(self, name: str?) -> TypeDecl? {
        if name is null { return null }
        return self.decls_by_name.get(name, null)
    }

    #: Every registered declaration, in registration order. The list itself, not
    #: a copy: laying a declaration out can register a type from a third module,
    #: and the caller's loop has to see the entries that appear behind it.
    pub fn all_decls(self) -> list[TypeDecl] {
        return self.decl_order
    }

    #: The name `t` is bound to in this file, or null when nothing binds it.
    #: `jaiTypeDeclBindingName`: a runtime guard names a class the way the
    #: module's globals do — the bare name or the alias, never the qualified one.
    pub fn binding_name(self, t: Type?) -> str? {
        if t is null or not t.is_named() { return null }
        for declared in self.decl_order {
            if declared.ty == t { return self.decl_aliases.get(id(declared), declared.name) }
        }
        return null
    }

    #: Every enum declaration a bare name reaches here, by that name.
    #: The emitter needs it: an enum pattern's tag is the variant's position in
    #: the declaration, which for an imported enum is in another file.
    #: Every type name this compilation knows and what each was declared as --
    #: the file's own declarations and the imported ones alike.
    #:
    #: The prompt carries these from one input to the next. It cannot read the
    #: tree for them: `from m import Point` binds a name to a class as surely
    #: as `class Point` does, and only the checker has been to `m`.
    pub fn declared_kinds(self) -> dict[str, TyKind] {
        var out: dict[str, TyKind] = {}
        for name in self.decls_by_name.keys() {
            let declared = self.decls_by_name[name]
            var kind = TyKind.Class
            if declared.is_enum {
                kind = TyKind.Enum
            } elif declared.is_trait {
                kind = TyKind.Trait
            }
            out[name] = kind
        }
        return out
    }

    pub fn enum_decls(self) -> dict[str, Node] {
        var out: dict[str, Node] = {}
        for name in self.decls_by_name.keys() {
            let declared = self.decls_by_name[name]
            let node = declared.node
            if declared.is_enum and node is not null { out[name] = node }
        }
        return out
    }

    #: Record that `symbol` is bound to a declaration in another module.
    pub fn set_imported_decl(self, symbol: Symbol?, decl: Node?) -> void {
        if symbol is null or decl is null { return }
        self.imported_decls[id(symbol)] = decl
    }

    #: The declaration another module holds for `symbol`, or null.
    pub fn imported_decl(self, symbol: Symbol?) -> Node? {
        if symbol is null { return null }
        return self.imported_decls.get(id(symbol), null)
    }

    #: Resolve names in `origin`'s namespace until `leave_foreign` puts it back.
    #: `jaiChkForeignBegin`. The scope and the bag change together: a name the
    #: foreign module can see and this file cannot is not this file's mistake to
    #: report, so the bag is swapped for a scratch one and dropped. A null origin
    #: is a no-op with a real saved state, so a caller may wrap unconditionally.
    pub fn enter_foreign(self, origin: any?) -> tuple[any?, DiagnosticBag] {
        let saved = (self.foreign_origin, self.bag)
        self.foreign_origin = origin
        if origin is not null { self.bag = DiagnosticBag() }
        return saved
    }

    #: Undo `enter_foreign`, discarding whatever the foreign scope reported.
    pub fn leave_foreign(self, saved: tuple[any?, DiagnosticBag]) -> void {
        self.foreign_origin = saved[0]
        self.bag = saved[1]
    }

    #: The module `node` was parsed from, or null for one written in this file.
    #: `jaiChkOriginOfNode`: every span carries the file it came from.
    pub fn origin_of_node(self, node: Node?) -> any? {
        if node is null { return null }
        let loader = self.sig_loader
        if loader is null { return null }
        return loader.for_file(node.span.file)
    }

    #: The type a written annotation resolved to, if it has been resolved. The C
    #: caches this on `AstType.resolved`; a `TypeExpr` has no such field.
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
    #: `jaiChkIdentTypeDecl`. Anything the resolver did not call a class answers
    #: node, which is how an imported binding is recognised.
    pub fn ident_type_decl(self, symbol: Symbol?) -> TypeDecl? {
        if symbol is null { return null }
        if is_type_symbol(symbol.kind) { return self.decl_named(symbol.name) }
        # An imported binding keeps the resolver's kind — it occupies a local or
        # a global slot and the emitter relies on that — so what it names is
        # reached through the declaration attached to the symbol instead.
        let imported = self.imported_decl(symbol)
        if imported is not null {
            if imported.kind == NodeKind.ClassDecl or
                imported.kind == NodeKind.TraitDecl or
                imported.kind == NodeKind.EnumDecl {
                return self.decl_for_node(imported)
            }
            return null
        }
        return self.decl_for_node(symbol.decl)
    }

    #: Where the type-name scope stack stands, so a declaration can unwind here.
    #: `jaiChkTypeNameMark`. The stack holds names that denote a type only inside
    #: the declaration that introduced them, and is consulted before the registry.
    pub fn type_name_mark(self) -> int {
        return self.type_names.len()
    }

    #: Bind `name` to `ty` until the enclosing declaration ends.
    #: `jaiChkPushTypeName`. A null name or type is dropped.
    pub fn push_type_name(self, name: str?, ty: Type?) -> void {
        if name is null or ty is null { return }
        self.type_names.push((name, ty))
    }

    #: Drop every binding pushed since `mark`. `jaiChkTypeNameRestore`, down to
    #: its tolerance of a mark outside the stack, which it ignores.
    pub fn type_name_restore(self, mark: int) -> void {
        if mark < 0 or mark > self.type_names.len() { return }
        while self.type_names.len() > mark { let _dropped = self.type_names.pop() }
    }

    #: The type `name` denotes in the innermost scope that binds it, or null.
    #: The first half of `jaiChkLookupTypeName`, consulted before the registry.
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
    #: `invert` takes the other edge, which is how an `else` arm reuses them.
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

    #: The type a binding was declared with, before any live fact rewrote it. A
    #: fact only replaces an already-saved type, so the oldest save is the one.
    pub fn declared_type(self, symbol: Symbol?) -> Type? {
        if symbol is null { return null }
        for (position, saved) in self.narrow_symbols.enumerate() {
            if saved == symbol { return self.narrow_saved[position] }
        }
        return self.symbol_type(symbol)
    }

    #: End the fact a store to `symbol` invalidated. `jaiChkNarrowInvalidate`. The
    #: save stack is left alone: `narrow_restore` still has to unwind in order.
    pub fn narrow_invalidate(self, symbol: Symbol?) -> void {
        let declared = self.declared_type(symbol)
        if declared is not null { self.set_symbol_type(symbol, declared) }
    }

    #: What `cond` proves, on the edge `negate` names. `jaiChkCollectFacts`.
    #: `a and b` gives both facts on the true edge and `not (a or b)` both on the
    #: false edge; anything else is a disjunction guaranteeing no single fact.
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

            # `isinstance(x, T)` is the call form of `x is T` and narrows alike.
            NodeKind.Call => {
                let callee = cond.child("callee")
                if callee is null or callee.kind != NodeKind.Ident { return }
                if callee.opt_text("name") != "isinstance" { return }
                let args = cond.records("args")
                if args.len() != 2 { return }
                let symbol = self._narrow_target(arg_value(args[0]))
                let target = self._type_operand(arg_value(args[1]))
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
            # `null` is a unit type with one value (§2.1), so `x is null` narrows
            # as `x == null`. It is not an identifier, so `_type_operand` misses it.
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
        if symbol is not null and is_type_symbol(symbol.kind) {
            let known = self.symbol_type(symbol)
            if known is not null { return known }
            return self.lookup_type_name(self, symbol.name) ?? self.universe.t_any
        }
        # `str`, `int`, `list`... name types as well as being bound as builtin
        # conversion functions (spec §9), so the type reading wins.
        if symbol is not null and symbol.kind != SymbolKind.Builtin { return null }
        return self.lookup_type_name(self, node.opt_text("name"))
    }
}

pub fn is_type_symbol(kind: SymbolKind) -> bool {
    if kind == SymbolKind.Class or kind == SymbolKind.Trait { return true }
    return kind == SymbolKind.Enum
}

fn _is_null_literal(node: Node?) -> bool {
    return node is not null and node.kind == NodeKind.NullLit
}

# A port of `jaiChkRequireAssignable` in src/sema/check.c, where the `any -> T`
# guard and the `int -> float` conversion enter the tree.
fn _default_require_assignable(
    c: Ctx,
    node: Node?,
    source: Type?,
    target: Type?,
    code: str,
    what: str
) -> bool {
    if target is null or source is null { return true }
    if is_never(source) { return true }

    # §2.2's widening first: it is the one assignment whose value changes.
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
    # float to int loses information, so it stays an error and keeps E0411.
    let narrowing = target.kind == TyKind.Int and source.kind == TyKind.Float
    let span = span_of(node)
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
        # Arithmetic over literals is still a literal: `let x: float = 1 + 2`.
        if node.kind == NodeKind.Binary and is_arithmetic_context(node.op()) {
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

    # Container literals adopt the expected type itself, not just the element
    # context: without it `let s: list[Shape] = [Circle(), Square()]` infers
    # `list[Circle | Square]` and is rejected against its own annotation.
    match expected.kind {
        TyKind.List | TyKind.Set => {
            # A comprehension builds a fresh container and takes the annotation.
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
            # checked against the position it is going into, not as a whole.
            c.set_type(node, expected)
        }

        _ => { }
    }
}

# A port of `jaiChkWidenToFloat`. An int literal becomes a float literal; anything
# else gets §2.2's conversion node, which the emitter turns into `OP_TO_FLOAT`.
fn _default_widen_to_float(c: Ctx, node: Node?, to: Type) -> void {
    if node is null { return }
    if _retype_int_literal(c, node) { return }
    _wrap_in_cast(c, node, to, true)
}

# `checker.jai` supplies this; until then every annotation reads as `any`.
fn _default_resolve_ast_type(c: Ctx, written: TypeExpr?) -> Type? {
    return null
}

# The type-name scope stack, then the registry: the first two of the four things
# `jaiChkLookupTypeName` consults. `checker.jai` adds the other two.
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
# `resolve_ast_type` can make of each written type — what `checkArgs` reads.
fn _default_function_type(c: Ctx, node: Node?) -> Type {
    let u = c.universe
    if node is null { return u.t_any }
    var params: list[Type] = []
    var flags = 0
    for parameter in node.records("params") {
        params.push(c.resolve_ast_type(c, param_type(parameter)) ?? u.t_any)
        if param_variadic(parameter) { flags = flags | 1 }
        if param_kwrest(parameter) { flags = flags | 2 }
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

# `jaiChkIsAny`. Statically unknown: `any` by construction, and an unsubstituted
# generic parameter because generics are erased (spec §6.1).
pub fn is_any(t: Type?) -> bool {
    if t is null { return true }
    return t.kind == TyKind.Any or t.kind == TyKind.GenericParam
}

pub fn is_never(t: Type?) -> bool {
    return t is not null and t.kind == TyKind.Never
}

# `jaiChkIsVoid`. `void` is `null` under the name a return position uses, so
# `t_void == t_null` and this is always false, in the C too. Ported because
# E0409 is.
pub fn is_void(c: Ctx, t: Type?) -> bool {
    if t is null { return false }
    return t == c.universe.t_void and c.universe.t_void != c.universe.t_null
}

pub fn span_of(node: Node?) -> Span {
    if node is null { return Span.none() }
    return node.span
}

# `jaiChkDeclType`: the type a declaration declares, or `any` when it has none.
pub fn decl_type(c: Ctx, decl: TypeDecl?) -> Type {
    if decl is null { return c.universe.t_any }
    return decl.ty ?? c.universe.t_any
}

# `jaiChkTypeDeclOf`: the declaration behind a class, trait or enum type.
# Interning already attached it here, so the field is the answer.
pub fn type_decl_of(t: Type?) -> TypeDecl? {
    if t is null { return null }
    if not t.is_named() { return null }
    return t.decl
}

# One half of §2.2's implicit conversion: respelling an int literal as a float
# literal. Not the assignment rule — this decides only whether a conversion node
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
pub fn is_arithmetic_context(op: OpKind) -> bool {
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

#: The name a runtime type guard checks against.
#:
#: `runtimeTypeName` in `src/codegen/codegen.c`. NOT the rendered type: the run
#: time has erased every type argument, so `list[int]` guards as `list`. A class,
#: trait or enum answers its *bare* binding name, or the alias it was imported
#: under, because the VM resolves this operand against the module's globals.
pub fn runtime_type_name(c: Ctx?, t: Type?) -> str? {
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
        # A union has no single runtime name: the C answers NULL and its caller
        # emits a tuple of the members' names, which `_guard_target` builds.
        TyKind.Union => { return null }
        _ => { return _binding_name(c, t) }
    }
}

# The name a nominal type answers to in the file being checked: what the registry
# says it is bound as, and otherwise the qualified name's last component.
fn _binding_name(c: Ctx?, t: Type) -> str {
    if c is not null {
        let bound = c.binding_name(t)
        if bound is not null { return bound }
    }
    let name = t.name
    if name is null { return "any" }
    let cut = name.rfind(".")
    if cut < 0 { return name }
    return name[cut + 1:]
}

#: A union wider than this guards as `any`: dropping members would turn the
#: check into a wrong rejection. `MAX_UNION_GUARD` in `src/codegen/codegen.c`.
pub const MAX_UNION_GUARD: int = 32

# What a cast records as the thing to check against: a plain name, or a `Union`
# type expression of one name per member, which the emitter turns into a tuple.
fn _guard_target(c: Ctx, to: Type) -> TypeExpr {
    let name = runtime_type_name(c, to)
    if name is not null { return TypeExpr.named(name) }
    if to.arg_count() > MAX_UNION_GUARD { return TypeExpr.named("any") }

    var members: list[TypeExpr] = []
    for member in to.args {
        # Unions are flattened, so a member is never itself a union.
        members.push(TypeExpr.named(runtime_type_name(c, member) ?? "any"))
    }
    return TypeExpr(SynKind.Union, null, members)
}

fn _wrap_in_cast(c: Ctx, node: Node?, to: Type?, widen: bool) -> void {
    if node is null or to is null or node.kind == NodeKind.Cast { return }
    # Erased: a guard would look up a class named `T`.
    if not widen and _mentions_generic_param(to) { return }

    let inner = Node(node.kind, node.fields, node.span)
    c.set_type(inner, c.type_of(node))
    # The resolver keys symbols on node identity and the operand is a *new* node,
    # so without re-binding a wrapped identifier loses its symbol and is reported
    # undefined.
    c.resolution.bind(inner, c.resolution.symbol_of(node))
    # And the scope, for the same reason: a wrapped lambda that loses its scope
    # loses its capture list, and compiled to a closure over nothing.
    let scope = c.resolution.scope_of(node)
    if scope is not null { c.resolution.set_scope(inner, scope) }
    # `Resolution` keys four tables on node identity; the operand has to carry
    # all four or the emitter silently emits a different program.
    let base = c.resolution.capture_base_of(node)
    if base != -1 { c.resolution.set_capture_base(inner, base) }
    if c.resolution.is_type_args(node) { c.resolution.set_type_args(inner) }

    # `widen` goes on the node, not only into the side table: the emitter reads
    # it to choose between `OP_TO_FLOAT` and `OP_TYPE_GUARD`.
    let cast = Node(
        NodeKind.Cast,
        {"operand": inner, "target": _guard_target(c, to), "widen": widen},
        node.span
    )
    node.kind = cast.kind
    node.fields = cast.fields
    c.set_type(node, to)
}

pub fn arg_value(record: dict[str, any]) -> Node? {
    let value: Node? = record.get("value", null)
    return value
}

pub fn param_type(record: dict[str, any]) -> TypeExpr? {
    let value: TypeExpr? = record.get("type", null)
    return value
}

pub fn param_variadic(record: dict[str, any]) -> bool {
    let value: bool = record.get("isVariadic", false)
    return value
}

pub fn param_kwrest(record: dict[str, any]) -> bool {
    let value: bool = record.get("isKwRest", false)
    return value
}

#: The subject of a help that shows the fix: a bare name quoted back verbatim,
#: and `fallback` for anything larger.
pub fn subject_of(node: Node?, fallback: str) -> str {
    if node is null or node.kind != NodeKind.Ident { return fallback }
    return node.opt_text("name") ?? fallback
}
