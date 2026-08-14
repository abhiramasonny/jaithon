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
#: this file is the `Resolver` pass that builds them.
from jaithon.ast import DeclKind, Node, NodeKind, Span, TypeExpr, Visibility
from jaithon.compile.diag import DiagnosticBag
from jaithon.compile.symbol import BUILTIN_NAMES, FunctionScope, MAX_LOCALS, MAX_UPVALUES, Resolution, SemaCode, Symbol, SymbolKind, UpvalueRef

#: The kind of region a scope covers.
pub enum ScopeKind { Module, Function, Block, Class, Loop }

class Scope {
    pub let kind: ScopeKind
    pub let depth: int
    pub let fn_index: int
    pub let saved_slot: int
    pub var names: dict[str, Symbol]
    pub var order: list[Symbol]
    pub var pending: dict[str, Span]
    pub var cleared: set[str]
    pub var label: str?
    #: Lowest slot this scope must close on the way out, or -1. ScopeKind.Loop:
    #: anything declared anywhere inside the loop, closed at every iteration
    #: edge (`_mark_captured_in_loop`). ScopeKind.Block: only what the block
    #: declares itself, closed when it exits (`_mark_captured_in_block`).
    pub var capture_base: int

    fn init(self, kind: ScopeKind, depth: int, fn_index: int, saved_slot: int) {
        self.kind = kind
        self.depth = depth
        self.fn_index = fn_index
        self.saved_slot = saved_slot
        self.names = {}
        self.order = []
        self.pending = {}
        self.cleared = set()
        self.label = null
        self.capture_base = -1
    }
}

class ClassInfo {
    pub let decl: Node
    pub let symbol: Symbol?
    pub let scope: Scope
    pub var super_symbol: Symbol?
    pub var total: int
    pub var state: int  # 0 unvisited, 1 visiting, 2 done

    fn init(self, decl: Node, symbol: Symbol?, scope: Scope) {
        self.decl = decl
        self.symbol = symbol
        self.scope = scope
        self.super_symbol = null
        self.total = 0
        self.state = 0
    }
}

#: Resolve one program, reporting into a shared `DiagnosticBag`.
#:
#: Use one instance per program: the scope tree, the symbols and the resolution
#: table all belong to that program.
pub class Resolver {
    #: Where every diagnostic goes; share it with the lexer and the parser so
    #: that a file's diagnostics come out in source order.
    pub let bag: DiagnosticBag
    #: The symbols and function scopes this pass produced.
    pub var out: Resolution
    #: When true an unresolved name becomes a module global instead of `E0200`,
    #: which is what the REPL needs.
    pub let repl_mode: bool

    var scopes: list[Scope]
    var functions: list[FunctionScope]
    var loop_labels: list[str?]
    var classes: list[ClassInfo]
    var self_ok: bool
    var super_ok: bool
    var wildcard_import: bool

    #: Create a resolver reporting into `bag`.
    pub fn init(self, bag: DiagnosticBag? = null, repl_mode: bool = false) {
        self.bag = bag ?? DiagnosticBag()
        self.out = Resolution()
        self.repl_mode = repl_mode
        self.scopes = []
        self.functions = []
        self.loop_labels = []
        self.classes = []
        self.self_ok = false
        self.super_ok = false
        self.wildcard_import = false
    }

    #: Resolve `program`, returning the table the emitter reads.
    #:
    #: The module body is itself a function (`BYTECODE.md` §5), so top-level
    #: blocks still allocate slots even though top-level names are globals.
    #: Ask `bag.has_errors()` whether the result is usable.
    pub fn resolve_program(self, program: Node) -> Resolution {
        if program.kind != NodeKind.Program and program.kind != NodeKind.Block {
            self.bag.error(
                SemaCode.E0902_INTERNAL_ERROR,
                program.span,
                "the resolver expects a program node"
            )
            return self.out
        }
        let module_fn = FunctionScope(program)
        self.functions.push(module_fn)
        self.out.set_scope(program, module_fn)
        self._push(ScopeKind.Module)

        self._block_contents(program.child_list("stmts"))
        for info in self.classes { self._assign_field_slots(info) }

        self._pop()
        self.functions.pop()
        return self.out
    }

    # ---------------------------------------------------------------- #
    # Scopes and slots                                                  #
    # ---------------------------------------------------------------- #
    fn _current(self) -> Scope { return self.scopes[self.scopes.len() - 1] }

    fn _current_fn(self) -> FunctionScope {
        return self.functions[self.functions.len() - 1]
    }

    fn _push(self, kind: ScopeKind) -> Scope {
        let depth = if self.scopes.len() == 0 { 0 } else { self._current().depth + 1 }
        let scope = Scope(kind, depth, self.functions.len() - 1, self._current_fn().next_slot)
        self.scopes.push(scope)
        return scope
    }

    fn _pop(self) -> void {
        let scope = self.scopes.pop()
        for symbol in scope.order { self._report_unused(symbol) }

        let owner = self.functions[scope.fn_index]
        if owner.next_slot > owner.local_count { owner.local_count = owner.next_slot }
        if scope.kind != ScopeKind.Function and scope.kind != ScopeKind.Module {
            owner.next_slot = scope.saved_slot  # sibling blocks reuse the slots
        }
    }

    fn _report_unused(self, symbol: Symbol) -> void {
        if symbol.is_used or symbol.name.starts_with("_") { return }
        if symbol.visibility == Visibility.Public { return }
        let kind = symbol.decl?.kind
        if kind == null { return }

        if kind == NodeKind.Import or kind == NodeKind.FromImport {
            self.bag.warn(
                SemaCode.W0100_UNUSED_IMPORT,
                symbol.decl_span,
                f"unused import `{symbol.name}`",
                "remove the import, or prefix the binding with `_`"
            )
        } elif kind == NodeKind.VarDecl or
            kind == NodeKind.For or
            kind == NodeKind.Try or
            kind == NodeKind.Match or
            kind == NodeKind.MatchExpr or
            kind == NodeKind.Comprehension {
            self.bag.warn(
                SemaCode.W0101_UNUSED_BINDING,
                symbol.decl_span,
                f"unused binding `{symbol.name}`",
                "prefix the name with `_` if this is intentional"
            )
        }
    }

    fn _alloc_slot(self, span: Span) -> int {
        let owner = self._current_fn()
        if owner.next_slot >= MAX_LOCALS {
            if owner.next_slot == MAX_LOCALS {
                self.bag.error(
                    SemaCode.E0208_TOO_MANY_LOCALS,
                    span,
                    f"too many locals in one function (max {MAX_LOCALS})"
                )
                owner.next_slot += 1
            }
            return MAX_LOCALS - 1
        }
        let slot = owner.next_slot
        owner.next_slot += 1
        if owner.next_slot > owner.local_count { owner.local_count = owner.next_slot }
        return slot
    }

    #: Declare `name` in the innermost scope and return its symbol.
    #:
    #: Redeclaring a name in the same scope is `E0302`, or `E0709` in a class
    #: body. `_` is a deliberate discard: redeclaring it is legal and silent.
    pub fn declare(self, name: str, kind: SymbolKind, span: Span, mutability: DeclKind) -> Symbol {
        let scope = self._current()
        let is_discard = name == "_"
        let previous = scope.names.get(name, null)
        let duplicate = previous != null and not is_discard

        if duplicate {
            let in_class = scope.kind == ScopeKind.Class
            var code = SemaCode.E0302_DUPLICATE_DECLARATION
            var message = f"`{name}` is already declared in this scope"
            if in_class {
                code = SemaCode.E0709_DUPLICATE_MEMBER
                message = f"duplicate member `{name}`"
            }
            let diag = self.bag.error(code, span, message)
            diag.label(previous?.decl_span ?? Span.none(), f"previous declaration of `{name}`")
        } elif not name.starts_with("_") and _kind_is_slotted(kind) {
            let shadowed = self._find_shadowed(name)
            if shadowed != null {
                let diag = self.bag.warn(
                    SemaCode.W0103_SHADOWED_BINDING,
                    span,
                    f"declaration of `{name}` shadows an outer " + f"binding"
                )
                diag.label(shadowed.decl_span, f"`{name}` first declared here")
            }
        }

        let symbol = Symbol(kind, name, span, mutability)
        symbol.depth = scope.depth
        if symbol.is_slotted() { symbol.slot = self._alloc_slot(span) }
        scope.names[name] = symbol
        if not duplicate and not is_discard { scope.order.push(symbol) }
        return symbol
    }

    fn _find_shadowed(self, name: str) -> Symbol? {
        var index = self.scopes.len() - 2
        while index >= 0 {
            let scope = self.scopes[index]
            if scope.kind != ScopeKind.Class {
                let found = scope.names.get(name, null)
                if found != null { return if found.is_slotted() { found } else { null } }
            }
            index -= 1
        }
        return null
    }

    fn _binding_kind(self) -> SymbolKind {
        if self._current().kind == ScopeKind.Module { return SymbolKind.Global }
        return SymbolKind.Local
    }

    # ---------------------------------------------------------------- #
    # Name resolution and captures                                      #
    # ---------------------------------------------------------------- #
    fn _add_upvalue(
        self,
        owner: FunctionScope,
        index: int,
        is_local: bool,
        by_value: bool,
        span: Span
    ) -> int {
        for (position, existing) in owner.upvalues.enumerate() {
            let same_slot = existing.index == index and existing.is_local == is_local
            if same_slot and existing.by_value == by_value { return position }
        }
        if owner.upvalues.len() >= MAX_UPVALUES {
            self.bag.error(
                SemaCode.E0209_TOO_MANY_UPVALUES,
                span,
                f"too many captured variables (max {MAX_UPVALUES})"
            )
            return 0
        }
        owner.upvalues.push(UpvalueRef(index, is_local, by_value))
        owner.captures_anything = true
        return owner.upvalues.len() - 1
    }

    fn _capture(self, owner_index: int, slot: int, by_value: bool, span: Span) -> int {
        var index = self._add_upvalue(self.functions[owner_index + 1], slot, true, by_value, span)
        var level = owner_index + 2
        while level < self.functions.len() {
            index = self._add_upvalue(self.functions[level], index, false, by_value, span)
            level += 1
        }
        return index
    }

    #: Spec §6: `var` is captured by reference, `let` and `const` by value. A
    #: nested `fn` declaration is neither — it names the closure being built,
    #: and snapshotting it would capture the null that is there a moment before
    #: the assignment, breaking recursion.
    fn _captures_by_value(self, symbol: Symbol) -> bool {
        if symbol.mutability != DeclKind.Let and symbol.mutability != DeclKind.Const {
            return false
        }
        return symbol.decl == null or symbol.decl.kind != NodeKind.FnDecl
    }

    fn _mark_captured_in_loop(self, scope_index: int, slot: int) -> void {
        if slot < 0 { return }
        var index = scope_index
        while index >= 0 {
            let scope = self.scopes[index]
            if scope.kind == ScopeKind.Loop {
                if scope.capture_base < 0 or slot < scope.capture_base {
                    scope.capture_base = slot
                }
                return
            }
            if scope.kind == ScopeKind.Function or scope.kind == ScopeKind.Module {
                return
            }
            index -= 1
        }
    }

    fn _mark_captured_in_block(self, scope_index: int, slot: int) -> void {
        if slot < 0 { return }
        let scope = self.scopes[scope_index]
        if scope.kind != ScopeKind.Block { return }
        if scope.capture_base < 0 or slot < scope.capture_base {
            scope.capture_base = slot
        }
    }

    fn _upvalue_alias(self, symbol: Symbol, index: int) -> Symbol {
        let owner = self._current_fn()
        for alias in owner.aliases {
            if alias.slot == index and alias.name == symbol.name { return alias }
        }
        let alias = Symbol(SymbolKind.Upvalue, symbol.name, symbol.decl_span, symbol.mutability)
        alias.slot = index
        alias.visibility = symbol.visibility
        alias.is_initialized = true
        alias.is_used = true
        alias.decl = symbol.decl
        owner.aliases.push(alias)
        return alias
    }

    #: Bind `name` to the symbol it denotes here, or report why it cannot be.
    #:
    #: Class bodies are skipped: fields and methods are reachable only through
    #: `self.x` or `Class.x`, never as bare names.
    pub fn resolve_name(self, name: str, span: Span, mark_used: bool) -> Symbol? {
        var cross_fn: Span? = null
        var cross_module = false
        let current_fn = self.functions.len() - 1
        var index = self.scopes.len() - 1

        while index >= 0 {
            let scope = self.scopes[index]
            if scope.kind != ScopeKind.Class {
                let symbol = scope.names.get(name, null)
                if symbol != null {
                    if mark_used { symbol.is_used = true }
                    if symbol.is_slotted() and scope.fn_index != current_fn {
                        symbol.is_captured = true
                        let by_value = self._captures_by_value(symbol)
                        if not by_value {
                            self._mark_captured_in_block(index, symbol.slot)
                            self._mark_captured_in_loop(index, symbol.slot)
                        }
                        let slot = self._capture(scope.fn_index, symbol.slot, by_value, span)
                        return self._upvalue_alias(symbol, slot)
                    }
                    return symbol
                }
            }
            if name in scope.pending and name not in scope.cleared {
                if scope.fn_index == current_fn {
                    self._use_before_decl(name, span, scope.pending[name])
                    return null
                }
                if scope.kind == ScopeKind.Module {
                    cross_module = true
                } elif cross_fn == null {
                    cross_fn = scope.pending[name]
                }
            }
            index -= 1
        }

        if name in BUILTIN_NAMES {
            let symbol = Symbol(SymbolKind.Builtin, name, Span.none(), DeclKind.Const)
            symbol.visibility = Visibility.Public
            symbol.is_initialized = true
            symbol.is_used = true
            return symbol
        }
        if cross_fn != null {
            self._use_before_decl(name, span, cross_fn)
            return null
        }
        if cross_module { return null }
        if self.repl_mode or self.wildcard_import {
            return self._deferred_global(name, span)
        }
        self.bag.error(SemaCode.E0200_UNDEFINED_NAME, span, f"undefined name `{name}`")
        return null
    }

    fn _use_before_decl(self, name: str, use_span: Span, decl_span: Span) -> void {
        let diag = self.bag.error(
            SemaCode.E0201_USE_BEFORE_DECLARATION,
            use_span,
            f"cannot use `{name}` before it is declared",
            "move the declaration above this use"
        )
        diag.label(decl_span, f"`{name}` is declared here, later in this scope")
    }

    fn _deferred_global(self, name: str, span: Span) -> Symbol {
        let root = self.scopes[0]
        let existing = root.names.get(name, null)
        if existing != null { return existing }

        let symbol = Symbol(SymbolKind.Global, name, span, DeclKind.Var)
        symbol.depth = root.depth
        symbol.is_initialized = true
        symbol.is_used = true
        root.names[name] = symbol
        return symbol
    }

    fn _ident(self, node: Node, mark_used: bool) -> void {
        self.out.bind(node, self.resolve_name(node.text("name"), node.span, mark_used))
    }

    fn _is_generic_instantiation(self, node: Node) -> bool {
        let object = node.child("object")
        if object == null { return false }

        if object.kind == NodeKind.Member {
            let base = object.child("object")
            if base == null or base.kind != NodeKind.Ident { return false }
            return _is_import_binding(self.out.symbol_of(base)) and
                self._type_argument_like(node.child("index"), 0)
        }
        if object.kind != NodeKind.Ident { return false }

        let symbol = self.out.symbol_of(object)
        if symbol == null { return false }
        if _names_a_type(symbol.kind) { return true }
        return _is_import_binding(symbol) and self._type_argument_like(node.child("index"), 0)
    }

    fn _type_argument_like(self, node: Node?, depth: int) -> bool {
        if node == null or depth > 8 { return false }
        if node.kind == NodeKind.Ident {
            let name = node.text("name")
            if name in _TYPE_WORDS { return true }
            let symbol = self._lookup_anywhere(name)
            if symbol == null { return false }
            if _names_a_type(symbol.kind) { return true }
            return _is_import_binding(symbol)
        }
        if node.kind == NodeKind.Index {
            return self._type_argument_like(node.child("object"), depth + 1) and
                self._type_argument_like(node.child("index"), depth + 1)
        }
        if node.kind == NodeKind.TupleLit {
            let items = node.child_list("items")
            for item in items {
                if not self._type_argument_like(item, depth + 1) { return false }
            }
            return items.len() > 0
        }
        return false
    }

    fn _lookup_anywhere(self, name: str) -> Symbol? {
        var index = self.scopes.len() - 1
        while index >= 0 {
            let found = self.scopes[index].names.get(name, null)
            if found != null { return found }
            index -= 1
        }
        return null
    }

    fn _mark_type_used(self, declared: TypeExpr?) -> Symbol? {
        return self._mark_named_type(declared?.name)
    }

    fn _mark_named_type(self, name: str?) -> Symbol? {
        if name == null { return null }
        var index = self.scopes.len() - 1
        while index >= 0 {
            let found = self.scopes[index].names.get(name, null)
            if found != null {
                found.is_used = true
                return found
            }
            index -= 1
        }
        return null
    }

    # ---------------------------------------------------------------- #
    # Hoisting                                                          #
    # ---------------------------------------------------------------- #
    fn _hoist_named(
        self,
        name: str?,
        kind: SymbolKind,
        span: Span,
        decl: Node,
        visibility: Visibility
    ) -> Symbol? {
        if name == null { return null }
        let symbol = self.declare(name, kind, span, DeclKind.Let)
        symbol.decl = decl
        symbol.visibility = visibility
        symbol.is_initialized = true
        return symbol
    }

    fn _slot_hoisted(self, symbol: Symbol?, span: Span) -> void {
        if symbol == null { return }
        if self._current().kind == ScopeKind.Module { return }
        symbol.slot = self._alloc_slot(span)
    }

    fn _hoist(self, stmts: list[Node]) -> void {
        for node in stmts {
            let kind = node.kind
            if kind == NodeKind.FnDecl {
                self.out.bind(
                    node,
                    self._hoist_named(
                        node.opt_text("name"),
                        self._binding_kind(),
                        node.span,
                        node,
                        node.visibility()
                    )
                )
            } elif kind == NodeKind.ClassDecl or kind == NodeKind.EnumDecl {
                var symbol_kind = SymbolKind.Enum
                if kind == NodeKind.ClassDecl { symbol_kind = SymbolKind.Class }
                let symbol = self._hoist_named(
                    node.opt_text("name"),
                    symbol_kind,
                    node.span,
                    node,
                    node.visibility()
                )
                self._slot_hoisted(symbol, node.span)
                self.out.bind(node, symbol)
            } elif kind == NodeKind.TraitDecl or kind == NodeKind.TypeDecl {
                var symbol_kind = SymbolKind.TypeAlias
                if kind == NodeKind.TraitDecl { symbol_kind = SymbolKind.Trait }
                self.out.bind(
                    node,
                    self._hoist_named(
                        node.opt_text("name"),
                        symbol_kind,
                        node.span,
                        node,
                        node.visibility()
                    )
                )
            } elif kind == NodeKind.Import {
                let symbol = self._hoist_named(
                    _import_binding_name(node),
                    SymbolKind.Module,
                    node.span,
                    node,
                    Visibility.Private
                )
                self._slot_hoisted(symbol, node.span)
                self.out.bind(node, symbol)
            } elif kind == NodeKind.FromImport {
                if node.flag("isWildcard") {
                    self.wildcard_import = true
                } else {
                    for item in node.records("items") {
                        let bound: str? = item["alias"] ?? item["name"]
                        let span: Span = item["span"]
                        let symbol = self._hoist_named(
                            bound,
                            self._binding_kind(),
                            span,
                            node,
                            Visibility.Private
                        )
                        self._slot_hoisted(symbol, span)
                        self.out.bind(item, symbol)
                    }
                }
            } elif kind == NodeKind.VarDecl {
                self._pattern_names(node.child("pattern"), false)
            }
        }
    }

    fn _pattern_names(self, pattern: Node?, clear: bool) -> void {
        if pattern == null { return }
        let kind = pattern.kind
        if kind == NodeKind.PatBind {
            let scope = self._current()
            let name = pattern.text("name")
            if clear {
                scope.cleared.add(name)
            } elif name not in scope.pending {
                scope.pending[name] = pattern.span
            }
        } elif kind == NodeKind.PatTuple or kind == NodeKind.PatList or kind == NodeKind.PatOr {
            for element in pattern.child_list("elems") {
                self._pattern_names(element, clear)
            }
        } elif kind == NodeKind.PatClass or kind == NodeKind.PatEnum {
            for element in pattern.child_list("subPatterns") {
                self._pattern_names(element, clear)
            }
        }
    }

    # ---------------------------------------------------------------- #
    # Patterns                                                          #
    # ---------------------------------------------------------------- #
    fn _declare_binding(
        self,
        name: str,
        span: Span,
        mutability: DeclKind,
        decl: Node?,
        or_alternative: bool
    ) -> Symbol {
        if or_alternative {
            let existing = self._current().names.get(name, null)
            if existing != null { return existing }
        }
        let symbol = self.declare(name, self._binding_kind(), span, mutability)
        symbol.decl = decl
        symbol.is_initialized = true
        return symbol
    }

    fn _pattern(
        self,
        pattern: Node?,
        declare: bool,
        mutability: DeclKind,
        decl: Node?,
        or_alternative: bool
    ) -> void {
        if pattern == null { return }
        let kind = pattern.kind

        if kind == NodeKind.PatWildcard {
            return
        } elif kind == NodeKind.PatBind {
            let name = pattern.text("name")
            if declare {
                self.out.bind(
                    pattern,
                    self._declare_binding(name, pattern.span, mutability, decl, or_alternative)
                )
            } else {
                self.out.bind(pattern, self.resolve_name(name, pattern.span, true))
            }
        } elif kind == NodeKind.PatLiteral {
            self._node(pattern.child("value"))
        } elif kind == NodeKind.PatRange {
            self._node(pattern.child("lo"))
            self._node(pattern.child("hi"))
        } elif kind == NodeKind.PatTuple or kind == NodeKind.PatList {
            for element in pattern.child_list("elems") {
                self._pattern(element, declare, mutability, decl, or_alternative)
            }
        } elif kind == NodeKind.PatOr {
            for (position, element) in pattern.child_list("elems").enumerate() {
                self._pattern(element, declare, mutability, decl, position > 0 or or_alternative)
            }
        } elif kind == NodeKind.PatClass or kind == NodeKind.PatEnum {
            self._mark_named_type(pattern.opt_text("typeName"))
            for element in pattern.child_list("subPatterns") {
                self._pattern(element, declare, mutability, decl, or_alternative)
            }
        } else {
            self._node(pattern)  # the parser put an expression here
        }
    }

    # ---------------------------------------------------------------- #
    # Functions                                                         #
    # ---------------------------------------------------------------- #
    fn _declare_generics(self, generics: list[dict[str, any]], decl: Node) -> void {
        for generic in generics {
            let symbol = self.declare(
                generic["name"],
                SymbolKind.GenericParam,
                generic["span"],
                DeclKind.Const
            )
            symbol.decl = decl
            symbol.is_initialized = true
            symbol.is_used = true
        }
    }

    fn _function(
        self,
        node: Node,
        is_method: bool,
        require_self: bool,
        class_has_super: bool
    ) -> void {
        let scope_fn = FunctionScope(node)
        self.functions.push(scope_fn)
        self.out.set_scope(node, scope_fn)

        let saved_loops = self.loop_labels
        let saved_self = self.self_ok
        let saved_super = self.super_ok
        self.loop_labels = []  # break and continue never cross a function

        self._push(ScopeKind.Function)
        self._declare_generics(node.records("generics"), node)

        let params = node.records("params")
        let is_static = node.flag("isStatic")
        var first_param = 0

        if is_method and not is_static {
            let receiver = self.declare("self", SymbolKind.Param, node.span, DeclKind.Let)
            receiver.slot = 0
            receiver.decl = node
            receiver.is_initialized = true
            receiver.is_used = true
            scope_fn.next_slot = 1
            scope_fn.local_count = 1
            self.self_ok = true
            self.super_ok = class_has_super

            if params.len() > 0 and params[0]["name"] == "self" {
                self.out.bind(params[0], receiver)
                first_param = 1
            } elif require_self {
                let shown = node.opt_text("name") ?? "?"
                self.bag.error(
                    SemaCode.E0703_MISSING_SELF,
                    node.span,
                    f"instance method `{shown}` must take `self` as its " + f"first parameter",
                    f"write `fn {shown}(self, ...)`, or mark it `static`"
                )
            }
        } else {
            self.self_ok = false
            self.super_ok = false
            if is_static and params.len() > 0 and params[0]["name"] == "self" {
                let shown = node.opt_text("name") ?? "?"
                self.bag.error(
                    SemaCode.E0707_STATIC_WITH_SELF,
                    params[0]["span"],
                    f"static method `{shown}` must not take `self`"
                )
                first_param = 1
            }
        }

        var index = first_param
        while index < params.len() {
            let param = params[index]
            self._node(param["defaultValue"])
            let symbol = self.declare(param["name"], SymbolKind.Param, param["span"], DeclKind.Var)
            symbol.decl = node
            symbol.is_initialized = true
            self.out.bind(param, symbol)
            index += 1
        }

        let body = node.child("body")
        if body != null {
            if node.flag("isExprBody") or body.kind != NodeKind.Block {
                self._node(body)
            } else {
                self._block_contents(body.child_list("stmts"))
            }
        }

        if scope_fn.next_slot > scope_fn.local_count {
            scope_fn.local_count = scope_fn.next_slot
        }
        self._pop()
        self.functions.pop()
        if scope_fn.is_generator { node.set("isGenerator", true) }

        self.loop_labels = saved_loops
        self.self_ok = saved_self
        self.super_ok = saved_super
    }

    # ---------------------------------------------------------------- #
    # Classes, traits, enums                                            #
    # ---------------------------------------------------------------- #
    fn _declare_member(self, method: Node, reuse_existing: bool) -> void {
        let name = method.opt_text("name")
        if name == null { return }
        if reuse_existing {
            let existing = self._current().names.get(name, null)
            if existing != null {
                self.out.bind(method, existing)
                return
            }
        }
        let symbol = self.declare(name, SymbolKind.Method, method.span, DeclKind.Let)
        symbol.decl = method
        symbol.visibility = method.visibility()
        symbol.is_initialized = true
        symbol.is_used = true
        self.out.bind(method, symbol)
    }

    fn _resolve_methods(self, methods: list[Node], require_self: bool, has_super: bool) -> void {
        for method in methods {
            if method.kind == NodeKind.FnDecl {
                self._function(
                    method,
                    true,
                    require_self and not method.flag("isStatic"),
                    has_super
                )
            }
        }
    }

    fn _class_decl(self, node: Node) -> void {
        if self.out.symbol_of(node) == null {
            self.out.bind(
                node,
                self._hoist_named(
                    node.opt_text("name"),
                    SymbolKind.Class,
                    node.span,
                    node,
                    node.visibility()
                )
            )
        }
        let superclass = node.type_field("superclass")
        let super_symbol = self._mark_type_used(superclass)
        for trait_type in node.type_list("traits") { self._mark_type_used(trait_type) }

        let scope = self._push(ScopeKind.Class)
        let info = ClassInfo(node, self.out.symbol_of(node), scope)
        info.super_symbol = super_symbol
        self.classes.push(info)
        self._declare_generics(node.records("generics"), node)

        for field in node.records("fields") {
            let mutability = if field["isLet"] { DeclKind.Let } else { DeclKind.Var }
            let symbol = self.declare(field["name"], SymbolKind.Field, field["span"], mutability)
            symbol.decl = node
            symbol.visibility = field["visibility"]
            symbol.is_initialized = field["defaultValue"] != null
            symbol.is_used = true
            self.out.bind(field, symbol)
        }
        for method in node.child_list("methods") { self._declare_member(method, false) }
        for getter in node.child_list("getters") { self._declare_member(getter, false) }
        for setter in node.child_list("setters") { self._declare_member(setter, true) }

        let saved_self = self.self_ok
        self.self_ok = false
        for field in node.records("fields") { self._node(field["defaultValue"]) }
        self.self_ok = saved_self

        let has_super = superclass != null
        self._resolve_methods(node.child_list("methods"), true, has_super)
        self._resolve_methods(node.child_list("getters"), false, has_super)
        self._resolve_methods(node.child_list("setters"), false, has_super)
        self._pop()
    }

    fn _trait_decl(self, node: Node) -> void {
        if self.out.symbol_of(node) == null {
            self.out.bind(
                node,
                self._hoist_named(
                    node.opt_text("name"),
                    SymbolKind.Trait,
                    node.span,
                    node,
                    node.visibility()
                )
            )
        }
        for parent in node.type_list("supers") { self._mark_type_used(parent) }
        self._push(ScopeKind.Class)
        self._declare_generics(node.records("generics"), node)
        for method in node.child_list("methods") { self._declare_member(method, false) }
        self._resolve_methods(node.child_list("methods"), true, false)
        self._pop()
    }

    fn _enum_decl(self, node: Node) -> void {
        if self.out.symbol_of(node) == null {
            self.out.bind(
                node,
                self._hoist_named(
                    node.opt_text("name"),
                    SymbolKind.Enum,
                    node.span,
                    node,
                    node.visibility()
                )
            )
        }
        self._push(ScopeKind.Class)
        self._declare_generics(node.records("generics"), node)

        for (tag, variant) in node.records("variants").enumerate() {
            let symbol = self.declare(
                variant["name"],
                SymbolKind.EnumVariant,
                variant["span"],
                DeclKind.Const
            )
            symbol.decl = node
            symbol.slot = tag  # the variant tag
            symbol.visibility = node.visibility()
            symbol.is_initialized = true
            symbol.is_used = true
            self.out.bind(variant, symbol)
        }
        for method in node.child_list("methods") { self._declare_member(method, false) }
        self._resolve_methods(node.child_list("methods"), true, false)
        self._pop()
    }

    fn _assign_field_slots(self, info: ClassInfo) -> void {
        if info.state == 2 { return }
        if info.state == 1 {
            let shown = info.decl.opt_text("name") ?? "?"
            self.bag.error(
                SemaCode.E0706_CYCLIC_INHERITANCE,
                info.decl.span,
                f"cyclic inheritance involving class `{shown}`",
                null,
                "a class may not be, directly or indirectly, its own " + "superclass"
            )
            self._assign_own_field_slots(info, 0)
            return
        }

        info.state = 1
        var base = 0
        let parent = self._find_class_info(info.super_symbol)
        if parent != null {
            self._assign_field_slots(parent)
            if info.state == 2 { return }
            base = parent.total
        }
        self._assign_own_field_slots(info, base)
    }

    fn _assign_own_field_slots(self, info: ClassInfo, base: int) -> void {
        var count = 0
        for field in info.decl.records("fields") {
            if field["isStatic"] { continue }  # class-level: no instance slot
            let symbol = info.scope.names.get(field["name"], null)
            if symbol != null and symbol.kind == SymbolKind.Field {
                symbol.slot = base + count
            }
            count += 1
        }
        info.total = base + count
        info.state = 2
    }

    fn _find_class_info(self, symbol: Symbol?) -> ClassInfo? {
        if symbol == null { return null }
        for info in self.classes {
            if info.symbol is symbol { return info }
        }
        return null
    }

    # ---------------------------------------------------------------- #
    # Statements                                                        #
    # ---------------------------------------------------------------- #
    fn _block_contents(self, stmts: list[Node]) -> void {
        self._hoist(stmts)
        for statement in stmts { self._node(statement) }
    }

    fn _var_decl(self, node: Node) -> void {
        let init = node.child("init")
        let decl_kind: DeclKind = node.get("varDeclKind")
        if init != null {
            self._node(init)
        } elif decl_kind != DeclKind.Var {
            let word = if decl_kind == DeclKind.Let { "let" } else { "const" }
            self.bag.error(
                SemaCode.E0304_MISSING_INITIALIZER,
                node.span,
                f"`{word}` declaration requires an initialiser",
                "give it a value, or declare it with `var`"
            )
        }

        let pattern = node.child("pattern")
        self._pattern_names(pattern, true)
        self._pattern(pattern, true, decl_kind, node, false)

        if pattern != null and pattern.kind == NodeKind.PatBind {
            let symbol = self.out.symbol_of(pattern)
            self.out.bind(node, symbol)
            if symbol != null {
                symbol.visibility = node.visibility()
                symbol.is_initialized = init != null
            }
        }
    }

    fn _assign_target(self, node: Node?, compound: bool) -> void {
        if node == null { return }
        if node.kind != NodeKind.Ident {
            self._node(node)
            return
        }
        self._ident(node, compound)
        let symbol = self.out.symbol_of(node)
        if symbol == null { return }

        if symbol.kind == SymbolKind.Builtin {
            self.bag.error(
                SemaCode.E0301_ASSIGN_TO_IMMUTABLE,
                node.span,
                f"cannot assign to built-in `{symbol.name}`"
            )
        } elif symbol.mutability == DeclKind.Const and symbol.kind != SymbolKind.Field {
            let diag = self.bag.error(
                SemaCode.E0305_ASSIGN_TO_CONST,
                node.span,
                f"cannot assign to constant `{symbol.name}`"
            )
            diag.label(symbol.decl_span, f"`{symbol.name}` declared here")
        } elif symbol.mutability == DeclKind.Let {
            let diag = self.bag.error(
                SemaCode.E0301_ASSIGN_TO_IMMUTABLE,
                node.span,
                f"cannot assign to immutable binding `{symbol.name}`",
                f"change the declaration to `var {symbol.name}`"
            )
            diag.label(symbol.decl_span, f"`{symbol.name}` declared immutable here")
        }
    }

    fn _loop(self, node: Node) -> void {
        self._node(node.child("cond"))  # null for `loop`
        let label = node.opt_text("label")
        self.loop_labels.push(label)
        let scope = self._push(ScopeKind.Loop)
        scope.label = label
        self._node(node.child("body"))
        self.out.set_capture_base(node, scope.capture_base)
        self._pop()
        self.loop_labels.pop()
    }

    fn _for(self, node: Node) -> void {
        self._node(node.child("iterable"))
        let label = node.opt_text("label")
        self.loop_labels.push(label)
        let scope = self._push(ScopeKind.Loop)
        scope.label = label

        self._pattern(node.child("pattern"), true, DeclKind.Let, node, false)
        self._node(node.child("body"))
        self.out.set_capture_base(node, scope.capture_base)
        self._pop()
        self.loop_labels.pop()
    }

    fn _jump(self, node: Node, is_break: bool) -> void {
        let word = if is_break { "break" } else { "continue" }
        if self.loop_labels.len() == 0 {
            var code = SemaCode.E0204_CONTINUE_OUTSIDE_LOOP
            if is_break { code = SemaCode.E0203_BREAK_OUTSIDE_LOOP }
            self.bag.error(code, node.span, f"`{word}` outside of a loop")
            return
        }
        let label = node.opt_text("label")
        if label == null { return }
        for candidate in self.loop_labels {
            if candidate == label { return }
        }
        self.bag.error(
            SemaCode.E0202_UNDEFINED_LABEL,
            node.span,
            f"undefined loop label `{label}`",
            f"label the target loop, e.g. `{label}: while ...`"
        )
    }

    fn _try(self, node: Node) -> void {
        self._node(node.child("body"))
        for handler in node.records("catches") {
            self._push(ScopeKind.Block)
            let caught: list[TypeExpr] = handler["types"]
            for entry in caught { self._mark_type_used(entry) }
            let name: str? = handler["name"]
            if name != null {
                let symbol = self.declare(name, SymbolKind.Local, handler["span"], DeclKind.Let)
                symbol.decl = node
                symbol.is_initialized = true
                self.out.bind(handler, symbol)
            }
            self._node(handler["body"])
            self._pop()
        }
        self._node(node.child("finallyBlock"))
    }

    fn _match(self, node: Node) -> void {
        self._node(node.child("subject"))
        for arm in node.records("arms") {
            self._push(ScopeKind.Block)
            self._pattern(arm["pattern"], true, DeclKind.Let, node, false)
            self._node(arm["guard"])
            self._node(arm["body"])
            self._pop()
        }
    }

    fn _comprehension(self, node: Node) -> void {
        self._push(ScopeKind.Block)
        for clause in node.records("clauses") {
            self._node(clause["iterable"])
            self._pattern(clause["pattern"], true, DeclKind.Let, node, false)
            let conditions: list[Node] = clause["conditions"]
            for condition in conditions { self._node(condition) }
        }
        self._node(node.child("keyExpr"))
        self._node(node.child("element"))
        self._pop()
    }

    fn _export(self, node: Node) -> void {
        let names: list[str?] = node.get("names")
        for name in names {
            if name == null { continue }
            var found: Symbol? = null
            var index = self.scopes.len() - 1
            while index >= 0 and found == null {
                found = self.scopes[index].names.get(name, null)
                index -= 1
            }
            if found == null {
                self.bag.error(
                    SemaCode.E0200_UNDEFINED_NAME,
                    node.span,
                    f"cannot export `{name}`: no such declaration here"
                )
                continue
            }
            found.visibility = Visibility.Public
            found.is_used = true
        }
    }

    fn _in_function_body(self) -> bool {
        return self._current_fn().decl.kind != NodeKind.Program
    }

    # ---------------------------------------------------------------- #
    # The walk                                                          #
    # ---------------------------------------------------------------- #
    fn _each(self, nodes: list[Node]) -> void {
        for node in nodes { self._node(node) }
    }

    fn _node(self, node: Node?) -> void {
        if node == null { return }
        match node.kind {
            NodeKind.IntLit | NodeKind.FloatLit | NodeKind.StrLit | NodeKind.BoolLit | NodeKind.NullLit | NodeKind.ModuleDecl | NodeKind.TypeDecl | NodeKind.Import | NodeKind.FromImport => { }
            NodeKind.FString => { self._each(node.child_list("parts")) }
            NodeKind.Ident => { self._ident(node, true) }
            NodeKind.Self => {
                if not self.self_ok {
                    let captured = self.resolve_name("self", node.span, true)
                    if captured != null {
                        self.out.bind(node, captured)
                    } else {
                        self.bag.error(
                            SemaCode.E0206_SELF_OUTSIDE_METHOD,
                            node.span,
                            "`self` is only valid inside a method"
                        )
                    }
                }
            }
            NodeKind.Super => {
                if not self.super_ok {
                    self.bag.error(
                        SemaCode.E0207_SUPER_OUTSIDE_SUBCLASS,
                        node.span,
                        "`super` is only valid inside a method of a class " + "that has a superclass"
                    )
                }
            }
            NodeKind.ListLit | NodeKind.SetLit | NodeKind.TupleLit => {
                self._each(node.child_list("items"))
            }
            NodeKind.DictLit => {
                self._each(node.child_list("keys"))
                self._each(node.child_list("values"))
            }
            NodeKind.Unary | NodeKind.Cast | NodeKind.Await | NodeKind.ThrowExpr => {
                self._node(node.child("operand"))
            }
            NodeKind.Binary | NodeKind.Logical | NodeKind.Coalesce => {
                self._node(node.child("left"))
                self._node(node.child("right"))
            }
            NodeKind.CompareChain => { self._each(node.child_list("operands")) }
            NodeKind.Ternary => {
                self._node(node.child("cond"))
                self._node(node.child("thenExpr"))
                self._node(node.child("elseExpr"))
            }
            NodeKind.Call => {
                self._node(node.child("callee"))
                for argument in node.records("args") { self._node(argument["value"]) }
            }
            NodeKind.Index => {
                self._node(node.child("object"))
                if self._is_generic_instantiation(node) {
                    self.out.set_type_args(node)
                } else {
                    self._node(node.child("index"))
                }
            }
            NodeKind.Slice => {
                self._node(node.child("object"))
                self._node(node.child("start"))
                self._node(node.child("stop"))
                self._node(node.child("step"))
            }
            NodeKind.Member | NodeKind.OptMember => { self._node(node.child("object")) }
            NodeKind.Lambda | NodeKind.AnonFn => { self._function(node, false, false, false) }
            NodeKind.Comprehension => { self._comprehension(node) }
            NodeKind.Range => {
                self._node(node.child("start"))
                self._node(node.child("stop"))
            }
            NodeKind.IfExpr | NodeKind.If => {
                self._node(node.child("cond"))
                self._node(node.child("thenBranch"))
                self._node(node.child("elseBranch"))
            }
            NodeKind.Match | NodeKind.MatchExpr => { self._match(node) }
            NodeKind.Yield => {
                if not self._in_function_body() {
                    self.bag.error(
                        SemaCode.E0210_YIELD_OUTSIDE_GENERATOR,
                        node.span,
                        "`yield` is only valid inside a function"
                    )
                } else {
                    self._current_fn().is_generator = true
                }
                self._node(node.child("operand"))
            }
            NodeKind.Block => {
                let scope = self._push(ScopeKind.Block)
                self._block_contents(node.child_list("stmts"))
                self.out.set_capture_base(node, scope.capture_base)
                self._pop()
            }
            NodeKind.Program => { self._block_contents(node.child_list("stmts")) }
            NodeKind.ExprStmt => { self._node(node.child("expr")) }
            NodeKind.VarDecl => { self._var_decl(node) }
            NodeKind.Assign => {
                self._node(node.child("value"))
                self._assign_target(node.child("target"), node.flag("isCompound"))
            }
            NodeKind.While | NodeKind.Loop => { self._loop(node) }
            NodeKind.For => { self._for(node) }
            NodeKind.Break => { self._jump(node, true) }
            NodeKind.Continue => { self._jump(node, false) }
            NodeKind.Return => {
                if not self._in_function_body() {
                    self.bag.error(
                        SemaCode.E0205_RETURN_OUTSIDE_FUNCTION,
                        node.span,
                        "`return` outside of a function"
                    )
                }
                self._node(node.child("value"))
            }
            NodeKind.Throw => { self._node(node.child("value")) }
            NodeKind.Try => { self._try(node) }
            NodeKind.Defer => {
                if not self._in_function_body() {
                    self.bag.error(
                        SemaCode.E0211_DEFER_OUTSIDE_FUNCTION,
                        node.span,
                        "`defer` is only valid inside a function"
                    )
                } else {
                    self._current_fn().has_defer = true
                }
                self._node(node.child("body"))
            }
            NodeKind.Assert => {
                self._node(node.child("cond"))
                self._node(node.child("message"))
            }
            NodeKind.FnDecl => {
                if self.out.symbol_of(node) == null {
                    self.out.bind(
                        node,
                        self._hoist_named(
                            node.opt_text("name"),
                            self._binding_kind(),
                            node.span,
                            node,
                            node.visibility()
                        )
                    )
                }
                self._function(node, false, false, false)
            }
            NodeKind.ClassDecl => { self._class_decl(node) }
            NodeKind.TraitDecl => { self._trait_decl(node) }
            NodeKind.EnumDecl => { self._enum_decl(node) }
            NodeKind.Export => { self._export(node) }
            _ => { self._pattern(node, false, DeclKind.Let, null, false) }
        }
    }
}

let _TYPE_WORDS: set[str] = {
    "any",
    "void",
    "null",
    "bool",
    "int",
    "float",
    "str",
    "bytes",
    "range",
    "never",
    "list",
    "set",
    "dict",
    "tuple",
}

fn _names_a_type(kind: SymbolKind) -> bool {
    return kind == SymbolKind.Class or
        kind == SymbolKind.Trait or
        kind == SymbolKind.Enum or
        kind == SymbolKind.TypeAlias or
        kind == SymbolKind.GenericParam
}

fn _is_import_binding(symbol: Symbol?) -> bool {
    if symbol == null { return false }
    if symbol.kind == SymbolKind.Module { return true }
    let decl = symbol.decl
    if decl == null { return false }
    return decl.kind == NodeKind.Import or decl.kind == NodeKind.FromImport
}

fn _kind_is_slotted(kind: SymbolKind) -> bool {
    return kind == SymbolKind.Local or kind == SymbolKind.Param
}

fn _import_binding_name(node: Node) -> str? {
    let path = node.opt_text("alias") ?? node.opt_text("path")
    if path == null { return null }
    let parts = path.split(".")
    return parts[parts.len() - 1]
}

#: Resolve `program`, returning the table `emit.jai` reads.
#:
#: Diagnostics go into `bag`; ask `bag.has_errors()` whether the result is
#: usable.
pub fn resolve(program: Node, bag: DiagnosticBag, repl_mode: bool = false) -> Resolution {
    return Resolver(bag, repl_mode).resolve_program(program)
}
