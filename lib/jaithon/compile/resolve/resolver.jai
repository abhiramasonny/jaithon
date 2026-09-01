#: The `Resolver` class -- the scope stack, the function stack, the loop
#: labels, the class list -- and every operation that answers *where does this
#: name live and which slot does it get*: pushing and popping a scope,
#: allocating a frame slot, declaring a name, reporting one that was never
#: used, capturing across a function boundary, and resolving a use to the
#: symbol it denotes.
#:
#: Those stayed methods on purpose. They are called from both production
#: files, and a method call is what the moved code already wrote, so keeping
#: them here turned `self._push(...)` into `r._push(...)` and left every call
#: shape alone. The cost is that they had to be marked `pub` to be reachable
#: at all; the leading underscore still says what they are.
#:
#: The seams. Jaithon rejects a circular import, so `resolve/` is ordered
#: `scope, predicate -> resolver -> decl -> walk` and each file imports only
#: the files before it. Three calls run the other way -- `resolve_program`
#: needs the block walk and the field-slot pass, and `decl.jai` needs the node
#: walk -- and each is a hook here behind a one-line function any file may
#: call. The indirect call sits inside that one line rather than inside the
#: body that made it, so a caller keeps whatever the JIT was already doing for
#: it. `wire.jai` fills the hooks in as it is imported, which is why a
#: `Resolver` built by hand needs no wiring call of its own, and why an unwired
#: hook is a thrown error naming itself rather than a wrong slot number.
#:
#: What is NOT here: anything that walks the tree. A declaration is `decl.jai`
#: and a statement or an expression is `walk.jai`; both reach this class
#: through `r`.
from jaithon.ast import DeclKind, Node, NodeKind, Span, TypeExpr, Visibility
from jaithon.compile.diag import DiagnosticBag
from jaithon.compile.symbol import BUILTIN_NAMES, FunctionScope, MAX_LOCALS, MAX_UPVALUES, Resolution, SemaCode, Symbol, SymbolKind, UpvalueRef
from jaithon.compile.resolve.scope import ClassInfo, Scope, ScopeKind
from jaithon.compile.resolve.predicate import _TYPE_WORDS, _is_import_binding, _kind_is_slotted, _names_a_type

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

    pub var scopes: list[Scope]
    pub var functions: list[FunctionScope]
    pub var loop_labels: list[str?]
    pub var classes: list[ClassInfo]
    pub var self_ok: bool
    pub var super_ok: bool
    pub var wildcard_import: bool

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

        resolve_block_contents(self, program.child_list("stmts"))
        for info in self.classes { assign_field_slots(self, info) }

        self._pop()
        self.functions.pop()
        return self.out
    }

    # --- scopes and slots ---------------------------------------------
    pub fn _current(self) -> Scope { return self.scopes[self.scopes.len() - 1] }

    pub fn _current_fn(self) -> FunctionScope {
        return self.functions[self.functions.len() - 1]
    }

    pub fn _push(self, kind: ScopeKind) -> Scope {
        let depth = if self.scopes.len() == 0 { 0 } else { self._current().depth + 1 }
        let scope = Scope(kind, depth, self.functions.len() - 1, self._current_fn().next_slot)
        self.scopes.push(scope)
        return scope
    }

    pub fn _pop(self) -> void {
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

    pub fn _alloc_slot(self, span: Span) -> int {
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

    pub fn _binding_kind(self) -> SymbolKind {
        if self._current().kind == ScopeKind.Module { return SymbolKind.Global }
        return SymbolKind.Local
    }

    # --- name resolution and captures ---------------------------------
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

    pub fn _ident(self, node: Node, mark_used: bool) -> void {
        self.out.bind(node, self.resolve_name(node.text("name"), node.span, mark_used))
    }

    pub fn _is_generic_instantiation(self, node: Node) -> bool {
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

    pub fn _mark_type_used(self, declared: TypeExpr?) -> Symbol? {
        return self._mark_named_type(declared?.name)
    }

    pub fn _mark_named_type(self, name: str?) -> Symbol? {
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
}

# --- the seams onto the two production files ---------------------------
fn _unwired_resolve_node(r: Resolver, node: Node?) -> void {
    throw ValueError("the resolver seam `resolve_node` was never wired")
}

fn _unwired_resolve_block_contents(r: Resolver, stmts: list[Node]) -> void {
    throw ValueError("the resolver seam `resolve_block_contents` was never wired")
}

fn _unwired_assign_field_slots(r: Resolver, info: ClassInfo) -> void {
    throw ValueError("the resolver seam `assign_field_slots` was never wired")
}

var _hook_resolve_node: fn(Resolver, Node?) -> void = _unwired_resolve_node
var _hook_resolve_block_contents: fn(Resolver, list[Node]) -> void = _unwired_resolve_block_contents
var _hook_assign_field_slots: fn(Resolver, ClassInfo) -> void = _unwired_assign_field_slots

#: `walk.jai`'s `_node`, for a function body and a field initialiser in
#: `decl.jai`, and for the expression a pattern position turned out to hold.
pub fn resolve_node(r: Resolver, node: Node?) -> void {
    _hook_resolve_node(r, node)
}

#: `walk.jai`'s `_block_contents`, for the module body here and for a `fn`
#: body in `decl.jai`. Hoist first, then walk: that two-pass order is what
#: makes mutually recursive declarations legal.
pub fn resolve_block_contents(r: Resolver, stmts: list[Node]) -> void {
    _hook_resolve_block_contents(r, stmts)
}

#: `decl.jai`'s `_assign_field_slots`, for the pass `resolve_program` runs
#: over `classes` once the whole program has been walked. It cannot run
#: earlier: a subclass's base offset is its superclass's field count, and the
#: superclass may be declared below it.
pub fn assign_field_slots(r: Resolver, info: ClassInfo) -> void {
    _hook_assign_field_slots(r, info)
}

#: Point every seam at the function that implements it. `wire.jai` is the
#: only caller, and it calls this as it is imported.
pub fn install_hooks(
    node: fn(Resolver, Node?) -> void,
    block_contents: fn(Resolver, list[Node]) -> void,
    field_slots: fn(Resolver, ClassInfo) -> void
) -> void {
    _hook_resolve_node = node
    _hook_resolve_block_contents = block_contents
    _hook_assign_field_slots = field_slots
}
