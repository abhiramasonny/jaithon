#: What a declaration binds: the hoisting pass that puts every mutually
#: recursive name in the scope before any statement runs, the pattern walk that
#: binds or resolves the names a pattern mentions, the function form -- its
#: generics, its receiver, its parameters, its body -- and the three type
#: bodies, `class`, `trait` and `enum`, together with the field-slot pass that
#: numbers a class's fields after its superclass has been numbered.
#:
#: One file rather than four because these share the same two moves. Every one
#: of them calls `r.declare` and then decorates the symbol it got back, and
#: every one of them pushes a scope that something further in has to walk. A
#: `class` is a scope full of methods, a method is a function, a function's
#: parameters are patterns, and a pattern is what a `for` binds -- cutting
#: anywhere along that chain buys a seam rather than an idea.
#:
#: What is NOT here: the walk that reaches these. `walk.jai` dispatches
#: `NodeKind.ClassDecl` here and this file calls back through `resolve_node`
#: for the bodies, which is the one direction the import order forbids.
from jaithon.ast import DeclKind, Node, NodeKind, Span, Visibility
from jaithon.compile.symbol import FunctionScope, SemaCode, Symbol, SymbolKind
from jaithon.compile.resolve.scope import ClassInfo, ScopeKind
from jaithon.compile.resolve.predicate import _import_binding_name
from jaithon.compile.resolve.resolver import Resolver, resolve_block_contents, resolve_node

# --- hoisting ---------------------------------------------------------
pub fn _hoist_named(
    r: Resolver,
    name: str?,
    kind: SymbolKind,
    span: Span,
    decl: Node,
    visibility: Visibility
) -> Symbol? {
    if name == null { return null }
    let symbol = r.declare(name, kind, span, DeclKind.Let)
    symbol.decl = decl
    symbol.visibility = visibility
    symbol.is_initialized = true
    return symbol
}

fn _slot_hoisted(r: Resolver, symbol: Symbol?, span: Span) -> void {
    if symbol == null { return }
    if r._current().kind == ScopeKind.Module { return }
    symbol.slot = r._alloc_slot(span)
}

pub fn _hoist(r: Resolver, stmts: list[Node]) -> void {
    for node in stmts {
        let kind = node.kind
        if kind == NodeKind.FnDecl {
            r.out.bind(
                node,
                _hoist_named(
                    r,
                    node.opt_text("name"),
                    r._binding_kind(),
                    node.span,
                    node,
                    node.visibility()
                )
            )
        } elif kind == NodeKind.ClassDecl or kind == NodeKind.EnumDecl {
            var symbol_kind = SymbolKind.Enum
            if kind == NodeKind.ClassDecl { symbol_kind = SymbolKind.Class }
            let symbol = _hoist_named(
                r,
                node.opt_text("name"),
                symbol_kind,
                node.span,
                node,
                node.visibility()
            )
            _slot_hoisted(r, symbol, node.span)
            r.out.bind(node, symbol)
        } elif kind == NodeKind.TraitDecl or kind == NodeKind.TypeDecl {
            var symbol_kind = SymbolKind.TypeAlias
            if kind == NodeKind.TraitDecl { symbol_kind = SymbolKind.Trait }
            r.out.bind(
                node,
                _hoist_named(
                    r,
                    node.opt_text("name"),
                    symbol_kind,
                    node.span,
                    node,
                    node.visibility()
                )
            )
        } elif kind == NodeKind.Import {
            let symbol = _hoist_named(
                r,
                _import_binding_name(node),
                SymbolKind.Module,
                node.span,
                node,
                Visibility.Private
            )
            _slot_hoisted(r, symbol, node.span)
            r.out.bind(node, symbol)
        } elif kind == NodeKind.FromImport {
            if node.flag("isWildcard") {
                r.wildcard_import = true
            } else {
                for item in node.records("items") {
                    let bound: str? = item["alias"] ?? item["name"]
                    let span: Span = item["span"]
                    let symbol = _hoist_named(
                        r,
                        bound,
                        r._binding_kind(),
                        span,
                        node,
                        Visibility.Private
                    )
                    _slot_hoisted(r, symbol, span)
                    r.out.bind(item, symbol)
                }
            }
        } elif kind == NodeKind.VarDecl {
            _pattern_names(r, node.child("pattern"), false)
        }
    }
}

pub fn _pattern_names(r: Resolver, pattern: Node?, clear: bool) -> void {
    if pattern == null { return }
    let kind = pattern.kind
    if kind == NodeKind.PatBind {
        let scope = r._current()
        let name = pattern.text("name")
        if clear {
            scope.cleared.add(name)
        } elif name not in scope.pending {
            scope.pending[name] = pattern.span
        }
    } elif kind == NodeKind.PatTuple or kind == NodeKind.PatList or kind == NodeKind.PatOr {
        for element in pattern.child_list("elems") {
            _pattern_names(r, element, clear)
        }
    } elif kind == NodeKind.PatClass or kind == NodeKind.PatEnum {
        for element in pattern.child_list("subPatterns") {
            _pattern_names(r, element, clear)
        }
    }
}

# --- patterns ---------------------------------------------------------
fn _declare_binding(
    r: Resolver,
    name: str,
    span: Span,
    mutability: DeclKind,
    decl: Node?,
    or_alternative: bool
) -> Symbol {
    if or_alternative {
        let existing = r._current().names.get(name, null)
        if existing != null { return existing }
    }
    let symbol = r.declare(name, r._binding_kind(), span, mutability)
    symbol.decl = decl
    symbol.is_initialized = true
    return symbol
}

pub fn _pattern(
    r: Resolver,
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
            r.out.bind(
                pattern,
                _declare_binding(r, name, pattern.span, mutability, decl, or_alternative)
            )
        } else {
            r.out.bind(pattern, r.resolve_name(name, pattern.span, true))
        }
    } elif kind == NodeKind.PatLiteral {
        resolve_node(r, pattern.child("value"))
    } elif kind == NodeKind.PatRange {
        resolve_node(r, pattern.child("lo"))
        resolve_node(r, pattern.child("hi"))
    } elif kind == NodeKind.PatTuple or kind == NodeKind.PatList {
        for element in pattern.child_list("elems") {
            _pattern(r, element, declare, mutability, decl, or_alternative)
        }
    } elif kind == NodeKind.PatOr {
        for (position, element) in pattern.child_list("elems").enumerate() {
            _pattern(r, element, declare, mutability, decl, position > 0 or or_alternative)
        }
    } elif kind == NodeKind.PatClass or kind == NodeKind.PatEnum {
        r._mark_named_type(pattern.opt_text("typeName"))
        for element in pattern.child_list("subPatterns") {
            _pattern(r, element, declare, mutability, decl, or_alternative)
        }
    } else {
        resolve_node(r, pattern)  # the parser put an expression here
    }
}

# --- functions --------------------------------------------------------
fn _declare_generics(r: Resolver, generics: list[dict[str, any]], decl: Node) -> void {
    for generic in generics {
        let symbol = r.declare(
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

pub fn _function(
    r: Resolver,
    node: Node,
    is_method: bool,
    require_self: bool,
    class_has_super: bool
) -> void {
    let scope_fn = FunctionScope(node)
    r.functions.push(scope_fn)
    r.out.set_scope(node, scope_fn)

    let saved_loops = r.loop_labels
    let saved_self = r.self_ok
    let saved_super = r.super_ok
    r.loop_labels = []  # break and continue never cross a function

    r._push(ScopeKind.Function)
    _declare_generics(r, node.records("generics"), node)

    let params = node.records("params")
    let is_static = node.flag("isStatic")
    var first_param = 0

    if is_method and not is_static {
        let receiver = r.declare("self", SymbolKind.Param, node.span, DeclKind.Let)
        receiver.slot = 0
        receiver.decl = node
        receiver.is_initialized = true
        receiver.is_used = true
        scope_fn.next_slot = 1
        scope_fn.local_count = 1
        r.self_ok = true
        r.super_ok = class_has_super

        if params.len() > 0 and params[0]["name"] == "self" {
            r.out.bind(params[0], receiver)
            first_param = 1
        } elif require_self {
            let shown = node.opt_text("name") ?? "?"
            r.bag.error(
                SemaCode.E0703_MISSING_SELF,
                node.span,
                f"instance method `{shown}` must take `self` as its " + f"first parameter",
                f"write `fn {shown}(self, ...)`, or mark it `static`"
            )
        }
    } else {
        r.self_ok = false
        r.super_ok = false
        if is_static and params.len() > 0 and params[0]["name"] == "self" {
            let shown = node.opt_text("name") ?? "?"
            r.bag.error(
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
        resolve_node(r, param["defaultValue"])
        let symbol = r.declare(param["name"], SymbolKind.Param, param["span"], DeclKind.Var)
        symbol.decl = node
        symbol.is_initialized = true
        r.out.bind(param, symbol)
        index += 1
    }

    let body = node.child("body")
    if body != null {
        if node.flag("isExprBody") or body.kind != NodeKind.Block {
            resolve_node(r, body)
        } else {
            resolve_block_contents(r, body.child_list("stmts"))
        }
    }

    if scope_fn.next_slot > scope_fn.local_count {
        scope_fn.local_count = scope_fn.next_slot
    }
    r._pop()
    r.functions.pop()
    if scope_fn.is_generator { node.set("isGenerator", true) }

    r.loop_labels = saved_loops
    r.self_ok = saved_self
    r.super_ok = saved_super
}

# --- classes, traits, enums -------------------------------------------
fn _declare_member(r: Resolver, method: Node, reuse_existing: bool) -> void {
    let name = method.opt_text("name")
    if name == null { return }
    if reuse_existing {
        let existing = r._current().names.get(name, null)
        if existing != null {
            r.out.bind(method, existing)
            return
        }
    }
    let symbol = r.declare(name, SymbolKind.Method, method.span, DeclKind.Let)
    symbol.decl = method
    symbol.visibility = method.visibility()
    symbol.is_initialized = true
    symbol.is_used = true
    r.out.bind(method, symbol)
}

fn _resolve_methods(r: Resolver, methods: list[Node], require_self: bool, has_super: bool) -> void {
    for method in methods {
        if method.kind == NodeKind.FnDecl {
            _function(r, method, true, require_self and not method.flag("isStatic"), has_super)
        }
    }
}

pub fn _class_decl(r: Resolver, node: Node) -> void {
    if r.out.symbol_of(node) == null {
        r.out.bind(
            node,
            _hoist_named(
                r,
                node.opt_text("name"),
                SymbolKind.Class,
                node.span,
                node,
                node.visibility()
            )
        )
    }
    let superclass = node.type_field("superclass")
    let super_symbol = r._mark_type_used(superclass)
    for trait_type in node.type_list("traits") { r._mark_type_used(trait_type) }

    let scope = r._push(ScopeKind.Class)
    let info = ClassInfo(node, r.out.symbol_of(node), scope)
    info.super_symbol = super_symbol
    r.classes.push(info)
    _declare_generics(r, node.records("generics"), node)

    for field in node.records("fields") {
        let mutability = if field["isLet"] { DeclKind.Let } else { DeclKind.Var }
        let symbol = r.declare(field["name"], SymbolKind.Field, field["span"], mutability)
        symbol.decl = node
        symbol.visibility = field["visibility"]
        symbol.is_initialized = field["defaultValue"] != null
        symbol.is_used = true
        r.out.bind(field, symbol)
    }
    for method in node.child_list("methods") { _declare_member(r, method, false) }
    for getter in node.child_list("getters") { _declare_member(r, getter, false) }
    for setter in node.child_list("setters") { _declare_member(r, setter, true) }

    let saved_self = r.self_ok
    r.self_ok = false
    for field in node.records("fields") { resolve_node(r, field["defaultValue"]) }
    r.self_ok = saved_self

    let has_super = superclass != null
    _resolve_methods(r, node.child_list("methods"), true, has_super)
    _resolve_methods(r, node.child_list("getters"), false, has_super)
    _resolve_methods(r, node.child_list("setters"), false, has_super)
    r._pop()
}

pub fn _trait_decl(r: Resolver, node: Node) -> void {
    if r.out.symbol_of(node) == null {
        r.out.bind(
            node,
            _hoist_named(
                r,
                node.opt_text("name"),
                SymbolKind.Trait,
                node.span,
                node,
                node.visibility()
            )
        )
    }
    for parent in node.type_list("supers") { r._mark_type_used(parent) }
    r._push(ScopeKind.Class)
    _declare_generics(r, node.records("generics"), node)
    for method in node.child_list("methods") { _declare_member(r, method, false) }
    _resolve_methods(r, node.child_list("methods"), true, false)
    r._pop()
}

pub fn _enum_decl(r: Resolver, node: Node) -> void {
    if r.out.symbol_of(node) == null {
        r.out.bind(
            node,
            _hoist_named(
                r,
                node.opt_text("name"),
                SymbolKind.Enum,
                node.span,
                node,
                node.visibility()
            )
        )
    }
    r._push(ScopeKind.Class)
    _declare_generics(r, node.records("generics"), node)

    for (tag, variant) in node.records("variants").enumerate() {
        let symbol = r.declare(
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
        r.out.bind(variant, symbol)
    }
    for method in node.child_list("methods") { _declare_member(r, method, false) }
    _resolve_methods(r, node.child_list("methods"), true, false)
    r._pop()
}

pub fn _assign_field_slots(r: Resolver, info: ClassInfo) -> void {
    if info.state == 2 { return }
    if info.state == 1 {
        let shown = info.decl.opt_text("name") ?? "?"
        r.bag.error(
            SemaCode.E0706_CYCLIC_INHERITANCE,
            info.decl.span,
            f"cyclic inheritance involving class `{shown}`",
            null,
            "a class may not be, directly or indirectly, its own " + "superclass"
        )
        _assign_own_field_slots(r, info, 0)
        return
    }

    info.state = 1
    var base = 0
    let parent = _find_class_info(r, info.super_symbol)
    if parent != null {
        _assign_field_slots(r, parent)
        if info.state == 2 { return }
        base = parent.total
    }
    _assign_own_field_slots(r, info, base)
}

fn _assign_own_field_slots(r: Resolver, info: ClassInfo, base: int) -> void {
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

fn _find_class_info(r: Resolver, symbol: Symbol?) -> ClassInfo? {
    if symbol == null { return null }
    for info in r.classes {
        if info.symbol is symbol { return info }
    }
    return null
}
