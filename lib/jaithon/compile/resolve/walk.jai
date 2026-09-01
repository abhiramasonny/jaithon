#: What a node uses: the statement forms that open a scope or check a
#: context -- `let`, assignment, the three loops, `break`/`continue`, `try`,
#: `match`, a comprehension and `export` -- and `_node`, the per-kind dispatch
#: that every other module reaches the tree through.
#:
#: Last in the directory's import order, which is what lets it be the one
#: module that needs no seam of its own: it may name everything in `decl.jai`
#: directly, and the two functions the seams point back at, `_node` and
#: `_block_contents`, are both here.
#:
#: The dispatch is one `match` over `NodeKind` and stays that way. It is the
#: only place in the pass that knows the whole language at once, so a kind
#: added to `jaithon.ast` and forgotten here falls into the final arm and is
#: read as a pattern -- which is the behaviour the single file had, and worth
#: keeping visible on one screen rather than spread over several files.
#:
#: What is NOT here: anything that declares. A form that binds a name calls
#: into `decl.jai`, even when the statement that owns it is here: `_var_decl`
#: is the `let` statement, but the names it binds are `_pattern`'s business.
from jaithon.ast import DeclKind, Node, NodeKind, TypeExpr, Visibility
from jaithon.compile.symbol import SemaCode, Symbol, SymbolKind
from jaithon.compile.resolve.scope import ScopeKind
from jaithon.compile.resolve.resolver import Resolver
from jaithon.compile.resolve.decl import _class_decl, _enum_decl, _function, _hoist
from jaithon.compile.resolve.decl import _hoist_named, _pattern, _pattern_names, _trait_decl

# --- statements -------------------------------------------------------
pub fn _block_contents(r: Resolver, stmts: list[Node]) -> void {
    _hoist(r, stmts)
    for statement in stmts { _node(r, statement) }
}

fn _var_decl(r: Resolver, node: Node) -> void {
    let init = node.child("init")
    let decl_kind: DeclKind = node.get("varDeclKind")
    if init != null {
        _node(r, init)
    } elif decl_kind != DeclKind.Var {
        let word = if decl_kind == DeclKind.Let { "let" } else { "const" }
        r.bag.error(
            SemaCode.E0304_MISSING_INITIALIZER,
            node.span,
            f"`{word}` declaration requires an initialiser",
            "give it a value, or declare it with `var`"
        )
    }

    let pattern = node.child("pattern")
    _pattern_names(r, pattern, true)
    _pattern(r, pattern, true, decl_kind, node, false)

    if pattern != null and pattern.kind == NodeKind.PatBind {
        let symbol = r.out.symbol_of(pattern)
        r.out.bind(node, symbol)
        if symbol != null {
            symbol.visibility = node.visibility()
            symbol.is_initialized = init != null
        }
    }
}

fn _assign_target(r: Resolver, node: Node?, compound: bool) -> void {
    if node == null { return }
    if node.kind != NodeKind.Ident {
        _node(r, node)
        return
    }
    r._ident(node, compound)
    let symbol = r.out.symbol_of(node)
    if symbol == null { return }

    if symbol.kind == SymbolKind.Builtin {
        r.bag.error(
            SemaCode.E0301_ASSIGN_TO_IMMUTABLE,
            node.span,
            f"cannot assign to built-in `{symbol.name}`"
        )
    } elif symbol.mutability == DeclKind.Const and symbol.kind != SymbolKind.Field {
        let diag = r.bag.error(
            SemaCode.E0305_ASSIGN_TO_CONST,
            node.span,
            f"cannot assign to constant `{symbol.name}`"
        )
        diag.label(symbol.decl_span, f"`{symbol.name}` declared here")
    } elif symbol.mutability == DeclKind.Let {
        let diag = r.bag.error(
            SemaCode.E0301_ASSIGN_TO_IMMUTABLE,
            node.span,
            f"cannot assign to immutable binding `{symbol.name}`",
            f"change the declaration to `var {symbol.name}`"
        )
        diag.label(symbol.decl_span, f"`{symbol.name}` declared immutable here")
    }
}

fn _loop(r: Resolver, node: Node) -> void {
    _node(r, node.child("cond"))  # null for `loop`
    let label = node.opt_text("label")
    r.loop_labels.push(label)
    let scope = r._push(ScopeKind.Loop)
    scope.label = label
    _node(r, node.child("body"))
    r.out.set_capture_base(node, scope.capture_base)
    r._pop()
    r.loop_labels.pop()
}

fn _for(r: Resolver, node: Node) -> void {
    _node(r, node.child("iterable"))
    let label = node.opt_text("label")
    r.loop_labels.push(label)
    let scope = r._push(ScopeKind.Loop)
    scope.label = label

    _pattern(r, node.child("pattern"), true, DeclKind.Let, node, false)
    _node(r, node.child("body"))
    r.out.set_capture_base(node, scope.capture_base)
    r._pop()
    r.loop_labels.pop()
}

fn _jump(r: Resolver, node: Node, is_break: bool) -> void {
    let word = if is_break { "break" } else { "continue" }
    if r.loop_labels.len() == 0 {
        var code = SemaCode.E0204_CONTINUE_OUTSIDE_LOOP
        if is_break { code = SemaCode.E0203_BREAK_OUTSIDE_LOOP }
        r.bag.error(code, node.span, f"`{word}` outside of a loop")
        return
    }
    let label = node.opt_text("label")
    if label == null { return }
    for candidate in r.loop_labels {
        if candidate == label { return }
    }
    r.bag.error(
        SemaCode.E0202_UNDEFINED_LABEL,
        node.span,
        f"undefined loop label `{label}`",
        f"label the target loop, e.g. `{label}: while ...`"
    )
}

fn _try(r: Resolver, node: Node) -> void {
    _node(r, node.child("body"))
    for handler in node.records("catches") {
        r._push(ScopeKind.Block)
        let caught: list[TypeExpr] = handler["types"]
        for entry in caught { r._mark_type_used(entry) }
        let name: str? = handler["name"]
        if name != null {
            let symbol = r.declare(name, SymbolKind.Local, handler["span"], DeclKind.Let)
            symbol.decl = node
            symbol.is_initialized = true
            r.out.bind(handler, symbol)
        }
        _node(r, handler["body"])
        r._pop()
    }
    _node(r, node.child("finallyBlock"))
}

fn _match(r: Resolver, node: Node) -> void {
    _node(r, node.child("subject"))
    for arm in node.records("arms") {
        r._push(ScopeKind.Block)
        _pattern(r, arm["pattern"], true, DeclKind.Let, node, false)
        _node(r, arm["guard"])
        _node(r, arm["body"])
        r._pop()
    }
}

fn _comprehension(r: Resolver, node: Node) -> void {
    r._push(ScopeKind.Block)
    for clause in node.records("clauses") {
        _node(r, clause["iterable"])
        _pattern(r, clause["pattern"], true, DeclKind.Let, node, false)
        let conditions: list[Node] = clause["conditions"]
        for condition in conditions { _node(r, condition) }
    }
    _node(r, node.child("keyExpr"))
    _node(r, node.child("element"))
    r._pop()
}

fn _export(r: Resolver, node: Node) -> void {
    let names: list[str?] = node.get("names")
    for name in names {
        if name == null { continue }
        var found: Symbol? = null
        var index = r.scopes.len() - 1
        while index >= 0 and found == null {
            found = r.scopes[index].names.get(name, null)
            index -= 1
        }
        if found == null {
            r.bag.error(
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

fn _in_function_body(r: Resolver) -> bool {
    return r._current_fn().decl.kind != NodeKind.Program
}

# --- the walk ---------------------------------------------------------
fn _each(r: Resolver, nodes: list[Node]) -> void {
    for node in nodes { _node(r, node) }
}

pub fn _node(r: Resolver, node: Node?) -> void {
    if node == null { return }
    match node.kind {
        NodeKind.IntLit | NodeKind.FloatLit | NodeKind.StrLit | NodeKind.BoolLit | NodeKind.NullLit | NodeKind.ModuleDecl | NodeKind.TypeDecl | NodeKind.Import | NodeKind.FromImport => { }
        NodeKind.FString => { _each(r, node.child_list("parts")) }
        NodeKind.Ident => { r._ident(node, true) }
        NodeKind.Self => {
            if not r.self_ok {
                let captured = r.resolve_name("self", node.span, true)
                if captured != null {
                    r.out.bind(node, captured)
                } else {
                    r.bag.error(
                        SemaCode.E0206_SELF_OUTSIDE_METHOD,
                        node.span,
                        "`self` is only valid inside a method"
                    )
                }
            }
        }
        NodeKind.Super => {
            if not r.super_ok {
                r.bag.error(
                    SemaCode.E0207_SUPER_OUTSIDE_SUBCLASS,
                    node.span,
                    "`super` is only valid inside a method of a class " + "that has a superclass"
                )
            }
        }
        NodeKind.ListLit | NodeKind.SetLit | NodeKind.TupleLit => {
            _each(r, node.child_list("items"))
        }
        NodeKind.DictLit => {
            _each(r, node.child_list("keys"))
            _each(r, node.child_list("values"))
        }
        NodeKind.Unary | NodeKind.Cast | NodeKind.Await | NodeKind.ThrowExpr => {
            _node(r, node.child("operand"))
        }
        NodeKind.Binary | NodeKind.Logical | NodeKind.Coalesce => {
            _node(r, node.child("left"))
            _node(r, node.child("right"))
        }
        NodeKind.CompareChain => { _each(r, node.child_list("operands")) }
        NodeKind.Ternary => {
            _node(r, node.child("cond"))
            _node(r, node.child("thenExpr"))
            _node(r, node.child("elseExpr"))
        }
        NodeKind.Call => {
            _node(r, node.child("callee"))
            for argument in node.records("args") { _node(r, argument["value"]) }
        }
        NodeKind.Index => {
            _node(r, node.child("object"))
            if r._is_generic_instantiation(node) {
                r.out.set_type_args(node)
            } else {
                _node(r, node.child("index"))
            }
        }
        NodeKind.Slice => {
            _node(r, node.child("object"))
            _node(r, node.child("start"))
            _node(r, node.child("stop"))
            _node(r, node.child("step"))
        }
        NodeKind.Member | NodeKind.OptMember => { _node(r, node.child("object")) }
        NodeKind.Lambda | NodeKind.AnonFn => { _function(r, node, false, false, false) }
        NodeKind.Comprehension => { _comprehension(r, node) }
        NodeKind.Range => {
            _node(r, node.child("start"))
            _node(r, node.child("stop"))
        }
        NodeKind.IfExpr | NodeKind.If => {
            _node(r, node.child("cond"))
            _node(r, node.child("thenBranch"))
            _node(r, node.child("elseBranch"))
        }
        NodeKind.Match | NodeKind.MatchExpr => { _match(r, node) }
        NodeKind.Yield => {
            if not _in_function_body(r) {
                r.bag.error(
                    SemaCode.E0210_YIELD_OUTSIDE_GENERATOR,
                    node.span,
                    "`yield` is only valid inside a function"
                )
            } else {
                r._current_fn().is_generator = true
            }
            _node(r, node.child("operand"))
        }
        NodeKind.Block => {
            let scope = r._push(ScopeKind.Block)
            _block_contents(r, node.child_list("stmts"))
            r.out.set_capture_base(node, scope.capture_base)
            r._pop()
        }
        NodeKind.Program => { _block_contents(r, node.child_list("stmts")) }
        NodeKind.ExprStmt => { _node(r, node.child("expr")) }
        NodeKind.VarDecl => { _var_decl(r, node) }
        NodeKind.Assign => {
            _node(r, node.child("value"))
            _assign_target(r, node.child("target"), node.flag("isCompound"))
        }
        NodeKind.While | NodeKind.Loop => { _loop(r, node) }
        NodeKind.For => { _for(r, node) }
        NodeKind.Break => { _jump(r, node, true) }
        NodeKind.Continue => { _jump(r, node, false) }
        NodeKind.Return => {
            if not _in_function_body(r) {
                r.bag.error(
                    SemaCode.E0205_RETURN_OUTSIDE_FUNCTION,
                    node.span,
                    "`return` outside of a function"
                )
            }
            _node(r, node.child("value"))
        }
        NodeKind.Throw => { _node(r, node.child("value")) }
        NodeKind.Try => { _try(r, node) }
        NodeKind.Defer => {
            if not _in_function_body(r) {
                r.bag.error(
                    SemaCode.E0211_DEFER_OUTSIDE_FUNCTION,
                    node.span,
                    "`defer` is only valid inside a function"
                )
            } else {
                r._current_fn().has_defer = true
            }
            _node(r, node.child("body"))
        }
        NodeKind.Assert => {
            _node(r, node.child("cond"))
            _node(r, node.child("message"))
        }
        NodeKind.FnDecl => {
            if r.out.symbol_of(node) == null {
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
            }
            _function(r, node, false, false, false)
        }
        NodeKind.ClassDecl => { _class_decl(r, node) }
        NodeKind.TraitDecl => { _trait_decl(r, node) }
        NodeKind.EnumDecl => { _enum_decl(r, node) }
        NodeKind.Export => { _export(r, node) }
        _ => { _pattern(r, node, false, DeclKind.Let, null, false) }
    }
}
