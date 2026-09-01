#: Everything a declaration keyword introduces: `class`, `trait`, `enum`,
#: `type`, `let`, `var` and `const`, the rest of a `fn` after its keyword,
#: parameter and generic lists, accessors, fields, variants, decorators, member
#: visibility, and the `import`, `from`-import, `export` and `module` forms.
#:
#: The largest file of `parse/`, and deliberately so. These productions share
#: the same visibility-and-start-index preamble and the same `_end_statement`
#: tail, and every further cut through them would have bought another seam
#: rather than another idea. It is last in the directory's import order, which
#: is what lets it be the one production file that needs no seam at all.
from jaithon.ast import DeclKind, Node, NodeKind, Span, TypeExpr, Visibility
from jaithon.compile.diag import Code
from jaithon.compile.token import TokenKind
from jaithon.compile.parse.state import Parser
from jaithon.compile.parse.node_build import _field, _generic, _import_item, _node, _param, _variant
from jaithon.compile.parse.type_expr import _parse_type, _parse_type_list
from jaithon.compile.parse.primary import _close_sequence
from jaithon.compile.parse.expr import _parse_expression
from jaithon.compile.parse.pattern import _parse_pattern
from jaithon.compile.parse.stmt import _expect_block, _parse_statement

pub fn _parse_generics(p: Parser) -> list[dict[str, any]] {
    var generics: list[dict[str, any]] = []
    if not p._check(TokenKind.LBracket) { return generics }

    let open_index = p.current
    p._advance()
    if not p._check(TokenKind.RBracket) {
        loop {
            let name_token = p._peek()
            let name = p._expect_ident_name("a type parameter name")
            if name == null { break }
            var bound: TypeExpr? = null
            if p._match(TokenKind.Colon) { bound = _parse_type(p) }
            p._check_reserved_type_name(name, name_token.span)
            generics.push(_generic(name, bound, name_token.span))
            if not p._match(TokenKind.Comma) { break }
            if p._check(TokenKind.RBracket) or p._check(TokenKind.Eof) { break }
        }
    }
    _close_sequence(p, TokenKind.RBracket, "]", "type parameter list", open_index)
    return generics
}

pub fn _parse_params(p: Parser, open_index: int) -> list[dict[str, any]] {
    var params: list[dict[str, any]] = []
    var rest_span: Span? = null
    var default_span: Span? = null

    p._skip_newlines()
    if not p._check(TokenKind.RParen) {
        loop {
            p._skip_newlines()
            let param_start = p.current
            let is_variadic = p._match(TokenKind.Ellipsis)
            let is_kw_rest = not is_variadic and p._match(TokenKind.StarStar)

            var name: str? = null
            if p._check(TokenKind.KwSelf) {
                p._advance()
                name = "self"
            } else {
                name = p._expect_ident_name("a parameter name")
                if name == null { break }
            }

            var declared: TypeExpr? = null
            var default_value: Node? = null
            if p._match(TokenKind.Colon) { declared = _parse_type(p) }
            if p._match(TokenKind.Eq) { default_value = _parse_expression(p) }
            let span = p._span_since(param_start)

            p._check_duplicate_param(params, name ?? "_", span)
            if rest_span != null {
                let diag = p._error_recovered(
                    span,
                    Code.E0112_PARAM_AFTER_VARIADIC,
                    f"parameter `{name}` cannot follow a " + "rest parameter"
                )
                if diag != null { diag.label(rest_span, "rest parameter declared here") }
            } elif default_span != null and
                default_value == null and
                not is_variadic and
                not is_kw_rest {
                let diag = p._error_recovered(
                    span,
                    Code.E0113_DEFAULT_BEFORE_REQUIRED,
                    f"required parameter `{name}` follows " + "a defaulted parameter"
                )
                if diag != null {
                    diag.label(default_span, "this parameter has a default value")
                }
            }
            if is_variadic or is_kw_rest { rest_span = span }
            if default_value != null { default_span = span }

            params.push(_param(name, declared, default_value, span, is_variadic, is_kw_rest))
            p._skip_newlines()
            if not p._match(TokenKind.Comma) { break }
            p._skip_newlines()
            if p._check(TokenKind.RParen) or p._check(TokenKind.Eof) or p.panic_mode {
                break
            }
        }
    }
    _close_sequence(p, TokenKind.RParen, ")", "parameter list", open_index)
    return params
}

pub fn _parse_fn_rest(
    p: Parser,
    visibility: Visibility,
    is_static: bool,
    start_index: int,
    named: bool,
    body_optional: bool
) -> Node? {
    p._advance()

    var name: str? = null
    if named {
        name = p._expect_ident_name("a function name")
        if name == null { return null }
    }

    let generics = _parse_generics(p)
    var params: list[dict[str, any]] = []
    let open_index = p.current
    if p._expect(TokenKind.LParen, "`(` to open the parameter list") {
        params = _parse_params(p, open_index)
    }
    var return_type: TypeExpr? = null
    if p._match(TokenKind.Arrow) { return_type = _parse_type(p) }

    let saved_yield = p.yield_count
    p.yield_count = 0
    var body: Node? = null
    var is_expr_body = false

    if body_optional and not p._check(TokenKind.LBrace) {
        body = null  # a trait requirement: a signature and no definition
    } elif p._check(TokenKind.FatArrow) {
        p._error_at(
            p._cur_span(),
            Code.E0104_EXPECTED_BLOCK,
            "a function body is a block, not `=>`",
            help: "write `{ return <expr> }`, or use a lambda `|x| <expr>`"
        )
        p._advance()
        body = _parse_expression(p)
        is_expr_body = true
    } else {
        body = _expect_block(p, "function")
    }

    let is_generator = p.yield_count > 0
    p.yield_count = saved_yield

    return _node(
        named ? NodeKind.FnDecl : NodeKind.AnonFn,
        p._span_since(start_index),
        {
            "name": name,
            "generics": generics,
            "params": params,
            "returnType": return_type,
            "body": body,
            "visibility": visibility,
            "isStatic": is_static,
            "isExprBody": is_expr_body,
            "isGenerator": is_generator,
            "isAsync": false,
            "decorators": [],
        }
    )
}

pub fn _parse_accessor(
    p: Parser,
    visibility: Visibility,
    is_static: bool,
    is_setter: bool
) -> Node? {
    let start_index = p.current
    p._advance()
    let name = p._expect_ident_name("a property name")
    if name == null { return null }

    var params: list[dict[str, any]] = []
    let open_index = p.current
    if p._expect(TokenKind.LParen, "`(` after the property name") {
        params = _parse_params(p, open_index)
    }
    var return_type: TypeExpr? = null
    if p._match(TokenKind.Arrow) { return_type = _parse_type(p) }

    let saved_yield = p.yield_count
    p.yield_count = 0
    let body = _expect_block(p, is_setter ? "set" : "get")
    p.yield_count = saved_yield

    return _node(
        NodeKind.FnDecl,
        p._span_since(start_index),
        {
            "name": name,
            "generics": [],
            "params": params,
            "returnType": return_type,
            "body": body,
            "visibility": visibility,
            "isStatic": is_static,
            "isExprBody": false,
            "isGenerator": false,
            "isAsync": false,
        }
    )
}

pub fn _parse_var_decl(p: Parser, visibility: Visibility, start_index: int) -> Node {
    let keyword = p._kind()
    var decl_kind = DeclKind.Const
    if keyword == TokenKind.KwLet {
        decl_kind = DeclKind.Let
    } elif keyword == TokenKind.KwVar {
        decl_kind = DeclKind.Var
    }
    p._advance()

    let pattern = _parse_pattern(p)
    if pattern.kind == NodeKind.PatBind {
        p._check_reserved_name(pattern.text("name"), pattern.span)
    }
    var declared: TypeExpr? = null
    if p._match(TokenKind.Colon) { declared = _parse_type(p) }

    var init: Node? = null
    if p._match(TokenKind.Eq) {
        init = _parse_expression(p)
    } elif decl_kind != DeclKind.Var and not p.panic_mode {
        let keyword_text = decl_kind == DeclKind.Let ? "let" : "const"
        p._error_at(
            p._span_since(start_index),
            Code.E0304_MISSING_INITIALIZER,
            f"`{keyword_text}` bindings must be initialised",
            help: "add `= <expression>`, or use `var` to declare it uninitialised"
        )
    }
    p._end_statement()

    return _node(
        NodeKind.VarDecl,
        p._span_since(start_index),
        {
            "varDeclKind": decl_kind,
            "pattern": pattern,
            "declaredType": declared,
            "init": init,
            "visibility": visibility,
        }
    )
}

pub fn _parse_class_decl(p: Parser, visibility: Visibility, start_index: int) -> Node? {
    p._advance()
    let name_token = p._peek()
    let name = p._expect_ident_name("a class name")
    if name == null { return null }
    p._check_reserved_type_name(name, name_token.span)

    let generics = _parse_generics(p)
    var superclass: TypeExpr? = null
    if p._match(TokenKind.KwExtends) { superclass = _parse_type(p) }
    var traits: list[TypeExpr] = []
    if p._match(TokenKind.Colon) { traits = _parse_type_list(p) }

    var fields: list[dict[str, any]] = []
    var methods: list[Node] = []
    var getters: list[Node] = []
    var setters: list[Node] = []

    let open_index = p.current
    if p._expect(TokenKind.LBrace, "`{` to open the class body") {
        p._skip_newlines()
        while not p._check(TokenKind.RBrace) and not p._check(TokenKind.Eof) {
            let before = p.current
            let decorators = _parse_decorators(p)
            let member_start = if decorators.len() > 0 { before } else { p.current }
            let member_visibility = _parse_member_visibility(p)
            let member_static = p._match(TokenKind.KwStatic)

            if p._check(TokenKind.KwVar) or p._check(TokenKind.KwLet) {
                let field = _parse_field(p, member_visibility, member_static)
                if field != null { fields.push(field) }
            } elif p._check(TokenKind.KwConst) {
                p._error_at(
                    p._cur_span(),
                    Code.E0100_UNEXPECTED_TOKEN,
                    "class fields are declared with `var` or `let`",
                    help: "write `static let NAME: <type> = <value>`"
                )
            } elif p._check(TokenKind.KwFn) {
                let method = _parse_fn_rest(
                    p,
                    member_visibility,
                    member_static,
                    member_start,
                    true,
                    false
                )
                if method != null {
                    if decorators.len() > 0 { method.set("decorators", decorators) }
                    methods.push(method)
                }
            } elif p._ident_text(0, "get") and p._kind_at(1) == TokenKind.Ident {
                let getter = _parse_accessor(p, member_visibility, member_static, false)
                if getter != null { getters.push(getter) }
            } elif p._ident_text(0, "set") and p._kind_at(1) == TokenKind.Ident {
                let setter = _parse_accessor(p, member_visibility, member_static, true)
                if setter != null { setters.push(setter) }
            } else {
                p._error_at(
                    p._cur_span(),
                    Code.E0103_EXPECTED_STATEMENT,
                    f"expected a class member, " + f"found {p._describe(p._peek())}",
                    help: "members are `var`/`let` fields, `fn` methods, " + "or `get`/`set` properties"
                )
            }

            if p.panic_mode { p._synchronize() }
            if p.current == before { p._advance() }
            p._skip_newlines()
        }
        _close_sequence(p, TokenKind.RBrace, "}", "class body", open_index)
    }

    return _node(
        NodeKind.ClassDecl,
        p._span_since(start_index),
        {
            "name": name,
            "generics": generics,
            "superclass": superclass,
            "traits": traits,
            "fields": fields,
            "methods": methods,
            "getters": getters,
            "setters": setters,
            "visibility": visibility,
            "isAbstract": false,
        }
    )
}

pub fn _parse_member_visibility(p: Parser) -> Visibility {
    if p._match(TokenKind.KwPub) { return Visibility.Public }
    if p._match(TokenKind.KwProt) { return Visibility.Protected }
    return Visibility.Private
}

#: Read `@name` decorators that precede a function declaration.
pub fn _parse_decorators(p: Parser) -> list[str] {
    var decorators: list[str] = []
    while p._check(TokenKind.At) {
        p._advance()
        let name = p._expect_ident_name("a decorator name")
        if name == null { break }
        decorators.push(name)
        p._skip_newlines()
    }
    return decorators
}

pub fn _parse_field(p: Parser, visibility: Visibility, is_static: bool) -> dict[str, any]? {
    let field_start = p.current
    let is_let = p._check(TokenKind.KwLet)
    p._advance()

    let name_token = p._peek()
    let name = p._expect_ident_name("a field name")
    if name == null {
        p._end_statement()
        return null
    }

    var declared: TypeExpr? = null
    if p._match(TokenKind.Colon) {
        declared = _parse_type(p)
    } else {
        let keyword_text = is_let ? "let" : "var"
        p._error_at(
            name_token.span,
            Code.E0105_EXPECTED_TYPE,
            f"field `{name}` needs a type annotation",
            help: f"write `{keyword_text} {name}: <type>`"
        )
    }
    var default_value: Node? = null
    if p._match(TokenKind.Eq) { default_value = _parse_expression(p) }
    let span = p._span_since(field_start)
    p._end_statement()

    return _field(name, declared, default_value, visibility, is_static, is_let, span)
}

pub fn _parse_trait_decl(p: Parser, visibility: Visibility, start_index: int) -> Node? {
    p._advance()
    let name_token = p._peek()
    let name = p._expect_ident_name("a trait name")
    if name == null { return null }
    p._check_reserved_type_name(name, name_token.span)

    let generics = _parse_generics(p)
    var supers: list[TypeExpr] = []
    if p._match(TokenKind.Colon) { supers = _parse_type_list(p) }

    var methods: list[Node] = []
    let open_index = p.current
    if p._expect(TokenKind.LBrace, "`{` to open the trait body") {
        p._skip_newlines()
        while not p._check(TokenKind.RBrace) and not p._check(TokenKind.Eof) {
            let before = p.current
            let member_visibility = _parse_member_visibility(p)
            let member_static = p._match(TokenKind.KwStatic)

            if p._check(TokenKind.KwFn) {
                let method = _parse_fn_rest(
                    p,
                    member_visibility,
                    member_static,
                    p.current,
                    true,
                    true
                )
                if method != null {
                    methods.push(method)
                    if method.child("body") == null { p._end_statement() }
                }
            } else {
                p._error_at(
                    p._cur_span(),
                    Code.E0103_EXPECTED_STATEMENT,
                    f"expected `fn` in a trait body, " + f"found {p._describe(p._peek())}"
                )
            }
            if p.panic_mode { p._synchronize() }
            if p.current == before { p._advance() }
            p._skip_newlines()
        }
        _close_sequence(p, TokenKind.RBrace, "}", "trait body", open_index)
    }

    return _node(
        NodeKind.TraitDecl,
        p._span_since(start_index),
        {
            "name": name,
            "generics": generics,
            "supers": supers,
            "methods": methods,
            "visibility": visibility,
        }
    )
}

pub fn _parse_enum_decl(p: Parser, visibility: Visibility, start_index: int) -> Node? {
    p._advance()
    let name_token = p._peek()
    let name = p._expect_ident_name("an enum name")
    if name == null { return null }
    p._check_reserved_type_name(name, name_token.span)

    let generics = _parse_generics(p)
    var variants: list[dict[str, any]] = []
    var methods: list[Node] = []

    let open_index = p.current
    if p._expect(TokenKind.LBrace, "`{` to open the enum body") {
        p._skip_newlines()
        while not p._check(TokenKind.RBrace) and not p._check(TokenKind.Eof) {
            let before = p.current
            if p._check(TokenKind.KwFn) or p._check(TokenKind.KwPub) or p._check(TokenKind.KwStatic) {
                var member_visibility = Visibility.Private
                if p._match(TokenKind.KwPub) { member_visibility = Visibility.Public }
                let member_static = p._match(TokenKind.KwStatic)
                if p._check(TokenKind.KwFn) {
                    let method = _parse_fn_rest(
                        p,
                        member_visibility,
                        member_static,
                        p.current,
                        true,
                        false
                    )
                    if method != null { methods.push(method) }
                } else {
                    p._error_at_cur(
                        Code.E0103_EXPECTED_STATEMENT,
                        "expected `fn` after `pub`/`static`"
                    )
                }
            } else {
                let variant = _parse_variant(p)
                if variant != null { variants.push(variant) }
                p._match(TokenKind.Comma)
            }
            if p.panic_mode { p._synchronize() }
            if p.current == before { p._advance() }
            p._skip_newlines()
        }
        _close_sequence(p, TokenKind.RBrace, "}", "enum body", open_index)
    }

    return _node(
        NodeKind.EnumDecl,
        p._span_since(start_index),
        {
            "name": name,
            "generics": generics,
            "variants": variants,
            "methods": methods,
            "visibility": visibility,
        }
    )
}

pub fn _parse_variant(p: Parser) -> dict[str, any]? {
    let variant_start = p.current
    let name = p._expect_ident_name("a variant name")
    if name == null { return null }

    var params: list[dict[str, any]] = []
    if p._check(TokenKind.LParen) {
        let payload_index = p.current
        p._advance()
        params = _parse_params(p, payload_index)
    }
    return _variant(name, params, p._span_since(variant_start))
}

pub fn _parse_type_decl(p: Parser, visibility: Visibility, start_index: int) -> Node? {
    p._advance()
    let name_token = p._peek()
    let name = p._expect_ident_name("a type name")
    if name == null { return null }
    p._check_reserved_type_name(name, name_token.span)

    let generics = _parse_generics(p)
    p._expect(TokenKind.Eq, "`=` in a type alias")
    let aliased = _parse_type(p)
    p._end_statement()

    return _node(
        NodeKind.TypeDecl,
        p._span_since(start_index),
        {"name": name, "generics": generics, "aliased": aliased, "visibility": visibility}
    )
}

pub fn _parse_module_path(p: Parser) -> str? {
    var path = ""
    while p._check(TokenKind.Dot) or p._check(TokenKind.DotDot) or p._check(TokenKind.Ellipsis) {
        path = path + p._advance().text
    }

    if not p._check(TokenKind.Ident) {
        p._error_at(
            p._cur_span(),
            Code.E0804_INVALID_MODULE_PATH,
            f"expected a module path, found {p._describe(p._peek())}"
        )
        return null
    }
    path = path + p._advance().text
    while p._check(TokenKind.Dot) and p._kind_at(1) == TokenKind.Ident {
        p._advance()
        path = path + "." + p._advance().text
    }
    return path
}

pub fn _parse_import(p: Parser, start_index: int) -> Node? {
    p._advance()
    let path = _parse_module_path(p)
    if path == null { return null }
    var alias: str? = null
    if p._match(TokenKind.KwAs) { alias = p._expect_ident_name("an import alias") }
    p._end_statement()

    return _node(NodeKind.Import, p._span_since(start_index), {"path": path, "alias": alias})
}

pub fn _parse_from_import(p: Parser, start_index: int) -> Node? {
    p._advance()
    let path = _parse_module_path(p)
    if path == null { return null }
    p._expect(TokenKind.KwImport, "`import` after the module path")

    var wildcard = false
    var items: list[dict[str, any]] = []
    if p._match(TokenKind.Star) {
        wildcard = true
    } else {
        let parens = p._match(TokenKind.LParen)
        loop {
            p._skip_newlines()
            let name_token = p._peek()
            let name = p._expect_ident_name("an imported name")
            if name == null { break }
            var alias: str? = null
            if p._match(TokenKind.KwAs) {
                alias = p._expect_ident_name("an import alias")
            }
            items.push(_import_item(name, alias, name_token.span))
            if not p._match(TokenKind.Comma) { break }
            if p._check(TokenKind.Eof) or p.panic_mode { break }
        }
        if parens {
            p._skip_newlines()
            p._expect(TokenKind.RParen, "`)` to close the import list")
        }
    }
    p._end_statement()

    return _node(
        NodeKind.FromImport,
        p._span_since(start_index),
        {"path": path, "items": items, "isWildcard": wildcard}
    )
}

pub fn _parse_export_decl(p: Parser, start_index: int) -> Node {
    p._advance()
    var names: list[str] = []

    let open_index = p.current
    if p._expect(TokenKind.LBrace, "`{` to open the export list") {
        p._skip_newlines()
        if not p._check(TokenKind.RBrace) {
            loop {
                p._skip_newlines()
                let name = p._expect_ident_name("an exported name")
                if name == null { break }
                names.push(name)
                p._skip_newlines()
                if not p._match(TokenKind.Comma) { break }
                p._skip_newlines()
                if p._check(TokenKind.RBrace) or p._check(TokenKind.Eof) { break }
            }
        }
        p._skip_newlines()
        _close_sequence(p, TokenKind.RBrace, "}", "export list", open_index)
    }
    p._end_statement()

    return _node(NodeKind.Export, p._span_since(start_index), {"names": names})
}

pub fn _parse_module_decl(p: Parser, start_index: int) -> Node? {
    p._advance()
    let name = _parse_module_path(p)
    if name == null { return null }
    p._end_statement()

    return _node(NodeKind.ModuleDecl, p._span_since(start_index), {"name": name})
}

pub fn _parse_declaration(p: Parser) -> Node? {
    let decorator_start = p.current
    let decorators = _parse_decorators(p)
    let start_index = p.current
    var visibility = Visibility.Private
    var saw_visibility = false
    if p._match(TokenKind.KwPub) {
        visibility = Visibility.Public
        saw_visibility = true
    } elif p._match(TokenKind.KwProt) {
        visibility = Visibility.Protected
        saw_visibility = true
    }
    let is_static = p._match(TokenKind.KwStatic)

    let kind = p._kind()
    if kind == TokenKind.KwClass { return _parse_class_decl(p, visibility, start_index) }
    if kind == TokenKind.KwTrait { return _parse_trait_decl(p, visibility, start_index) }
    if kind == TokenKind.KwEnum { return _parse_enum_decl(p, visibility, start_index) }
    if kind == TokenKind.KwType { return _parse_type_decl(p, visibility, start_index) }
    if kind == TokenKind.KwLet or kind == TokenKind.KwVar or kind == TokenKind.KwConst {
        return _parse_var_decl(p, visibility, start_index)
    }
    if kind == TokenKind.KwFn and p._kind_at(1) == TokenKind.Ident {
        let fn_start = if decorators.len() > 0 { decorator_start } else { start_index }
        let decl = _parse_fn_rest(p, visibility, is_static, fn_start, true, false)
        if decl != null and decorators.len() > 0 { decl.set("decorators", decorators) }
        return decl
    }
    if kind == TokenKind.KwImport or
        kind == TokenKind.KwFrom or
        kind == TokenKind.KwExport or
        kind == TokenKind.KwModule {
        if saw_visibility or is_static {
            p._error_at(
                p._span_since(start_index),
                Code.E0100_UNEXPECTED_TOKEN,
                "imports and exports take no visibility modifier",
                help: "delete the modifier"
            )
            return null
        }
        if kind == TokenKind.KwImport { return _parse_import(p, start_index) }
        if kind == TokenKind.KwFrom { return _parse_from_import(p, start_index) }
        if kind == TokenKind.KwExport { return _parse_export_decl(p, start_index) }
        return _parse_module_decl(p, start_index)
    }

    if saw_visibility or is_static {
        var modifier = visibility == Visibility.Public ? "pub" : "prot"
        if is_static { modifier = "static" }
        p._error_at(
            p._span_since(start_index),
            Code.E0103_EXPECTED_STATEMENT,
            f"expected a declaration after `{modifier}`, " + f"found {p._describe(p._peek())}",
            help: "`pub` may precede `fn`, `class`, `trait`, `enum`, `type`, " +
                "`let`, `var`, or `const`"
        )
        return null
    }

    return _parse_statement(p)
}
