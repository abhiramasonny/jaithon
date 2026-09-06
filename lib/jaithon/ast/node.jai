#: `Node` itself, the sub-structure builders its own static builders call, and
#: `children_of`.
#:
#: One file because Jaithon rejects an import cycle and these three genuinely
#: point at each other: `Node.call` builds its arguments with `arg()`, and
#: `Node.children()` answers with `children_of`. A sibling holding either would
#: import `node` and be imported back by it. Moving the import inside the
#: function does not rescue that layout -- a function-local import loads a
#: second copy of the module, so the `Node` the sibling names stops being this
#: `Node` and every signature between them stops type-checking.
#:
#: What is NOT here: the walk built on top of `children_of` -- `walk`,
#: `preorder`, `find_all` -- and the predicates that read a finished tree.
#: Those only ever consume a `Node`, so they sit downstream in `tree.jai`.
from jaithon.ast.kinds import DeclKind, NodeKind, Visibility, kind_name
from jaithon.ast.operators import OpKind, op_text
from jaithon.ast.schema import _ELEMENT_TAG, _FIELD_INDEX, _RECORDS, _default_field, new_slots
from jaithon.ast.schema import record_fields, tags_of
from jaithon.ast.span import Span
from jaithon.ast.types import TypeExpr

#: One node of the tree: a kind, a span, and the fields that kind declares.
#:
#: The constructor fills every field the kind declares with its default and then
#: overlays `fields`, so a node is always complete and `to_json` never has to
#: guess. A name the kind does not declare raises `KeyError` — that is how a
#: typo in a tree-building pass is caught at the point of the mistake.
#:
#: The fields live in a slot list positioned by `_FIELD_INDEX[kind]`, not in a
#: dict of their own: a dict per node was 401k of the run's allocations and the
#: one dict behind most of the tier's "the live dict is empty or holds more
#: than one kind of value" refusals. Nothing outside this class needs to know
#: that — the seven accessors below take the same names and raise the same
#: errors, and the slot list is in the order `fields_of(kind)` lists, which is
#: the order the C emits and the order `ast_encode` writes.
#:
#: The accessors read `_FIELD_INDEX` rather than a function over it, and
#: `schema.jai` says why: a call in front of a container leaves the tier a kind
#: it cannot name, which refuses `get` outright and costs 10.0M interpreted
#: instructions of `check lib/jaithon`.
#:
#: ```
#: let call = Node.call(Node.ident("f"), [Node.int_lit(1)])
#: to_source(call)                  # "f(1)"
#: call.require("callee").text("name")   # "f"
#: ```
pub class Node {
    #: Which kind of node this is.
    pub var kind: NodeKind
    #: Where the node was written.
    pub var span: Span
    #: The kind's fields, one slot per field, in the order `fields_of` lists.
    pub var fields: list[any]

    #: Create a node of `kind`, overlaying `fields` on that kind's defaults.
    pub fn init(self, kind: NodeKind, fields: dict[str, any]? = null, span: Span? = null) {
        self.kind = kind
        self.span = span ?? Span.none()
        self.fields = new_slots(kind)
        if fields is null { return }
        # The overlay stays interpreted whatever this is written as: the tier's
        # index-store arm wants one kind of value, and `fields` carries every
        # kind a field can hold. What the slot list buys here is the loop that
        # is GONE -- the walk of `fields_of(kind)` that called `_default_field`
        # once per field of every node built.
        let index = _FIELD_INDEX[kind]
        for (name, value) in fields.items() {
            if name not in index {
                throw KeyError(f"{kind_name(kind)} has no field `{name}`")
            }
            self.fields[index[name]] = value
        }
    }

    #: An identifier reference.
    pub static fn ident(name: str) -> Node {
        return Node(NodeKind.Ident, {"name": name})
    }

    #: An integer literal.
    pub static fn int_lit(value: int) -> Node {
        return Node(NodeKind.IntLit, {"intLit": value})
    }

    #: A float literal.
    pub static fn float_lit(value: float) -> Node {
        return Node(NodeKind.FloatLit, {"floatLit": value})
    }

    #: A string literal holding `text`, with its length filled in.
    pub static fn str_lit(text: str) -> Node {
        return Node(NodeKind.StrLit, {"chars": text, "length": text.len()})
    }

    #: A boolean literal.
    pub static fn bool_lit(value: bool) -> Node {
        return Node(NodeKind.BoolLit, {"boolLit": value})
    }

    #: The `null` literal.
    pub static fn null_lit() -> Node {
        return Node(NodeKind.NullLit)
    }

    #: A unary operation, such as `-x` or `not done`.
    pub static fn unary(op: OpKind, operand: Node) -> Node {
        return Node(NodeKind.Unary, {"op": op, "operand": operand})
    }

    #: A binary operation. Use `logical` for `and` and `or`.
    pub static fn binary(op: OpKind, left: Node, right: Node) -> Node {
        return Node(NodeKind.Binary, {"op": op, "left": left, "right": right})
    }

    #: A short-circuiting `and` or `or`.
    pub static fn logical(op: OpKind, left: Node, right: Node) -> Node {
        if op != OpKind.And and op != OpKind.Or {
            throw ValueError(f"a logical node needs `and` or `or`, got {op_text(op)}")
        }
        return Node(NodeKind.Logical, {"op": op, "left": left, "right": right})
    }

    #: A call with positional arguments.
    #:
    #: Pass `arg()` records directly through `Node(NodeKind.Call, ...)` when an
    #: argument is named or spread.
    pub static fn call(callee: Node, args: list[Node]) -> Node {
        return Node(NodeKind.Call, {"callee": callee, "args": [arg(value) for value in args]})
    }

    #: A field or method access, `object.name`.
    pub static fn member(object: Node, name: str) -> Node {
        return Node(NodeKind.Member, {"object": object, "name": name})
    }

    #: A subscript, `object[index]`.
    pub static fn index(object: Node, index: Node) -> Node {
        return Node(NodeKind.Index, {"object": object, "index": index})
    }

    #: A binding pattern, the leaf of a `let` or a match arm.
    pub static fn bind(name: str, declared_type: TypeExpr? = null) -> Node {
        return Node(NodeKind.PatBind, {"name": name, "type": declared_type})
    }

    #: A braced block of statements.
    pub static fn block(stmts: list[Node]) -> Node {
        return Node(NodeKind.Block, {"stmts": stmts})
    }

    #: An expression evaluated for its effect.
    pub static fn expr_stmt(expr: Node) -> Node {
        return Node(NodeKind.ExprStmt, {"expr": expr})
    }

    #: A `let`, `var` or `const` declaration.
    pub static fn var_decl(
        decl: DeclKind,
        pattern: Node,
        declared_type: TypeExpr? = null,
        init: Node? = null
    ) -> Node {
        return Node(
            NodeKind.VarDecl,
            {
                "varDeclKind": decl,
                "pattern": pattern,
                "declaredType": declared_type,
                "init": init,
            }
        )
    }

    #: A plain `target = value` assignment.
    pub static fn assign(target: Node, value: Node) -> Node {
        return Node(NodeKind.Assign, {"target": target, "value": value})
    }

    #: A compound assignment such as `total += n`.
    pub static fn compound_assign(op: OpKind, target: Node, value: Node) -> Node {
        return Node(
            NodeKind.Assign,
            {"target": target, "value": value, "op": op, "isCompound": true}
        )
    }

    #: A `return`, with no value by default.
    pub static fn return_stmt(value: Node? = null) -> Node {
        return Node(NodeKind.Return, {"value": value})
    }

    #: An `if` statement; `else_branch` may be a block or another `if`.
    pub static fn if_stmt(cond: Node, then_branch: Node, else_branch: Node? = null) -> Node {
        return Node(
            NodeKind.If,
            {"cond": cond, "thenBranch": then_branch, "elseBranch": else_branch}
        )
    }

    #: A `while` loop.
    pub static fn while_stmt(cond: Node, body: Node, label: str? = null) -> Node {
        return Node(NodeKind.While, {"cond": cond, "body": body, "label": label})
    }

    #: A `for` loop over `iterable`.
    pub static fn for_stmt(pattern: Node, iterable: Node, body: Node, label: str? = null) -> Node {
        return Node(
            NodeKind.For,
            {"pattern": pattern, "iterable": iterable, "body": body, "label": label}
        )
    }

    #: A function declaration built from `param()` records.
    pub static fn fn_decl(
        name: str,
        params: list[dict[str, any]],
        body: Node,
        return_type: TypeExpr? = null,
        visibility: Visibility = Visibility.Private
    ) -> Node {
        return Node(
            NodeKind.FnDecl,
            {
                "name": name,
                "params": params,
                "returnType": return_type,
                "body": body,
                "visibility": visibility,
            }
        )
    }

    #: A whole module: the root of every parsed file.
    pub static fn program(stmts: list[Node]) -> Node {
        return Node(NodeKind.Program, {"stmts": stmts})
    }

    #: Whether this kind declares a field called `name`.
    pub fn has(self, name: str) -> bool {
        return name in _FIELD_INDEX[self.kind]
    }

    #: The raw value of `name`, raising `KeyError` when the kind has no such field.
    pub fn get(self, name: str) -> any {
        let index = _FIELD_INDEX[self.kind]
        if name not in index {
            throw KeyError(f"{kind_name(self.kind)} has no field `{name}`")
        }
        return self.fields[index[name]]
    }

    #: Set `name` and return self, so builders chain.
    #:
    #: ```
    #: Node.ident("x").set("name", "y").text("name")   # "y"
    #: ```
    pub fn set(self, name: str, value: any) -> Node {
        let index = _FIELD_INDEX[self.kind]
        if name not in index {
            throw KeyError(f"{kind_name(self.kind)} has no field `{name}`")
        }
        self.fields[index[name]] = value
        return self
    }

    #: Return self with `span` replacing the current one.
    pub fn with_span(self, span: Span) -> Node {
        self.span = span
        return self
    }

    #: The child stored under `name`, or null when there is none.
    pub fn child(self, name: str) -> Node? {
        return self.get(name)
    }

    #: The child stored under `name`, raising `ValueError` when it is absent.
    pub fn require(self, name: str) -> Node {
        let value = self.get(name)
        if value is null {
            throw ValueError(f"{kind_name(self.kind)} is missing its `{name}` child")
        }
        return value
    }

    #: The list of children stored under `name`.
    pub fn child_list(self, name: str) -> list[Node] {
        return self.get(name)
    }

    #: The list of sub-structure dicts stored under `name`, such as a call's args.
    pub fn records(self, name: str) -> list[dict[str, any]] {
        return self.get(name)
    }

    #: The string stored under `name`, raising `ValueError` when it is null.
    pub fn text(self, name: str) -> str {
        let value = self.get(name)
        if value is null {
            throw ValueError(f"{kind_name(self.kind)} has no `{name}` text")
        }
        return value
    }

    #: The string stored under `name`, or null.
    pub fn opt_text(self, name: str) -> str? {
        return self.get(name)
    }

    #: The boolean stored under `name`.
    pub fn flag(self, name: str) -> bool {
        return self.get(name)
    }

    #: The integer stored under `name`.
    pub fn number(self, name: str) -> int {
        return self.get(name)
    }

    #: The type expression stored under `name`, or null.
    pub fn type_field(self, name: str) -> TypeExpr? {
        return self.get(name)
    }

    #: The list of type expressions stored under `name`.
    pub fn type_list(self, name: str) -> list[TypeExpr] {
        return self.get(name)
    }

    #: The operator of a unary, binary, logical or assignment node.
    pub fn op(self) -> OpKind {
        return self.get("op")
    }

    #: The visibility of a declaration node.
    pub fn visibility(self) -> Visibility {
        return self.get("visibility")
    }

    #: Every direct child node, in the order `walk` visits them.
    pub fn children(self) -> list[Node] {
        return children_of(self)
    }

    fn __eq__(self, other: Node) -> bool {
        return self.kind == other.kind and self.span == other.span and self.fields == other.fields
    }

    fn __str__(self) -> str {
        return f"{kind_name(self.kind)}[{self.span.start}:{self.span.end}]"
    }
}

fn _record(tag: str, values: dict[str, any]) -> dict[str, any] {
    let record: dict[str, any] = {}
    for (name, field_tag) in record_fields(tag) {
        record[name] = values.get(name, _default_field(name, field_tag))
    }
    return record
}

#: Build a parameter of a function, lambda or enum variant.
pub fn param(
    name: str,
    declared_type: TypeExpr? = null,
    default_value: Node? = null,
    variadic: bool = false,
    kw_rest: bool = false
) -> dict[str, any] {
    if variadic and kw_rest {
        throw ValueError(f"parameter {name} cannot be both `...` and `**`")
    }
    return _record(
        "param",
        {
            "name": name,
            "type": declared_type,
            "defaultValue": default_value,
            "isVariadic": variadic,
            "isKwRest": kw_rest,
        }
    )
}

#: Build a call argument: positional, named when `name` is given, or spread.
pub fn arg(value: Node, name: str? = null, spread: bool = false) -> dict[str, any] {
    return _record("arg", {"name": name, "value": value, "isSpread": spread})
}

#: Build a generic parameter, optionally bounded by a trait.
pub fn generic(name: str, bound: TypeExpr? = null) -> dict[str, any] {
    return _record("generic", {"name": name, "bound": bound})
}

#: Build one arm of a `match`, with an optional guard.
pub fn match_arm(pattern: Node, body: Node, guard: Node? = null) -> dict[str, any] {
    return _record("arm", {"pattern": pattern, "guard": guard, "body": body})
}

#: Build one `catch` clause; an empty `types` catches everything.
pub fn catch_clause(name: str?, body: Node, types: list[TypeExpr] = []) -> dict[str, any] {
    return _record("catch", {"name": name, "types": types, "body": body})
}

#: Build one `for` clause of a comprehension, with its `if` filters.
pub fn comp_clause(pattern: Node, iterable: Node, conditions: list[Node] = []) -> dict[str, any] {
    return _record("clause", {"pattern": pattern, "iterable": iterable, "conditions": conditions})
}

#: Build a class field declaration.
pub fn class_field(
    name: str,
    declared_type: TypeExpr,
    default_value: Node? = null,
    visibility: Visibility = Visibility.Private,
    is_static: bool = false,
    is_let: bool = false
) -> dict[str, any] {
    return _record(
        "field",
        {
            "name": name,
            "type": declared_type,
            "defaultValue": default_value,
            "visibility": visibility,
            "isStatic": is_static,
            "isLet": is_let,
        }
    )
}

#: Build an enum variant, with `param()` records for its payload.
pub fn variant(name: str, params: list[dict[str, any]] = []) -> dict[str, any] {
    return _record("variant", {"name": name, "params": params})
}

#: Build one item of a `from ... import` list.
pub fn import_item(name: str, alias: str? = null) -> dict[str, any] {
    return _record("item", {"name": name, "alias": alias})
}

#: Every direct child of `node`, in the order the C `jaiAstWalk` visits them.
#:
#: Null children are dropped, exactly as the C walk skips them.
pub fn children_of(node: Node) -> list[Node] {
    let out: list[Node] = []

    if node.kind == NodeKind.DictLit {
        let keys = node.child_list("keys")
        let values = node.child_list("values")
        for index in 0..max(keys.len(), values.len()) {
            if index < keys.len() { _push_child(out, keys[index]) }
            if index < values.len() { _push_child(out, values[index]) }
        }
        return out
    }
    if node.kind == NodeKind.Comprehension {
        _push_child(out, node.child("keyExpr"))
        _push_child(out, node.child("element"))
        _collect_children(out, node.records("clauses"), "clause[]")
        return out
    }

    for (slot, tag) in tags_of(node.kind).enumerate() {
        _collect_children(out, node.fields[slot], tag)
    }
    return out
}

fn _push_child(out: list[Node], value: any) -> void {
    if value is not null { out.push(value) }
}

fn _collect_children(out: list[Node], value: any, tag: str) -> void {
    if value is null { return }
    if tag == "node" {
        _push_child(out, value)
        return
    }
    if tag == "nodes" {
        for item in value { _push_child(out, item) }
        return
    }
    let element = _ELEMENT_TAG.get(tag, "")
    if element not in _RECORDS { return }
    for record in value {
        for (name, inner) in _RECORDS[element] {
            _collect_children(out, record.get(name, null), inner)
        }
    }
}
