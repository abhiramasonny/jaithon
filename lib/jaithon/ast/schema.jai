#: The field schema: which fields each `NodeKind` declares, in the order the C
#: emits them, with what tag, and what an unset one defaults to.
#:
#: `Node.init` and `children_of` are the only readers inside this directory,
#: but `ast_encode.jai` reads the same tables without building anything, which
#: is why `_ELEMENT_TAG`, `_RECORDS` and `_default_field` were already `pub`
#: before the directory existed and are re-exported by the `jaithon.ast`
#: facade. Keeping the tables here rather than in `node.jai` is what lets a
#: reader of the encoder stop at one file.
#:
#: A node stores its fields in a slot list rather than a dict, so the four
#: tables at the bottom of this file are what turn a field NAME into a
#: position: `_FIELD_INDEX`, and the three behind `tags_of` and `new_slots`.
#: All four are derived from the same `_SCHEMAS` that `fields_of` answers
#: from, and all four are built once, at module load.
#:
#: What is NOT here: the sub-structure builders. `param()`, `arg()` and their
#: siblings fill a record out of `_RECORDS` and `_default_field`, but they
#: build `Node`-carrying dicts and `Node.call` calls one of them, so they live
#: with `Node`.
from jaithon.ast.kinds import CompKind, DeclKind, NodeKind, Visibility, _KINDS, _KIND_INDEX
from jaithon.ast.operators import OpKind
from jaithon.ast.span import Span

pub let _ELEMENT_TAG: dict[str, str] = {
    "nodes": "node",
    "strs": "str",
    "str[]": "str",
    "types": "type",
    "ops": "op",
    "param[]": "param",
    "arg[]": "arg",
    "generic[]": "generic",
    "clause[]": "clause",
    "arm[]": "arm",
    "catch[]": "catch",
    "field[]": "field",
    "variant[]": "variant",
    "item[]": "item",
}

pub let _RECORDS: dict[str, list[tuple[str, str]]] = {
    "param": [
        ("name", "str"),
        ("type", "type"),
        ("defaultValue", "node"),
        ("span", "span"),
        ("isVariadic", "bool"),
        ("isKwRest", "bool"),
    ],
    "arg": [
        ("name", "str"),
        ("value", "node"),
        ("isSpread", "bool"),
        ("span", "span"),
    ],
    "generic": [
        ("name", "str"),
        ("bound", "type"),
        ("span", "span"),
    ],
    "clause": [
        ("pattern", "node"),
        ("iterable", "node"),
        ("conditions", "nodes"),
        ("span", "span"),
    ],
    "arm": [
        ("pattern", "node"),
        ("guard", "node"),
        ("body", "node"),
        ("span", "span"),
    ],
    "catch": [
        ("name", "str"),
        ("types", "types"),
        ("body", "node"),
        ("span", "span"),
    ],
    "field": [
        ("name", "str"),
        ("type", "type"),
        ("defaultValue", "node"),
        ("visibility", "vis"),
        ("isStatic", "bool"),
        ("isLet", "bool"),
        ("span", "span"),
    ],
    "variant": [
        ("name", "str"),
        ("params", "param[]"),
        ("span", "span"),
    ],
    "item": [
        ("name", "str"),
        ("alias", "str"),
        ("span", "span"),
    ],
}

fn _build_schemas() -> list[list[tuple[str, str]]] {
    let schemas: list[list[tuple[str, str]]] = [[] for _kind in _KINDS]

    fn define(kinds: list[NodeKind], fields: list[tuple[str, str]]) -> void {
        for kind in kinds { schemas[_KIND_INDEX[kind]] = fields }
    }

    define([NodeKind.IntLit], [("intLit", "int")])
    define([NodeKind.FloatLit], [("floatLit", "float")])
    define([NodeKind.StrLit], [("chars", "str"), ("length", "int")])
    define([NodeKind.BoolLit], [("boolLit", "bool")])
    define([NodeKind.NullLit, NodeKind.Self, NodeKind.Super, NodeKind.PatWildcard], [])
    define([NodeKind.FString], [("parts", "nodes")])
    define([NodeKind.Ident, NodeKind.ModuleDecl], [("name", "str")])
    define([NodeKind.ListLit, NodeKind.SetLit, NodeKind.TupleLit], [("items", "nodes")])
    define([NodeKind.DictLit], [("keys", "nodes"), ("values", "nodes")])
    define([NodeKind.Unary], [("op", "op"), ("operand", "node")])
    define([NodeKind.Binary, NodeKind.Logical], [("op", "op"), ("left", "node"), ("right", "node")])
    define([NodeKind.CompareChain], [("operands", "nodes"), ("ops", "ops")])
    define([NodeKind.Ternary], [("cond", "node"), ("thenExpr", "node"), ("elseExpr", "node")])
    define([NodeKind.Coalesce], [("left", "node"), ("right", "node")])
    define([NodeKind.Call], [("callee", "node"), ("args", "arg[]")])
    define([NodeKind.Index], [("object", "node"), ("index", "node")])
    define(
        [NodeKind.Slice],
        [("object", "node"), ("start", "node"), ("stop", "node"), ("step", "node")]
    )
    define([NodeKind.Member, NodeKind.OptMember], [("object", "node"), ("name", "str")])
    define(
        [NodeKind.Lambda, NodeKind.AnonFn, NodeKind.FnDecl],
        [
            ("name", "str"),
            ("generics", "generic[]"),
            ("params", "param[]"),
            ("returnType", "type"),
            ("body", "node"),
            ("visibility", "vis"),
            ("isStatic", "bool"),
            ("isExprBody", "bool"),
            ("isGenerator", "bool"),
            ("isAsync", "bool"),
            ("decorators", "str[]"),
        ]
    )
    define(
        [NodeKind.Comprehension],
        [("compKind", "compk"), ("element", "node"), ("keyExpr", "node"), ("clauses", "clause[]")]
    )
    define([NodeKind.Range], [("start", "node"), ("stop", "node"), ("inclusive", "bool")])
    define(
        [NodeKind.IfExpr, NodeKind.If],
        [("cond", "node"), ("thenBranch", "node"), ("elseBranch", "node")]
    )
    define([NodeKind.Match, NodeKind.MatchExpr], [("subject", "node"), ("arms", "arm[]")])
    define([NodeKind.Cast], [("operand", "node"), ("target", "type"), ("widen", "bool")])
    define([NodeKind.Yield, NodeKind.Await, NodeKind.ThrowExpr], [("operand", "node")])
    define([NodeKind.Block, NodeKind.Program], [("stmts", "nodes")])
    define([NodeKind.ExprStmt], [("expr", "node")])
    define(
        [NodeKind.VarDecl],
        [
            ("varDeclKind", "vdk"),
            ("pattern", "node"),
            ("declaredType", "type"),
            ("init", "node"),
            ("visibility", "vis"),
        ]
    )
    define(
        [NodeKind.Assign],
        [("target", "node"), ("value", "node"), ("op", "op"), ("isCompound", "bool")]
    )
    define([NodeKind.While, NodeKind.Loop], [("cond", "node"), ("body", "node"), ("label", "str")])
    define(
        [NodeKind.For],
        [("pattern", "node"), ("iterable", "node"), ("body", "node"), ("label", "str")]
    )
    define([NodeKind.Break, NodeKind.Continue], [("label", "str")])
    define([NodeKind.Return, NodeKind.Throw], [("value", "node")])
    define([NodeKind.Try], [("body", "node"), ("catches", "catch[]"), ("finallyBlock", "node")])
    define([NodeKind.Defer], [("body", "node")])
    define([NodeKind.Assert], [("cond", "node"), ("message", "node")])
    define(
        [NodeKind.ClassDecl],
        [
            ("name", "str"),
            ("generics", "generic[]"),
            ("superclass", "type"),
            ("traits", "types"),
            ("fields", "field[]"),
            ("methods", "nodes"),
            ("getters", "nodes"),
            ("setters", "nodes"),
            ("visibility", "vis"),
            ("isAbstract", "bool"),
        ]
    )
    define(
        [NodeKind.TraitDecl],
        [
            ("name", "str"),
            ("generics", "generic[]"),
            ("supers", "types"),
            ("methods", "nodes"),
            ("visibility", "vis"),
        ]
    )
    define(
        [NodeKind.EnumDecl],
        [
            ("name", "str"),
            ("generics", "generic[]"),
            ("variants", "variant[]"),
            ("methods", "nodes"),
            ("visibility", "vis"),
        ]
    )
    define(
        [NodeKind.TypeDecl],
        [("name", "str"), ("generics", "generic[]"), ("aliased", "type"), ("visibility", "vis")]
    )
    define([NodeKind.Import], [("path", "str"), ("alias", "str")])
    define([NodeKind.FromImport], [("path", "str"), ("items", "item[]"), ("isWildcard", "bool")])
    define([NodeKind.Export], [("names", "strs")])
    define([NodeKind.PatBind], [("name", "str"), ("type", "type")])
    define([NodeKind.PatLiteral], [("value", "node")])
    define([NodeKind.PatRange], [("lo", "node"), ("hi", "node"), ("inclusive", "bool")])
    define(
        [NodeKind.PatTuple, NodeKind.PatList, NodeKind.PatOr],
        [("elems", "nodes"), ("restIndex", "int")]
    )
    define(
        [NodeKind.PatClass, NodeKind.PatEnum],
        [
            ("typeName", "str"),
            ("variantName", "str"),
            ("fieldNames", "strs"),
            ("subPatterns", "nodes"),
        ]
    )
    return schemas
}

let _SCHEMAS: list[list[tuple[str, str]]] = _build_schemas()

#: Return the fields of `kind` as `(name, tag)` pairs, in C emission order.
#:
#: ```
#: fields_of(NodeKind.Index)   # [("object", "node"), ("index", "node")]
#: ```
pub fn fields_of(kind: NodeKind) -> list[tuple[str, str]] {
    return _SCHEMAS[_KIND_INDEX[kind]]
}

#: Return the fields of the sub-structure named by a list tag such as `"arm[]"`.
#:
#: ```
#: record_fields("item[]")   # [("name", "str"), ("alias", "str"), ("span", "span")]
#: ```
pub fn record_fields(tag: str) -> list[tuple[str, str]] {
    let element = _ELEMENT_TAG.get(tag, tag)
    let fields = _RECORDS.get(element, null)
    if fields is null { throw ValueError(f"no AST sub-structure is tagged {tag}") }
    return fields
}

pub fn _default_field(name: str, tag: str) -> any {
    # A sequence pattern with no `...rest` has restIndex -1; 0 would claim the
    # first element is the rest.
    if name == "restIndex" { return -1 }
    if tag in _ELEMENT_TAG { return [] }
    # A sub-structure always carries a span, synthesised when it has no source.
    if tag == "span" { return Span.none() }
    if tag == "bool" { return false }
    if tag == "int" { return 0 }
    if tag == "float" { return 0.0 }
    if tag == "op" { return OpKind.Add }
    if tag == "vis" { return Visibility.Private }
    if tag == "vdk" { return DeclKind.Let }
    if tag == "compk" { return CompKind.List }
    return null
}

# Four views of _SCHEMAS, one per question a slot list gets asked: which slot
# does this name occupy, what tag does this slot carry, what does a fresh slot
# list look like, and which of its slots must not be shared. They are built
# here, below `_default_field`, because `_build_default_slots` calls it.
#
# Keyed by the kind itself rather than by `_KIND_INDEX[kind]`, which is how
# `_SCHEMAS` is indexed. That is one dict read where the two-table form is a
# dict read and then a list read, and these four are read once per node built
# and once per field touched.
fn _build_field_index() -> dict[NodeKind, dict[str, int]] {
    let indexes: dict[NodeKind, dict[str, int]] = {}
    for kind in _KINDS {
        let index: dict[str, int] = {}
        var slot = 0
        for (name, _tag) in _SCHEMAS[_KIND_INDEX[kind]] {
            index[name] = slot
            slot = slot + 1
        }
        indexes[kind] = index
    }
    return indexes
}

fn _build_tags() -> dict[NodeKind, list[str]] {
    let tags: dict[NodeKind, list[str]] = {}
    for kind in _KINDS {
        tags[kind] = [tag for (_name, tag) in _SCHEMAS[_KIND_INDEX[kind]]]
    }
    return tags
}

fn _build_default_slots() -> dict[NodeKind, list[any]] {
    let defaults: dict[NodeKind, list[any]] = {}
    for kind in _KINDS {
        let fields = _SCHEMAS[_KIND_INDEX[kind]]
        defaults[kind] = [_default_field(name, tag) for (name, tag) in fields]
    }
    return defaults
}

# The slots `_default_field` answers with a fresh `[]`. A prebuilt row is
# copied rather than rebuilt, and a shallow copy would hand every node of a
# kind the SAME list, so these are the slots `new_slots` refills.
#
# Every row ends in -1. Most kinds declare no list field at all, and an empty
# row leaves the tier nothing to sample -- it refuses the whole of `new_slots`
# with "the live list is empty, so there is no exemplar", which costs every
# node its slot list. The terminator is also the loop's condition, so it pays
# for itself.
fn _build_list_slots() -> dict[NodeKind, list[int]] {
    let lists: dict[NodeKind, list[int]] = {}
    for kind in _KINDS {
        let slots: list[int] = []
        var slot = 0
        for (_name, tag) in _SCHEMAS[_KIND_INDEX[kind]] {
            if tag in _ELEMENT_TAG { slots.push(slot) }
            slot = slot + 1
        }
        slots.push(-1)
        lists[kind] = slots
    }
    return lists
}

#: The slot each field of `kind` occupies, keyed by the name `fields_of` lists.
#:
#: A table rather than a `field_index(kind)` in front of it, and `pub` for the
#: same reason `_RECORDS` is: `Node`'s accessors read it on every field
#: touched, and a call in front of a container leaves the tier a kind it cannot
#: name. That alone refused `Node.get` and cost 10.0M interpreted instructions
#: of `check lib/jaithon`. It is NOT re-exported by the `jaithon.ast` facade —
#: `node.jai` is the only reader, so the public surface is what it was.
#:
#: ```
#: _FIELD_INDEX[NodeKind.Index]   # {"object": 0, "index": 1}
#: ```
pub let _FIELD_INDEX: dict[NodeKind, dict[str, int]] = _build_field_index()

let _TAGS: dict[NodeKind, list[str]] = _build_tags()
let _DEFAULT_SLOTS: dict[NodeKind, list[any]] = _build_default_slots()
let _LIST_SLOTS: dict[NodeKind, list[int]] = _build_list_slots()

#: The tag of each field of `kind`, in slot order.
#:
#: ```
#: tags_of(NodeKind.Index)   # ["node", "node"]
#: ```
pub fn tags_of(kind: NodeKind) -> list[str] {
    return _TAGS[kind]
}

#: A fresh slot list for `kind`, every slot holding that field's default.
#:
#: ```
#: new_slots(NodeKind.StrLit)   # [null, 0], the `chars` and `length` defaults
#: ```
pub fn new_slots(kind: NodeKind) -> list[any] {
    # Annotated, and not because the checker needs it: `.copy()` answers a
    # container of no known kind, and the tier then refuses the store below
    # with "an index store into an object that is not a dict" -- which is the
    # whole of this function, once per node.
    let slots: list[any] = _DEFAULT_SLOTS[kind].copy()
    let refill = _LIST_SLOTS[kind]
    var at = 0
    while refill[at] >= 0 {
        slots[refill[at]] = []
        at = at + 1
    }
    return slots
}
