#: The Jaithon abstract syntax tree, as Jaithon data.
#:
#: This module mirrors `src/lang/ast.h` case for case and field for field: one
#: `NodeKind` per C `AstKind`, one `OpKind` per C `OpKind`, and field names
#: spelled exactly as `jaiAstToJson` writes them. Three properties hold across
#: this file and its two encoding siblings (see below), and the formatter, the
#: documentation generator and the self-hosted front end all rest on them:
#:
#: - `from_json` reads, unchanged, what `jaithon --emit=ast` prints.
#: - `from_json(to_json(node))` equals `node`.
#: - Re-parsing `to_source(node)` yields `node` again, spans aside.
#:
#: A node is a kind, a span, and a `fields` dict keyed by the C field names.
#: `fields_of(kind)` lists a kind's fields in the order the C emits them, each
#: tagged with the shape of its value:
#:
#: | tag | Jaithon value |
#: |---|---|
#: | `"node"` | `Node?` |
#: | `"nodes"` | `list[Node]` |
#: | `"str"` | `str?` |
#: | `"strs"` | `list[str?]` |
#: | `"bool"`, `"int"`, `"float"` | the scalar of that name |
#: | `"type"`, `"types"` | `TypeExpr?`, `list[TypeExpr]` |
#: | `"op"`, `"ops"` | `OpKind`, `list[OpKind]` |
#: | `"vis"`, `"vdk"`, `"compk"` | `Visibility`, `DeclKind`, `CompKind` |
#: | `"param[]"` and its eight siblings | `list[dict[str, any]]` |
#:
#: The bracketed tags are the C sub-structures — `AstParam`, `AstArg`,
#: `AstMatchArm` and the rest. They are dicts rather than classes because they
#: carry no behaviour; build them with `param()`, `arg()`, `match_arm()` and
#: their siblings so that every key is present, and read their shapes back with
#: `record_fields()`.
#:
#: ```
#: let call = Node.call(Node.ident("f"), [Node.int_lit(1)])
#: call.require("callee").text("name")            # "f"
#: fields_of(NodeKind.Index)                       # [("object", "node"), ("index", "node")]
#: ```
#:
#: ## What lives here, and what moved out
#:
#: This file keeps the spine every node-shaped thing hangs off: the
#: `NodeKind`/`OpKind` vocabulary and the field schema built from it, `Span`,
#: `TypeExpr` and `Node` themselves, the sub-structure builders `Node`'s own
#: builders call (`param`, `arg`, ...), and the traversal (`children_of`,
#: `walk`, ...) and divergence predicates (`always_diverges`, ...) that
#: `Node.children()` and the type checker call directly. Those pieces reach
#: into each other's private state too much to separate without either an
#: import cycle back into this file or turning a schema table into public API
#: for no reason.
#:
#: Three things that only *read* an already-built tree do not need any of
#: that, so they moved to siblings in this directory, each taking the private
#: helpers only it used: rendering a tree as debug text or as the C's JSON
#: wire format lives in `ast_encode.jai` (`to_sexp`, `to_json`/`from_json`);
#: rendering it back as Jaithon source lives in `ast_unparse.jai`
#: (`to_source`), by far the largest single piece and the reason this file was
#: over 3800 lines; and the per-kind dispatch base class lives in
#: `ast_visitor.jai` (`Visitor`), which nothing in this tree currently
#: imports. The only change made here to let them reach the schema is marking
#: `_ELEMENT_TAG`, `_RECORDS` and `_default_field` (below) `pub` — `ast_encode`
#: needs the same field tags `Node.init` and `children_of` already rely on.

#: Every node kind, in the order of the C `AstKind` enum.
#:
#: Expressions come first, then statements, then declarations, then patterns,
#: with `Program` last. `is_expression`, `is_statement`, `is_declaration` and
#: `is_pattern` answer which group a kind belongs to.
pub enum NodeKind {
    IntLit,
    FloatLit,
    StrLit,
    BoolLit,
    NullLit,
    FString,
    Ident,
    Self,
    Super,
    ListLit,
    DictLit,
    SetLit,
    TupleLit,
    Unary,
    Binary,
    Logical,
    CompareChain,
    Ternary,
    Coalesce,
    Call,
    Index,
    Slice,
    Member,
    OptMember,
    Lambda,
    AnonFn,
    Comprehension,
    Range,
    IfExpr,
    MatchExpr,
    Cast,
    Yield,
    Await,
    ThrowExpr,

    Block,
    ExprStmt,
    VarDecl,
    Assign,
    If,
    While,
    Loop,
    For,
    Match,
    Break,
    Continue,
    Return,
    Throw,
    Try,
    Defer,
    Assert,

    FnDecl,
    ClassDecl,
    TraitDecl,
    EnumDecl,
    TypeDecl,
    Import,
    FromImport,
    Export,
    ModuleDecl,

    PatWildcard,
    PatBind,
    PatLiteral,
    PatRange,
    PatTuple,
    PatList,
    PatClass,
    PatEnum,
    PatOr,

    Program,
}

#: Every operator code, in the order of the C `OpKind` enum.
#:
#: `AddWrap`, `SubWrap` and `MulWrap` are the wrapping operators `+%`, `-%` and
#: `*%`; `Neg`, `Pos`, `Not` and `Bnot` are the four unary operators.
pub enum OpKind {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    AddWrap,
    SubWrap,
    MulWrap,
    Band,
    Bor,
    Bxor,
    Shl,
    Shr,
    MatMul,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Is,
    IsNot,
    In,
    NotIn,
    And,
    Or,
    Neg,
    Pos,
    Not,
    Bnot,
}

#: The visibility a declaration was written with.
pub enum Visibility { Private, Protected, Public }

#: Which of `let`, `var` and `const` introduced a binding.
pub enum DeclKind { Let, Var, Const }

#: Which brackets a comprehension was written with.
pub enum CompKind { List, Dict, Set, Generator }

#: The shape of a type expression, mirroring the C `AstTypeKind`.
#: `Const` is an integer literal in type position (`Tensor[float, 2, 3]`).
pub enum TypeKind { Name, Generic, Optional, Union, Fn, Tuple, Infer, Const }

# ------------------------------------------------------------------
# Name tables
# ------------------------------------------------------------------
let _KIND_NAMES: list[str] = [
    "AST_INT_LIT",
    "AST_FLOAT_LIT",
    "AST_STR_LIT",
    "AST_BOOL_LIT",
    "AST_NULL_LIT",
    "AST_FSTRING",
    "AST_IDENT",
    "AST_SELF",
    "AST_SUPER",
    "AST_LIST_LIT",
    "AST_DICT_LIT",
    "AST_SET_LIT",
    "AST_TUPLE_LIT",
    "AST_UNARY",
    "AST_BINARY",
    "AST_LOGICAL",
    "AST_COMPARE_CHAIN",
    "AST_TERNARY",
    "AST_COALESCE",
    "AST_CALL",
    "AST_INDEX",
    "AST_SLICE",
    "AST_MEMBER",
    "AST_OPT_MEMBER",
    "AST_LAMBDA",
    "AST_ANON_FN",
    "AST_COMPREHENSION",
    "AST_RANGE",
    "AST_IF_EXPR",
    "AST_MATCH_EXPR",
    "AST_CAST",
    "AST_YIELD",
    "AST_AWAIT",
    "AST_THROW_EXPR",
    "AST_BLOCK",
    "AST_EXPR_STMT",
    "AST_VAR_DECL",
    "AST_ASSIGN",
    "AST_IF",
    "AST_WHILE",
    "AST_LOOP",
    "AST_FOR",
    "AST_MATCH",
    "AST_BREAK",
    "AST_CONTINUE",
    "AST_RETURN",
    "AST_THROW",
    "AST_TRY",
    "AST_DEFER",
    "AST_ASSERT",
    "AST_FN_DECL",
    "AST_CLASS_DECL",
    "AST_TRAIT_DECL",
    "AST_ENUM_DECL",
    "AST_TYPE_DECL",
    "AST_IMPORT",
    "AST_FROM_IMPORT",
    "AST_EXPORT",
    "AST_MODULE_DECL",
    "AST_PAT_WILDCARD",
    "AST_PAT_BIND",
    "AST_PAT_LITERAL",
    "AST_PAT_RANGE",
    "AST_PAT_TUPLE",
    "AST_PAT_LIST",
    "AST_PAT_CLASS",
    "AST_PAT_ENUM",
    "AST_PAT_OR",
    "AST_PROGRAM",
]

let _KINDS: list[NodeKind] = [
    NodeKind.IntLit,
    NodeKind.FloatLit,
    NodeKind.StrLit,
    NodeKind.BoolLit,
    NodeKind.NullLit,
    NodeKind.FString,
    NodeKind.Ident,
    NodeKind.Self,
    NodeKind.Super,
    NodeKind.ListLit,
    NodeKind.DictLit,
    NodeKind.SetLit,
    NodeKind.TupleLit,
    NodeKind.Unary,
    NodeKind.Binary,
    NodeKind.Logical,
    NodeKind.CompareChain,
    NodeKind.Ternary,
    NodeKind.Coalesce,
    NodeKind.Call,
    NodeKind.Index,
    NodeKind.Slice,
    NodeKind.Member,
    NodeKind.OptMember,
    NodeKind.Lambda,
    NodeKind.AnonFn,
    NodeKind.Comprehension,
    NodeKind.Range,
    NodeKind.IfExpr,
    NodeKind.MatchExpr,
    NodeKind.Cast,
    NodeKind.Yield,
    NodeKind.Await,
    NodeKind.ThrowExpr,
    NodeKind.Block,
    NodeKind.ExprStmt,
    NodeKind.VarDecl,
    NodeKind.Assign,
    NodeKind.If,
    NodeKind.While,
    NodeKind.Loop,
    NodeKind.For,
    NodeKind.Match,
    NodeKind.Break,
    NodeKind.Continue,
    NodeKind.Return,
    NodeKind.Throw,
    NodeKind.Try,
    NodeKind.Defer,
    NodeKind.Assert,
    NodeKind.FnDecl,
    NodeKind.ClassDecl,
    NodeKind.TraitDecl,
    NodeKind.EnumDecl,
    NodeKind.TypeDecl,
    NodeKind.Import,
    NodeKind.FromImport,
    NodeKind.Export,
    NodeKind.ModuleDecl,
    NodeKind.PatWildcard,
    NodeKind.PatBind,
    NodeKind.PatLiteral,
    NodeKind.PatRange,
    NodeKind.PatTuple,
    NodeKind.PatList,
    NodeKind.PatClass,
    NodeKind.PatEnum,
    NodeKind.PatOr,
    NodeKind.Program,
]

let _KIND_INDEX: dict[NodeKind, int] = {kind: index for (index, kind) in _KINDS.enumerate()}
let _KIND_BY_NAME: dict[str, NodeKind] = {name: kind for (name, kind) in zip(_KIND_NAMES, _KINDS)}

const _FIRST_STATEMENT = 34
const _FIRST_DECLARATION = 50
const _FIRST_PATTERN = 59
const _PROGRAM_INDEX = 68

assert _KIND_NAMES.len() == _KINDS.len(), "the AST kind tables have different lengths"
assert _KIND_NAMES[_FIRST_STATEMENT] == "AST_BLOCK", "the AST kind table drifted from AstKind"
assert _KIND_NAMES[_PROGRAM_INDEX] == "AST_PROGRAM", "the AST kind table drifted from AstKind"

let _OP_NAMES: list[str] = [
    "OPK_ADD",
    "OPK_SUB",
    "OPK_MUL",
    "OPK_DIV",
    "OPK_FLOORDIV",
    "OPK_MOD",
    "OPK_POW",
    "OPK_ADD_WRAP",
    "OPK_SUB_WRAP",
    "OPK_MUL_WRAP",
    "OPK_BAND",
    "OPK_BOR",
    "OPK_BXOR",
    "OPK_SHL",
    "OPK_SHR",
    "OPK_MATMUL",
    "OPK_EQ",
    "OPK_NE",
    "OPK_LT",
    "OPK_LE",
    "OPK_GT",
    "OPK_GE",
    "OPK_IS",
    "OPK_IS_NOT",
    "OPK_IN",
    "OPK_NOT_IN",
    "OPK_AND",
    "OPK_OR",
    "OPK_NEG",
    "OPK_POS",
    "OPK_NOT",
    "OPK_BNOT",
]

let _OPS: list[OpKind] = [
    OpKind.Add,
    OpKind.Sub,
    OpKind.Mul,
    OpKind.Div,
    OpKind.FloorDiv,
    OpKind.Mod,
    OpKind.Pow,
    OpKind.AddWrap,
    OpKind.SubWrap,
    OpKind.MulWrap,
    OpKind.Band,
    OpKind.Bor,
    OpKind.Bxor,
    OpKind.Shl,
    OpKind.Shr,
    OpKind.MatMul,
    OpKind.Eq,
    OpKind.Ne,
    OpKind.Lt,
    OpKind.Le,
    OpKind.Gt,
    OpKind.Ge,
    OpKind.Is,
    OpKind.IsNot,
    OpKind.In,
    OpKind.NotIn,
    OpKind.And,
    OpKind.Or,
    OpKind.Neg,
    OpKind.Pos,
    OpKind.Not,
    OpKind.Bnot,
]

let _OP_SOURCE: list[str] = [
    "+",
    "-",
    "*",
    "/",
    "//",
    "%",
    "**",
    "+%",
    "-%",
    "*%",
    "&",
    "|",
    "^",
    "<<",
    ">>",
    "@",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "is",
    "is not",
    "in",
    "not in",
    "and",
    "or",
    "-",
    "+",
    "not",
    "~",
]

let _OP_INDEX: dict[OpKind, int] = {op: index for (index, op) in _OPS.enumerate()}
let _OP_BY_NAME: dict[str, OpKind] = {name: op for (name, op) in zip(_OP_NAMES, _OPS)}

const _FIRST_COMPARISON = 16
const _LAST_COMPARISON = 25

assert _OP_NAMES.len() == _OPS.len(), "the operator tables have different lengths"
assert _OP_SOURCE.len() == _OPS.len(), "the operator source table has the wrong length"

#: Return the C enum name of `kind`, the spelling used in the JSON encoding.
#:
#: ```
#: kind_name(NodeKind.IntLit)   # "AST_INT_LIT"
#: ```
pub fn kind_name(kind: NodeKind) -> str {
    return _KIND_NAMES[_KIND_INDEX[kind]]
}

#: Return the kind whose C name is `name`.
#:
#: Raises `ValueError` for a name no version of the C AST ever emitted, which is
#: the only useful reaction to a tree from a future compiler.
#:
#: ```
#: kind_from_name("AST_CALL")   # NodeKind.Call
#: ```
pub fn kind_from_name(name: str) -> NodeKind {
    let kind = _KIND_BY_NAME.get(name, null)
    if kind is null { throw ValueError(f"unknown AST kind: {name}") }
    return kind
}

#: Return the C enum name of `op`.
pub fn op_name(op: OpKind) -> str {
    return _OP_NAMES[_OP_INDEX[op]]
}

#: Return the operator whose C name is `name`.
pub fn op_from_name(name: str) -> OpKind {
    let op = _OP_BY_NAME.get(name, null)
    if op is null { throw ValueError(f"unknown operator kind: {name}") }
    return op
}

#: Return the source spelling of `op`.
#:
#: ```
#: op_text(OpKind.Pow)     # "**"
#: op_text(OpKind.IsNot)   # "is not"
#: ```
pub fn op_text(op: OpKind) -> str {
    return _OP_SOURCE[_OP_INDEX[op]]
}

#: Whether `op` is one of the ten comparison operators, which chain.
pub fn op_is_comparison(op: OpKind) -> bool {
    let index = _OP_INDEX[op]
    return index >= _FIRST_COMPARISON and index <= _LAST_COMPARISON
}

#: Whether nodes of `kind` produce a value.
pub fn is_expression(kind: NodeKind) -> bool {
    return _KIND_INDEX[kind] < _FIRST_STATEMENT
}

#: Whether nodes of `kind` may appear directly in a block.
#:
#: Declarations are statements, exactly as in the C predicate: they live in
#: blocks. `AST_PROGRAM` is neither a statement nor an expression.
pub fn is_statement(kind: NodeKind) -> bool {
    let index = _KIND_INDEX[kind]
    return index >= _FIRST_STATEMENT and index < _FIRST_PATTERN
}

#: Whether `kind` declares a name: a function, class, trait, enum, type alias,
#: import, export or module header.
pub fn is_declaration(kind: NodeKind) -> bool {
    let index = _KIND_INDEX[kind]
    return index >= _FIRST_DECLARATION and index < _FIRST_PATTERN
}

#: Whether `kind` is one of the nine pattern kinds.
pub fn is_pattern(kind: NodeKind) -> bool {
    let index = _KIND_INDEX[kind]
    return index >= _FIRST_PATTERN and index < _PROGRAM_INDEX
}

# ------------------------------------------------------------------
# Field schema
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Spans
# ------------------------------------------------------------------

#: A half-open range of source text, as the C lexer records it.
#:
#: `start` and `end` are UTF-8 **byte** offsets into the file, so they can be
#: compared with the C compiler's own diagnostics. Lines and columns are 1-based
#: and count Unicode scalar values, which is what §1.1 of the language spec
#: requires and what a caret under an emoji needs.
#:
#: ```
#: Span(0, 4, 5).line_column("let x = 1")   # (1, 5)
#: Span(0, 4, 5).text("let x = 1")          # "x"
#: ```
pub class Span {
    #: Index of the source file in the compilation's file table, or -1.
    pub let file: int
    #: Byte offset of the first character of the span.
    pub let start: int
    #: Byte offset one past the last character of the span.
    pub let end: int

    #: Create a span over `[start, end)` of the file numbered `file`.
    pub fn init(self, file: int, start: int, end: int) {
        if end < start { throw ValueError(f"span end {end} is before its start {start}") }
        self.file = file
        self.start = start
        self.end = end
    }

    #: The span of a synthesised node: file -1, an empty range at offset 0.
    pub static fn none() -> Span {
        return Span(-1, 0, 0)
    }

    #: Whether this span points into a real file.
    pub fn is_known(self) -> bool {
        return self.file >= 0
    }

    #: The number of bytes the span covers.
    pub fn length(self) -> int {
        return self.end - self.start
    }

    #: The smallest span covering both `self` and `other`.
    #:
    #: Mirrors `jaiSpanJoin`: a synthesised span contributes nothing, so joining
    #: a real span with `Span.none()` gives the real one back.
    pub fn join(self, other: Span) -> Span {
        if not other.is_known() { return self }
        if not self.is_known() { return other }
        return Span(self.file, min(self.start, other.start), max(self.end, other.end))
    }

    #: The slice of `source` this span covers.
    pub fn text(self, source: str) -> str {
        return source[_char_index(source, self.start):_char_index(source, self.end)]
    }

    #: The 1-based line and column of `start` in `source`.
    pub fn line_column(self, source: str) -> tuple[int, int] {
        var offset = 0
        var line = 1
        var column = 1
        for character in source {
            if offset >= self.start { return (line, column) }
            offset += _utf8_length(character)
            if character == "\n" {
                line += 1
                column = 1
            } else {
                column += 1
            }
        }
        return (line, column)
    }

    #: The 1-based line of `start` in `source`.
    pub fn line(self, source: str) -> int {
        let (line, _column) = self.line_column(source)
        return line
    }

    #: The 1-based column of `start` in `source`.
    pub fn column(self, source: str) -> int {
        let (_line, column) = self.line_column(source)
        return column
    }

    #: The whole source line containing `start`, without its terminator.
    #:
    #: This is the line a diagnostic prints above its caret.
    pub fn line_text(self, source: str) -> str {
        let lines = source.split("\n")
        let index = self.line(source) - 1
        if index < 0 or index >= lines.len() { return "" }
        return lines[index]
    }

    fn __eq__(self, other: Span) -> bool {
        return self.file == other.file and self.start == other.start and self.end == other.end
    }

    fn __hash__(self) -> int {
        return hash((self.file, self.start, self.end))
    }

    fn __str__(self) -> str {
        return f"Span({self.file}, {self.start}, {self.end})"
    }
}

fn _utf8_length(character: str) -> int {
    let code = ord(character)
    if code < 0x80 { return 1 }
    if code < 0x800 { return 2 }
    if code < 0x10000 { return 3 }
    return 4
}

fn _char_index(source: str, offset: int) -> int {
    var byte = 0
    var index = 0
    for character in source {
        if byte >= offset { return index }
        byte += _utf8_length(character)
        index += 1
    }
    return index
}

# ------------------------------------------------------------------
# Type expressions
# ------------------------------------------------------------------

#: A type as it was written, before the checker resolves it.
#:
#: Mirrors the C `AstType`: `args` holds a generic's arguments, a union's
#: members, a tuple's members or a function's parameters, and `inner` holds an
#: optional's element or a function's return type.
#:
#: ```
#: TypeExpr.generic("list", [TypeExpr.named("int")]).to_source()   # "list[int]"
#: TypeExpr.optional(TypeExpr.named("str")).to_source()            # "str?"
#: ```
pub class TypeExpr {
    #: Which kind of type expression this is.
    pub let kind: TypeKind
    #: The identifier of a `Name` or `Generic` type, null otherwise.
    pub let name: str?
    #: Generic arguments, union members, tuple members or function parameters.
    pub let args: list[TypeExpr]
    #: An optional's element or a function's return type, null otherwise.
    pub let inner: TypeExpr?
    #: Where the type was written.
    pub let span: Span

    #: Create a type expression of `kind`; prefer the named constructors below.
    pub fn init(
        self,
        kind: TypeKind,
        name: str? = null,
        args: list[TypeExpr] = [],
        inner: TypeExpr? = null,
        span: Span? = null
    ) {
        self.kind = kind
        self.name = name
        self.args = args
        self.inner = inner
        self.span = span ?? Span.none()
    }

    #: A plain named type, such as `int` or `Account`.
    pub static fn named(name: str) -> TypeExpr {
        return TypeExpr(TypeKind.Name, name: name)
    }

    #: An applied generic, such as `dict[str, int]`.
    pub static fn generic(name: str, args: list[TypeExpr]) -> TypeExpr {
        if args.len() == 0 { throw ValueError(f"generic type {name} needs at least one argument") }
        return TypeExpr(TypeKind.Generic, name: name, args: args)
    }

    #: `inner?`, the union of `inner` and null.
    pub static fn optional(inner: TypeExpr) -> TypeExpr {
        return TypeExpr(TypeKind.Optional, inner: inner)
    }

    #: `A | B | ...`, flattened exactly as the parser flattens it.
    pub static fn union_of(members: list[TypeExpr]) -> TypeExpr {
        if members.len() < 2 { throw ValueError("a union type needs at least two members") }
        return TypeExpr(TypeKind.Union, args: members)
    }

    #: `fn(params) -> returns`.
    pub static fn function(params: list[TypeExpr], returns: TypeExpr) -> TypeExpr {
        return TypeExpr(TypeKind.Fn, args: params, inner: returns)
    }

    #: `(A, B)`.
    pub static fn tuple_of(members: list[TypeExpr]) -> TypeExpr {
        return TypeExpr(TypeKind.Tuple, args: members)
    }

    #: The absent annotation, which the checker fills in.
    pub static fn infer() -> TypeExpr {
        return TypeExpr(TypeKind.Infer)
    }

    #: Render this type as source. `TypeKind.Infer` renders as the empty string.
    pub fn to_source(self) -> str {
        match self.kind {
            TypeKind.Name => { return self.name ?? "_" }
            TypeKind.Generic => {
                return (self.name ?? "_") + "[" + ", ".join(self._arg_sources()) + "]"
            }
            TypeKind.Optional => {
                let inner = self.inner
                if inner is null { return "null" }
                return inner.to_source() + "?"
            }
            TypeKind.Union => { return " | ".join(self._arg_sources()) }
            TypeKind.Fn => {
                let head = "fn(" + ", ".join(self._arg_sources()) + ") -> "
                let returns = self.inner
                if returns is null { return head + "void" }
                return head + returns.to_source()
            }
            TypeKind.Tuple => { return "(" + ", ".join(self._arg_sources()) + ")" }
            TypeKind.Infer => { return "" }
            TypeKind.Const => { return self.name ?? "0" }
        }
    }

    fn _arg_sources(self) -> list[str] {
        return [argument.to_source() for argument in self.args]
    }

    fn __eq__(self, other: TypeExpr) -> bool {
        return self.kind == other.kind and
            self.name == other.name and
            self.args == other.args and
            self.inner == other.inner and
            self.span == other.span
    }

    fn __str__(self) -> str {
        return self.to_source()
    }
}

# ------------------------------------------------------------------
# Nodes
# ------------------------------------------------------------------

#: One node of the tree: a kind, a span, and the fields that kind declares.
#:
#: The constructor fills every field the kind declares with its default and then
#: overlays `fields`, so a node is always complete and `to_json` never has to
#: guess. A name the kind does not declare raises `KeyError` — that is how a
#: typo in a tree-building pass is caught at the point of the mistake.
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
    #: The kind's fields, keyed by the C field names `fields_of` lists.
    pub var fields: dict[str, any]

    #: Create a node of `kind`, overlaying `fields` on that kind's defaults.
    pub fn init(self, kind: NodeKind, fields: dict[str, any]? = null, span: Span? = null) {
        self.kind = kind
        self.span = span ?? Span.none()
        self.fields = {}
        for (name, tag) in fields_of(kind) { self.fields[name] = _default_field(name, tag) }
        for (name, value) in (fields ?? {}).items() {
            if name not in self.fields {
                throw KeyError(f"{kind_name(kind)} has no field `{name}`")
            }
            self.fields[name] = value
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
        return name in self.fields
    }

    #: The raw value of `name`, raising `KeyError` when the kind has no such field.
    pub fn get(self, name: str) -> any {
        if name not in self.fields {
            throw KeyError(f"{kind_name(self.kind)} has no field `{name}`")
        }
        return self.fields[name]
    }

    #: Set `name` and return self, so builders chain.
    #:
    #: ```
    #: Node.ident("x").set("name", "y").text("name")   # "y"
    #: ```
    pub fn set(self, name: str, value: any) -> Node {
        if name not in self.fields {
            throw KeyError(f"{kind_name(self.kind)} has no field `{name}`")
        }
        self.fields[name] = value
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

# ------------------------------------------------------------------
# Sub-structure builders
# ------------------------------------------------------------------
fn _record(tag: str, values: dict[str, any]) -> dict[str, any] {
    let record: dict[str, any] = {}
    for (name, field_tag) in record_fields(tag) {
        record[name] = values.get(name, _default_field(name, field_tag))
    }
    return record
}

fn _flag(record: dict[str, any], name: str) -> bool {
    let value: bool = record.get(name, false)
    return value
}

fn _text_at(record: dict[str, any], name: str) -> str? {
    let value: str? = record.get(name, null)
    return value
}

fn _node_at(record: dict[str, any], name: str) -> Node? {
    let value: Node? = record.get(name, null)
    return value
}

fn _type_at(record: dict[str, any], name: str) -> TypeExpr? {
    let value: TypeExpr? = record.get(name, null)
    return value
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

# ------------------------------------------------------------------
# Traversal
# ------------------------------------------------------------------

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

    for (name, tag) in fields_of(node.kind) {
        _collect_children(out, node.fields.get(name, null), tag)
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

#: Walk the tree rooted at `node`, mirroring `jaiAstWalk`.
#:
#: `pre` runs on a node before its children and prunes them by returning false;
#: `post` runs afterwards either way, so a visitor can keep balanced enter and
#: leave state across a pruned subtree. A pre-hook that never prunes returns
#: true unconditionally.
#:
#: ```
#: var seen = 0
#: walk(Node.call(Node.ident("f"), [Node.ident("x")]), fn(node: Node) -> bool {
#:     seen += 1
#:     return node.kind != NodeKind.Call
#: }, fn(_node: Node) -> void {})
#: seen   # 1
#: ```
pub fn walk(node: Node, pre: fn(Node) -> bool, post: fn(Node) -> void) -> void {
    if pre(node) {
        for child in children_of(node) { walk(child, pre, post) }
    }
    post(node)
}

#: Every node of the tree in pre-order, root first.
#:
#: ```
#: preorder(Node.call(Node.ident("f"), [Node.ident("x")])).len()   # 3
#: ```
pub fn preorder(node: Node) -> list[Node] {
    let out: list[Node] = []
    walk(node, fn(current: Node) -> bool {
        out.push(current)
        return true
    }, fn(_current: Node) -> void {})
    return out
}

#: Every node of the tree for which `predicate` holds, in pre-order.
#:
#: ```
#: let tree = Node.call(Node.ident("f"), [Node.ident("x")])
#: find_all(tree, fn(node: Node) -> bool { return node.kind == NodeKind.Ident }).len()   # 2
#: ```
pub fn find_all(node: Node, predicate: fn(Node) -> bool) -> list[Node] {
    return [found for found in preorder(node) if predicate(found)]
}

# ------------------------------------------------------------------
# Predicates
# ------------------------------------------------------------------

#: Whether `node` may appear on the left of `=`.
#:
#: ```
#: is_assignable(Node.member(Node.ident("a"), "b"))   # true
#: is_assignable(Node.int_lit(1))                     # false
#: ```
pub fn is_assignable(node: Node) -> bool {
    match node.kind {
        NodeKind.Ident | NodeKind.Member | NodeKind.Index | NodeKind.Slice | NodeKind.PatBind | NodeKind.PatWildcard => {
            return true
        }
        NodeKind.OptMember => { return false }
        NodeKind.TupleLit | NodeKind.ListLit => { return _all_assignable(node.child_list("items")) }
        NodeKind.PatTuple | NodeKind.PatList => { return _all_assignable(node.child_list("elems")) }
        _ => { return false }
    }
}

fn _all_assignable(elements: list[Node]) -> bool {
    if elements.len() == 0 { return false }
    for element in elements {
        if not is_assignable(element) { return false }
    }
    return true
}

#: Whether `node` always transfers control instead of falling through.
#:
#: Conservative in the direction that keeps the missing-return check honest: a
#: loop that any `break` can leave does not diverge, whether or not that break
#: is reachable.
#:
#: ```
#: always_diverges(Node.return_stmt())                     # true
#: always_diverges(Node.expr_stmt(Node.int_lit(1)))        # false
#: ```
pub fn always_diverges(node: Node) -> bool {
    match node.kind {
        NodeKind.Return | NodeKind.Throw | NodeKind.ThrowExpr | NodeKind.Break | NodeKind.Continue => {
            return true
        }
        NodeKind.Block | NodeKind.Program => {
            for stmt in node.child_list("stmts") {
                if always_diverges(stmt) { return true }
            }
            return false
        }
        NodeKind.ExprStmt => { return always_diverges(node.require("expr")) }
        NodeKind.If | NodeKind.IfExpr => {
            let alternative = node.child("elseBranch")
            if alternative is null { return false }
            return always_diverges(node.require("thenBranch")) and always_diverges(alternative)
        }
        NodeKind.Loop => { return not _loop_has_break(node) }
        NodeKind.While => {
            let cond = node.child("cond")
            if cond is null or cond.kind != NodeKind.BoolLit { return false }
            if not cond.flag("boolLit") { return false }
            return not _loop_has_break(node)
        }
        NodeKind.For => { return false }
        NodeKind.Match | NodeKind.MatchExpr => {
            let arms = node.records("arms")
            if arms.len() == 0 { return false }
            for current in arms {
                if not always_diverges(_node_at(current, "body") ?? Node.null_lit()) {
                    return false
                }
            }
            return true
        }
        NodeKind.Try => {
            let ensure = node.child("finallyBlock")
            if ensure is not null and always_diverges(ensure) { return true }
            if not always_diverges(node.require("body")) { return false }
            for handler in node.records("catches") {
                if not always_diverges(_node_at(handler, "body") ?? Node.null_lit()) {
                    return false
                }
            }
            return true
        }
        _ => { return false }
    }
}

fn _is_loop_kind(kind: NodeKind) -> bool {
    return kind == NodeKind.While or kind == NodeKind.Loop or kind == NodeKind.For
}

fn _is_function_kind(kind: NodeKind) -> bool {
    return kind == NodeKind.FnDecl or kind == NodeKind.Lambda or kind == NodeKind.AnonFn
}

fn _loop_has_break(node: Node) -> bool {
    let label = node.opt_text("label")
    let body = node.child("body")
    if body is null { return false }
    var depth = 0
    var found = false

    fn enter(current: Node) -> bool {
        if _is_function_kind(current.kind) or
            current.kind == NodeKind.ClassDecl or
            current.kind == NodeKind.TraitDecl or
            current.kind == NodeKind.EnumDecl {
            return false
        }
        if _is_loop_kind(current.kind) {
            depth += 1
            return not found
        }
        if current.kind == NodeKind.Break {
            let target = current.opt_text("label")
            if target is null {
                if depth == 0 { found = true }
            } elif label is not null and target == label {
                found = true
            }
        }
        return not found
    }

    fn leave(current: Node) -> void {
        if _is_loop_kind(current.kind) { depth -= 1 }
    }

    walk(body, enter, leave)
    return found
}
