#: The Jaithon abstract syntax tree, as Jaithon data.
#:
#: This module mirrors `src/lang/ast.h` case for case and field for field: one
#: `NodeKind` per C `AstKind`, one `OpKind` per C `OpKind`, and field names
#: spelled exactly as `jaiAstToJson` writes them. Three properties hold, and the
#: formatter, the documentation generator and the self-hosted front end all rest
#: on them:
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
#: from_json(to_json(Node.ident("x"))).text("name")   # "x"
#: to_source(Node.binary(OpKind.Add, Node.ident("a"), Node.int_lit(1)))   # "a + 1"
#: ```
import std.json
from std.math import INF, NAN, is_inf, is_nan
from std.str import StringBuilder, escape, quote, repeat

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
pub enum TypeKind { Name, Generic, Optional, Union, Fn, Tuple, Infer }

# ------------------------------------------------------------------
# Name tables
# ------------------------------------------------------------------

# Parallel to `_KINDS`; both are in C declaration order and the asserts below
# are this module's version of the C `_Static_assert` on the same tables.
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

# Group boundaries in `_KINDS`, matching the C category table.
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

# `is` through `not in` are the comparison operators, as in `jaiOpKindIsComparison`.
const _FIRST_COMPARISON = 16
const _LAST_COMPARISON = 25

assert _OP_NAMES.len() == _OPS.len(), "the operator tables have different lengths"
assert _OP_SOURCE.len() == _OPS.len(), "the operator source table has the wrong length"

let _VISIBILITY_NAMES: dict[Visibility, str] = {
    Visibility.Private: "AST_VIS_PRIVATE",
    Visibility.Protected: "AST_VIS_PROTECTED",
    Visibility.Public: "AST_VIS_PUBLIC",
}

let _DECL_KIND_NAMES: dict[DeclKind, str] = {
    DeclKind.Let: "VD_LET",
    DeclKind.Var: "VD_VAR",
    DeclKind.Const: "VD_CONST",
}

let _COMP_KIND_NAMES: dict[CompKind, str] = {
    CompKind.List: "COMP_LIST",
    CompKind.Dict: "COMP_DICT",
    CompKind.Set: "COMP_SET",
    CompKind.Generator: "COMP_GENERATOR",
}

let _TYPE_KIND_NAMES: dict[TypeKind, str] = {
    TypeKind.Name: "TYPE_NAME",
    TypeKind.Generic: "TYPE_GENERIC",
    TypeKind.Optional: "TYPE_OPTIONAL",
    TypeKind.Union: "TYPE_UNION",
    TypeKind.Fn: "TYPE_FN",
    TypeKind.Tuple: "TYPE_TUPLE",
    TypeKind.Infer: "TYPE_INFER",
}

let _VISIBILITY_BY_NAME: dict[str, Visibility] = {name: value for (value, name) in _VISIBILITY_NAMES.items()}
let _DECL_KIND_BY_NAME: dict[str, DeclKind] = {name: value for (value, name) in _DECL_KIND_NAMES.items()}
let _COMP_KIND_BY_NAME: dict[str, CompKind] = {name: value for (value, name) in _COMP_KIND_NAMES.items()}
let _TYPE_KIND_BY_NAME: dict[str, TypeKind] = {name: value for (value, name) in _TYPE_KIND_NAMES.items()}

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

# List tags and the tag of their elements. Everything else is a scalar.
let _ELEMENT_TAG: dict[str, str] = {
    "nodes": "node",
    "strs": "str",
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

# The C sub-structures, keyed by the element tag above, in emission order.
let _RECORDS: dict[str, list[tuple[str, str]]] = {
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
    # `widen` marks §2.2's int-to-float conversion, as it does on the C's
    # `AST_CAST` (src/lang/ast.h:277). It is the one field here the C does not
    # serialise — neither its text dump nor its JSON emits it — so a node read
    # back from the C carries the default, which is correct: only a checker
    # inserts a cast, and a C-produced tree has already had its casts turned
    # into bytecode. The emitter needs it because it is the whole difference
    # between `OP_TO_FLOAT` and `OP_TYPE_GUARD`, and without it the self-hosted
    # front end cannot emit either correctly.
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

# The value a field holds when the encoder left it out or a builder omitted it.
fn _default_field(name: str, tag: str) -> any {
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

# Bytes in the UTF-8 encoding of one scalar value.
fn _utf8_length(character: str) -> int {
    let code = ord(character)
    if code < 0x80 { return 1 }
    if code < 0x800 { return 2 }
    if code < 0x10000 { return 3 }
    return 4
}

# Character index of a byte offset, so that byte spans can slice a `str`.
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
    #: Which of the seven shapes this type has.
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

# Typed reads of a sub-structure's fields, the counterpart of the builders below.
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
# S-expression printing
# ------------------------------------------------------------------
#
# `jaiAstPrint` in src/lang/ast.c, which is what `jaithon ast` and the prompt's
# `:ast` print. The shape is `(kind attr=value` followed by one line per child
# slot, and an absent optional slot prints `nil` so that positional slots stay
# unambiguous in a golden.
#
# `scripts/ast_parity.sh` diffs this against the C printer over every source in
# the tree. That script exists only while both printers do.
fn _emit(out: StringBuilder, text: str) -> void {
    out.push(text)
}

fn _indent(out: StringBuilder, depth: int) -> void {
    var at = 0
    while at < depth {
        out.push("  ")
        at += 1
    }
}

#: `NodeKind.IntLit` -> `"int_lit"`: the C tag without its `AST_` prefix.
fn _kind_tag(kind: NodeKind) -> str {
    var name = kind_name(kind)
    if name.starts_with("AST_") { name = name[4:] }
    return name.lower()
}

fn _quoted(text: str) -> str {
    let out = StringBuilder()
    out.push("\"")
    for character in text {
        let code = ord(character)
        if character == "\"" {
            out.push("\\\"")
        } elif character == "\\" {
            out.push("\\\\")
        } elif character == "\n" {
            out.push("\\n")
        } elif character == "\r" {
            out.push("\\r")
        } elif character == "\t" {
            out.push("\\t")
        } elif code < 0x20 {
            out.push(f"\\x{_hex_byte(code)}")
        } else {
            out.push(character)
        }
    }
    out.push("\"")
    return out.build()
}

fn _hex_byte(value: int) -> str {
    let digits = "0123456789abcdef"
    return digits[value // 16] + digits[value % 16]
}

fn _float_text(value: float) -> str {
    if is_nan(value) { return "nan" }
    if is_inf(value) {
        if value > 0.0 { return "inf" }
        return "-inf"
    }
    return f"{value:.17g}"
}

fn _name_attr(out: StringBuilder, key: str, name: str?) -> void {
    if name is null { return }
    out.push(f" {key}={name}")
}

#: The written form of a type annotation, or `_` when there is none.
fn _type_text(written: TypeExpr?) -> str {
    if written is null { return "_" }
    match written.kind {
        TypeKind.Name => { return written.name ?? "_" }
        TypeKind.Generic => {
            let head = written.name ?? "_"
            return f"{head}[{_type_list(written.args)}]"
        }
        TypeKind.Optional => { return _type_text(written.inner) + "?" }
        TypeKind.Union => {
            var parts: list[str] = []
            for argument in written.args { parts.push(_type_text(argument)) }
            return " | ".join(parts)
        }
        TypeKind.Fn => { return f"fn({_type_list(written.args)}) -> {_type_text(written.inner)}" }
        TypeKind.Tuple => { return f"({_type_list(written.args)})" }
        TypeKind.Infer => { return "_" }
    }
}

fn _type_list(types: list[TypeExpr]) -> str {
    var parts: list[str] = []
    for argument in types { parts.push(_type_text(argument)) }
    return ", ".join(parts)
}

fn _type_attr(out: StringBuilder, key: str, written: TypeExpr?) -> void {
    if written is null { return }
    out.push(f" {key}={_type_text(written)}")
}

fn _child(out: StringBuilder, child: Node?, depth: int) -> void {
    out.push("\n")
    if child is null {
        _indent(out, depth)
        out.push("nil")
    } else {
        _print_node(out, child, depth)
    }
}

fn _children(out: StringBuilder, nodes: list[Node], depth: int) -> void {
    for child in nodes { _child(out, child, depth) }
}

fn _params(out: StringBuilder, params: list[dict[str, any]], depth: int) -> void {
    for param in params {
        out.push("\n")
        _indent(out, depth)
        out.push("(param")
        _name_attr(out, "name", _record_str(param, "name"))
        _type_attr(out, "type", _record_type(param, "type"))
        if _record_flag(param, "isVariadic") { out.push(" variadic") }
        if _record_flag(param, "isKwRest") { out.push(" kwrest") }
        let default_value = _record_node(param, "defaultValue")
        if default_value is not null { _child(out, default_value, depth + 1) }
        out.push(")")
    }
}

fn _generics(out: StringBuilder, generics: list[dict[str, any]], depth: int) -> void {
    for generic in generics {
        out.push("\n")
        _indent(out, depth)
        out.push("(generic")
        _name_attr(out, "name", _record_str(generic, "name"))
        _type_attr(out, "bound", _record_type(generic, "bound"))
        out.push(")")
    }
}

fn _record_str(record: dict[str, any], key: str) -> str? {
    let value = record.get(key, null)
    if value is str { return value }
    return null
}

fn _record_node(record: dict[str, any], key: str) -> Node? {
    let value = record.get(key, null)
    if value is Node { return value }
    return null
}

fn _record_nodes(record: dict[str, any], key: str) -> list[Node] {
    let value = record.get(key, null)
    if value is list { return value }
    return []
}

fn _record_type(record: dict[str, any], key: str) -> TypeExpr? {
    let value = record.get(key, null)
    if value is TypeExpr { return value }
    return null
}

fn _record_flag(record: dict[str, any], key: str) -> bool {
    let value = record.get(key, null)
    if value is bool { return value }
    return false
}

fn _record_int(record: dict[str, any], key: str) -> int {
    let value = record.get(key, null)
    if value is int { return value }
    return 0
}

# The per-kind body: attributes first, in the order `fields_of` lists them,
# then one line per child slot. A transliteration of the switch in
# `printNode` (src/lang/ast.c), checked against it by scripts/ast_parity.sh.
fn _print_body(out: StringBuilder, node: Node, child: int) -> void {
    match node.kind {
        NodeKind.IntLit => { _emit(out, " " + str(node.number("intLit"))) }

        NodeKind.FloatLit => { _emit(out, " " + _float_text(node.get("floatLit"))) }

        NodeKind.StrLit => { _emit(out, " " + _quoted(node.opt_text("chars") ?? "")) }

        NodeKind.BoolLit => { _emit(out, if node.flag("boolLit") { " true" } else { " false" }) }

        # The kind tag is the whole node.
        NodeKind.NullLit | NodeKind.Self | NodeKind.Super => { }

        NodeKind.FString => { _children(out, node.child_list("parts"), child) }

        NodeKind.Ident => { _name_attr(out, "name", node.opt_text("name")) }

        NodeKind.ListLit | NodeKind.SetLit | NodeKind.TupleLit => {
            _children(out, node.child_list("items"), child)
        }

        NodeKind.DictLit => {
            for (key, value) in zip(node.child_list("keys"), node.child_list("values")) {
                _child(out, key, child)
                _child(out, value, child)
            }
        }

        NodeKind.Unary => {
            _emit(out, f" op={op_text(node.op())}")
            _child(out, node.child("operand"), child)
        }

        NodeKind.Binary | NodeKind.Logical => {
            _emit(out, f" op={op_text(node.op())}")
            _child(out, node.child("left"), child)
            _child(out, node.child("right"), child)
        }

        NodeKind.CompareChain => {
            _emit(out, " ops=")
            for (index, op) in node.get("ops").enumerate() {
                if index > 0 { _emit(out, ",") }
                _emit(out, op_text(op))
            }
            _children(out, node.child_list("operands"), child)
        }

        NodeKind.Ternary => {
            _child(out, node.child("cond"), child)
            _child(out, node.child("thenExpr"), child)
            _child(out, node.child("elseExpr"), child)
        }

        NodeKind.Coalesce => {
            _child(out, node.child("left"), child)
            _child(out, node.child("right"), child)
        }

        NodeKind.Call => {
            _child(out, node.child("callee"), child)
            for argument in node.records("args") {
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(arg")
                _name_attr(out, "name", _record_str(argument, "name"))
                if _record_flag(argument, "isSpread") { _emit(out, " spread") }
                _child(out, _record_node(argument, "value"), child + 1)
                _emit(out, ")")
            }
        }

        NodeKind.Index => {
            _child(out, node.child("object"), child)
            _child(out, node.child("index"), child)
        }

        NodeKind.Slice => {
            _child(out, node.child("object"), child)
            _child(out, node.child("start"), child)
            _child(out, node.child("stop"), child)
            _child(out, node.child("step"), child)
        }

        NodeKind.Member | NodeKind.OptMember => {
            _name_attr(out, "name", node.opt_text("name"))
            _child(out, node.child("object"), child)
        }

        NodeKind.Lambda | NodeKind.AnonFn => {
            _name_attr(out, "name", node.opt_text("name"))
            _type_attr(out, "returns", node.type_field("returnType"))
            let visibility = node.visibility()
            if visibility != Visibility.Private {
                _emit(out, f" vis={_VISIBILITY_NAMES[visibility]}")
            }
            if node.flag("isStatic") { _emit(out, " static") }
            if node.flag("isGenerator") { _emit(out, " generator") }
            if node.flag("isAsync") { _emit(out, " async") }
            _generics(out, node.records("generics"), child)
            _params(out, node.records("params"), child)
            _child(out, node.child("body"), child)
        }

        NodeKind.Comprehension => {
            let comp_kind = node.get("compKind")
            _emit(out, f" kind={_COMP_KIND_NAMES[comp_kind]}")
            _child(out, node.child("element"), child)
            _child(out, node.child("keyExpr"), child)  # dict comprehensions only
            for clause in node.records("clauses") {
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(clause")
                _child(out, _record_node(clause, "pattern"), child + 1)
                _child(out, _record_node(clause, "iterable"), child + 1)
                _children(out, _record_nodes(clause, "conditions"), child + 1)
                _emit(out, ")")
            }
        }

        NodeKind.Range => {
            if node.flag("inclusive") { _emit(out, " inclusive") }
            _child(out, node.child("start"), child)
            _child(out, node.child("stop"), child)
        }

        NodeKind.IfExpr => {
            _child(out, node.child("cond"), child)
            _child(out, node.child("thenBranch"), child)
            _child(out, node.child("elseBranch"), child)
        }

        NodeKind.Match | NodeKind.MatchExpr => {
            _child(out, node.child("subject"), child)
            for arm in node.records("arms") {
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(arm")
                _child(out, _record_node(arm, "pattern"), child + 1)
                _child(out, _record_node(arm, "guard"), child + 1)
                _child(out, _record_node(arm, "body"), child + 1)
                _emit(out, ")")
            }
        }

        NodeKind.Cast => {
            _type_attr(out, "target", node.type_field("target"))
            _child(out, node.child("operand"), child)
        }

        NodeKind.Yield | NodeKind.Await | NodeKind.ThrowExpr => {
            _child(out, node.child("operand"), child)
        }

        NodeKind.If => {
            _child(out, node.child("cond"), child)
            _child(out, node.child("thenBranch"), child)
            _child(out, node.child("elseBranch"), child)
        }

        NodeKind.Block | NodeKind.Program => { _children(out, node.child_list("stmts"), child) }

        NodeKind.ExprStmt => { _child(out, node.child("expr"), child) }

        NodeKind.VarDecl => {
            let decl_kind: DeclKind = node.get("varDeclKind")
            _emit(out, " kind=" + _DECL_KIND_NAMES[decl_kind])
            _type_attr(out, "type", node.type_field("declaredType"))
            if node.visibility() != Visibility.Private {
                _emit(out, " vis=" + _VISIBILITY_NAMES[node.visibility()])
            }
            _child(out, node.child("pattern"), child)
            _child(out, node.child("init"), child)
        }

        NodeKind.Assign => {
            if node.flag("isCompound") { _emit(out, " op=" + op_text(node.op()) + "=") }
            _child(out, node.child("target"), child)
            _child(out, node.child("value"), child)
        }

        NodeKind.While | NodeKind.Loop => {
            _name_attr(out, "label", node.opt_text("label"))
            _child(out, node.child("cond"), child)
            _child(out, node.child("body"), child)
        }

        NodeKind.For => {
            _name_attr(out, "label", node.opt_text("label"))
            _child(out, node.child("pattern"), child)
            _child(out, node.child("iterable"), child)
            _child(out, node.child("body"), child)
        }

        NodeKind.Break | NodeKind.Continue => { _name_attr(out, "label", node.opt_text("label")) }

        NodeKind.Return | NodeKind.Throw => { _child(out, node.child("value"), child) }

        NodeKind.Try => {
            _child(out, node.child("body"), child)
            for handler in node.records("catches") {
                let name: str? = handler.get("name", null)
                let types: list[TypeExpr] = handler.get("types", [])
                let body: Node? = handler.get("body", null)
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(catch")
                _name_attr(out, "name", name)
                for (index, caught) in types.enumerate() {
                    _emit(out, index == 0 ? " types=" : " | ")
                    _emit(out, _type_text(caught))
                }
                _child(out, body, child + 1)
                _emit(out, ")")
            }
            _child(out, node.child("finallyBlock"), child)
        }

        NodeKind.Defer => { _child(out, node.child("body"), child) }

        NodeKind.Assert => {
            _child(out, node.child("cond"), child)
            _child(out, node.child("message"), child)
        }

        NodeKind.FnDecl => {
            _name_attr(out, "name", node.opt_text("name"))
            _type_attr(out, "returns", node.type_field("returnType"))
            let vis = node.visibility()
            if vis != Visibility.Private { _emit(out, f" vis={_VISIBILITY_NAMES[vis]}") }
            if node.flag("isStatic") { _emit(out, " static") }
            if node.flag("isGenerator") { _emit(out, " generator") }
            if node.flag("isAsync") { _emit(out, " async") }
            _generics(out, node.records("generics"), child)
            _params(out, node.records("params"), child)
            _child(out, node.child("body"), child)
        }

        NodeKind.ClassDecl => {
            _name_attr(out, "name", node.opt_text("name"))
            _type_attr(out, "extends", node.type_field("superclass"))
            let vis = node.visibility()
            if vis != Visibility.Private { _emit(out, f" vis={_VISIBILITY_NAMES[vis]}") }
            if node.flag("isAbstract") { _emit(out, " abstract") }
            for (index, trait_type) in node.type_list("traits").enumerate() {
                _emit(out, if index == 0 { " traits=" } else { "," })
                _emit(out, _type_text(trait_type))
            }
            _generics(out, node.records("generics"), child)
            for current in node.records("fields") {
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(field")
                _name_attr(out, "name", _record_str(current, "name"))
                _type_attr(out, "type", _record_type(current, "type"))
                # A field's visibility prints even when private; a declaration's does not.
                let field_vis: Visibility = current.get("visibility", Visibility.Private)
                _emit(out, f" vis={_VISIBILITY_NAMES[field_vis]}")
                if _record_flag(current, "isStatic") { _emit(out, " static") }
                _emit(out, if _record_flag(current, "isLet") { " let" } else { " var" })
                let default_value = _record_node(current, "defaultValue")
                if default_value is not null { _child(out, default_value, child + 1) }
                _emit(out, ")")
            }
            _children(out, node.child_list("methods"), child)
            _children(out, node.child_list("getters"), child)
            _children(out, node.child_list("setters"), child)
        }

        NodeKind.TraitDecl => {
            _name_attr(out, "name", node.opt_text("name"))
            let vis = node.visibility()
            if vis != Visibility.Private { _emit(out, f" vis={_VISIBILITY_NAMES[vis]}") }
            for (index, super_type) in node.type_list("supers").enumerate() {
                _emit(out, if index == 0 { " supers=" } else { "," })
                _emit(out, _type_text(super_type))
            }
            _generics(out, node.records("generics"), child)
            _children(out, node.child_list("methods"), child)
        }

        NodeKind.EnumDecl => {
            _name_attr(out, "name", node.opt_text("name"))
            let vis = node.visibility()
            if vis != Visibility.Private { _emit(out, f" vis={_VISIBILITY_NAMES[vis]}") }
            _generics(out, node.records("generics"), child)
            for current in node.records("variants") {
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(variant")
                _name_attr(out, "name", _record_str(current, "name"))
                _params(out, current.get("params", []), child + 1)
                _emit(out, ")")
            }
            _children(out, node.child_list("methods"), child)
        }

        NodeKind.TypeDecl => {
            _name_attr(out, "name", node.opt_text("name"))
            _type_attr(out, "aliased", node.type_field("aliased"))
            let vis = node.visibility()
            if vis != Visibility.Private { _emit(out, f" vis={_VISIBILITY_NAMES[vis]}") }
            _generics(out, node.records("generics"), child)
        }

        NodeKind.Import => {
            _name_attr(out, "path", node.opt_text("path"))
            _name_attr(out, "alias", node.opt_text("alias"))
        }

        NodeKind.FromImport => {
            _name_attr(out, "path", node.opt_text("path"))
            if node.flag("isWildcard") { _emit(out, " wildcard") }
            for current in node.records("items") {
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, "(item")
                _name_attr(out, "name", _record_str(current, "name"))
                _name_attr(out, "alias", _record_str(current, "alias"))
                _emit(out, ")")
            }
        }

        NodeKind.Export => {
            let names: list[str?] = node.get("names")
            for (index, name) in names.enumerate() {
                _emit(out, if index == 0 { " names=" } else { "," })
                _emit(out, name ?? "?")
            }
        }

        NodeKind.ModuleDecl => { _name_attr(out, "name", node.opt_text("name")) }

        NodeKind.PatWildcard => { }

        NodeKind.PatBind => {
            _name_attr(out, "name", node.opt_text("name"))
            _type_attr(out, "type", node.type_field("type"))
        }

        NodeKind.PatLiteral => { _child(out, node.child("value"), child) }

        NodeKind.PatRange => {
            if node.flag("inclusive") { _emit(out, " inclusive") }
            _child(out, node.child("lo"), child)
            _child(out, node.child("hi"), child)
        }

        NodeKind.PatTuple | NodeKind.PatList | NodeKind.PatOr => {
            let rest = node.number("restIndex")
            if rest >= 0 { _emit(out, f" rest={rest}") }
            _children(out, node.child_list("elems"), child)
        }

        NodeKind.PatClass | NodeKind.PatEnum => {
            _name_attr(out, "typeName", node.opt_text("typeName"))
            _name_attr(out, "variantName", node.opt_text("variantName"))
            let names = node.get("fieldNames")
            # A null name is a positional sub-pattern; a named one gets its own line.
            for (index, sub) in node.child_list("subPatterns").enumerate() {
                let name = _at_or_null(names, index)
                if name is null {
                    _child(out, sub, child)
                    continue
                }
                _emit(out, "\n")
                _indent(out, child)
                _emit(out, f"(field name={name}")
                _child(out, sub, child + 1)
                _emit(out, ")")
            }
        }
    }
}

fn _print_node(out: StringBuilder, node: Node, depth: int) -> void {
    _indent(out, depth)
    out.push("(")
    out.push(_kind_tag(node.kind))

    let child = depth + 1
    _print_body(out, node, child)
    out.push(")")
}

#: Print `node` the way `jaithon ast` prints it, with no trailing newline.
#:
#: ```
#: to_sexp(Node.int_lit(1))       # "(int_lit 1)"
#: ```
pub fn to_sexp(node: Node) -> str {
    let out = StringBuilder()
    _print_node(out, node, 0)
    return out.build()
}

# ------------------------------------------------------------------
# JSON
# ------------------------------------------------------------------

#: Encode `node` as the JSON `jaiAstToJson` writes.
#:
#: Keys appear in the C's order, so the two front ends' output can be diffed
#: directly. Two differences are deliberate: a float may be spelled differently
#: though never valued differently, and the trailing newline the C writes after
#: the document is the caller's to add.
#:
#: ```
#: let encoded = to_json(Node.ident("x"))
#: from_json(encoded).text("name")   # "x"
#: ```
pub fn to_json(node: Node) -> str {
    return json.stringify(to_value(node))
}

#: Decode a node from the JSON `jaiAstToJson` wrote.
#:
#: Raises `ParseError` for malformed JSON and `ValueError` for JSON that is
#: well formed but is not an AST.
pub fn from_json(text: str) -> Node {
    return from_value(json.parse(text))
}

#: Convert `node` to plain dicts, lists and scalars.
#:
#: This is the same shape `to_json` serialises, for tools that would rather walk
#: the data than re-parse it. A null node converts to null.
pub fn to_value(node: Node?) -> any {
    if node is null { return null }
    let out: dict[str, any] = {"kind": kind_name(node.kind), "span": _span_value(node.span)}
    for (name, tag) in fields_of(node.kind) {
        out[name] = _value_of(node.fields.get(name, _default_field(name, tag)), tag)
    }
    return out
}

#: Rebuild a node from the plain data `to_value` produces.
pub fn from_value(value: any) -> Node {
    let node = _node_from(value)
    if node is null { throw ValueError("expected an AST node object, got null") }
    return node
}

fn _value_of(value: any, tag: str) -> any {
    if tag in _ELEMENT_TAG {
        let element = _ELEMENT_TAG[tag]
        return [_value_of(item, element) for item in value]
    }
    if tag == "node" { return to_value(value) }
    if tag == "type" { return _type_value(value) }
    if tag == "span" { return _span_value(value) }
    if tag == "op" { return op_name(value) }
    if tag == "vis" { return _VISIBILITY_NAMES[value] }
    if tag == "vdk" { return _DECL_KIND_NAMES[value] }
    if tag == "compk" { return _COMP_KIND_NAMES[value] }
    if tag == "float" { return _float_value(value) }
    if tag in _RECORDS { return _record_value(value, _RECORDS[tag]) }
    return value
}

fn _record_value(record: dict[str, any], schema: list[tuple[str, str]]) -> dict[str, any] {
    let out: dict[str, any] = {}
    for (name, tag) in schema {
        out[name] = _value_of(record.get(name, _default_field(name, tag)), tag)
    }
    return out
}

fn _span_value(span: Span) -> dict[str, any] {
    return {"start": span.start, "end": span.end, "file": span.file}
}

fn _type_value(type_expr: TypeExpr?) -> any {
    if type_expr is null { return null }
    return {
        "kind": _TYPE_KIND_NAMES[type_expr.kind],
        "span": _span_value(type_expr.span),
        "name": type_expr.name,
        "args": [_type_value(argument) for argument in type_expr.args],
        "inner": _type_value(type_expr.inner),
    }
}

# JSON has no infinity or NaN, so the C writes them as strings and so do we.
fn _float_value(value: float) -> any {
    if is_nan(value) { return "NaN" }
    if is_inf(value) {
        if value > 0.0 { return "Infinity" }
        return "-Infinity"
    }
    return value
}

fn _node_from(value: any) -> Node? {
    if value is null { return null }
    if not isinstance(value, dict) {
        throw ValueError(f"expected an AST node object, got {repr(value)}")
    }
    let name = value.get("kind", null)
    if name is null { throw ValueError("an AST node object needs a `kind`") }
    let kind = kind_from_name(name)
    let node = Node(kind, span: _span_from(value.get("span", null)))
    for (field, tag) in fields_of(kind) {
        node.fields[field] = _from_value(value.get(field, null), field, tag)
    }
    return node
}

fn _from_value(value: any, name: str, tag: str) -> any {
    if value is null { return _default_field(name, tag) }
    if tag in _ELEMENT_TAG {
        let element = _ELEMENT_TAG[tag]
        return [_from_value(item, name, element) for item in value]
    }
    if tag == "node" { return _node_from(value) }
    if tag == "type" { return _type_from(value) }
    if tag == "span" { return _span_from(value) }
    if tag == "op" { return op_from_name(value) }
    if tag == "vis" { return _enum_from(_VISIBILITY_BY_NAME, value, "visibility") }
    if tag == "vdk" { return _enum_from(_DECL_KIND_BY_NAME, value, "declaration kind") }
    if tag == "compk" { return _enum_from(_COMP_KIND_BY_NAME, value, "comprehension kind") }
    if tag == "float" { return _float_from(value) }
    if tag in _RECORDS { return _record_from(value, _RECORDS[tag]) }
    return value
}

fn _record_from(value: any, schema: list[tuple[str, str]]) -> dict[str, any] {
    if not isinstance(value, dict) {
        throw ValueError(f"expected an AST sub-structure object, got {repr(value)}")
    }
    let record: dict[str, any] = {}
    for (name, tag) in schema {
        record[name] = _from_value(value.get(name, null), name, tag)
    }
    return record
}

fn _span_from(value: any) -> Span {
    if value is null { return Span.none() }
    return Span(value.get("file", -1), value.get("start", 0), value.get("end", 0))
}

fn _type_from(value: any) -> TypeExpr? {
    if value is null { return null }
    let raw = value.get("kind", null)
    let kind = _TYPE_KIND_BY_NAME.get(raw, null)
    if kind is null { throw ValueError(f"unknown AST type kind: {repr(raw)}") }
    let args: list[TypeExpr] = []
    for raw_arg in value.get("args", []) {
        let argument = _type_from(raw_arg)
        if argument is not null { args.push(argument) }
    }
    return TypeExpr(
        kind,
        name: value.get("name", null),
        args: args,
        inner: _type_from(value.get("inner", null)),
        span: _span_from(value.get("span", null))
    )
}

fn _enum_from(table: dict[str, any], name: any, what: str) -> any {
    let value = table.get(name, null)
    if value is null { throw ValueError(f"unknown {what}: {repr(name)}") }
    return value
}

fn _float_from(value: any) -> float {
    if isinstance(value, str) {
        if value == "NaN" { return NAN }
        if value == "Infinity" { return INF }
        if value == "-Infinity" { return -INF }
        throw ValueError(f"expected a float, got the string {quote(value)}")
    }
    return float(value)
}

# ------------------------------------------------------------------
# Traversal
# ------------------------------------------------------------------

#: Every direct child of `node`, in the order the C `jaiAstWalk` visits them.
#:
#: Null children are dropped, exactly as the C walk skips them.
pub fn children_of(node: Node) -> list[Node] {
    let out: list[Node] = []

    # Two orders differ from the field order and both matter to a rewriting
    # pass: a dict literal alternates key and value, and a comprehension's key
    # expression is visited before its element.
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

#: A visitor with one hook per node kind, dispatched by kind.
#:
#: Subclass it, override the hooks you care about, and call `visit`. Every hook
#: defaults to `default_visit`, which visits the children and returns null, so an
#: override that does not call it prunes that subtree. Hooks are `prot`, so a
#: subclass may override them but outside code goes through `visit`.
#:
#: ```
#: class NameCollector extends Visitor {
#:     pub var names: list[str]
#:     pub fn init(self) { self.names = [] }
#:     prot fn visit_ident(self, node: Node) -> any {
#:         self.names.push(node.text("name"))
#:         return null
#:     }
#: }
#: let collector = NameCollector()
#: collector.visit(Node.call(Node.ident("f"), [Node.ident("x")]))
#: collector.names   # ["f", "x"]
#: ```
pub class Visitor {
    #: Dispatch `node` to the hook for its kind and return whatever it returns.
    #:
    #: The match below is exhaustive over `NodeKind` on purpose: adding a kind
    #: to the enum stops this file compiling until the kind has a hook, exactly
    #: as adding one to `AstKind` stops `jaiAstWalk` compiling.
    pub fn visit(self, node: Node) -> any {
        match node.kind {
            NodeKind.IntLit => { return self.visit_int_lit(node) }
            NodeKind.FloatLit => { return self.visit_float_lit(node) }
            NodeKind.StrLit => { return self.visit_str_lit(node) }
            NodeKind.BoolLit => { return self.visit_bool_lit(node) }
            NodeKind.NullLit => { return self.visit_null_lit(node) }
            NodeKind.FString => { return self.visit_fstring(node) }
            NodeKind.Ident => { return self.visit_ident(node) }
            NodeKind.Self => { return self.visit_self(node) }
            NodeKind.Super => { return self.visit_super(node) }
            NodeKind.ListLit => { return self.visit_list_lit(node) }
            NodeKind.DictLit => { return self.visit_dict_lit(node) }
            NodeKind.SetLit => { return self.visit_set_lit(node) }
            NodeKind.TupleLit => { return self.visit_tuple_lit(node) }
            NodeKind.Unary => { return self.visit_unary(node) }
            NodeKind.Binary => { return self.visit_binary(node) }
            NodeKind.Logical => { return self.visit_logical(node) }
            NodeKind.CompareChain => { return self.visit_compare_chain(node) }
            NodeKind.Ternary => { return self.visit_ternary(node) }
            NodeKind.Coalesce => { return self.visit_coalesce(node) }
            NodeKind.Call => { return self.visit_call(node) }
            NodeKind.Index => { return self.visit_index(node) }
            NodeKind.Slice => { return self.visit_slice(node) }
            NodeKind.Member => { return self.visit_member(node) }
            NodeKind.OptMember => { return self.visit_opt_member(node) }
            NodeKind.Lambda => { return self.visit_lambda(node) }
            NodeKind.AnonFn => { return self.visit_anon_fn(node) }
            NodeKind.Comprehension => { return self.visit_comprehension(node) }
            NodeKind.Range => { return self.visit_range(node) }
            NodeKind.IfExpr => { return self.visit_if_expr(node) }
            NodeKind.MatchExpr => { return self.visit_match_expr(node) }
            NodeKind.Cast => { return self.visit_cast(node) }
            NodeKind.Yield => { return self.visit_yield(node) }
            NodeKind.Await => { return self.visit_await(node) }
            NodeKind.ThrowExpr => { return self.visit_throw_expr(node) }
            NodeKind.Block => { return self.visit_block(node) }
            NodeKind.ExprStmt => { return self.visit_expr_stmt(node) }
            NodeKind.VarDecl => { return self.visit_var_decl(node) }
            NodeKind.Assign => { return self.visit_assign(node) }
            NodeKind.If => { return self.visit_if(node) }
            NodeKind.While => { return self.visit_while(node) }
            NodeKind.Loop => { return self.visit_loop(node) }
            NodeKind.For => { return self.visit_for(node) }
            NodeKind.Match => { return self.visit_match(node) }
            NodeKind.Break => { return self.visit_break(node) }
            NodeKind.Continue => { return self.visit_continue(node) }
            NodeKind.Return => { return self.visit_return(node) }
            NodeKind.Throw => { return self.visit_throw(node) }
            NodeKind.Try => { return self.visit_try(node) }
            NodeKind.Defer => { return self.visit_defer(node) }
            NodeKind.Assert => { return self.visit_assert(node) }
            NodeKind.FnDecl => { return self.visit_fn_decl(node) }
            NodeKind.ClassDecl => { return self.visit_class_decl(node) }
            NodeKind.TraitDecl => { return self.visit_trait_decl(node) }
            NodeKind.EnumDecl => { return self.visit_enum_decl(node) }
            NodeKind.TypeDecl => { return self.visit_type_decl(node) }
            NodeKind.Import => { return self.visit_import(node) }
            NodeKind.FromImport => { return self.visit_from_import(node) }
            NodeKind.Export => { return self.visit_export(node) }
            NodeKind.ModuleDecl => { return self.visit_module_decl(node) }
            NodeKind.PatWildcard => { return self.visit_pat_wildcard(node) }
            NodeKind.PatBind => { return self.visit_pat_bind(node) }
            NodeKind.PatLiteral => { return self.visit_pat_literal(node) }
            NodeKind.PatRange => { return self.visit_pat_range(node) }
            NodeKind.PatTuple => { return self.visit_pat_tuple(node) }
            NodeKind.PatList => { return self.visit_pat_list(node) }
            NodeKind.PatClass => { return self.visit_pat_class(node) }
            NodeKind.PatEnum => { return self.visit_pat_enum(node) }
            NodeKind.PatOr => { return self.visit_pat_or(node) }
            NodeKind.Program => { return self.visit_program(node) }
        }
    }

    #: Visit every child of `node`.
    pub fn visit_children(self, node: Node) -> void {
        for child in children_of(node) { self.visit(child) }
    }

    #: Visit the children and return null: what every hook does by default.
    pub fn default_visit(self, node: Node) -> any {
        self.visit_children(node)
        return null
    }

    prot fn visit_int_lit(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_float_lit(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_str_lit(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_bool_lit(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_null_lit(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_fstring(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_ident(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_self(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_super(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_list_lit(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_dict_lit(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_set_lit(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_tuple_lit(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_unary(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_binary(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_logical(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_compare_chain(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_ternary(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_coalesce(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_call(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_index(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_slice(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_member(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_opt_member(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_lambda(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_anon_fn(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_comprehension(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_range(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_if_expr(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_match_expr(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_cast(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_yield(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_await(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_throw_expr(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_block(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_expr_stmt(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_var_decl(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_assign(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_if(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_while(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_loop(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_for(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_match(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_break(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_continue(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_return(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_throw(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_try(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_defer(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_assert(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_fn_decl(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_class_decl(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_trait_decl(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_enum_decl(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_type_decl(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_import(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_from_import(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_export(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_module_decl(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_pat_wildcard(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_pat_bind(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_pat_literal(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_pat_range(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_pat_tuple(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_pat_list(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_pat_class(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_pat_enum(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_pat_or(self, node: Node) -> any { return self.default_visit(node) }

    prot fn visit_program(self, node: Node) -> any { return self.default_visit(node) }
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
        # `x?.y = v` has no meaning: the point of `?.` is that it may reach no
        # object at all.
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
        # Once one statement diverges the rest of the block is unreachable, so
        # the block diverges too.
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
        # `loop` and `while true` run forever unless something breaks out.
        NodeKind.Loop => { return not _loop_has_break(node) }
        NodeKind.While => {
            let cond = node.child("cond")
            if cond is null or cond.kind != NodeKind.BoolLit { return false }
            if not cond.flag("boolLit") { return false }
            return not _loop_has_break(node)
        }
        # A `for` may iterate zero times, so it can always fall through.
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
        # A `finally` that diverges wins outright; otherwise the body and every
        # handler have to diverge.
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

# Whether any `break` inside this loop's body binds to this loop.
fn _loop_has_break(node: Node) -> bool {
    let label = node.opt_text("label")
    let body = node.child("body")
    if body is null { return false }
    var depth = 0
    var found = false

    fn enter(current: Node) -> bool {
        # Control never crosses a function or type boundary, so a `break` down
        # there cannot be ours.
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
                # Unlabelled: it binds to the innermost enclosing loop.
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

# ------------------------------------------------------------------
# Unparsing
# ------------------------------------------------------------------

# Precedence levels, from §4.1 of the language spec. A child is parenthesised
# exactly when its level is below the level its position requires, which is what
# makes `to_source` round-trip without wrapping everything in brackets.
const _P_ATOM = 17
const _P_POSTFIX = 16
const _P_POW = 15
const _P_UNARY = 14
const _P_MUL = 13
const _P_ADD = 12
const _P_SHIFT = 11
const _P_BAND = 10
const _P_BXOR = 9
const _P_BOR = 8
const _P_RANGE = 7
const _P_COMPARE = 6
const _P_NOT = 5
const _P_AND = 4
const _P_OR = 3
const _P_COALESCE = 2
const _P_LOWEST = 1

let _OP_PRECEDENCE: dict[OpKind, int] = {
    OpKind.Pow: _P_POW,
    OpKind.Mul: _P_MUL,
    OpKind.Div: _P_MUL,
    OpKind.FloorDiv: _P_MUL,
    OpKind.Mod: _P_MUL,
    OpKind.MatMul: _P_MUL,
    OpKind.MulWrap: _P_MUL,
    OpKind.Add: _P_ADD,
    OpKind.Sub: _P_ADD,
    OpKind.AddWrap: _P_ADD,
    OpKind.SubWrap: _P_ADD,
    OpKind.Shl: _P_SHIFT,
    OpKind.Shr: _P_SHIFT,
    OpKind.Band: _P_BAND,
    OpKind.Bxor: _P_BXOR,
    OpKind.Bor: _P_BOR,
    OpKind.Eq: _P_COMPARE,
    OpKind.Ne: _P_COMPARE,
    OpKind.Lt: _P_COMPARE,
    OpKind.Le: _P_COMPARE,
    OpKind.Gt: _P_COMPARE,
    OpKind.Ge: _P_COMPARE,
    OpKind.Is: _P_COMPARE,
    OpKind.IsNot: _P_COMPARE,
    OpKind.In: _P_COMPARE,
    OpKind.NotIn: _P_COMPARE,
    OpKind.And: _P_AND,
    OpKind.Or: _P_OR,
    OpKind.Neg: _P_UNARY,
    OpKind.Pos: _P_UNARY,
    OpKind.Bnot: _P_UNARY,
    OpKind.Not: _P_NOT,
}

let _DECL_KEYWORDS: dict[DeclKind, str] = {
    DeclKind.Let: "let",
    DeclKind.Var: "var",
    DeclKind.Const: "const",
}

# Top-level declarations that read better with a blank line around them.
let _BLOCK_DECLARATIONS: set[NodeKind] = {
    NodeKind.FnDecl,
    NodeKind.ClassDecl,
    NodeKind.TraitDecl,
    NodeKind.EnumDecl,
}

#: Render `node` as Jaithon source.
#:
#: Parsing the result yields a tree equal to `node` in every field except its
#: spans, which move because the text moved. That property is what makes the
#: formatter possible: it parses, prints, and trusts that nothing but layout
#: changed. Three details are worth knowing:
#:
#: - `Cast` nodes are inserted by the type checker and have no surface syntax,
#:   so they print as their operand.
#: - `isAbstract`, `isGenerator` and `isAsync` are inferred by the parser rather
#:   than written, so they carry no output of their own.
#: - An empty set literal prints as `set()`, since `{}` is the empty dict.
#:
#: ```
#: let sum = Node.binary(OpKind.Add, Node.int_lit(1), Node.int_lit(2))
#: to_source(Node.binary(OpKind.Mul, sum, Node.int_lit(3)))   # "(1 + 2) * 3"
#: ```
pub fn to_source(node: Node) -> str {
    return _Printer().render(node)
}

fn _precedence(node: Node) -> int {
    return match node.kind {
        NodeKind.Binary | NodeKind.Logical | NodeKind.Unary => _OP_PRECEDENCE[node.op()],
        NodeKind.CompareChain => _P_COMPARE,
        NodeKind.Range => _P_RANGE,
        NodeKind.Coalesce => _P_COALESCE,
        NodeKind.Call | NodeKind.Index | NodeKind.Slice | NodeKind.Member | NodeKind.OptMember => _P_POSTFIX,
        NodeKind.Ternary | NodeKind.Lambda | NodeKind.IfExpr | NodeKind.MatchExpr | NodeKind.ThrowExpr | NodeKind.Yield | NodeKind.Await => _P_LOWEST,
        NodeKind.Cast => _precedence(node.require("operand")),
        _ => _P_ATOM,
    }
}

class _Printer {
    let out: StringBuilder
    var indent: int
    var indent_pending: bool

    fn init(self) {
        self.out = StringBuilder()
        self.indent = 0
        self.indent_pending = false
    }

    #: Render `node` and return the text. The one entry point; everything else
    #: here is internal to the printer.
    pub fn render(self, node: Node) -> str {
        if node.kind == NodeKind.Program {
            self.program(node)
        } elif is_pattern(node.kind) {
            self.pattern(node)
        } elif is_expression(node.kind) {
            self.expr(node, _P_LOWEST)
        } else {
            self.stmt(node)
        }
        return self.out.build()
    }

    fn emit(self, text: str) -> void {
        if self.indent_pending {
            self.out.push(repeat("    ", self.indent))
            self.indent_pending = false
        }
        self.out.push(text)
    }

    # A blank line stays blank: the indent is only written once something else is.
    fn line(self) -> void {
        self.out.push("\n")
        self.indent_pending = true
    }

    fn program(self, node: Node) -> void {
        let stmts = node.child_list("stmts")
        for (index, stmt) in stmts.enumerate() {
            if index > 0 and (_is_block_decl(stmts[index - 1]) or _is_block_decl(stmt)) {
                self.line()
            }
            self.stmt(stmt)
            self.line()
        }
    }

    fn block(self, node: Node) -> void {
        let stmts = node.child_list("stmts")
        if stmts.len() == 0 {
            self.emit("{}")
            return
        }
        self.emit("{")
        self.line()
        self.indent += 1
        for stmt in stmts {
            self.stmt(stmt)
            self.line()
        }
        self.indent -= 1
        self.emit("}")
    }

    # A branch is a block in every tree the parser builds; anything else still
    # has to print inside braces to parse back as one statement.
    fn body(self, node: Node?) -> void {
        if node is null {
            self.emit("{}")
            return
        }
        if node.kind == NodeKind.Block {
            self.block(node)
            return
        }
        self.emit("{ ")
        self.stmt(node)
        self.emit(" }")
    }

    fn stmt(self, node: Node?) -> void {
        if node is null { return }
        match node.kind {
            NodeKind.Block => { self.block(node) }
            NodeKind.ExprStmt => { self.expr(node.child("expr"), _P_LOWEST) }
            NodeKind.VarDecl => { self.var_decl(node) }
            NodeKind.Assign => { self.assign(node) }
            NodeKind.If => { self.conditional(node) }
            NodeKind.While => {
                self.label(node)
                self.emit("while ")
                self.expr(node.child("cond"), _P_LOWEST)
                self.emit(" ")
                self.body(node.child("body"))
            }
            NodeKind.Loop => {
                self.label(node)
                self.emit("loop ")
                self.body(node.child("body"))
            }
            NodeKind.For => {
                self.label(node)
                self.emit("for ")
                self.pattern(node.child("pattern"))
                self.emit(" in ")
                self.expr(node.child("iterable"), _P_LOWEST)
                self.emit(" ")
                self.body(node.child("body"))
            }
            NodeKind.Match => { self.match_node(node) }
            NodeKind.Break => { self.jump(node, "break") }
            NodeKind.Continue => { self.jump(node, "continue") }
            NodeKind.Return => {
                self.emit("return")
                let value = node.child("value")
                if value is not null {
                    self.emit(" ")
                    self.expr(value, _P_LOWEST)
                }
            }
            NodeKind.Throw => {
                self.emit("throw ")
                self.expr(node.child("value"), _P_LOWEST)
            }
            NodeKind.Try => { self.try_stmt(node) }
            NodeKind.Defer => {
                self.emit("defer ")
                self.body(node.child("body"))
            }
            NodeKind.Assert => {
                self.emit("assert ")
                self.expr(node.child("cond"), _P_LOWEST)
                let message = node.child("message")
                if message is not null {
                    self.emit(", ")
                    self.expr(message, _P_LOWEST)
                }
            }
            NodeKind.FnDecl => { self.function(node, "fn") }
            NodeKind.ClassDecl => { self.class_decl(node) }
            NodeKind.TraitDecl => { self.trait_decl(node) }
            NodeKind.EnumDecl => { self.enum_decl(node) }
            NodeKind.TypeDecl => {
                self.visibility(node.visibility())
                self.emit("type " + node.text("name"))
                self.generics(node.records("generics"))
                self.emit(" = " + _type_source(node.type_field("aliased")))
            }
            NodeKind.Import => {
                self.emit("import " + node.text("path"))
                let alias = node.opt_text("alias")
                if alias is not null { self.emit(" as " + alias) }
            }
            NodeKind.FromImport => { self.from_import(node) }
            NodeKind.Export => { self.emit("export { " + ", ".join(node.get("names")) + " }") }
            NodeKind.ModuleDecl => { self.emit("module " + node.text("name")) }
            NodeKind.Program => { self.program(node) }
            _ => { self.expr(node, _P_LOWEST) }
        }
    }

    fn label(self, node: Node) -> void {
        let label = node.opt_text("label")
        if label is not null { self.emit("'" + label + ": ") }
    }

    fn jump(self, node: Node, keyword: str) -> void {
        self.emit(keyword)
        let label = node.opt_text("label")
        if label is not null { self.emit(" '" + label) }
    }

    fn visibility(self, value: Visibility) -> void {
        if value == Visibility.Public {
            self.emit("pub ")
        } elif value == Visibility.Protected {
            self.emit("prot ")
        }
    }

    fn var_decl(self, node: Node) -> void {
        self.visibility(node.visibility())
        self.emit(_DECL_KEYWORDS[node.get("varDeclKind")] + " ")
        self.pattern(node.child("pattern"))
        let declared = node.type_field("declaredType")
        if declared is not null and declared.kind != TypeKind.Infer {
            self.emit(": " + declared.to_source())
        }
        let init = node.child("init")
        if init is not null {
            self.emit(" = ")
            self.expr(init, _P_LOWEST)
        }
    }

    fn assign(self, node: Node) -> void {
        let target = node.child("target")
        if target is not null and is_pattern(target.kind) {
            self.pattern(target)
        } else {
            self.expr(target, _P_LOWEST)
        }
        if node.flag("isCompound") {
            self.emit(" " + op_text(node.op()) + "= ")
        } else {
            self.emit(" = ")
        }
        self.expr(node.child("value"), _P_LOWEST)
    }

    # `elif` is an else branch that is itself an `if`; `else { if ... }` keeps
    # its block and so keeps its braces.
    fn conditional(self, node: Node) -> void {
        self.emit("if ")
        self.expr(node.child("cond"), _P_LOWEST)
        self.emit(" ")
        self.body(node.child("thenBranch"))
        var alternative = node.child("elseBranch")
        while alternative is not null {
            if alternative.kind == NodeKind.If or alternative.kind == NodeKind.IfExpr {
                self.emit(" elif ")
                self.expr(alternative.child("cond"), _P_LOWEST)
                self.emit(" ")
                self.body(alternative.child("thenBranch"))
                alternative = alternative.child("elseBranch")
            } else {
                self.emit(" else ")
                self.body(alternative)
                alternative = null
            }
        }
    }

    fn match_node(self, node: Node) -> void {
        self.emit("match ")
        self.expr(node.child("subject"), _P_LOWEST)
        self.emit(" {")
        self.line()
        self.indent += 1
        for current in node.records("arms") {
            self.pattern(_node_at(current, "pattern"))
            let guard = _node_at(current, "guard")
            if guard is not null {
                self.emit(" if ")
                self.expr(guard, _P_LOWEST)
            }
            self.emit(" => ")
            let body = _node_at(current, "body")
            if body is not null and body.kind == NodeKind.Block {
                self.block(body)
            } else {
                self.expr(body, _P_LOWEST)
                self.emit(",")
            }
            self.line()
        }
        self.indent -= 1
        self.emit("}")
    }

    fn try_stmt(self, node: Node) -> void {
        self.emit("try ")
        self.body(node.child("body"))
        for handler in node.records("catches") {
            self.emit(" catch")
            let name = _text_at(handler, "name")
            if name is not null { self.emit(" " + name) }
            let types = handler.get("types", [])
            if types.len() > 0 {
                self.emit(": " + " | ".join([caught.to_source() for caught in types]))
            }
            self.emit(" ")
            self.body(_node_at(handler, "body"))
        }
        let ensure = node.child("finallyBlock")
        if ensure is not null {
            self.emit(" finally ")
            self.body(ensure)
        }
    }

    fn from_import(self, node: Node) -> void {
        self.emit("from " + node.text("path") + " import ")
        if node.flag("isWildcard") {
            self.emit("*")
            return
        }
        for (index, item) in node.records("items").enumerate() {
            if index > 0 { self.emit(", ") }
            self.emit(_text_at(item, "name") ?? "_")
            let alias = _text_at(item, "alias")
            if alias is not null { self.emit(" as " + alias) }
        }
    }

    fn function(self, node: Node, keyword: str) -> void {
        self.visibility(node.visibility())
        if node.flag("isStatic") { self.emit("static ") }
        self.emit(keyword)
        let name = node.opt_text("name")
        if name is not null { self.emit(" " + name) }
        self.generics(node.records("generics"))
        self.params(node.records("params"))
        let returns = node.type_field("returnType")
        if returns is not null { self.emit(" -> " + returns.to_source()) }
        # A trait method without a default implementation is a signature alone.
        let body = node.child("body")
        if body is null { return }
        self.emit(" ")
        self.body(body)
    }

    fn generics(self, items: list[dict[str, any]]) -> void {
        if items.len() == 0 { return }
        self.emit("[")
        for (index, current) in items.enumerate() {
            if index > 0 { self.emit(", ") }
            self.emit(_text_at(current, "name") ?? "_")
            let bound = _type_at(current, "bound")
            if bound is not null { self.emit(": " + bound.to_source()) }
        }
        self.emit("]")
    }

    fn params(self, items: list[dict[str, any]]) -> void {
        self.emit("(")
        for (index, current) in items.enumerate() {
            if index > 0 { self.emit(", ") }
            if _flag(current, "isVariadic") { self.emit("...") }
            if _flag(current, "isKwRest") { self.emit("**") }
            self.emit(_text_at(current, "name") ?? "_")
            let declared = _type_at(current, "type")
            if declared is not null { self.emit(": " + declared.to_source()) }
            let fallback = _node_at(current, "defaultValue")
            if fallback is not null {
                self.emit(" = ")
                self.expr(fallback, _P_LOWEST)
            }
        }
        self.emit(")")
    }

    fn class_decl(self, node: Node) -> void {
        self.visibility(node.visibility())
        self.emit("class " + node.text("name"))
        self.generics(node.records("generics"))
        let superclass = node.type_field("superclass")
        if superclass is not null { self.emit(" extends " + superclass.to_source()) }
        self.implements_list(node.type_list("traits"))
        let fields = node.records("fields")
        let methods = node.child_list("methods")
        let getters = node.child_list("getters")
        let setters = node.child_list("setters")
        if fields.len() + methods.len() + getters.len() + setters.len() == 0 {
            self.emit(" {}")
            return
        }
        self.emit(" {")
        self.line()
        self.indent += 1
        for current in fields {
            self.class_field(current)
            self.line()
        }
        var written = fields.len() > 0
        written = self.members(methods, "fn", written)
        written = self.members(getters, "get", written)
        written = self.members(setters, "set", written)
        self.indent -= 1
        self.emit("}")
    }

    # Members are separated by a blank line; `written` says whether anything is
    # above them already.
    fn members(self, nodes: list[Node], keyword: str, written: bool) -> bool {
        var any_written = written
        for member in nodes {
            if any_written { self.line() }
            self.function(member, keyword)
            self.line()
            any_written = true
        }
        return any_written
    }

    fn class_field(self, field: dict[str, any]) -> void {
        let value: Visibility = field.get("visibility", Visibility.Private)
        self.visibility(value)
        if _flag(field, "isStatic") { self.emit("static ") }
        if _flag(field, "isLet") {
            self.emit("let ")
        } else {
            self.emit("var ")
        }
        self.emit(_text_at(field, "name") ?? "_")
        let declared = _type_at(field, "type")
        if declared is not null { self.emit(": " + declared.to_source()) }
        let fallback = _node_at(field, "defaultValue")
        if fallback is not null {
            self.emit(" = ")
            self.expr(fallback, _P_LOWEST)
        }
    }

    fn trait_decl(self, node: Node) -> void {
        self.visibility(node.visibility())
        self.emit("trait " + node.text("name"))
        self.generics(node.records("generics"))
        self.implements_list(node.type_list("supers"))
        let methods = node.child_list("methods")
        if methods.len() == 0 {
            self.emit(" {}")
            return
        }
        self.emit(" {")
        self.line()
        self.indent += 1
        let _written = self.members(methods, "fn", false)
        self.indent -= 1
        self.emit("}")
    }

    fn enum_decl(self, node: Node) -> void {
        self.visibility(node.visibility())
        self.emit("enum " + node.text("name"))
        self.generics(node.records("generics"))
        self.emit(" {")
        self.line()
        self.indent += 1
        for current in node.records("variants") {
            self.emit(_text_at(current, "name") ?? "_")
            let params = current.get("params", [])
            if params.len() > 0 { self.params(params) }
            self.emit(",")
            self.line()
        }
        let _written = self.members(node.child_list("methods"), "fn", true)
        self.indent -= 1
        self.emit("}")
    }

    fn implements_list(self, types: list[TypeExpr]) -> void {
        if types.len() == 0 { return }
        self.emit(": " + ", ".join([one.to_source() for one in types]))
    }

    fn expr(self, node: Node?, min_precedence: int) -> void {
        if node is null { return }
        # A sign-leading literal is one token, not an operator and a number
        # (§1.6: `-9223372036854775808` is the only one, since every other
        # negative literal parses as `-` applied to a positive one). Let any
        # operator stand in front of it and that minus binds to the operator
        # instead, leaving digits that overflow `int`: `- -9223372036854775808`
        # and `1 - -9223372036854775808` are both rejected. Brackets are the
        # only spelling that survives in every position, so it always wears
        # them.
        let wrap = _precedence(node) < min_precedence or _is_signed_literal(node)
        if wrap { self.emit("(") }
        self.bare_expr(node)
        if wrap { self.emit(")") }
    }

    # `1.foo` would lex as a float, so an integer receiver keeps its brackets
    # even though its precedence says it needs none.
    fn receiver(self, node: Node?) -> void {
        if node is not null and node.kind == NodeKind.IntLit {
            self.emit("(")
            self.bare_expr(node)
            self.emit(")")
            return
        }
        self.expr(node, _P_POSTFIX)
    }

    fn bare_expr(self, node: Node) -> void {
        match node.kind {
            NodeKind.IntLit => { self.emit(str(node.number("intLit"))) }
            NodeKind.FloatLit => { self.emit(_float_source(node.get("floatLit"))) }
            NodeKind.StrLit => { self.emit(quote(node.opt_text("chars") ?? "")) }
            NodeKind.BoolLit => { self.emit(node.flag("boolLit") ? "true" : "false") }
            NodeKind.NullLit => { self.emit("null") }
            NodeKind.Self => { self.emit("self") }
            NodeKind.Super => { self.emit("super") }
            NodeKind.Ident => { self.emit(node.text("name")) }
            NodeKind.FString => { self.fstring(node) }
            NodeKind.ListLit => { self.sequence(node.child_list("items"), "[", "]", false) }
            NodeKind.SetLit => { self.set_lit(node) }
            NodeKind.TupleLit => { self.sequence(node.child_list("items"), "(", ")", true) }
            NodeKind.DictLit => { self.dict_lit(node) }
            NodeKind.Unary => { self.unary(node) }
            NodeKind.Binary | NodeKind.Logical => { self.binary(node) }
            NodeKind.CompareChain => { self.compare_chain(node) }
            NodeKind.Ternary => {
                self.expr(node.child("cond"), _P_COALESCE)
                self.emit(" ? ")
                self.expr(node.child("thenExpr"), _P_LOWEST)
                self.emit(" : ")
                self.expr(node.child("elseExpr"), _P_LOWEST)
            }
            NodeKind.Coalesce => {
                self.expr(node.child("left"), _P_COALESCE + 1)
                self.emit(" ?? ")
                self.expr(node.child("right"), _P_COALESCE)
            }
            NodeKind.Call => { self.call(node) }
            NodeKind.Index => {
                self.receiver(node.child("object"))
                self.emit("[")
                # `a[x, y]` parses to a tuple index, and the two spell the same
                # tree — but `Dict[K, V]` is how a generic instantiation is
                # written, so print the bare list and not `Dict[(K, V)]`. A
                # one-element tuple keeps its brackets: `a[x,]` is not `a[x]`.
                let subscript = node.child("index")
                if subscript is not null and
                    subscript.kind == NodeKind.TupleLit and
                    subscript.child_list("items").len() > 1 {
                    self.sequence(subscript.child_list("items"), "", "", false)
                } else {
                    self.expr(subscript, _P_LOWEST)
                }
                self.emit("]")
            }
            NodeKind.Slice => { self.slice(node) }
            NodeKind.Member => {
                self.receiver(node.child("object"))
                self.emit("." + node.text("name"))
            }
            NodeKind.OptMember => {
                self.receiver(node.child("object"))
                self.emit("?." + node.text("name"))
            }
            NodeKind.Lambda => { self.lambda(node) }
            NodeKind.AnonFn => { self.function(node, "fn") }
            NodeKind.Comprehension => { self.comprehension(node) }
            NodeKind.Range => {
                self.expr(node.child("start"), _P_RANGE + 1)
                self.emit(node.flag("inclusive") ? "..=" : "..")
                self.expr(node.child("stop"), _P_RANGE + 1)
            }
            NodeKind.IfExpr => { self.conditional(node) }
            NodeKind.Match | NodeKind.MatchExpr => { self.match_node(node) }
            # The checker inserts casts; they have no syntax of their own.
            NodeKind.Cast => { self.bare_expr(node.require("operand")) }
            NodeKind.Yield => {
                self.emit("yield")
                let operand = node.child("operand")
                if operand is not null {
                    self.emit(" ")
                    self.expr(operand, _P_LOWEST)
                }
            }
            NodeKind.Await => {
                self.emit("await ")
                self.expr(node.child("operand"), _P_LOWEST)
            }
            NodeKind.ThrowExpr => {
                self.emit("throw ")
                self.expr(node.child("operand"), _P_LOWEST)
            }
            _ => {
                if is_pattern(node.kind) {
                    self.pattern(node)
                } else {
                    throw ValueError(f"{kind_name(node.kind)} is not an expression")
                }
            }
        }
    }

    fn sequence(self, items: list[Node], open: str, close: str, tuple_comma: bool) -> void {
        self.emit(open)
        for (index, item) in items.enumerate() {
            if index > 0 { self.emit(", ") }
            self.expr(item, _P_LOWEST)
        }
        # `(x)` is just `x`; a one-element tuple needs its trailing comma.
        if tuple_comma and items.len() == 1 { self.emit(",") }
        self.emit(close)
    }

    fn set_lit(self, node: Node) -> void {
        let items = node.child_list("items")
        # `{}` is the empty dict, so the empty set has to be spelled as a call.
        if items.len() == 0 {
            self.emit("set()")
            return
        }
        self.sequence(items, "{", "}", false)
    }

    fn dict_lit(self, node: Node) -> void {
        let keys = node.child_list("keys")
        let values = node.child_list("values")
        self.emit("{")
        for (index, key) in keys.enumerate() {
            if index > 0 { self.emit(", ") }
            self.expr(key, _P_LOWEST)
            self.emit(": ")
            if index < values.len() { self.expr(values[index], _P_LOWEST) }
        }
        self.emit("}")
    }

    fn unary(self, node: Node) -> void {
        let op = node.op()
        let operand = node.child("operand")
        if op == OpKind.Not {
            self.emit("not ")
            self.expr(operand, _P_COMPARE)
            return
        }
        self.emit(op_text(op))
        # `- -x`: without the space the two signs would read as one token. A
        # sign-leading *literal* needs no space here — `expr` brackets it.
        if _leads_with_sign(operand) { self.emit(" ") }
        self.expr(operand, _P_UNARY)
    }

    fn binary(self, node: Node) -> void {
        let op = node.op()
        let level = _OP_PRECEDENCE[op]
        if op == OpKind.Pow {
            # `**` is the one right-associative binary operator, and its left
            # operand may only be a postfix expression.
            self.expr(node.child("left"), _P_POSTFIX)
            self.emit(" ** ")
            self.expr(node.child("right"), _P_UNARY)
            return
        }
        # Comparison is chained, not left-associative (§4.2): printing the left
        # operand of `(a > b) != c` without its brackets would re-parse as the
        # three-operand chain `a > b and b != c`, a different program. Every
        # other level is left-associative and needs no bracket on the left.
        let left_level = if level == _P_COMPARE { level + 1 } else { level }
        self.expr(node.child("left"), left_level)
        self.emit(" " + op_text(op) + " ")
        self.expr(node.child("right"), level + 1)
    }

    fn compare_chain(self, node: Node) -> void {
        let operands = node.child_list("operands")
        let ops = node.get("ops")
        for (index, operand) in operands.enumerate() {
            if index > 0 { self.emit(" " + op_text(ops[index - 1]) + " ") }
            self.expr(operand, _P_COMPARE + 1)
        }
    }

    fn call(self, node: Node) -> void {
        self.receiver(node.child("callee"))
        self.emit("(")
        for (index, current) in node.records("args").enumerate() {
            if index > 0 { self.emit(", ") }
            if _flag(current, "isSpread") { self.emit("...") }
            let name = _text_at(current, "name")
            if name is not null { self.emit(name + ": ") }
            self.expr(_node_at(current, "value"), _P_LOWEST)
        }
        self.emit(")")
    }

    fn slice(self, node: Node) -> void {
        self.receiver(node.child("object"))
        self.emit("[")
        self.expr(node.child("start"), _P_LOWEST)
        self.emit(":")
        self.expr(node.child("stop"), _P_LOWEST)
        let step = node.child("step")
        if step is not null {
            self.emit(":")
            self.expr(step, _P_LOWEST)
        }
        self.emit("]")
    }

    fn lambda(self, node: Node) -> void {
        self.emit("|")
        for (index, current) in node.records("params").enumerate() {
            if index > 0 { self.emit(", ") }
            self.emit(_text_at(current, "name") ?? "_")
        }
        self.emit("| ")
        self.expr(node.child("body"), _P_LOWEST)
    }

    fn comprehension(self, node: Node) -> void {
        let kind = node.get("compKind")
        var open = "["
        var close = "]"
        if kind == CompKind.Dict or kind == CompKind.Set {
            open = "{"
            close = "}"
        } elif kind == CompKind.Generator {
            open = "("
            close = ")"
        }
        self.emit(open)
        if kind == CompKind.Dict {
            self.expr(node.child("keyExpr"), _P_LOWEST)
            self.emit(": ")
        }
        self.expr(node.child("element"), _P_LOWEST)
        for clause in node.records("clauses") {
            self.emit(" for ")
            self.pattern(_node_at(clause, "pattern"))
            self.emit(" in ")
            # A ternary iterable would swallow the clause's own `if`.
            self.expr(_node_at(clause, "iterable"), _P_COALESCE)
            for condition in clause.get("conditions", []) {
                self.emit(" if ")
                self.expr(condition, _P_COALESCE)
            }
        }
        self.emit(close)
    }

    fn fstring(self, node: Node) -> void {
        self.emit("f\"")
        for part in node.child_list("parts") {
            if part.kind == NodeKind.StrLit {
                self.emit(_fstring_chunk(part.opt_text("chars") ?? ""))
                continue
            }
            let spec = _format_spec(part)
            self.emit("{")
            if spec is null {
                self.expr(part, _P_LOWEST)
            } else {
                self.expr(part.require("callee").child("object"), _P_LOWEST)
                self.emit(":" + spec)
            }
            self.emit("}")
        }
        self.emit("\"")
    }

    fn pattern(self, node: Node?) -> void {
        if node is null {
            self.emit("_")
            return
        }
        match node.kind {
            NodeKind.PatWildcard => { self.emit("_") }
            NodeKind.PatBind => {
                self.emit(node.text("name"))
                let declared = node.type_field("type")
                if declared is not null { self.emit(": " + declared.to_source()) }
            }
            NodeKind.PatLiteral => { self.expr(node.child("value"), _P_UNARY) }
            NodeKind.PatRange => {
                self.expr(node.child("lo"), _P_RANGE + 1)
                self.emit(node.flag("inclusive") ? "..=" : "..")
                self.expr(node.child("hi"), _P_RANGE + 1)
            }
            NodeKind.PatTuple => { self.sequence_pattern(node, "(", ")") }
            NodeKind.PatList => { self.sequence_pattern(node, "[", "]") }
            NodeKind.PatOr => {
                for (index, element) in node.child_list("elems").enumerate() {
                    if index > 0 { self.emit(" | ") }
                    self.pattern(element)
                }
            }
            NodeKind.PatClass => { self.class_pattern(node) }
            NodeKind.PatEnum => { self.enum_pattern(node) }
            _ => { self.expr(node, _P_LOWEST) }
        }
    }

    fn sequence_pattern(self, node: Node, open: str, close: str) -> void {
        let elements = node.child_list("elems")
        let rest = node.number("restIndex")
        self.emit(open)
        for (index, element) in elements.enumerate() {
            if index > 0 { self.emit(", ") }
            if index == rest { self.emit("...") }
            self.pattern(element)
        }
        if open == "(" and elements.len() == 1 and rest < 0 { self.emit(",") }
        self.emit(close)
    }

    fn class_pattern(self, node: Node) -> void {
        self.emit(node.text("typeName"))
        let names = node.get("fieldNames")
        let subs = node.child_list("subPatterns")
        # `Some(x)` encodes as a fieldNames array of nulls, one per sub-pattern,
        # so emptiness is not the test for the positional form.
        if not _has_field_names(names) {
            self.emit("(")
            for (index, sub) in subs.enumerate() {
                if index > 0 { self.emit(", ") }
                self.pattern(sub)
            }
            self.emit(")")
            return
        }
        self.emit("{")
        for (index, sub) in subs.enumerate() {
            if index > 0 { self.emit(", ") }
            let name = _at_or_null(names, index)
            if name is null {
                self.pattern(sub)
                continue
            }
            # `Point{x}` is `Point{x: x}` written the short way.
            if _is_plain_bind(sub, name) {
                self.emit(name)
            } else {
                self.emit(name + ": ")
                self.pattern(sub)
            }
        }
        self.emit("}")
    }

    fn enum_pattern(self, node: Node) -> void {
        self.emit(node.text("typeName") + "." + node.text("variantName"))
        let subs = node.child_list("subPatterns")
        if subs.len() == 0 { return }
        self.emit("(")
        for (index, sub) in subs.enumerate() {
            if index > 0 { self.emit(", ") }
            self.pattern(sub)
        }
        self.emit(")")
    }
}

fn _is_block_decl(node: Node) -> bool {
    return node.kind in _BLOCK_DECLARATIONS
}

# Whether a class pattern named its fields, `Point{x: a}`, rather than matching
# them positionally, `Point(a)`.
fn _has_field_names(names: list[str?]) -> bool {
    for name in names {
        if name is not null { return true }
    }
    return false
}

fn _at_or_null(names: list[str?], index: int) -> str? {
    if index >= names.len() { return null }
    return names[index]
}

fn _is_plain_bind(node: Node?, name: str) -> bool {
    if node is null or node.kind != NodeKind.PatBind { return false }
    if node.type_field("type") is not null { return false }
    return node.text("name") == name
}

fn _type_source(type_expr: TypeExpr?) -> str {
    if type_expr is null { return "any" }
    return type_expr.to_source()
}

# True when `node` renders as a `-` or `+` applied to something, so that a
# prefix `-` or `+` in front of it needs a space to stay a token of its own.
fn _leads_with_sign(node: Node?) -> bool {
    if node is null or node.kind != NodeKind.Unary { return false }
    return node.op() == OpKind.Neg or node.op() == OpKind.Pos
}

# A literal whose own text opens with a sign. `expr` brackets one wherever it
# appears, because a space in front is not enough: the sign has to stay part of
# the token. `freeze_literals` rewrites a literal into an `Ident` holding its
# spelling, so that shape counts too.
fn _is_signed_literal(node: Node?) -> bool {
    if node is null { return false }
    match node.kind {
        NodeKind.Ident => {
            let name = node.opt_text("name") ?? ""
            return name.starts_with("-") or name.starts_with("+")
        }
        NodeKind.IntLit => { return node.number("intLit") < 0 }
        NodeKind.FloatLit => { return node.get("floatLit") < 0.0 }
        _ => { return false }
    }
}

# A float literal has to keep its point: `1` would parse back as an int.
fn _float_source(value: float) -> str {
    let text = repr(value)
    if "." in text or "e" in text or "E" in text { return text }
    return text + ".0"
}

# The literal chunks of an f-string carry their braces doubled.
fn _fstring_chunk(text: str) -> str {
    let out = StringBuilder()
    for character in text {
        if character == "{" {
            out.push("{{")
        } elif character == "}" {
            out.push("}}")
        } else {
            out.push(escape(character))
        }
    }
    return out.build()
}

# The parser rewrites `{x:.2f}` into `x.__format__(".2f")`; recover the spec so
# that the hole prints the way it was written.
fn _format_spec(node: Node) -> str? {
    if node.kind != NodeKind.Call { return null }
    let callee = node.child("callee")
    if callee is null or callee.kind != NodeKind.Member { return null }
    if callee.text("name") != "__format__" { return null }
    let args = node.records("args")
    if args.len() != 1 { return null }
    let spec = _node_at(args[0], "value")
    if spec is null or spec.kind != NodeKind.StrLit { return null }
    if _text_at(args[0], "name") is not null { return null }
    return spec.opt_text("chars")
}
