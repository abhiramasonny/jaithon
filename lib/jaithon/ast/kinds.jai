#: The node-kind vocabulary: `NodeKind`, the three small enums that tag a
#: node's fields, the C `AstKind` name table `NodeKind` is checked against, and
#: the four group predicates that read that table's boundaries.
#:
#: `_KINDS` and `_KIND_INDEX` are `pub` so that `schema.jai` can build and index
#: the per-kind field table, which is the one thing outside this file that has
#: to know a kind's position rather than its name. That is the same concession
#: `_ELEMENT_TAG`, `_RECORDS` and `_default_field` already make for
#: `ast_encode.jai`. The `jaithon.ast` facade does not re-export either, so the
#: public surface is exactly what it was; reaching them means naming this
#: module.
#:
#: What is NOT here: `OpKind` and its tables, which share no state with these
#: and live in `operators.jai`; and `TypeKind`, which tags a `TypeExpr` rather
#: than a node and travels with it in `types.jai`.

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

#: The visibility a declaration was written with.
pub enum Visibility { Private, Protected, Public }

#: Which of `let`, `var` and `const` introduced a binding.
pub enum DeclKind { Let, Var, Const }

#: Which brackets a comprehension was written with.
pub enum CompKind { List, Dict, Set, Generator }

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

pub let _KINDS: list[NodeKind] = [
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

pub let _KIND_INDEX: dict[NodeKind, int] = {kind: index for (index, kind) in _KINDS.enumerate()}
let _KIND_BY_NAME: dict[str, NodeKind] = {name: kind for (name, kind) in zip(_KIND_NAMES, _KINDS)}

const _FIRST_STATEMENT = 34
const _FIRST_DECLARATION = 50
const _FIRST_PATTERN = 59
const _PROGRAM_INDEX = 68

assert _KIND_NAMES.len() == _KINDS.len(), "the AST kind tables have different lengths"
assert _KIND_NAMES[_FIRST_STATEMENT] == "AST_BLOCK", "the AST kind table drifted from AstKind"
assert _KIND_NAMES[_PROGRAM_INDEX] == "AST_PROGRAM", "the AST kind table drifted from AstKind"

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
