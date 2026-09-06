#: The Jaithon abstract syntax tree, as Jaithon data.
#:
#: Facade: imports and re-exports every module of the `ast/` directory.
#:
#: This package mirrors `src/lang/ast.h` case for case and field for field: one
#: `NodeKind` per C `AstKind`, one `OpKind` per C `OpKind`, and field names
#: spelled exactly as `jaiAstToJson` writes them. Three properties hold across
#: it and its two encoding siblings (see below), and the formatter, the
#: documentation generator and the self-hosted front end all rest on them:
#:
#: - `from_json` reads, unchanged, what `jaithon --emit=ast` prints.
#: - `from_json(to_json(node))` equals `node`.
#: - Re-parsing `to_source(node)` yields `node` again, spans aside.
#:
#: A node is a kind, a span, and a `fields` slot list, one slot per field the
#: kind declares. `fields_of(kind)` lists those fields in the order the C emits
#: them, which is the order the slots hold them, each tagged with the shape of
#: its value:
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
#: ## What lives where
#:
#: Import from `jaithon.ast` and take no interest in the directory: this facade
#: re-exports exactly the names the single file exported, so the split below is
#: invisible to every caller.
#:
#: The vocabulary comes first and depends on nothing: `kinds.jai` holds
#: `NodeKind`, the three enums that tag a node's fields and the group
#: predicates; `operators.jai` holds `OpKind` and the source spelling of each
#: operator; `span.jai` holds `Span`, which the lexer and the diagnostics want
#: without wanting a tree. On top of those, `types.jai` holds `TypeExpr` and
#: its `TypeKind`, and `schema.jai` holds the per-kind field table. `node.jai`
#: holds `Node`, the sub-structure builders and `children_of` — one file
#: because `Node.call` calls `arg()` and `Node.children()` calls
#: `children_of`, and separating them would need an import cycle. `tree.jai`
#: sits last and consumes a finished tree: the walk, and the assignability and
#: divergence predicates the checker calls.
#:
#: Three things that only *read* a built tree are siblings of this file rather
#: than modules of the directory, since they need none of the private state
#: above: rendering a tree as debug text or as the C's JSON wire format lives
#: in `ast_encode.jai` (`to_sexp`, `to_json`/`from_json`); rendering it back as
#: Jaithon source lives in `ast_unparse.jai` (`to_source`), by far the largest
#: single piece and the reason this file was once over 3800 lines; and the
#: per-kind dispatch base class lives in `ast_visitor.jai` (`Visitor`), which
#: nothing in this tree currently imports. What lets `ast_encode` reach the
#: schema is that `_ELEMENT_TAG`, `_RECORDS` and `_default_field` are `pub` and
#: re-exported below — it needs the same field tags `Node.init` and
#: `children_of` already rely on.
#:
#: `kinds.jai` marks `_KINDS` and `_KIND_INDEX` `pub` for the same kind of
#: reason, so that `schema.jai` can build and index the field table. Those two
#: are deliberately NOT re-exported here: reaching them means naming
#: `jaithon.ast.kinds`, and the surface of `jaithon.ast` stays what it was.
from jaithon.ast.kinds import CompKind, DeclKind, NodeKind, Visibility
from jaithon.ast.kinds import is_declaration, is_expression, is_pattern, is_statement
from jaithon.ast.kinds import kind_from_name, kind_name
from jaithon.ast.operators import OpKind, op_from_name, op_is_comparison, op_name, op_text
from jaithon.ast.span import Span
from jaithon.ast.types import TypeExpr, TypeKind
from jaithon.ast.schema import _ELEMENT_TAG, _RECORDS, _default_field, fields_of, record_fields
from jaithon.ast.node import Node, arg, catch_clause, children_of, class_field, comp_clause
from jaithon.ast.node import generic, import_item, match_arm, param, variant
from jaithon.ast.tree import always_diverges, find_all, is_assignable, preorder, walk

export { CompKind, DeclKind, NodeKind, Visibility }
export { is_declaration, is_expression, is_pattern, is_statement }
export { kind_from_name, kind_name }
export { OpKind, op_from_name, op_is_comparison, op_name, op_text }
export { Span }
export { TypeExpr, TypeKind }
export { _ELEMENT_TAG, _RECORDS, _default_field, fields_of, record_fields }
export { Node, arg, catch_clause, children_of, class_field, comp_clause }
export { generic, import_item, match_arm, param, variant }
export { always_diverges, find_all, is_assignable, preorder, walk }
