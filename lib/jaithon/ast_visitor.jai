#: The per-kind dispatch base class for walking an AST, split out of ast.jai.
#:
#: `Visitor` has no state of its own beyond what a subclass adds and touches
#: nothing but `Node`, `NodeKind` and `children_of` -- it moved here because
#: nothing else in ast.jai depended on it and, as of this split, nothing in
#: this tree imports it either. It is kept as public API for callers outside
#: the compiler who want one hook per node kind instead of building their own
#: `match` over `NodeKind`.
from jaithon.ast import Node, NodeKind, children_of

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
