#: The three shapes the pass keeps beside the tree: what kind of region a
#: scope covers, the scope itself, and the per-class record that field-slot
#: assignment walks after every class body has been seen.
#:
#: They are here rather than with the `Resolver` because they are the
#: vocabulary the pass is written in and because none of them has any
#: behaviour: a `Scope` is a bag of names, and a `ClassInfo` is a bag of what
#: one `class` declaration turned out to say. `Scope` and `ClassInfo` had to
#: be marked `pub` for the sibling modules to name them at all, and are
#: deliberately not re-exported by `resolve.jai`, since nothing outside this
#: directory has ever needed either.
#:
#: What is NOT here: `Symbol`, `FunctionScope` and `Resolution`. Those are the
#: pass's *output*, they outlive it, and they live in
#: `jaithon.compile.symbol`. These three are its scaffolding and are dropped
#: on the way out.
from jaithon.ast import Node, Span
from jaithon.compile.symbol import Symbol

#: The kind of region a scope covers.
pub enum ScopeKind { Module, Function, Block, Class, Loop }

pub class Scope {
    pub let kind: ScopeKind
    pub let depth: int
    pub let fn_index: int
    pub let saved_slot: int
    pub var names: dict[str, Symbol]
    pub var order: list[Symbol]
    pub var pending: dict[str, Span]
    pub var cleared: set[str]
    pub var label: str?
    #: Lowest slot this scope must close on the way out, or -1. ScopeKind.Loop:
    #: anything declared anywhere inside the loop, closed at every iteration
    #: edge (`_mark_captured_in_loop`). ScopeKind.Block: only what the block
    #: declares itself, closed when it exits (`_mark_captured_in_block`).
    pub var capture_base: int

    pub fn init(self, kind: ScopeKind, depth: int, fn_index: int, saved_slot: int) {
        self.kind = kind
        self.depth = depth
        self.fn_index = fn_index
        self.saved_slot = saved_slot
        self.names = {}
        self.order = []
        self.pending = {}
        self.cleared = set()
        self.label = null
        self.capture_base = -1
    }
}

pub class ClassInfo {
    pub let decl: Node
    pub let symbol: Symbol?
    pub let scope: Scope
    pub var super_symbol: Symbol?
    pub var total: int
    pub var state: int  # 0 unvisited, 1 visiting, 2 done

    pub fn init(self, decl: Node, symbol: Symbol?, scope: Scope) {
        self.decl = decl
        self.symbol = symbol
        self.scope = scope
        self.super_symbol = null
        self.total = 0
        self.state = 0
    }
}
