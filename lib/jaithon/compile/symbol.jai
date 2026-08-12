#: The resolver's output model: every type `resolve.jai`'s `Resolver` builds
#: and hands to the checker and the emitter, plus the fixed inputs it checks
#: bindings against. Data classes only (`SemaCode`'s codes, `BUILTIN_NAMES`
#: and the two frame limits are static; `Symbol`, `UpvalueRef`,
#: `FunctionScope` and `Resolution` are populated by `Resolver` but never
#: read back by it once a pass moves on), so this file imports from nowhere
#: in `jaithon.compile` and everything downstream can import it directly
#: instead of reaching through `resolve.jai`.
from jaithon.ast import DeclKind, Node, Span, Visibility

#: Most local slots one function frame can hold, matching `JAI_MAX_LOCALS`.
pub const MAX_LOCALS: int = 65535

#: Most variables one function may capture, matching `JAI_MAX_UPVALUES`.
pub const MAX_UPVALUES: int = 255

#: Stable diagnostic codes for name resolution, spelled as in
#: `src/common/diag.h`, continuing the `E00xx`/`E01xx` groups `Code` in
#: `jaithon.compile.lexer` owns.
pub class SemaCode {
    pub static let E0200_UNDEFINED_NAME: str = "E0200"
    pub static let E0201_USE_BEFORE_DECLARATION: str = "E0201"
    pub static let E0202_UNDEFINED_LABEL: str = "E0202"
    pub static let E0203_BREAK_OUTSIDE_LOOP: str = "E0203"
    pub static let E0204_CONTINUE_OUTSIDE_LOOP: str = "E0204"
    pub static let E0205_RETURN_OUTSIDE_FUNCTION: str = "E0205"
    pub static let E0206_SELF_OUTSIDE_METHOD: str = "E0206"
    pub static let E0207_SUPER_OUTSIDE_SUBCLASS: str = "E0207"
    pub static let E0208_TOO_MANY_LOCALS: str = "E0208"
    pub static let E0209_TOO_MANY_UPVALUES: str = "E0209"
    pub static let E0210_YIELD_OUTSIDE_GENERATOR: str = "E0210"
    pub static let E0211_DEFER_OUTSIDE_FUNCTION: str = "E0211"
    pub static let E0301_ASSIGN_TO_IMMUTABLE: str = "E0301"
    pub static let E0302_DUPLICATE_DECLARATION: str = "E0302"
    pub static let E0304_MISSING_INITIALIZER: str = "E0304"
    pub static let E0305_ASSIGN_TO_CONST: str = "E0305"
    pub static let E0703_MISSING_SELF: str = "E0703"
    pub static let E0706_CYCLIC_INHERITANCE: str = "E0706"
    pub static let E0707_STATIC_WITH_SELF: str = "E0707"
    pub static let E0709_DUPLICATE_MEMBER: str = "E0709"
    pub static let E0902_INTERNAL_ERROR: str = "E0902"
    pub static let W0100_UNUSED_IMPORT: str = "W0100"
    pub static let W0101_UNUSED_BINDING: str = "W0101"
    pub static let W0103_SHADOWED_BINDING: str = "W0103"
}

#: Names that resolve without a declaration: the built-in free functions, the
#: primitive namespace, and the exception hierarchy of `LANGUAGE.md` §5.4.
#:
#: A name outside this set and outside every enclosing scope is `E0200`.
pub let BUILTIN_NAMES: set[str] = {
    "print",
    "input",
    "len",
    "range",
    "type_of",
    "repr",
    "str",
    "int",
    "float",
    "bool",
    "id",
    "hash",
    "abs",
    "min",
    "max",
    "sum",
    "sorted",
    "reversed",
    "enumerate",
    "zip",
    "map",
    "filter",
    "any",
    "all",
    "chr",
    "ord",
    "isinstance",
    "callable",
    "dir",
    "exit",
    "list",
    "dict",
    "set",
    "tuple",
    "bytes",
    "assert_eq",
    "__prim__",
    "Error",
    "AssertionError",
    "ArithmeticError",
    "DivisionByZeroError",
    "OverflowError",
    "LookupError",
    "IndexError",
    "KeyError",
    "NameError",
    "TypeError",
    "ValueError",
    "ParseError",
    "AttributeError",
    "IOError",
    "FileNotFoundError",
    "PermissionError",
    "OSError",
    "RuntimeError",
    "RecursionError",
    "StopIteration",
    "ImportError",
}

#: What a bound name denotes, mirroring `SymbolKind` in `src/sema/resolve.h`.
pub enum SymbolKind {
    Local,
    Upvalue,
    Global,
    Param,
    Field,
    Method,
    Class,
    Trait,
    Enum,
    EnumVariant,
    Module,
    TypeAlias,
    GenericParam,
    Builtin,
}

#: One bound name, and everything the emitter needs to address it.
pub class Symbol {
    #: What this name denotes.
    pub var kind: SymbolKind
    #: The identifier text.
    pub let name: str
    #: Where it was declared; `Span.none()` for a built-in.
    pub let decl_span: Span
    #: Local slot, upvalue index, instance-field slot or enum variant tag; -1
    #: when the symbol lives in no numbered position.
    pub var slot: int
    #: Scope depth of the declaration.
    pub var depth: int
    #: Whether the binding was introduced by `let`, `var` or `const`.
    pub var mutability: DeclKind
    #: Declared visibility; private unless marked `pub`.
    pub var visibility: Visibility
    #: True when some inner function closes over this binding.
    pub var is_captured: bool
    #: True once the name has been read.
    pub var is_used: bool
    #: True once the binding holds a value.
    pub var is_initialized: bool
    #: The declaring node, or null.
    pub var decl: Node?

    #: Create an unslotted, unused symbol for `name`.
    pub fn init(self, kind: SymbolKind, name: str, decl_span: Span, mutability: DeclKind) {
        self.kind = kind
        self.name = name
        self.decl_span = decl_span
        self.slot = -1
        self.depth = 0
        self.mutability = mutability
        self.visibility = Visibility.Private
        self.is_captured = false
        self.is_used = false
        self.is_initialized = false
        self.decl = null
    }

    #: True when this binding occupies a frame slot, and so can be captured.
    pub fn is_slotted(self) -> bool {
        return self.kind == SymbolKind.Local or self.kind == SymbolKind.Param
    }

    fn __str__(self) -> str {
        return f"Symbol({self.name}, slot {self.slot})"
    }
}

#: One entry of a function's capture list, in `OP_CLOSURE` operand order.
pub class UpvalueRef {
    #: The enclosing function's local slot, or its upvalue index.
    pub let index: int
    #: True when `index` names a local of the immediately enclosing function.
    pub let is_local: bool
    #: True when the capture is a snapshot rather than an alias: spec §6 makes
    #: `let` and `const` bindings captured by value and `var` by reference.
    pub let by_value: bool

    #: Record one capture.
    pub fn init(self, index: int, is_local: bool, by_value: bool) {
        self.index = index
        self.is_local = is_local
        self.by_value = by_value
    }
}

#: Per-function resolution state: what `OP_CLOSURE` and the record's header need.
pub class FunctionScope {
    #: The `fn`, lambda or program node this describes.
    pub let decl: Node
    #: Peak slot usage, which becomes the record's `maxSlots`.
    pub var local_count: int
    #: Next free slot; restored when a block scope pops.
    pub var next_slot: int
    #: Capture list, in `OP_CLOSURE` operand order.
    pub var upvalues: list[UpvalueRef]
    #: Upvalue symbols handed out so far, one per capture.
    pub var aliases: list[Symbol]
    #: True when the function captures anything and so needs a closure.
    pub var captures_anything: bool
    #: True when the body registers a `defer`.
    pub var has_defer: bool
    #: True when the body contains `yield`.
    pub var is_generator: bool

    #: Create the scope for `decl`, with slot 0 reserved for callee or self.
    pub fn init(self, decl: Node) {
        self.decl = decl
        self.local_count = 1
        self.next_slot = 1
        self.upvalues = []
        self.aliases = []
        self.captures_anything = false
        self.has_defer = false
        self.is_generator = false
    }
}

#: What the resolver learned, keyed by the identity of the node or record it
#: belongs to.
#:
#: A `jaithon.ast` node carries only the fields the C AST serialises, so there is
#: nowhere on it to hang a symbol. Identity is the key because two structurally
#: equal nodes — `x` in two different scopes — must not share a binding.
pub class Resolution {
    var symbols: dict[int, Symbol]
    var scopes: dict[int, FunctionScope]
    var loop_bases: dict[int, int]
    var type_args: set[int]

    #: Create an empty table.
    pub fn init(self) {
        self.symbols = {}
        self.scopes = {}
        self.loop_bases = {}
        self.type_args = set()
    }

    #: True when an `Index` node's brackets held type arguments rather than a
    #: subscript. `Box[int](0)` parses as an index because only the symbol
    #: table can tell the two apart; generics are erased at run time (spec
    #: §6.1), so the arguments have no value meaning and the emitter skips
    #: them. Recorded here rather than on the node because a `jaithon.ast` node
    #: holds only what the C AST serialises.
    pub fn is_type_args(self, node: Node) -> bool {
        return id(node) in self.type_args
    }

    #: Mark `node`'s brackets as type arguments.
    pub fn set_type_args(self, node: Node) -> void {
        self.type_args.add(id(node))
    }

    #: The symbol bound to `node`, which may be a node or a sub-structure
    #: record such as a parameter or a `catch` clause.
    pub fn symbol_of(self, node: any) -> Symbol? {
        return self.symbols.get(id(node), null)
    }

    #: Bind `symbol` to `node`.
    pub fn bind(self, node: any, symbol: Symbol?) -> void {
        if symbol != null { self.symbols[id(node)] = symbol }
    }

    #: The function scope of a `fn`, lambda or program node.
    pub fn scope_of(self, node: Node) -> FunctionScope? {
        return self.scopes.get(id(node), null)
    }

    #: Record `scope` as the function scope of `node`.
    pub fn set_scope(self, node: Node, scope: FunctionScope) -> void {
        self.scopes[id(node)] = scope
    }

    #: Lowest slot a scope-ending node must close on the way out, or -1 when
    #: nothing it owns is captured by reference.
    #:
    #: A `for`/`while`/`loop` node closes at the end of every iteration and owns
    #: everything declared inside it; a `Block` node closes when the block exits
    #: and owns only what it declares itself.
    pub fn capture_base_of(self, node: Node) -> int {
        return self.loop_bases.get(id(node), -1)
    }

    #: Record the close slot of a loop or block node.
    pub fn set_capture_base(self, node: Node, base: int) -> void {
        self.loop_bases[id(node)] = base
    }

    #: Frame window size of `node`, which is 1 when it was never resolved.
    pub fn slot_count(self, node: Node) -> int {
        let scope = self.scope_of(node)
        if scope == null { return 1 }
        return scope.local_count
    }
}
