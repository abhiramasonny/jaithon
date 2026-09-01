#: The `Emitter` class -- the function-context stack, the resolver's output,
#: the enum table a pattern needs -- and every helper that writes a byte or
#: moves the cursor without walking a production: the opcode writers, the
#: constant and string pool, the operand-stack depth model, the temporary
#: allocator, the jump-and-patch pair, the symbol load and store, the binding
#: form of a pattern, the scope and loop bookkeeping, and the diagnostic for a
#: construct the instruction set cannot express.
#:
#: Those helpers stayed methods on purpose. They are called from all seven
#: production files, and a method call is what the moved code already wrote, so
#: keeping them here turned `self._op(...)` into `e._op(...)` and left every
#: call shape alone.
#:
#: The seams. Jaithon rejects a circular import and the productions are one
#: knot of mutual recursion, so `emit/` is ordered `exception -> expr ->
#: function -> decl -> pattern -> loop -> stmt` and each file imports only the
#: files below it. Three calls run the other way -- a nested STATEMENT, which
#: every one of them needs; a `match` in expression position, which `expr.jai`
#: reaches before `pattern.jai` exists; and a CLOSURE for a lambda, which
#: `expr.jai` reaches before `function.jai` exists. Each is a hook here behind
#: a one-line function any file may call. The indirect call sits inside that
#: one line rather than inside the hot body that made it, so a caller keeps
#: whatever the JIT was already doing for it. `wire.jai` fills the hooks in as
#: it is imported, which is why an `Emitter` built by hand needs no wiring call
#: of its own, and why an unwired hook is a thrown error naming itself rather
#: than a wrong chunk.
#:
#: What is NOT here: every production. They are free functions in the seven
#: sibling files and reach this class through `e`.
from jaithon.ast import Node, NodeKind, Span, Visibility
from jaithon.compile.diag import DiagnosticBag
from jaithon.compile.symbol import Resolution, SemaCode, Symbol, SymbolKind
from jaithon.compile.emit.opcode import Op
from jaithon.compile.emit.spec import Const, ConstTag, FN_GENERATOR
from jaithon.compile.emit.chunk import Chunk, FuncProto
from jaithon.compile.emit.shape import _symbol_is_global_like

pub fn _no_optimize(proto: FuncProto, level: int) -> void { }

pub class LoopFrame {
    pub let label: str?
    pub let continue_target: int
    #: Lowest slot to close when an iteration ends, or -1. See `_close_iteration`.
    pub let capture_base: int
    #: Operand-stack depth `continue` and `break` unwind to. A `for` leaves its
    #: iterator pushed across the body, so `continue` keeps it and `break` drops it.
    pub let continue_depth: int
    pub let break_depth: int
    pub var breaks: list[int]

    pub fn init(
        self,
        label: str?,
        continue_target: int,
        capture_base: int,
        continue_depth: int,
        break_depth: int
    ) {
        self.label = label
        self.continue_target = continue_target
        self.capture_base = capture_base
        self.continue_depth = continue_depth
        self.break_depth = break_depth
        self.breaks = []
    }
}

pub class FnCtx {
    pub let proto: FuncProto
    pub var loops: list[LoopFrame]
    #: Lowest capture base of the blocks open around the cursor, or -1.
    #: `return` leaves all of them at once and closes at this.
    pub var close_base: int
    #: Watermark, not a stack: the lowest base of any block or loop *opened*
    #: since it was last reset. `_try` resets it around the guarded region,
    #: since an exception skips every block close inside that region.
    pub var deep_close_base: int
    #: Next free temporary slot, handed out above the resolver's peak. A
    #: temporary lives in the frame window, not on the operand stack, so a
    #: pattern test can fail without unwinding partially destructured values.
    pub var next_temp: int
    #: High-water mark of `next_temp`; the frame reserves this many slots
    #: before the operand stack starts.
    pub var max_slot: int
    #: The `finally` blocks open around the cursor, innermost last. An explicit
    #: `return`, `break` or `continue` leaving them inlines a copy of each; the
    #: exception path uses the exception table instead.
    pub var finallys: list[Node]
    #: Number of loop frames open when each `finally` was entered, so that a
    #: `break` knows which of them it is leaving.
    pub var finally_loop_depth: list[int]
    #: Above zero inside a `try`, `catch`, `finally` or deferred block. A
    #: `return` there may not reuse the frame: see `_can_tail_call`.
    pub var protect_depth: int
    #: True once this frame has registered a `defer`. Set where PUSH_DEFER is
    #: emitted, so a `return` above the `defer` sees false, exactly as the VM
    #: would find no defer registered yet.
    pub var has_defer: bool

    pub fn init(self, proto: FuncProto, local_count: int) {
        self.proto = proto
        self.loops = []
        self.close_base = -1
        self.deep_close_base = -1
        self.next_temp = if local_count < 1 { 1 } else { local_count }
        self.max_slot = self.next_temp
        self.finallys = []
        self.finally_loop_depth = []
        self.protect_depth = 0
        self.has_defer = false
    }
}

#: Lower a resolved program to a `FuncProto` for the module body.
#:
#: Run `jaithon.compile.resolve` first: the emitter reads the symbols, slots and
#: capture lists that pass left behind and looks nothing up itself.
pub class Emitter {
    #: The resolver's output: symbols by node identity, and function scopes.
    pub let resolution: Resolution
    #: Where a construct the instruction set cannot express is reported.
    pub let bag: DiagnosticBag
    #: When true `assert` compiles to nothing, as `--release` requires.
    pub let strip_asserts: bool
    #: The `--O` level handed to `optimize`: 0 disables passes 3-7 (spec §8).
    pub let opt_level: int
    #: The optimiser run over every function this emitter finishes; see
    #: `_no_optimize` for why it is a parameter.
    pub let optimize: fn(FuncProto, int) -> void

    #: The file's source text, so an assert can bake in its condition. Null
    #: falls back to the bare `"assertion failed"`, as the C does.
    pub var source: str?
    #: Enum declarations this file imported, by the name they are bound to here.
    #: An enum pattern needs the variant's tag, which is the declaration's
    #: variant order and for an imported enum lives in another file. The C reads
    #: it out of the checker's registry (`jaiTypeDeclFind` in
    #: `src/codegen/codegen_pattern.c`); here the checker hands the enums over.
    pub var imported_enums: dict[str, Node]
    pub var stack: list[FnCtx]
    pub var enums: dict[str, Node]
    #: The METHOD names of each class whose body is being lowered, innermost
    #: last. `_call` needs it to tell `self.m()` -- a real method -- from
    #: `self.f()`, a field that happens to hold a callable; see the tail-call
    #: note there. Methods only: a getter's `self.g()` calls what the getter
    #: RETURNS, and an inherited method is not in this class's own list, so
    #: both keep the old shape rather than being answered wrongly.
    pub var class_methods: list[list[str]]

    #: Create an emitter over `resolution`, reporting into `bag`.
    pub fn init(
        self,
        resolution: Resolution,
        bag: DiagnosticBag,
        strip_asserts: bool = false,
        opt_level: int = 2,
        optimize: fn(FuncProto, int) -> void = _no_optimize
    ) {
        self.resolution = resolution
        self.bag = bag
        self.strip_asserts = strip_asserts
        self.opt_level = opt_level
        self.optimize = optimize
        self.source = null
        self.imported_enums = {}
        self.stack = []
        self.enums = {}
        self.class_methods = []
    }

    #: Lower `program` and return the module body's function record.
    pub fn compile_program(self, program: Node, module_name: str? = null) -> FuncProto {
        let proto = FuncProto(module_name ?? "__main__", 0, 0)
        for name in self.imported_enums.keys() { self.enums[name] = self.imported_enums[name] }
        self._index_enums(program.child_list("stmts"))
        self.stack.push(FnCtx(proto, self.resolution.slot_count(program)))
        let stmts = program.child_list("stmts")
        for statement in stmts {
            if statement.kind == NodeKind.FnDecl { lower_stmt(self, statement) }
        }
        for statement in stmts {
            if statement.kind != NodeKind.FnDecl { lower_stmt(self, statement) }
        }
        self._run_defers(program.span)
        self._op(Op.ReturnNull, program.span)
        self._finish()
        return proto
    }

    pub fn _finish(self) -> void {
        let ctx = self._ctx()
        ctx.proto.max_slots = ctx.max_slot + ctx.proto.chunk.max_depth
        self.optimize(ctx.proto, self.opt_level)
        self.stack.pop()
    }

    pub fn _index_enums(self, stmts: list[Node]) -> void {
        for node in stmts {
            if node.kind == NodeKind.EnumDecl {
                self.enums[node.text("name")] = node
            } elif node.kind == NodeKind.Block {
                self._index_enums(node.child_list("stmts"))
            }
        }
    }

    pub fn _chunk(self) -> Chunk { return self.stack[self.stack.len() - 1].proto.chunk }

    pub fn _ctx(self) -> FnCtx { return self.stack[self.stack.len() - 1] }

    pub fn _op(self, op: Op, span: Span) -> int {
        return self._chunk().write_op(op, span)
    }

    pub fn _op_u8(self, op: Op, value: int, span: Span) -> void {
        self._chunk().write_op(op, span)
        self._chunk().write_byte(value)
    }

    pub fn _op_u16(self, op: Op, value: int, span: Span) -> void {
        self._chunk().write_op(op, span)
        self._chunk().write_u16(value)
    }

    pub fn _op_k(self, op: Op, index: int, span: Span) -> void {
        self._chunk().write_op(op, span)
        self._chunk().write_u24(index)
    }

    pub fn _op_cached(self, op: Op, index: int, span: Span) -> void {
        self._chunk().write_op(op, span)
        self._chunk().write_u24(index)
        self._chunk().write_u16(self._chunk().add_cache())
    }

    pub fn _string(self, text: str) -> int {
        return self._chunk().add_constant(Const(ConstTag.Str, text))
    }

    pub fn _depth(self) -> int { return self._chunk().depth }

    pub fn _adjust(self, delta: int) -> void { self._chunk().adjust(delta) }

    pub fn _set_depth(self, value: int) -> void { self._chunk().set_depth(value) }

    pub fn _pop_n(self, count: int, span: Span) -> void {
        var left = count
        while left > 0 {
            if left == 1 {
                self._op(Op.Pop, span)
                return
            }
            let step = if left > 255 { 255 } else { left }
            self._op_u8(Op.PopN, step, span)
            self._adjust(-step)
            left -= step
        }
    }

    pub fn _alloc_temp(self) -> int {
        let ctx = self._ctx()
        let slot = ctx.next_temp
        ctx.next_temp += 1
        if ctx.next_temp > ctx.max_slot { ctx.max_slot = ctx.next_temp }
        return slot
    }

    pub fn _free_temps(self, count: int) -> void {
        let ctx = self._ctx()
        ctx.next_temp -= count
        if ctx.next_temp < 0 { ctx.next_temp = 0 }
    }

    pub fn _get_local(self, slot: int, span: Span) -> void {
        self._op_u16(Op.GetLocal, if slot < 0 { 0 } else { slot }, span)
    }

    pub fn _emit_int(self, value: int, span: Span) -> void {
        if value >= -32768 and value <= 32767 {
            self._chunk().write_op(Op.Int, span)
            self._chunk().write_i16(value)
            return
        }
        self._op_k(Op.Const, self._chunk().add_constant(Const(ConstTag.Int, value)), span)
    }

    pub fn _bind_local(self, slot: int, span: Span) -> void {
        self._op_u16(Op.Bind, if slot < 0 { 0 } else { slot }, span)
    }

    pub fn _emit_jump(self, op: Op, span: Span) -> int {
        self._chunk().write_op(op, span)
        let operand = self._chunk().here()
        self._chunk().write_u16(0)
        return operand
    }

    pub fn _patch(self, operand: int) -> void {
        self._patch_from(operand, operand + 2)
    }

    pub fn _patch_from(self, operand: int, base: int) -> void {
        self._chunk().patch_i16(operand, self._chunk().here() - base)
    }

    pub fn _patch_all(self, operands: list[int]) -> void {
        for operand in operands { self._patch(operand) }
    }

    pub fn _loop_back(self, target: int, span: Span) -> void {
        self._chunk().write_op(Op.Loop, span)
        let operand = self._chunk().here()
        self._chunk().write_i16(target - (operand + 2))
    }

    pub fn _unsupported(self, span: Span, what: str) -> void {
        self.bag.error(
            SemaCode.E0902_INTERNAL_ERROR,
            span,
            f"the self-hosted back end cannot lower {what}"
        )
    }

    pub fn _load_name(self, node: Node, name: str) -> void {
        self._load_symbol(self.resolution.symbol_of(node), name, node.span)
    }

    pub fn _load_symbol(self, symbol: Symbol?, name: str, span: Span) -> void {
        if symbol == null {
            self._op_cached(Op.GetGlobal, self._string(name), span)
            return
        }
        if symbol.is_slotted() {
            self._get_local(symbol.slot, span)
        } elif symbol.kind == SymbolKind.Upvalue {
            self._op_u8(Op.GetUpvalue, if symbol.slot < 0 { 0 } else { symbol.slot }, span)
        } elif symbol.kind == SymbolKind.Field {
            self._op_u16(Op.GetLocal, 0, span)
            self._op_cached(Op.GetField, self._string(symbol.name), span)
        } else {
            self._op_cached(Op.GetGlobal, self._string(symbol.name), span)
        }
    }

    pub fn _store_symbol(self, symbol: Symbol?, name: str, span: Span, declaring: bool) -> void {
        if symbol == null or _symbol_is_global_like(symbol) {
            let bound = if symbol == null { name } else { symbol.name }
            if declaring {
                self._op_k(Op.DefGlobal, self._string(bound), span)
            } else {
                self._op_cached(Op.SetGlobal, self._string(bound), span)
                self._op(Op.Pop, span)
            }
            return
        }
        if symbol.is_slotted() {
            self._bind_local(symbol.slot, span)
        } elif symbol.kind == SymbolKind.Upvalue {
            self._op_u8(Op.SetUpvalue, if symbol.slot < 0 { 0 } else { symbol.slot }, span)
            self._op(Op.Pop, span)
        } else {
            self._op_u16(Op.GetLocal, 0, span)
            self._op(Op.Swap, span)
            self._op_cached(Op.SetField, self._string(symbol.name), span)
        }
    }

    pub fn _bind_pattern(self, pattern: Node?, span: Span) -> void {
        if pattern == null {
            self._op(Op.Pop, span)
            return
        }
        let kind = pattern.kind
        if kind == NodeKind.PatWildcard {
            self._op(Op.Pop, span)
        } elif kind == NodeKind.PatBind {
            let symbol = self.resolution.symbol_of(pattern)
            if symbol == null or symbol.kind == SymbolKind.Global {
                let name = if symbol == null { pattern.text("name") } else { symbol.name }
                self._op_k(Op.DefGlobal, self._string(name), span)
            } elif symbol.kind == SymbolKind.Upvalue {
                self._op_u8(Op.SetUpvalue, symbol.slot, span)
                self._op(Op.Pop, span)
            } else {
                self._op_u16(Op.Bind, symbol.slot, span)
            }
        } elif kind == NodeKind.PatTuple or kind == NodeKind.PatList {
            let elements = pattern.child_list("elems")
            let rest = pattern.number("restIndex")
            self._chunk().write_op(Op.Unpack, span)
            self._chunk().write_byte(elements.len())
            self._chunk().write_byte(if rest < 0 { 255 } else { rest })
            self._adjust(elements.len() - 1)
            for element in elements { self._bind_pattern(element, span) }
        } else {
            self._unsupported(pattern.span, "this pattern in a binding position")
            self._op(Op.Pop, span)
        }
    }

    pub fn _invoke_name(self, name: str, argc: int, span: Span) -> void {
        let index = self._string(name)
        let cache = self._chunk().add_cache()
        self._chunk().write_op(Op.Invoke, span)
        self._chunk().write_u24(index)
        self._chunk().write_byte(argc)
        self._chunk().write_u16(cache)
        self._adjust(-argc)
    }

    pub fn _class_spec(
        self,
        kind: int,
        name: str,
        super_name: str?,
        is_abstract: bool,
        required: list[Const],
        variants: list[Const]
    ) -> int {
        var items: list[Const] = [Const(ConstTag.Int, kind), Const(ConstTag.Str, name), if super_name == null {
            Const(ConstTag.Null, null)
        } else {
            Const(ConstTag.Str, super_name)
        }, Const(ConstTag.Bool, is_abstract), Const(ConstTag.Tuple, required), Const(ConstTag.Tuple, variants)]
        return self._chunk().add_constant(Const(ConstTag.Tuple, items))
    }

    pub fn _enter_loop(
        self,
        label: str?,
        continue_target: int,
        base: int,
        continue_depth: int,
        break_depth: int
    ) -> void {
        let ctx = self._ctx()
        ctx.loops.push(LoopFrame(label, continue_target, base, continue_depth, break_depth))
        if base >= 0 and (ctx.deep_close_base < 0 or base < ctx.deep_close_base) {
            ctx.deep_close_base = base
        }
    }

    pub fn _close_loop(self) -> void {
        let frame = self._ctx().loops.pop()
        self._set_depth(frame.break_depth)
        self._patch_all(frame.breaks)
    }

    pub fn _close_scope(self, base: int, span: Span) -> void {
        if base < 0 { return }
        self._op_u16(Op.CloseUpvalue, base, span)
    }

    pub fn _close_iteration(self, base: int, span: Span) -> void {
        self._close_scope(base, span)
    }

    pub fn _enter_close_scope(self, base: int) -> int {
        let ctx = self._ctx()
        let saved = ctx.close_base
        if base >= 0 {
            if ctx.close_base < 0 or base < ctx.close_base { ctx.close_base = base }
            if ctx.deep_close_base < 0 or base < ctx.deep_close_base {
                ctx.deep_close_base = base
            }
        }
        return saved
    }

    pub fn _return_close_base(self) -> int {
        var base = self._ctx().close_base
        let loops = self._close_base_through(0)
        if loops >= 0 and (base < 0 or loops < base) { base = loops }
        return base
    }

    pub fn _can_tail_call(self, value: Node) -> bool {
        if value.kind != NodeKind.Call { return false }
        let ctx = self._ctx()
        if ctx.protect_depth > 0 or ctx.has_defer { return false }
        if ctx.proto.flags & FN_GENERATOR != 0 { return false }
        if ctx.finallys.len() > 0 { return false }

        let callee = value.child("callee")
        if callee == null { return false }
        if callee.kind == NodeKind.Super { return false }
        if callee.kind == NodeKind.Member and callee.require("object").kind == NodeKind.Super {
            return false
        }
        for argument in value.records("args") {
            if argument["isSpread"] { return false }
            if argument["name"] != null { return false }
        }
        return true
    }

    pub fn _run_defers(self, span: Span) -> void {
        if self._ctx().has_defer { self._op(Op.RunDefers, span) }
    }

    pub fn _close_base_through(self, target: int) -> int {
        let loops = self._ctx().loops
        var base = -1
        var index = loops.len() - 1
        while index >= target {
            let candidate = loops[index].capture_base
            if candidate >= 0 and (base < 0 or candidate < base) { base = candidate }
            index -= 1
        }
        return base
    }

    #: The frame slot `for <pattern> in <range>` binds each value into, or -1
    #: when the pattern is not one `OP_FOR_RANGE_BIND` can carry.
    #:
    #: It carries a slot number and nothing else, so a global, an upvalue and a
    #: destructuring pattern all fall back to the general loop — a range never
    #: yields anything unpackable anyway, so the tuple case is an error program
    #: either way and belongs on the path that reports it.
    #:
    #: A wildcard has no symbol and needs a destination nothing will read, so it
    #: takes a temporary — but only when `allocate` says this call is the one
    #: about to emit the loop, not the one deciding whether to.
    pub fn _range_bind_slot(self, pattern: Node?, allocate: bool) -> int {
        if pattern == null { return -1 }
        if pattern.kind == NodeKind.PatWildcard {
            return if allocate { self._alloc_temp() } else { 0 }
        }
        if pattern.kind != NodeKind.PatBind { return -1 }
        let symbol = self.resolution.symbol_of(pattern)
        if symbol == null or not symbol.is_slotted() { return -1 }
        return symbol.slot
    }

    pub fn _reserve_jump(self) -> int {
        let operand = self._chunk().here()
        self._chunk().write_u16(0)
        return operand
    }

    pub fn _literal_constant(self, node: Node?) -> int {
        if node == null { return self._chunk().add_constant(Const(ConstTag.Null, null)) }
        match node.kind {
            NodeKind.IntLit => {
                return self._chunk().add_constant(Const(ConstTag.Int, node.number("intLit")))
            }
            NodeKind.FloatLit => {
                return self._chunk().add_constant(Const(ConstTag.Float, node.get("floatLit")))
            }
            NodeKind.StrLit => { return self._string(node.text("chars")) }
            NodeKind.BoolLit => {
                return self._chunk().add_constant(Const(ConstTag.Bool, node.flag("boolLit")))
            }
            NodeKind.NullLit => { return self._chunk().add_constant(Const(ConstTag.Null, null)) }
            _ => {
                self._unsupported(node.span, "a non-literal pattern constant")
                return self._chunk().add_constant(Const(ConstTag.Null, null))
            }
        }
    }

    pub fn _variant_tag(self, pattern: Node) -> int {
        let name = pattern.opt_text("variantName")
        let owner = pattern.opt_text("typeName")
        if name == null or owner == null { return -1 }
        let decl = self.enums.get(owner, null)
        if decl == null { return -1 }
        for (tag, variant) in decl.records("variants").enumerate() {
            if variant["name"] == name { return tag }
        }
        return -1
    }
}

fn _unwired_stmt(e: Emitter, node: Node?) -> void {
    throw ValueError("the emitter seam `lower_stmt` was never wired")
}

fn _unwired_match(e: Emitter, node: Node, as_expression: bool) -> void {
    throw ValueError("the emitter seam `lower_match` was never wired")
}

fn _unwired_closure(e: Emitter, node: Node, name: str?, extra_flags: int) -> int {
    throw ValueError("the emitter seam `lower_closure` was never wired")
}

var _hook_stmt: fn(Emitter, Node?) -> void = _unwired_stmt
var _hook_match: fn(Emitter, Node, bool) -> void = _unwired_match
var _hook_closure: fn(Emitter, Node, str?, int) -> int = _unwired_closure

#: `stmt.jai`'s `_stmt`, for every file that lowers a nested statement: a
#: guarded body, a loop body, a catch body, a match arm, a function body, and
#: the copy of a `finally` an explicit exit inlines.
pub fn lower_stmt(e: Emitter, node: Node?) -> void {
    return _hook_stmt(e, node)
}

#: `pattern.jai`'s `_match`, for `match` in expression position in `expr.jai`.
pub fn lower_match(e: Emitter, node: Node, as_expression: bool) -> void {
    return _hook_match(e, node, as_expression)
}

#: `function.jai`'s `_closure`, for a lambda or an anonymous `fn` in `expr.jai`.
pub fn lower_closure(e: Emitter, node: Node, name: str?, extra_flags: int) -> int {
    return _hook_closure(e, node, name, extra_flags)
}

#: Point every seam at the production that implements it. `wire.jai` is the
#: only caller, and it calls this as it is imported.
pub fn install_hooks(
    stmt: fn(Emitter, Node?) -> void,
    match_form: fn(Emitter, Node, bool) -> void,
    closure: fn(Emitter, Node, str?, int) -> int
) -> void {
    _hook_stmt = stmt
    _hook_match = match_form
    _hook_closure = closure
}
