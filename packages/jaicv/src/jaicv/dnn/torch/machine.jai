#: The restricted pickle machine and the opcodes it runs.
from .bits import be_u32, f64_from_halves, le_int, le_u16, le_u32, le_u64
from .bits import STORAGE_DTYPES
from .pickle import Checkpoint, PickleGlobal, PickleTuple
from .rebuild import allowed_global, rebuild_tensor, refuse_global, tuple_items
from .tensor import TorchTensor

const OP_MARK = 0x28
const OP_EMPTY_TUPLE = 0x29
const OP_STOP = 0x2E
const OP_POP = 0x30
const OP_POP_MARK = 0x31
const OP_BINBYTES = 0x42
const OP_SHORT_BINBYTES = 0x43
const OP_BINFLOAT = 0x47
const OP_BININT = 0x4A
const OP_BININT1 = 0x4B
const OP_BININT2 = 0x4D
const OP_NONE = 0x4E
const OP_PERSID = 0x50
const OP_BINPERSID = 0x51
const OP_REDUCE = 0x52
const OP_BINUNICODE = 0x58
const OP_EMPTY_LIST = 0x5D
const OP_APPEND = 0x61
const OP_BUILD = 0x62
const OP_GLOBAL = 0x63
const OP_DICT = 0x64
const OP_APPENDS = 0x65
const OP_BINGET = 0x68
const OP_LONG_BINGET = 0x6A
const OP_LIST = 0x6C
const OP_BINPUT = 0x71
const OP_LONG_BINPUT = 0x72
const OP_SETITEM = 0x73
const OP_TUPLE = 0x74
const OP_SETITEMS = 0x75
const OP_EMPTY_DICT = 0x7D
const OP_PROTO = 0x80
const OP_NEWOBJ = 0x81
const OP_TUPLE1 = 0x85
const OP_TUPLE2 = 0x86
const OP_TUPLE3 = 0x87
const OP_NEWTRUE = 0x88
const OP_NEWFALSE = 0x89
const OP_LONG1 = 0x8A
const OP_LONG4 = 0x8B
const OP_SHORT_BINUNICODE = 0x8C
const OP_BINUNICODE8 = 0x8D
const OP_BINBYTES8 = 0x8E
const OP_EMPTY_SET = 0x8F
const OP_ADDITEMS = 0x90
const OP_FROZENSET = 0x91
const OP_NEWOBJ_EX = 0x92
const OP_STACK_GLOBAL = 0x93
const OP_MEMOIZE = 0x94
const OP_FRAME = 0x95

#: The restricted pickle machine.
#:
#: It is a stack machine like the real one, with the same memo and the same
#: marks, and it stops at the same `STOP`. What it does not have is a way to
#: reach a name the whitelist did not allow, and that is the only difference
#: that matters.
pub class Unpickler {
    pub let data: bytes
    pub let source: Checkpoint
    pub var at: int
    pub var stack: list[any]
    pub var marks: list[int]
    pub var memo: dict[int, any]

    pub fn init(self, data: bytes, source: Checkpoint) {
        self.data = data
        self.source = source
        self.at = 0
        self.stack = []
        self.marks = []
        self.memo = {}
    }

    #: Interpret the stream and return whatever `STOP` left on the stack.
    pub fn run(self) -> any {
        loop {
            if self.at >= self.data.len() {
                throw ValueError("the pickle ended before its STOP")
            }
            let opcode = self.data[self.at]
            self.at += 1
            if opcode == OP_STOP { return self._pop() }
            self._step(opcode)
        }
    }

    fn _step(self, opcode: int) -> void {
        if opcode == OP_BINPUT {
            self.memo[self._take_u8()] = self._peek()
            return
        }
        if opcode == OP_BINGET {
            self.stack.push(self._recall(self._take_u8()))
            return
        }
        if opcode == OP_BININT1 {
            self.stack.push(self._take_u8())
            return
        }
        if opcode == OP_BINUNICODE {
            self.stack.push(self._take_text(self._take_u32()))
            return
        }
        if opcode == OP_MARK {
            self.marks.push(self.stack.len())
            return
        }
        if opcode == OP_TUPLE {
            self.stack.push(PickleTuple(self._pop_to_mark()))
            return
        }
        if opcode == OP_REDUCE {
            self._reduce()
            return
        }
        if opcode == OP_BINPERSID {
            self.stack.push(self._resolve_persistent_id(self._pop()))
            return
        }
        if opcode == OP_TUPLE1 {
            self._tuple_of(1)
            return
        }
        if opcode == OP_TUPLE2 {
            self._tuple_of(2)
            return
        }
        if opcode == OP_TUPLE3 {
            self._tuple_of(3)
            return
        }
        if opcode == OP_EMPTY_DICT {
            self.stack.push(self._new_dict())
            return
        }
        if opcode == OP_EMPTY_LIST {
            self.stack.push(self._new_list())
            return
        }
        if opcode == OP_EMPTY_TUPLE {
            self.stack.push(PickleTuple([]))
            return
        }
        if opcode == OP_NEWFALSE {
            self.stack.push(false)
            return
        }
        if opcode == OP_NEWTRUE {
            self.stack.push(true)
            return
        }
        if opcode == OP_NONE {
            self.stack.push(null)
            return
        }
        if opcode == OP_SETITEMS {
            self._set_items(self._pop_to_mark())
            return
        }
        if opcode == OP_SETITEM {
            let value = self._pop()
            let key = self._pop()
            self._set_items([key, value])
            return
        }
        if opcode == OP_APPENDS {
            self._append_all(self._pop_to_mark())
            return
        }
        if opcode == OP_APPEND {
            self._append_all([self._pop()])
            return
        }
        if opcode == OP_GLOBAL {
            self._global(self._take_line(), self._take_line())
            return
        }
        if opcode == OP_STACK_GLOBAL {
            let name = self._pop()
            let module_name = self._pop()
            if not name is str or not module_name is str {
                throw ValueError("STACK_GLOBAL was not handed two strings")
            }
            self._global(module_name, name)
            return
        }
        if opcode == OP_MEMOIZE {
            self.memo[self.memo.len()] = self._peek()
            return
        }
        if opcode == OP_LONG_BINPUT {
            self.memo[self._take_u32()] = self._peek()
            return
        }
        if opcode == OP_LONG_BINGET {
            self.stack.push(self._recall(self._take_u32()))
            return
        }
        if opcode == OP_BININT2 {
            self.stack.push(self._take_u16())
            return
        }
        if opcode == OP_BININT {
            self.stack.push(self._take_i32())
            return
        }
        if opcode == OP_SHORT_BINUNICODE {
            self.stack.push(self._take_text(self._take_u8()))
            return
        }
        if opcode == OP_BINUNICODE8 {
            self.stack.push(self._take_text(self._take_u64()))
            return
        }
        if opcode == OP_SHORT_BINBYTES {
            self.stack.push(self._take_bytes(self._take_u8()))
            return
        }
        if opcode == OP_BINBYTES {
            self.stack.push(self._take_bytes(self._take_u32()))
            return
        }
        if opcode == OP_BINBYTES8 {
            self.stack.push(self._take_bytes(self._take_u64()))
            return
        }
        if opcode == OP_BINFLOAT {
            #: Alone among the binary opcodes, this one is big-endian.
            let at = self._advance(8)
            self.stack.push(f64_from_halves(be_u32(self.data, at), be_u32(self.data, at + 4)))
            return
        }
        if opcode == OP_LONG1 {
            self._long(self._take_u8())
            return
        }
        if opcode == OP_LONG4 {
            self._long(self._take_i32())
            return
        }
        if opcode == OP_DICT {
            self._dict_from_mark()
            return
        }
        if opcode == OP_LIST {
            let items = self._pop_to_mark()
            var made = self._new_list()
            for item in items { made.push(item) }
            self.stack.push(made)
            return
        }
        if opcode == OP_EMPTY_SET {
            self.stack.push(set([]))
            return
        }
        if opcode == OP_FROZENSET {
            self.stack.push(set(self._pop_to_mark()))
            return
        }
        if opcode == OP_ADDITEMS {
            #: Sets never carry weights, so the items are taken and dropped
            #: rather than folded into a set that nothing downstream reads.
            let _items = self._pop_to_mark()
            return
        }
        if opcode == OP_BUILD {
            self._build()
            return
        }
        if opcode == OP_PROTO {
            let version = self._take_u8()
            if version > 5 {
                throw ValueError(
                    f"this pickle is protocol {version}, and 5 is the highest there is"
                )
            }
            return
        }
        if opcode == OP_FRAME {
            let _length = self._take_u64()
            return
        }
        if opcode == OP_POP {
            let _dropped = self._pop()
            return
        }
        if opcode == OP_POP_MARK {
            let _dropped = self._pop_to_mark()
            return
        }
        if opcode == OP_PERSID {
            self.stack.push(self._resolve_persistent_id(self._take_line()))
            return
        }
        if opcode == OP_NEWOBJ or opcode == OP_NEWOBJ_EX {
            let _arguments = self._pop()
            if opcode == OP_NEWOBJ_EX { let _keywords = self._pop() }
            let target = self._pop()
            var named = "an object"
            if target is PickleGlobal {
                let global_target: PickleGlobal = target
                named = global_target.full_name()
            }
            throw ValueError(
                f"this pickle constructs {named} directly, which this reader will not do; " +
                    "only tensors and containers are rebuilt"
            )
        }
        throw ValueError(
            f"pickle opcode 0x{opcode} at byte {self.at - 1} is one this reader does not implement"
        )
    }

    fn _advance(self, count: int) -> int {
        if self.at + count > self.data.len() {
            throw ValueError("the pickle ends in the middle of an operand")
        }
        let begin = self.at
        self.at += count
        return begin
    }

    fn _take_u8(self) -> int { return self.data[self._advance(1)] }

    fn _take_u16(self) -> int { return le_u16(self.data, self._advance(2)) }

    fn _take_u32(self) -> int { return le_u32(self.data, self._advance(4)) }

    fn _take_i32(self) -> int { return le_int(self.data, self._advance(4), 4, true) }

    fn _take_u64(self) -> int { return le_u64(self.data, self._advance(8)) }

    fn _take_bytes(self, length: int) -> bytes {
        let begin = self._advance(length)
        return self.data.slice(begin, begin + length)
    }

    fn _take_text(self, length: int) -> str { return self._take_bytes(length).decode() }

    #: Everything up to the next newline, which is how the text opcodes end.
    fn _take_line(self) -> str {
        var end = self.at
        while end < self.data.len() and self.data[end] != 0x0A { end += 1 }
        if end >= self.data.len() {
            throw ValueError("the pickle ends in the middle of a line-terminated operand")
        }
        let text = self.data.slice(self.at, end).decode()
        self.at = end + 1
        return text
    }

    #: `LONG1` and `LONG4`: a little-endian two's complement integer of any
    #: width. Past eight bytes there is nothing here to hold it, and saying so
    #: is better than folding it to something wrong.
    fn _long(self, length: int) -> void {
        if length > 8 {
            throw ValueError(
                f"this pickle holds a {length}-byte integer, which is wider than an int"
            )
        }
        self.stack.push(le_int(self.data, self._advance(length), length, true))
    }

    fn _new_dict(self) -> dict[any, any] {
        var made: dict[any, any] = {}
        return made
    }

    fn _new_list(self) -> list[any] {
        var made: list[any] = []
        return made
    }

    fn _peek(self) -> any {
        if self.stack.len() == 0 { throw ValueError("the pickle read an empty stack") }
        return self.stack[self.stack.len() - 1]
    }

    fn _pop(self) -> any {
        if self.stack.len() == 0 { throw ValueError("the pickle popped an empty stack") }
        return self.stack.pop()
    }

    fn _recall(self, slot: int) -> any {
        if not self.memo.contains(slot) {
            throw ValueError(f"the pickle read memo slot {slot}, which nothing has filled")
        }
        return self.memo[slot]
    }

    fn _pop_to_mark(self) -> list[any] {
        if self.marks.len() == 0 { throw ValueError("the pickle closed a mark it never opened") }
        let floor = self.marks.pop()
        if floor > self.stack.len() { throw ValueError("the pickle's stack fell below its mark") }
        var taken: list[any] = []
        for index in floor..self.stack.len() { taken.push(self.stack[index]) }
        while self.stack.len() > floor { let _dropped = self.stack.pop() }
        return taken
    }

    fn _tuple_of(self, count: int) -> void {
        var items: list[any] = []
        for _index in 0..count { items.push(self._pop()) }
        var ordered: list[any] = []
        var index = items.len() - 1
        while index >= 0 {
            ordered.push(items[index])
            index -= 1
        }
        self.stack.push(PickleTuple(ordered))
    }

    fn _dict_from_mark(self) -> void {
        let flat = self._pop_to_mark()
        var made = self._new_dict()
        self.stack.push(made)
        self._set_items(flat)
    }

    fn _set_items(self, flat: list[any]) -> void {
        if flat.len() % 2 != 0 {
            throw ValueError("the pickle set an odd number of dictionary halves")
        }
        let target = self._peek()
        if not target is dict {
            throw ValueError("the pickle set items on something that is not a dictionary")
        }
        var into: dict[any, any] = target
        var index = 0
        while index < flat.len() {
            into[flat[index]] = flat[index + 1]
            index += 2
        }
    }

    fn _append_all(self, items: list[any]) -> void {
        let target = self._peek()
        if not target is list {
            throw ValueError("the pickle appended to something that is not a list")
        }
        var into: list[any] = target
        for item in items { into.push(item) }
    }

    fn _global(self, module_name: str, name: str) -> void {
        if not allowed_global(module_name, name) { refuse_global(module_name, name) }
        self.stack.push(PickleGlobal(module_name, name))
    }

    #: `BUILD` restores an object's state, which for pickle means setting its
    #: attributes -- not its items.
    #:
    #: The one thing here that ever carries state is the `OrderedDict` a state
    #: dict is, and the state it carries is its `_metadata` attribute, holding
    #: the module versions torch writes beside the weights. Those are
    #: attributes of the mapping and not entries in it, so folding them in
    #: would invent keys the checkpoint does not have. There is nowhere here to
    #: keep an attribute, so they are dropped; anything else with state is a
    #: real object, and rebuilding one of those is what this reader will not do.
    fn _build(self) -> void {
        let _state = self._pop()
        if self._peek() is dict { return }
        throw ValueError("this pickle restores object state this reader cannot rebuild")
    }

    fn _reduce(self) -> void {
        let arguments = self._pop()
        let target = self._pop()
        if not target is PickleGlobal {
            throw ValueError("the pickle called something that is not a whitelisted global")
        }
        let called: PickleGlobal = target
        let items = tuple_items(arguments)
        if items is null {
            throw ValueError(f"{called.full_name()} was called with something that is not a tuple")
        }
        let args: list[any] = items
        self.stack.push(self._apply(called, args))
    }

    fn _apply(self, called: PickleGlobal, args: list[any]) -> any {
        let full = called.full_name()
        if full == "torch._utils._rebuild_tensor_v2" or full == "torch._utils._rebuild_tensor" {
            return rebuild_tensor(args, full)
        }
        if full == "torch._utils._rebuild_parameter" {
            if args.len() == 0 or not args[0] is TorchTensor {
                throw ValueError("_rebuild_parameter was not handed a tensor to wrap")
            }
            return args[0]
        }
        if full == "collections.OrderedDict" or
            full == "builtins.dict" or
            full == "__builtin__.dict" {
            if args.len() != 0 {
                throw ValueError(
                    f"{full} was called with arguments, which this reader cannot apply"
                )
            }
            return self._new_dict()
        }
        if full == "builtins.list" or full == "__builtin__.list" {
            if args.len() == 0 { return self._new_list() }
            return self._sequence_of(args[0], full)
        }
        if full == "builtins.tuple" or full == "__builtin__.tuple" {
            if args.len() == 0 { return PickleTuple([]) }
            return PickleTuple(self._sequence_of(args[0], full))
        }
        if full == "builtins.set" or
            full == "builtins.frozenset" or
            full == "__builtin__.set" or
            full == "__builtin__.frozenset" {
            if args.len() == 0 { return set([]) }
            return set(self._sequence_of(args[0], full))
        }
        throw ValueError(f"{full} is on the whitelist as a value, and this pickle calls it")
    }

    fn _sequence_of(self, value: any, full: str) -> list[any] {
        let items = tuple_items(value)
        if items is not null {
            let taken: list[any] = items
            return taken
        }
        if value is list {
            let listed: list[any] = value
            return listed
        }
        throw ValueError(f"{full} was handed something that is not a sequence")
    }

    #: Resolve a persistent id, which for torch is always a storage.
    #:
    #: The id is the tuple `("storage", storage_type, key, location, numel)`,
    #: and it stands in for the archive member `data/<key>`. Nothing else is a
    #: persistent id in a checkpoint, so nothing else is accepted.
    fn _resolve_persistent_id(self, ident: any) -> any {
        let items = tuple_items(ident)
        if items is null {
            throw ValueError("this checkpoint's persistent ids are not the tuples torch writes")
        }
        let parts: list[any] = items
        if parts.len() < 5 or not parts[0] is str {
            throw ValueError("this checkpoint's persistent ids are not the tuples torch writes")
        }
        let kind: str = parts[0]
        if kind != "storage" {
            throw ValueError(
                f"this checkpoint holds a '{kind}' persistent id, and only storages are read"
            )
        }
        if not parts[1] is PickleGlobal {
            throw ValueError("a persistent id names a storage class this reader did not resolve")
        }
        let storage_class: PickleGlobal = parts[1]
        if storage_class.module_name != "torch" or not STORAGE_DTYPES.contains(storage_class.name) {
            throw ValueError(
                f"a persistent id names {storage_class.full_name()}, which is not a torch storage class"
            )
        }
        var key = ""
        if parts[2] is str {
            key = parts[2]
        } elif parts[2] is int {
            key = str(parts[2])
        } else {
            throw ValueError(
                "a persistent id names a storage key that is neither text nor a number"
            )
        }
        if not parts[4] is int {
            throw ValueError(f"the persistent id for storage '{key}' has no element count")
        }
        let count: int = parts[4]
        return self.source.storage(key, STORAGE_DTYPES[storage_class.name], count)
    }
}
