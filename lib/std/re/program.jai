#: Lower a parse tree to a Thompson NFA program: the instruction set and the
#: compiler that emits it.
from .parse import CharClass, Node
from .parse import NODE_ALTERNATE, NODE_ASSERT, NODE_CHAR, NODE_CONCAT
from .parse import NODE_EMPTY, NODE_GROUP

const MAX_PROGRAM = 20000

pub const OP_CHAR = 0
pub const OP_SPLIT = 1
pub const OP_JUMP = 2
pub const OP_SAVE = 3
pub const OP_ASSERT = 4
pub const OP_MATCH = 5

#: One NFA instruction. `x` and `y` are program counters, except in `OP_SAVE`
#: where `x` is a capture slot and in `OP_ASSERT` where it is an assertion
#: kind.
pub class Instruction {
    pub var op: int
    pub var x: int
    pub var y: int
    pub var char_set: CharClass?

    pub fn init(self, op: int, x: int = 0, y: int = 0, char_set: CharClass? = null) {
        self.op = op
        self.x = x
        self.y = y
        self.char_set = char_set
    }
}

#: Compile a parse tree into an NFA program.
#:
#: `Save 0` and `Save 1` around the whole tree record the span of the match
#: itself, so group 0 needs no special case anywhere else.
pub fn _compile(tree: Node) -> list[Instruction] {
    let program: list[Instruction] = []
    _emit(program, Instruction(OP_SAVE, 0))
    _compile_node(program, tree)
    _emit(program, Instruction(OP_SAVE, 1))
    _emit(program, Instruction(OP_MATCH))
    return program
}

fn _emit(program: list[Instruction], instruction: Instruction) -> int {
    if program.len() >= MAX_PROGRAM {
        throw ValueError(
            f"pattern expands past {MAX_PROGRAM} instructions; lower its repeat counts"
        )
    }
    program.push(instruction)
    return program.len() - 1
}

fn _compile_node(program: list[Instruction], node: Node) -> void {
    if node.kind == NODE_EMPTY {
        return
    }
    if node.kind == NODE_CHAR {
        _emit(program, Instruction(OP_CHAR, 0, 0, node.char_set))
    } elif node.kind == NODE_ASSERT {
        _emit(program, Instruction(OP_ASSERT, node.assertion))
    } elif node.kind == NODE_CONCAT {
        for child in node.children { _compile_node(program, child) }
    } elif node.kind == NODE_GROUP {
        if node.index < 0 {
            _compile_node(program, node.children[0])
        } else {
            _emit(program, Instruction(OP_SAVE, 2 * node.index))
            _compile_node(program, node.children[0])
            _emit(program, Instruction(OP_SAVE, 2 * node.index + 1))
        }
    } elif node.kind == NODE_ALTERNATE {
        _compile_alternate(program, node)
    } else {
        _compile_repeat(program, node)
    }
}

#: Chain the branches: each one but the last is guarded by a `Split` that
#: prefers it and falls through to the next, and ends with a `Jump` past them
#: all. Preferring the earlier branch is what makes alternation ordered.
fn _compile_alternate(program: list[Instruction], node: Node) -> void {
    let jumps: list[int] = []
    let last = node.children.len() - 1
    for (position, branch) in node.children.enumerate() {
        if position == last {
            _compile_node(program, branch)
        } else {
            let split = _emit(program, Instruction(OP_SPLIT))
            program[split].x = program.len()
            _compile_node(program, branch)
            jumps.push(_emit(program, Instruction(OP_JUMP)))
            program[split].y = program.len()
        }
    }
    for jump in jumps { program[jump].x = program.len() }
}

#: Expand a repeat into copies of its body.
#:
#: `a{2,4}` becomes `aa` and then two optional copies, and `a{2,}` becomes `a`
#: followed by `a+`. Expansion is why a bound is capped: the program, and so
#: the cost of every match, grows with it.
fn _compile_repeat(program: list[Instruction], node: Node) -> void {
    let body = node.children[0]
    if node.high < 0 {
        let required = node.low > 0 ? node.low - 1 : 0
        for _ in 0..required { _compile_node(program, body) }
        _compile_loop(program, body, node.lazy, node.low == 0)
        return
    }
    for _ in 0..node.low { _compile_node(program, body) }
    let splits: list[int] = []
    let starts: list[int] = []
    for _ in 0..node.high - node.low {
        splits.push(_emit(program, Instruction(OP_SPLIT)))
        starts.push(program.len())
        _compile_node(program, body)
    }
    let end = program.len()
    for (position, split) in splits.enumerate() {
        _patch(program, split, starts[position], end, node.lazy)
    }
}

#: Emit `body*` when `optional`, `body+` otherwise.
#:
#: A star tests before the body and jumps back after it; a plus falls into the
#: body and tests after it. Either way the `Split` prefers the body when greedy
#: and the exit when lazy — that one bit is the whole difference between `.*`
#: and `.*?`.
fn _compile_loop(program: list[Instruction], body: Node, lazy: bool, optional: bool) -> void {
    if optional {
        let split = _emit(program, Instruction(OP_SPLIT))
        let start = program.len()
        _compile_node(program, body)
        _emit(program, Instruction(OP_JUMP, split))
        _patch(program, split, start, program.len(), lazy)
        return
    }
    let start = program.len()
    _compile_node(program, body)
    let split = _emit(program, Instruction(OP_SPLIT))
    _patch(program, split, start, program.len(), lazy)
}

fn _patch(program: list[Instruction], split: int, preferred: int, other: int, lazy: bool) -> void {
    program[split].x = lazy ? other : preferred
    program[split].y = lazy ? preferred : other
}
