#: Every source file in the tree, parsed, formatted, and parsed again.
#:
#: `fmt --check` asks whether a file is already canonical. This asks the
#: question underneath it: whether formatting a file changes the PROGRAM. The
#: two are different, and only this one catches a printer that quietly drops
#: something. Two such bugs were shipped: the printer never wrote decorators,
#: so every `jaithon fmt` deleted `@trace` and `@gpu_kernel` and turned
#: jaitensor's hot paths back into ordinary functions; and it indented the
#: lines inside a triple-quoted string, so a formatted program's data was four
#: characters a line longer than the one that was written.
#:
#: The comparison is the s-expression form of the tree, which is why
#: `ast_encode` has to print everything that changes what a program does -- a
#: field it leaves out is a field this cannot see.
#:
#: The compact printer is checked the same way. `ast_unparse.to_source` is what
#: `jaithon ast` and the REPL print with, and it is a different body of code
#: from the formatter: it lost decorators too, and separately. Writing a tree
#: out and reading it back must give the same tree.
#:
#: The JSON encoding is NOT checked here, though it is the same class of
#: question. Encoding and decoding four hundred trees is thirty seconds on its
#: own -- more than everything else in this file put together -- and
#: `test_ast_encode` already round-trips the shapes it carries. Run it by hand
#: if the encoder changes.
#:
#: Files that do not parse are counted and skipped: the error-message tests are
#: meant not to parse, and this is not the gate for them.
from jaithon.ast_encode import to_sexp
from jaithon.ast_unparse import to_source
from jaithon.compile.diag import DiagnosticBag
from jaithon.ast import Node
from jaithon.compile.parser import parse
from jaithon.tool.fmt import format_source
from std.io import Path, read_file

let ROOTS = ["lib", "tests", "packages", "examples", "scripts"]

fn parse_of(source: str) -> Node? {
    let bag = DiagnosticBag()
    let node = parse(source, 0, bag)
    if node is null or bag.has_errors() { return null }
    return node
}

fn tree_of(source: str) -> str? {
    let node = parse_of(source)
    if node is null { return null }
    return to_sexp(node)
}

fn main() -> int {
    var checked = 0
    var changed = 0
    var skipped = 0
    for root in ROOTS {
        if not Path(root).is_dir() { continue }
        for entry in Path(root).walk() {
            let name = entry.to_str()
            if not name.ends_with(".jai") { continue }
            if name.contains("__jaicache__") { continue }
            let source = read_file(name)
            let parsed = parse_of(source)
            if parsed is null {
                skipped += 1
                continue
            }
            let before = to_sexp(parsed)
            let printed = tree_of(to_source(parsed))
            if printed is null or printed != before {
                changed += 1
                print(f"UNPARSE LOST       {name}")
                continue
            }
            var formatted = source
            try {
                formatted = format_source(source, name)
            } catch e: Error {
                print(f"FORMATTING FAILED  {name}: {e}")
                changed += 1
                continue
            }
            let after = tree_of(formatted)
            if after is null {
                print(f"UNPARSEABLE AFTER  {name}")
                changed += 1
                continue
            }
            checked += 1
            if before == after { continue }
            changed += 1
            print(f"AST CHANGED        {name}")
        }
    }
    if changed > 0 {
        print(f"fmt round trip FAILED: {changed} of {checked + changed} files changed")
        return 1
    }
    print(f"fmt round trip: {checked} files unchanged, {skipped} did not parse")
    return 0
}
