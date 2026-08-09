#: Print one file's syntax tree the way `jaithon ast` does, using the
#: self-hosted printer. `scripts/ast_parity.sh` diffs this against the C's.
from jaithon.ast import to_sexp
from jaithon.compile.lexer import DiagnosticBag
from jaithon.compile.parser import parse
from std.io import read_file
from std.os import argv, exit

fn main() -> int {
    let args = argv()
    if args.len() < 2 {
        print("usage: ast_sexp_dump.jai FILE")
        return 2
    }
    let path = args[1]
    let program = parse(read_file(path), 0, DiagnosticBag())
    if program is null { return 1 }
    print(to_sexp(program))
    return 0
}
