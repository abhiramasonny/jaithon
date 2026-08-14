#: The `jaithon` subcommands, written in Jaithon.
#:
#: Facade: imports and re-exports every sibling module in this directory.
#:
#: They live beside `std` rather than inside it, because no user program
#: imports them — a language whose tools cannot be written in it has not
#: finished being a language.
#:
#: ```
#: command_names()                 # ["bench", "doc", "fmt", "test"]
#: is_command("fmt")               # true
#: is_command("compile")           # false
#: ```
#:
#: ```text
#: jaithon run lib/jaithon/tool/mod.jai fmt --check lib
#: ```
from std.io import print_err

from .bench import main as bench_main
from .doc import main as doc_main
from .fmt import main as fmt_main
from .test import main as test_main

export { bench_main, doc_main, fmt_main, test_main }

let _COMMANDS: dict[str, fn(list[str]) -> int] = {
    "bench": bench_main,
    "doc": doc_main,
    "fmt": fmt_main,
    "test": test_main,
}

#: Return the subcommand names, in alphabetical order.
#:
#: ```
#: command_names()    # ["bench", "doc", "fmt", "test"]
#: ```
pub fn command_names() -> list[str] {
    return _COMMANDS.keys().sorted()
}

#: Report whether `name` is one of the subcommands.
#:
#: ```
#: is_command("doc")     # true
#: is_command("docs")    # false
#: ```
pub fn is_command(name: str) -> bool {
    return name in _COMMANDS
}

#: Run the subcommand named by `name` with `args`.
#:
#: `args` is passed through unchanged (name included) — each `main` expects
#: the CLI's full argv. Raises `KeyError` for an unknown name; check
#: `is_command` first if `name` came from a user.
pub fn run(name: str, args: list[str]) -> int {
    if name not in _COMMANDS {
        throw KeyError(f"no such subcommand: {name}, expected one of " + ", ".join(command_names()))
    }
    return _COMMANDS[name](args)
}

#: Dispatch on `args[1]`, so that this module is itself a runnable tool.
#:
#: ```text
#: jaithon run lib/jaithon/tool/mod.jai doc --out docs/api lib
#: ```
pub fn main(args: list[str]) -> int {
    if args.len() < 2 {
        print_err("usage: jaithon <" + " | ".join(command_names()) + "> [options] [paths...]")
        return 2
    }
    let name = args[1]
    if not is_command(name) {
        print_err(f"jaithon: no such subcommand: {name}")
        print_err("expected one of " + ", ".join(command_names()))
        return 2
    }
    return run(name, args[1:args.len()])
}
