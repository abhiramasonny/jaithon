#: The `jaithon doc` command line: one page per module, an index, and `search.json`.
from jaithon.tool.args import Arguments
from jaithon.tool.fmt import tail_arguments
from std.io import Path, print_err
from .collect import collect_all
from .html import render_html
from .index import render_index, search_index
from .markdown import render_markdown
from .model import _by_name, coverage

#: The default directory `jaithon doc` writes into.
pub const DEFAULT_OUTPUT: str = "docs/api"

#: Run `jaithon doc`.
#:
#: ```text
#: jaithon doc [--out dir] [--html] [paths...]
#: ```
#:
#: Writes one page per module, an index, and `search.json`. Exits 1 when no
#: module could be read.
pub fn main(args: list[str]) -> int {
    let options = _Options.parse(tail_arguments(args, "doc"))
    if options.usage_error is not null {
        print_err(f"jaithon doc: {options.usage_error}")
        print_err("usage: jaithon doc [--out dir] [--html] [paths...]")
        return 2
    }

    let modules = _by_name(collect_all(options.paths))
    if modules.len() == 0 {
        print_err("jaithon doc: nothing to document")
        return 1
    }

    let directory = Path(options.output)
    directory.mkdir(true, true)
    let extension = if options.html { "html" } else { "md" }
    for mod in modules {
        let page = if options.html {
            render_html(mod, modules)
        } else {
            render_markdown(mod, modules)
        }
        directory.join(mod.page(extension)).write_text(page)
    }
    directory.join("index." + extension).write_text(render_index(modules, options.html))
    directory.join("search.json").write_text(search_index(modules, extension))

    let (documented, total) = coverage(modules)
    let missing = total - documented
    print(
        f"{modules.len()} modules, {total} public declarations, " +
            f"{missing} undocumented -> {options.output}"
    )
    return 0
}

class _Options {
    pub var output: str
    pub var html: bool
    pub var paths: list[str]
    pub var usage_error: str?

    pub fn init(self) {
        self.output = DEFAULT_OUTPUT
        self.html = false
        self.paths = []
        self.usage_error = null
    }

    pub static fn parse(args: list[str]) -> _Options {
        let options = _Options()
        let reader = Arguments(args)
        loop {
            let argument = reader.next()
            if argument is null { break }
            if not argument.is_flag() {
                options.paths.push(argument.text)
                continue
            }
            if argument.name == "--out" {
                let value = reader.value_of(argument)
                if value is null { break }
                options.output = value
            } elif argument.name == "--html" {
                if reader.is_bare(argument) { options.html = true }
            } elif argument.name == "--markdown" {
                if reader.is_bare(argument) { options.html = false }
            } else {
                reader.unknown(argument)
            }
        }
        options.usage_error = reader.error
        if options.paths.len() == 0 { options.paths = ["lib"] }
        return options
    }
}
