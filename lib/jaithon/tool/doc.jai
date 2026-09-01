#: The documentation generator behind `jaithon doc`.
#:
#: Facade: imports and re-exports every sibling module in `doc/`.
#:
#: Documentation is the `#:` comments and the declarations they sit above —
#: there is no separate manual to fall out of date. Only `pub` declarations
#: are documented (`spec/STYLE.md` §4); an undocumented one is still listed,
#: since hiding it would hide the hole. Fenced examples are executed by
#: `jaithon test --doc`, so pages can't drift from the code they document.
#:
#: ```
#: module_name_of("lib/std/algo/sort.jai")    # "std.algo.sort"
#: module_name_of("lib/std/ds/mod.jai")       # "std.ds"
#: ```
from .doc.collect import collect_all, collect_module, module_name_of
from .doc.driver import DEFAULT_OUTPUT, main
from .doc.html import render_html
from .doc.index import render_index, search_index
from .doc.markdown import render_markdown
from .doc.model import DocComment, Entry, Example, ModuleDoc, build_links, coverage

export { collect_all, collect_module, module_name_of }
export { DEFAULT_OUTPUT, main }
export { render_html, render_index, search_index, render_markdown }
export { DocComment, Entry, Example, ModuleDoc, build_links, coverage }
