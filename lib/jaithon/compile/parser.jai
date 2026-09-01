#: The Jaithon parser: tokens in, a `jaithon.ast` tree out.
#:
#: ```
#: parse("let x = 1\n") != null      # true
#: parse("let x =\n") == null        # true: a file with an error has no tree
#: ```
#:
#: Facade: imports and re-exports the modules of the `parse/` directory. The
#: surface is three names, and it was three names before the split.
from jaithon.compile.parse.state import MAX_RECURSION, Parser
from jaithon.compile.parse.wire import parse

export { MAX_RECURSION, Parser, parse }
