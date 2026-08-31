# expect: E0902
#: The one live example of a diagnostic only the EMITTER produces -- the
#: checker accepts `import *` as a first-class re-export and says nothing.
#: Without this file no test in the tree exercises any of emit.jai's nine
#: `_unsupported()` sites, so a corpus diff makes dropping the emit pass from
#: `check` look safe when it is not. See jaiCheckFile in module.c.
from std.core import *
