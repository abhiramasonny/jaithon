#: The command-line convention every `jaithon` subcommand obeys.
#:
#: There is one rule, and both sides of the CLI boundary keep it: an option
#: that takes a value accepts `--name value` and `--name=value`, and the C
#: driver forwards what the user typed without rewriting it. The driver cannot
#: know which of a tool's flags take a value — it does not own their option
#: tables — so it forwards every argument after the subcommand verbatim and in
#: order, and the tool, which does know, pairs the flags with their values
#: here. Before this existed the driver rewrote `--out DIR` into `--out=DIR`
#: and `jaithon.tool.doc` rejected its own documented spelling.
#:
#: A tool's option table is then a loop over `Arguments`, and a usage error is
#: whatever the cursor recorded:
#:
#: ```
#: let reader = Arguments(["--out", "docs/api", "--html", "lib"])
#: reader.next()?.name              # "--out"
#: reader.next()?.text              # "docs/api"
#: reader.next()?.is_flag()         # true
#: reader.next()?.text              # "lib"
#: reader.next()                    # null
#: ```

#: One command-line argument, split into the parts an option table needs.
#:
#: `text` is exactly what the user typed. `name` is the option it names —
#: `--out` for both `--out docs` and `--out=docs` — or the empty string when
#: the argument is a path. `value` is the part after the `=`, and null when the
#: argument did not use that form, which is not the same as the empty value
#: `--out=` gives.
#:
#: ```
#: Argument("--out=docs").name       # "--out"
#: Argument("--out=docs").value      # "docs"
#: Argument("--html").value          # null
#: Argument("lib/std").is_flag()     # false
#: ```
pub class Argument {
    #: The argument exactly as it was given.
    pub let text: str

    #: The option name, or the empty string for a path.
    pub let name: str

    #: The value written after `=`, or null when there was none.
    pub let value: str?

    #: Split one raw argument into its name and inline value.
    pub fn init(self, text: str) {
        var name = ""
        var value: str? = null
        # `-` is a path (stdin, by convention) and `--` an end-of-options
        # marker the caller handles; neither names an option.
        if text.starts_with("-") and text != "-" and text != "--" {
            let equals = text.find("=")
            if equals < 0 {
                name = text
            } else {
                name = text[0:equals]
                value = text[equals + 1:text.len()]
            }
        }
        self.text = text
        self.name = name
        self.value = value
    }

    #: Whether this argument names an option rather than a path.
    #:
    #: ```
    #: Argument("--check").is_flag()   # true
    #: Argument("lib").is_flag()       # false
    #: ```
    pub fn is_flag(self) -> bool { return self.name.len() > 0 }
}

#: A cursor over an argument list that understands both option spellings.
#:
#: The cursor also carries the first usage error it saw, so an option table can
#: run to the end of the line and report once, and callers never have to invent
#: the same message twice.
#:
#: ```
#: let reader = Arguments(["--filter=slow", "tests"])
#: let flag = reader.next()
#: reader.value_of(flag)      # "slow"
#: reader.next()?.text        # "tests"
#: ```
pub class Arguments {
    #: The arguments, split, in the order they were given.
    pub let items: list[Argument]

    #: How many of them have been consumed.
    pub var index: int

    #: The first usage error recorded, or null while the line is well formed.
    pub var error: str?

    #: Build a cursor over `args`.
    pub fn init(self, args: list[str]) {
        self.items = [Argument(text) for text in args]
        self.index = 0
        self.error = null
    }

    #: The next argument, or null once they are all consumed.
    pub fn next(self) -> Argument? {
        if self.index >= self.items.len() { return null }
        let argument = self.items[self.index]
        self.index += 1
        return argument
    }

    #: The value belonging to `argument`: its `=` part, or the next word.
    #:
    #: Returns null and records the usage error when there is no value, or when
    #: the next word is itself an option — `--filter --verbose` is a forgotten
    #: value far more often than it is a search for the text `--verbose`, which
    #: `--filter=--verbose` still expresses.
    #:
    #: ```
    #: let reader = Arguments(["--out", "docs"])
    #: reader.value_of(reader.next())    # "docs"
    #: ```
    pub fn value_of(self, argument: Argument?) -> str? {
        #  `next()` answers `Argument?`, and the documented spelling of this
        #  call is `reader.value_of(reader.next())`, so the optional is the
        #  parameter type. There is nothing to find a value for once the
        #  arguments are exhausted, and no option name to name in an error.
        if argument is null { return null }
        let inline = argument.value
        if inline is not null { return inline }
        if self.index < self.items.len() and not self.items[self.index].is_flag() {
            let following = self.items[self.index]
            self.index += 1
            return following.text
        }
        self.fail(f"{argument.name} needs a value")
        return null
    }

    #: Accept `argument` as an option that takes no value.
    #:
    #: Returns false and records the usage error when a value was attached, so
    #: that `--html=yes` is a mistake the user hears about rather than a flag
    #: that quietly does nothing.
    #:
    #: ```
    #: let reader = Arguments([])
    #: reader.is_bare(Argument("--html"))       # true
    #: reader.is_bare(Argument("--html=yes"))   # false
    #: ```
    pub fn is_bare(self, argument: Argument) -> bool {
        if argument.value is null { return true }
        self.fail(f"{argument.name} takes no value")
        return false
    }

    #: Record `argument` as an option this tool does not have.
    pub fn unknown(self, argument: Argument) -> void {
        self.fail(f"unknown option {repr(argument.text)}")
    }

    #: Record `message` as the usage error, unless one is already recorded.
    #:
    #: The first error is the one worth printing: everything after it was
    #: parsed under a misunderstanding of the line.
    pub fn fail(self, message: str) -> void {
        if self.error is null { self.error = message }
    }
}
