#: The command-line convention every `jaithon` subcommand obeys.
#:
#: The C driver forwards every argument verbatim — it doesn't own a tool's
#: option table and can't tell which flags take a value — and the tool pairs
#: flags with values here. (Previously the driver rewrote `--out DIR` to
#: `--out=DIR`, breaking `jaithon.tool.doc`'s documented spelling.)
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
#: `value` is null when `=` wasn't used, which is not the same as the empty
#: value `--out=` gives.
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
#: It remembers the first usage error, so an option table can scan to the end
#: of the line and report once instead of per-flag.
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
    #: A following option-looking word counts as a missing value, not a value
    #: — use `--filter=--verbose` to search for that literal text.
    #:
    #: ```
    #: let reader = Arguments(["--out", "docs"])
    #: reader.value_of(reader.next())    # "docs"
    #: ```
    pub fn value_of(self, argument: Argument?) -> str? {
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
    #: Fails if a value was attached, so `--html=yes` errors instead of
    #: silently doing nothing.
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
