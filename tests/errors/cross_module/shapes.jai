#: Helper module for the cross-module diagnostic fixtures beside it.
#:
#: It is deliberately in a subdirectory: `tests/errors/*.jai` is the diagnostic
#: glob, and a file with no `# expect:` header there is reported as skipped.

#: A box with one integer side.
pub class Box {
    #: The side length.
    pub var side: int

    #: Create a box of side zero.
    pub fn init(self) { self.side = 0 }

    #: Set the side length.
    pub fn resize(self, side: int) -> void { self.side = side }
}

#: Return `a` scaled by `factor`, labelled with `label`.
pub fn scaled(a: int, factor: int, label: str) -> str { return f"{label}{a * factor}" }
