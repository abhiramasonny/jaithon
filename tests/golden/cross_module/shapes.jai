#: Helper module for `imported_alias.jai` beside it.
#:
#: In a subdirectory because `tests/golden/*.jai` is the golden glob and a file
#: there with no `.expected` next to it is not a test — the same reason
#: `tests/errors/cross_module` exists.

#: What a shape is drawn with.
pub enum Ink { Solid, Dotted }

#: A box with one integer side.
pub class Box {
    #: The side length.
    pub let side: int
    #: How it is drawn.
    pub let ink: Ink

    #: Create a box of `side`, solid unless told otherwise.
    pub fn init(self, side: int, ink: Ink = Ink.Solid) {
        self.side = side
        self.ink = ink
    }

    #: The area of the square this box describes.
    pub fn area(self) -> int { return self.side * self.side }
}

#: `box` with its side multiplied by `factor`.
pub fn scaled(box: Box, factor: int = 2) -> Box {
    return Box(box.side * factor, box.ink)
}
