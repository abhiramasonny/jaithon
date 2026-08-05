# A module for the REPL's import tests: something to import a class and a
# function from, so that `from bindings_module import Tag` has a Tag.
pub class Tag {
    pub var label: str

    fn init(self, label: str) {
        self.label = label
    }
}

pub fn shout(text: str) -> str {
    return text + "!"
}
