#: The message tree the prototxt and the caffemodel both parse into.

#: One parsed message, however it arrived.
#:
#: Values are held by field name in four buckets rather than one list of boxed
#: values, because a blob's `data` is hundreds of thousands of floats and one
#: object apiece for them is not affordable. A repeated field keeps every
#: value; a singular one that appears twice keeps both and the readers take the
#: last, which is protobuf's own merge rule.
pub class Message {
    pub let kind: str
    pub var ints: dict[str, list[int]]
    pub var floats: dict[str, list[float]]
    pub var strs: dict[str, list[str]]
    pub var children: dict[str, list[Message]]

    pub fn init(self, kind: str) {
        self.kind = kind
        self.ints = {}
        self.floats = {}
        self.strs = {}
        self.children = {}
    }

    #: Whether the file said anything at all about this field.
    pub fn has(self, name: str) -> bool {
        return name in self.ints or
            name in self.floats or
            name in self.strs or
            name in self.children
    }

    pub fn add_ints(self, name: str, values: list[int]) -> void {
        if name in self.ints {
            for value in values { self.ints[name].push(value) }
        } else {
            self.ints[name] = values
        }
    }

    pub fn add_floats(self, name: str, values: list[float]) -> void {
        if name in self.floats {
            for value in values { self.floats[name].push(value) }
        } else {
            self.floats[name] = values
        }
    }

    pub fn add_str(self, name: str, value: str) -> void {
        if name in self.strs {
            self.strs[name].push(value)
        } else {
            self.strs[name] = [value]
        }
    }

    pub fn add_child(self, name: str, value: Message) -> void {
        if name in self.children {
            self.children[name].push(value)
        } else {
            self.children[name] = [value]
        }
    }

    pub fn ints_of(self, name: str) -> list[int] { return self.ints.get(name, []) }

    pub fn floats_of(self, name: str) -> list[float] { return self.floats.get(name, []) }

    pub fn strs_of(self, name: str) -> list[str] { return self.strs.get(name, []) }

    pub fn children_of(self, name: str) -> list[Message] { return self.children.get(name, []) }

    pub fn int_of(self, name: str, fallback: int) -> int {
        let values = self.ints_of(name)
        if values.len() == 0 { return fallback }
        return values[values.len() - 1]
    }

    pub fn bool_of(self, name: str, fallback: bool) -> bool {
        let values = self.ints_of(name)
        if values.len() == 0 { return fallback }
        return values[values.len() - 1] != 0
    }

    pub fn float_of(self, name: str, fallback: float) -> float {
        let values = self.floats_of(name)
        if values.len() == 0 { return fallback }
        return values[values.len() - 1]
    }

    pub fn str_of(self, name: str, fallback: str) -> str {
        let values = self.strs_of(name)
        if values.len() == 0 { return fallback }
        return values[values.len() - 1]
    }

    pub fn child(self, name: str) -> Message? {
        let values = self.children_of(name)
        if values.len() == 0 { return null }
        return values[values.len() - 1]
    }

    #: A sub-message, or an empty one of the right kind so that every reader
    #: below gets the schema's default without a special case for "absent".
    pub fn param(self, name: str, kind: str) -> Message {
        return self.child(name) ?? Message(kind)
    }

    fn __str__(self) -> str { return f"Message({self.kind})" }
}
