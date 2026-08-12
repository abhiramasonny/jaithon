#: Operations on tuples, and named tuples built from a list of field names.
#:
#: Tuples are fixed-size and immutable, so everything here returns a new value.
const _MAX_ARITY = 8

#: The shape of a named tuple: a type name plus its field names.
pub class NamedTupleType {
    #: The name this shape prints under.
    pub let name: str
    #: The field names, in declaration order.
    pub let fields: list[str]
    let position: dict[str, int]

    #: Define a record shape whose field names are unique and non-empty.
    pub fn init(self, name: str, fields: list[str]) {
        if name.len() == 0 { throw ValueError("a named tuple needs a type name") }
        let position: dict[str, int] = {}
        for (index, field) in fields.enumerate() {
            if field.len() == 0 { throw ValueError(f"empty field name in {name}") }
            if field in position {
                throw ValueError(f"duplicate field {field} in {name}")
            }
            position[field] = index
        }
        self.name = name
        # Copy, so a later change to the caller's list cannot desync `position`.
        self.fields = fields[0:fields.len()]
        self.position = position
    }

    #: The position of `field` in this shape, or null when it has no such field.
    pub fn index_of(self, field: str) -> int? {
        return self.position.get(field, null)
    }

    #: Build a record from a dict keyed by field name.
    pub fn of_dict(self, values: dict[str, any]) -> NamedTuple {
        for key in values.keys() {
            if key not in self.position {
                throw ValueError(f"{self.name} has no field {key}")
            }
        }
        let ordered: list[any] = []
        for field in self.fields {
            if field not in values {
                throw ValueError(f"{self.name} is missing field {field}")
            }
            ordered.push(values[field])
        }
        return NamedTuple(self, ordered)
    }

    fn __call__(self, ...values: any) -> NamedTuple {
        if values.len() != self.fields.len() {
            throw ValueError(f"{self.name} takes {self.fields.len()} values, got {values.len()}")
        }
        return NamedTuple(self, values)
    }

    fn __eq__(self, other: NamedTupleType) -> bool {
        return self.name == other.name and self.fields == other.fields
    }

    fn __str__(self) -> str {
        let body = ", ".join(self.fields)
        return f"NamedTupleType({self.name}: {body})"
    }
}

#: One record of a `NamedTupleType`: fixed values reachable by name or position.
pub class NamedTuple {
    #: The shape this record was built from.
    pub let kind: NamedTupleType
    let values: list[any]

    #: Build a record directly; prefer calling its `NamedTupleType`.
    pub fn init(self, kind: NamedTupleType, values: list[any]) {
        self.kind = kind
        self.values = values[0:values.len()]
    }

    #: The value of `field`.
    #:
    #: Raises `KeyError` when the shape has no such field.
    pub fn get(self, field: str) -> any {
        let index = self.kind.index_of(field) ?? -1
        if index < 0 { throw KeyError(f"{self.kind.name} has no field {field}") }
        return self.values[index]
    }

    #: A copy of this record with `field` set to `value`.
    pub fn replace(self, field: str, value: any) -> NamedTuple {
        let index = self.kind.index_of(field) ?? -1
        if index < 0 { throw KeyError(f"{self.kind.name} has no field {field}") }
        let updated = self.values[0:self.values.len()]
        updated[index] = value
        return NamedTuple(self.kind, updated)
    }

    #: The values as a plain list, in field order.
    pub fn to_list(self) -> list[any] {
        return self.values[0:self.values.len()]
    }

    #: The record as a dict keyed by field name.
    pub fn to_dict(self) -> dict[str, any] {
        let out: dict[str, any] = {}
        for (index, field) in self.kind.fields.enumerate() {
            out[field] = self.values[index]
        }
        return out
    }

    fn __getitem__(self, key: any) -> any {
        if isinstance(key, str) { return self.get(key) }
        return self.values[key]
    }

    fn __len__(self) -> int {
        return self.values.len()
    }

    fn __eq__(self, other: NamedTuple) -> bool {
        return self.kind == other.kind and self.values == other.values
    }

    fn __hash__(self) -> int {
        # FNV-1a over the element hashes; the wrapping multiply keeps it from
        # raising OverflowError the way plain `*` would.
        var digest = 1469598103934665603
        for value in self.values {
            digest = (digest ^ hash(value)) *% 1099511628211
        }
        return digest
    }

    fn __str__(self) -> str {
        let parts: list[str] = []
        for (index, field) in self.kind.fields.enumerate() {
            parts.push(f"{field}={repr(self.values[index])}")
        }
        let body = ", ".join(parts)
        return f"{self.kind.name}({body})"
    }
}

#: Define a named-tuple shape and return the constructor for it.
#:
#: Jaithon has no dynamic attribute lookup, so `point["x"]` stands in for
#: `point.x`.
#:
#: ```
#: let Point = named_tuple("Point", "x y")
#: let point = Point(3, 4)
#: point["x"]                             # 3
#: point[1]                               # 4
#: point.to_dict() == {"x": 3, "y": 4}    # true
#: str(point.replace("x", 9))             # "Point(x=9, y=4)"
#: ```
pub fn named_tuple(type_name: str, fields: list[str] | str) -> NamedTupleType {
    let names = if isinstance(fields, str) { _parse_field_spec(fields) } else { fields }
    return NamedTupleType(type_name, names)
}

#: Exchange the two elements of a pair.
#:
#: ```
#: swap((1, "a"))    # ("a", 1)
#: ```
pub fn swap[A, B](pair: tuple[A, B]) -> tuple[B, A] {
    return (pair[1], pair[0])
}

#: Apply `transform` to the first element of a pair.
#:
#: ```
#: map_first((3, "a"), |n| n * 2)    # (6, "a")
#: ```
pub fn map_first[A, B, C](pair: tuple[A, B], transform: fn(A) -> C) -> tuple[C, B] {
    return (transform(pair[0]), pair[1])
}

#: Apply `transform` to the second element of a pair.
#:
#: ```
#: map_second((3, "a"), |text| text.upper())    # (3, "A")
#: ```
pub fn map_second[A, B, C](pair: tuple[A, B], transform: fn(B) -> C) -> tuple[A, C] {
    return (pair[0], transform(pair[1]))
}

#: Copy the elements of a tuple of any arity into a list.
#:
#: ```
#: to_list((1, 2, 3))    # [1, 2, 3]
#: ```
pub fn to_list(value: tuple) -> list[any] {
    return [value[index] for index in 0..len(value)]
}

#: Build a tuple holding the elements of `items`.
#:
#: A tuple's arity is part of its type, so the cases have to be spelled out:
#: this covers 1 to 8 elements and raises `ValueError` outside that range.
#:
#: ```
#: from_list([1, 2])    # (1, 2)
#: ```
pub fn from_list(items: list[any]) -> tuple {
    let count = items.len()
    if count < 1 or count > _MAX_ARITY {
        throw ValueError(f"from_list builds tuples of 1 to {_MAX_ARITY} elements, got {count}")
    }
    if count == 1 { return (items[0],) }
    if count == 2 { return (items[0], items[1]) }
    if count == 3 { return (items[0], items[1], items[2]) }
    if count == 4 { return (items[0], items[1], items[2], items[3]) }
    if count == 5 { return (items[0], items[1], items[2], items[3], items[4]) }
    if count == 6 { return (items[0], items[1], items[2], items[3], items[4], items[5]) }
    if count == 7 {
        return (items[0], items[1], items[2], items[3], items[4], items[5], items[6])
    }
    return (items[0], items[1], items[2], items[3], items[4], items[5], items[6], items[7])
}

#: Pair `a` and `b` up to the length of the longer, padding the shorter one.
#:
#: ```
#: zip_longest([1, 2], ["a"], 0, "-")    # [(1, "a"), (2, "-")]
#: ```
pub fn zip_longest[A, B](a: list[A], b: list[B], fill_a: A, fill_b: B) -> list[tuple[A, B]] {
    let out: list[tuple[A, B]] = []
    for index in 0..max(a.len(), b.len()) {
        let left = if index < a.len() { a[index] } else { fill_a }
        let right = if index < b.len() { b[index] } else { fill_b }
        out.push((left, right))
    }
    return out
}

#: Split a list of triples into three lists; the three-element `unzip`.
#:
#: ```
#: unzip3([(1, "a", true), (2, "b", false)])    # ([1, 2], ["a", "b"], [true, false])
#: ```
pub fn unzip3[A, B, C](triples: list[tuple[A, B, C]]) -> tuple[list[A], list[B], list[C]] {
    let firsts = [triple[0] for triple in triples]
    let seconds = [triple[1] for triple in triples]
    let thirds = [triple[2] for triple in triples]
    return (firsts, seconds, thirds)
}

fn _parse_field_spec(spec: str) -> list[str] {
    let names: list[str] = []
    for piece in spec.replace(",", " ").split(" ") {
        if piece.len() > 0 { names.push(piece) }
    }
    return names
}
