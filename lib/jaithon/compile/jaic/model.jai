#: What a `.jaic` file holds, as objects: `Module`, the header fields and the
#: mod body; `Import`, one row of the import section; and `StrTab`, the module
#: string table every tag-10 constant names by index.
#:
#: `StrTab` belongs to the file's shape rather than to one direction's code,
#: which is why it is not in `write.jai` even though the writer is what
#: interns into it -- the reader rebuilds the same table into `Reader.strings`
#: and indexes it the same way. It must stay byte-compatible with `StrTab` in
#: `src/vm/bytecode/serialize.c`.
#:
#: What is NOT here: the codecs. These three carry data, plus the one `write`
#: that is part of `StrTab`'s own definition.
from jaithon.compile.emit import FuncProto
from jaithon.compile.jaic.bits import utf8_bytes
from jaithon.compile.jaic.format import FLAG_DEBUG
from jaithon.compile.jaic.stream import Writer

#: One `.jaic` file: the header fields, the mod pool and the mod body.
pub class Module {
    #: Source path, recorded for tracebacks.
    pub var path: str
    #: FNV-1a 64 of the source text; the cache is used only when it matches.
    pub var source_hash: int
    #: `FLAG_DEBUG` and `FLAG_RELEASE`.
    pub var flags: int
    #: Last-modified time of the source, informational only.
    pub var mtime: int
    #: Names this mod exports.
    pub var exports: list[str]
    #: Imported mod paths, with the alias each was bound to or null.
    pub var imports: list[Import]
    #: The mod body's function record.
    pub var body: FuncProto

    #: Describe a mod.
    pub fn init(self, path: str, source_hash: int, body: FuncProto, flags: int = FLAG_DEBUG) {
        self.path = path
        self.source_hash = source_hash
        self.flags = flags
        self.mtime = 0
        self.exports = []
        self.imports = []
        self.body = body
    }
}

#: One row of the container's import section.
pub class Import {
    #: The dotted mod path.
    pub let path: str
    #: The name it was bound to, or null when it kept its own.
    pub let alias: str?

    #: Record an import.
    pub fn init(self, path: str, alias: str? = null) {
        self.path = path
        self.alias = alias
    }
}

#: Every string constant in the module, once, in first-use order.
#:
#: Two things pay for this: 62.5% of a `.jaic`'s string payload was the same
#: text repeated across nested function pools (73.8% in the seed), and interning
#: at load drops from one call per occurrence to one per distinct text.
#:
#: Must stay byte-compatible with `StrTab` in `src/vm/bytecode/serialize.c` —
#: the C runtime reads what this writes.
pub class StrTab {
    #: The distinct texts, in the order they were first seen.
    pub var items: list[str]
    #: text -> its index in `items`.
    pub var index: dict[str, int]

    pub fn init(self) -> void {
        self.items = []
        self.index = {}
    }

    #: The index of `text`, adding it if this is its first use.
    pub fn intern(self, text: str) -> int {
        let existing = self.index.get(text)
        if existing != null { return existing ?? 0 }
        let position = self.items.len()
        self.items.push(text)
        self.index[text] = position
        return position
    }

    #: `count`, then each string as {uleb128 length, bytes}. Varint lengths
    #: because 94.9% of strings are under 16 bytes, where a flat u32 length is
    #: mostly zeroes.
    pub fn write(self, out: Writer) -> void {
        out.uleb(self.items.len())
        for text in self.items {
            let encoded = utf8_bytes(text)
            out.uleb(encoded.len())
            out.bytes(encoded)
        }
    }
}
