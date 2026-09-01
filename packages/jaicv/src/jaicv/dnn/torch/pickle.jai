#: What a pickle builds with: storages, globals, tuples, and the checkpoint the
#: storages are read out of.
from .bits import dtype_size
from .zip import ZipArchive

#: One tensor storage: a flat run of elements that any number of tensors may
#: be views onto.
pub class Storage {
    pub let dtype: str
    pub let data: bytes
    pub let count: int

    pub fn init(self, dtype: str, data: bytes, count: int) {
        self.dtype = dtype
        self.data = data
        self.count = count
    }
}

#: A global the pickle named and the whitelist allowed.
pub class PickleGlobal {
    pub let module_name: str
    pub let name: str

    pub fn init(self, module_name: str, name: str) {
        self.module_name = module_name
        self.name = name
    }

    pub fn full_name(self) -> str { return self.module_name + "." + self.name }
}

#: A pickle tuple.
#:
#: Jaithon's own tuples are fixed-arity and a pickle's are not, so they travel
#: as their items behind a marker class. Keeping them apart from a list is not
#: pedantry: a persistent id is a tuple, and telling one from a list of two
#: tensors is how the reader knows which it is looking at.
pub class PickleTuple {
    pub let items: list[any]

    pub fn init(self, items: list[any]) { self.items = items }
}

#: The archive plus the prefix every member of it sits under.
pub class Checkpoint {
    pub let archive: ZipArchive
    pub let root: str
    pub var loaded: dict[str, Storage]

    pub fn init(self, archive: ZipArchive, root: str) {
        self.archive = archive
        self.root = root
        self.loaded = {}
    }

    #: The storage a persistent id names, read once however many tensors are
    #: views onto it. Caching is not only for speed: a shared storage that was
    #: deflated would otherwise be expanded again for every view of it.
    pub fn storage(self, key: str, dtype: str, count: int) -> Storage {
        if self.loaded.contains(key) {
            let known = self.loaded[key]
            if known.dtype != dtype {
                throw ValueError(
                    f"storage '{key}' is read as {known.dtype} in one place and {dtype} in another"
                )
            }
            return known
        }
        let member = self.root + "data/" + key
        if not self.archive.has(member) {
            throw ValueError(f"the checkpoint names a storage '{key}' that it does not carry")
        }
        let raw = self.archive.read(member)
        let wanted = count * dtype_size(dtype)
        if raw.len() != wanted {
            throw ValueError(
                f"storage '{key}' should hold {wanted} bytes for {count} {dtype} elements, " +
                    f"and holds {raw.len()}"
            )
        }
        let made = Storage(dtype, raw, count)
        self.loaded[key] = made
        return made
    }
}
