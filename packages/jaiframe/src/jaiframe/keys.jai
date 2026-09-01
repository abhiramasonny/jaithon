#: Row identity and row order — the substrate group-by, join, reshape and
#: sorting all stand on.
#:
#: Two questions come up in every one of those modules: *which rows carry the
#: same key*, and *what order do the rows go in*. Answering either one well is
#: a hash table or a sort on the device, and answering them four times over in
#: four modules is how a data frame library ends up with four subtly different
#: notions of what makes two rows equal. So both live here, over one encoding.
#:
#: `encode_keys` turns any list of key columns into a `KeyEncoding`: a packed
#: lane of uint32 digits per column, ordered most significant first, in which
#: unsigned integer order *is* the value's order and equality on every lane
#: *is* row equality. A float becomes its order-preserving bit pattern; the two
#: lanes of an `Int64` become a biased high digit and a raw low one; a
#: dictionary code becomes the rank of its level. After that one step nothing
#: downstream has to know a dtype: `factorize` hashes the lanes and probes an
#: open-addressed table, `argsort` runs a radix pass per digit, and both are
#: the same code for text, timestamps and floats.
#:
#: ```
#: let city = Column.from_strs(["dallas", "austin", "dallas"])
#: defer { city.free() }
#: let grouped = factorize([city])
#: defer { grouped[0].free() }
#: grouped[2]                    # 2 groups
#: ```
from .keys.lanes import NO_GROUP
from .keys.encode import KeyEncoding, encode_keys, hash_keys
from .keys.group import drop_duplicates, duplicated, factorize, mode, nunique
from .keys.group import unique, value_counts
from .keys.sort import argsort, nlargest, nsmallest, rank, sort_index, sort_values

export { NO_GROUP, KeyEncoding, encode_keys, hash_keys }
export { factorize, unique, nunique, value_counts, mode }
export { duplicated, drop_duplicates }
export { argsort, sort_values, sort_index, nlargest, nsmallest, rank }
