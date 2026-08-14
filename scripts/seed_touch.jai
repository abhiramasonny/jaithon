#: Compile every module the seed carries, writing each image into the
#: `__jaicache__` beside its source. `make reseed` runs this, then embeds
#: exactly what it built.
#:
#: It used to work by *importing* `std.list` and letting the importer populate
#: the cache as a side effect. That relied on the C front end: inside the
#: bootstrap window `src/runtime/modules/module.c` compiled with C, so the front end's
#: own closure was compiled even though the front end was not loaded yet. With
#: one front end the window has only the seed, and a seed that serves a module
#: means that module is never compiled from source -- so the next seed would be
#: built from whatever little remained. Measured once, before that was
#: understood: 39 modules to 10 in a single generation.
#:
#: So nothing here imports anything for its side effects. The front end is
#: loaded from the *previous* seed and then compiles the next one's sources
#: explicitly, which never enters the window. That is the whole of what
#: self-hosting means, and it is what makes `make fixpoint-check` a real
#: question: generation N's compiler builds generation N+1's images, and they
#: have to come out byte for byte the same.
from jaithon.compile.mod import build_file
from std.io import Path

#: The tree whose modules the seed carries.
#:
#: All of `lib`, not only `lib/jaithon`. The compiler imports std modules while
#: it is itself loading -- `std.json` among them -- and those imports happen
#: inside the bootstrap window, where the compiler does not exist yet. No "the
#: compiler is ready now" flag can help: they are needed *during* its load.
pub let SEED_ROOT: str = "lib"

fn cache_path_for(source: str) -> str {
    let path = Path(source)
    let name = path.name()
    let stem = name[0:name.len() - ".jai".len()]
    return f"{path.parent()}/__jaicache__/{stem}.jaic"
}

fn main() -> int {
    var built = 0
    var refused = 0

    for entry in Path(SEED_ROOT).walk() {
        let source = f"{entry}"
        if not source.ends_with(".jai") { continue }
        if source.contains("__jaicache__") { continue }

        let out = cache_path_for(source)
        Path(out).parent().mkdir(parents: true, exist_ok: true)
        if build_file(source, out).ok() {
            built += 1
        } else {
            refused += 1
            print(f"seed_touch: {source} did not compile")
        }
    }

    print(f"seed_touch: {built} compiled, {refused} refused")
    return if refused > 0 { 1 } else { 0 }
}
