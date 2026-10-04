/* module_path.c — turning a dotted module name into a file on disk.
 *
 * Split out of module.c (see that file's header comment for why the rest of
 * module loading stayed together as one unit). This piece has no opinion
 * about the bootstrap window, the .jaic cache, or the import stack: it only
 * knows about directories and spec §8's search rules. Two concerns, kept in
 * one file because the second is meaningless without the first:
 *
 *   - The search path itself: JAITHON_PATH, jaiModulePathAdd, and the
 *     library directories derived from wherever this executable lives.
 *   - The pure syntax of turning "std.math" plus a directory into
 *     "<dir>/std/math.jai" (or the package form "<dir>/std/math/mod.jai"),
 *     tried against each directory on the path in turn.
 *
 * importFailure, ensurePathReady, isRegularFile, storeResolved and
 * displayName cross the file boundary with module.c in both directions —
 * see module_internal.h for why each is where it is.
 */

#include <stdlib.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <sys/attr.h>
#include <sys/vnode.h>
#include <unistd.h>
#endif

#include "runtime/runtime.h"
#include "runtime/modules/module_internal.h"
#include "native/native.h"

#define JAI_PROJECT_MANIFEST "jaithon.package.json"

/* Directories listed in an E0800 note before the list is elided. */
#define JAI_MAX_SEARCH_REPORTED 8

/* ------------------------------------------------------------------ */
/* Search path                                                          */
/* ------------------------------------------------------------------ */

typedef JAI_VEC(char *) DirList;

static DirList sUserDirs;
static DirList sLibDirs;
static bool    sPathReady;

/* Everything on the path after the binary's own lib/ -- every package's src/
 * and the system-wide defaults -- is added the first time something has to
 * look past what is already there: a module the earlier directories do not
 * hold, or a caller that wants the whole list. A program that imports only the
 * standard library never needs it, and building it cost every `run` 13
 * realpath calls, ~45 stats and two directory listings: 3.3M of the 26.5M
 * instructions a cached hello world retired. realpath alone is ~160K
 * instructions on macOS, one getattrlist per path component, and a package's
 * src/ is a dozen components deep. The order is unchanged -- the deferred
 * directories are appended exactly where they used to stand -- so a lookup
 * finds what it found before. JAITHON_LAZY_MODULE_PATH=0 builds it eagerly. */
static bool    sPathComplete = true;
static char    sDeferredExecDir[JAI_MAX_PATH];
static bool    sDeferredDefaults;

static bool dirListHas(const DirList *list, const char *dir) {
    for (int i = 0; i < list->count; i++) {
        if (strcmp(list->data[i], dir) == 0) return true;
    }
    return false;
}

static void noteCanonicalDir(const char *dir);

static void dirListAdd(DirList *list, const char *dir) {
    if (dir == NULL || dir[0] == '\0') return;

    char absolute[JAI_MAX_PATH];
    bool canonical = jaiPathAbsolute(absolute, sizeof absolute, dir);
    const char *entry = canonical ? absolute : dir;
    if (canonical) noteCanonicalDir(absolute);
    if (dirListHas(list, entry)) return;

    char *copy = jaiStrdup(entry);
    if (copy == NULL) return;
    JAI_VEC_PUSH(char *, list, copy);
}

static void dirListClear(DirList *list) {
    for (int i = 0; i < list->count; i++) {
        char *s = list->data[i];
        if (s != NULL) (void)jaiRealloc(s, strlen(s) + 1, 0);
    }
    list->count = 0;
}

static void syncModulePathMirror(void) {
    if (vm.gc == NULL) return;

    vm.modulePath.count = 0;
    for (int i = 0; i < sUserDirs.count; i++) {
        JAI_VEC_PUSH(ObjString *, &vm.modulePath,
                     jaiStringInternC(sUserDirs.data[i]));
    }
    for (int i = 0; i < sLibDirs.count; i++) {
        JAI_VEC_PUSH(ObjString *, &vm.modulePath,
                     jaiStringInternC(sLibDirs.data[i]));
    }
}

static void addLibDir(const char *dir) {
    if (dir == NULL || dir[0] == '\0' || !jaiPathIsDir(dir)) return;
    dirListAdd(&sLibDirs, dir);
}

static void addLibDirRelative(const char *base, const char *suffix) {
    char candidate[JAI_MAX_PATH];
    jaiPathJoin(candidate, sizeof candidate, base, suffix);
    addLibDir(candidate);
}

static void freePackageNames(char **names, int count) {
    if (names == NULL) return;
    for (int i = 0; i < count; i++) {
        if (names[i] != NULL)
            JAI_FREE_ARRAY(char, names[i], strlen(names[i]) + 1);
    }
    JAI_FREE_ARRAY(char *, names, count + 1);
}

static int comparePackageNames(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

static void addPackageSourceDirs(const char *packagesDir) {
    if (packagesDir == NULL || !jaiPathIsDir(packagesDir)) return;

    int count = 0;
    char **names = jaiListDir(packagesDir, &count);
    if (names == NULL) return;
    if (count > 1)
        qsort(names, (size_t)count, sizeof(char *), comparePackageNames);

    for (int i = 0; i < count; i++) {
        const char *name = names[i];
        if (name == NULL || name[0] == '.') continue;

        char package[JAI_MAX_PATH];
        char manifest[JAI_MAX_PATH];
        char source[JAI_MAX_PATH];
        jaiPathJoin(package, sizeof package, packagesDir, name);
        jaiPathJoin(manifest, sizeof manifest, package, JAI_PROJECT_MANIFEST);
        jaiPathJoin(source, sizeof source, package, "src");
        if (isRegularFile(manifest)) addLibDir(source);
    }
    freePackageNames(names, count);
}

static void addPackageDirsRelative(const char *base, const char *suffix) {
    char candidate[JAI_MAX_PATH];
    jaiPathJoin(candidate, sizeof candidate, base, suffix);
    addPackageSourceDirs(candidate);
}

/* ------------------------------------------------------------------ */
/* Resolution memo                                                      */
/* ------------------------------------------------------------------ */

/* (importing directory, dotted name) -> the path a successful resolution
 * produced. Every `import` statement a module top level runs resolves its
 * name again, whether or not the module is already loaded -- the module table
 * is keyed by the canonical path, so the path has to be found first -- and a
 * resolution is a stat per candidate plus a realpath of the winner, which on
 * macOS is one getattrlist per path component: ~157K instructions for a file a
 * dozen directories deep. `import jaicv` ran 2,618 of them for 332 distinct
 * files, 5,007 stats beside them, and spent more than half of its 854M
 * instructions there.
 *
 * Only successes are kept, so a module that does not exist yet is looked for
 * again and every diagnostic is produced exactly as before. The search path
 * only ever grows at its end once built (completePath appends), which cannot
 * change where an earlier success was found; anything that rebuilds or
 * prepends to it clears the memo. What the memo does give up is noticing a
 * file that appears, mid-run, earlier on the path than one already found --
 * the same trade Python's sys.modules makes. A resolution that looked in a
 * directory named relative to the current one is not kept at all, nor is a
 * relative candidate's realpath: after an os.chdir the REPL's `import m` must
 * find the new directory's m, as it always did. JAITHON_RESOLVE_MEMO=0 turns
 * it off. */
typedef struct {
    char    *key;      /* NULL for an empty slot */
    size_t   keyLen;
    char    *path;
    uint32_t hash;
} MemoEntry;

typedef struct {
    MemoEntry *slots;
    size_t     cap;    /* zero or a power of two */
    size_t     count;
} Memo;

/* (fromDir NUL dottedName NUL) -> resolved path. */
static Memo sResolved;
/* A candidate path -> what realpath made of it. Different importers name the
 * same file through different (fromDir, name) keys, so this second table is
 * what keeps the 932 realpaths `import jaicv` still made with only the first
 * down to the 332 distinct files it has. */
static Memo sCanonical;

static bool memoEnabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_RESOLVE_MEMO");
        cached = (v != NULL && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached == 1;
}

static uint32_t memoHash(const char *key, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= (unsigned char)key[i];
        h *= 16777619u;
    }
    return h;
}

/* The resolution key, or 0 when it does not fit. */
static size_t memoKey(char *buf, size_t size, const char *fromDir,
                      const char *dottedName) {
    const char *dir = fromDir != NULL ? fromDir : "";
    size_t a = strlen(dir);
    size_t b = strlen(dottedName);
    if (a + b + 2 > size) return 0;
    memcpy(buf, dir, a);
    buf[a] = '\0';
    memcpy(buf + a + 1, dottedName, b);
    buf[a + 1 + b] = '\0';
    return a + b + 2;
}

static MemoEntry *memoSlot(MemoEntry *table, size_t cap, const char *key,
                           size_t len, uint32_t hash) {
    for (size_t i = hash & (cap - 1);; i = (i + 1) & (cap - 1)) {
        MemoEntry *e = &table[i];
        if (e->key == NULL) return e;
        if (e->hash == hash && e->keyLen == len && memcmp(e->key, key, len) == 0)
            return e;
    }
}

static const char *memoFind(const Memo *m, const char *key, size_t len,
                            uint32_t hash) {
    if (m->cap == 0) return NULL;
    const MemoEntry *e = memoSlot(m->slots, m->cap, key, len, hash);
    return e->key != NULL ? e->path : NULL;
}

static void memoInsert(Memo *m, const char *key, size_t len, uint32_t hash,
                       const char *path) {
    if ((m->count + 1) * 2 > m->cap) {
        size_t cap = m->cap == 0 ? 64 : m->cap * 2;
        MemoEntry *table = JAI_ALLOC_ZEROED(MemoEntry, cap);
        for (size_t i = 0; i < m->cap; i++) {
            MemoEntry *old = &m->slots[i];
            if (old->key != NULL) *memoSlot(table, cap, old->key, old->keyLen, old->hash) = *old;
        }
        if (m->slots != NULL) JAI_FREE_ARRAY(MemoEntry, m->slots, m->cap);
        m->slots = table;
        m->cap = cap;
    }
    MemoEntry *e = memoSlot(m->slots, m->cap, key, len, hash);
    if (e->key != NULL) return;
    e->key = jaiMemdup(key, len);
    e->keyLen = len;
    e->path = jaiStrdup(path);
    e->hash = hash;
    m->count++;
}

static void memoClear(Memo *m) {
    for (size_t i = 0; i < m->cap; i++) {
        MemoEntry *e = &m->slots[i];
        if (e->key == NULL) continue;
        JAI_FREE_ARRAY(char, e->key, e->keyLen + 1);   /* jaiMemdup adds a NUL */
        JAI_FREE_ARRAY(char, e->path, strlen(e->path) + 1);
        e->key = NULL;
        e->path = NULL;
    }
    m->count = 0;
}

/* Directories known to be canonical: every search-path directory realpath
 * accepted, and the directory of every path realpath produced (each prefix of
 * a canonical path is canonical). Never cleared -- it records facts about the
 * filesystem, not about the search path. */
static Memo sCanonicalDirs;

static void noteCanonicalDir(const char *dir) {
    if (dir[0] == '\0') return;
    size_t len = strlen(dir);
    uint32_t hash = memoHash(dir, len);
    if (memoFind(&sCanonicalDirs, dir, len, hash) == NULL)
        memoInsert(&sCanonicalDirs, dir, len, hash, "");
}

static void noteCanonicalFile(const char *path) {
    char dir[JAI_MAX_PATH];
    jaiPathDirname(dir, sizeof dir, path);
    noteCanonicalDir(dir);
}

static bool isCanonicalDir(const char *dir) {
    size_t len = strlen(dir);
    return memoFind(&sCanonicalDirs, dir, len, memoHash(dir, len)) != NULL;
}

static bool envOn(const char *name, bool fallback) {
    const char *v = getenv(name);
    if (v == NULL || v[0] == '\0') return fallback;
    return strcmp(v, "0") != 0;
}

#ifdef __APPLE__
/* realpath of `dir`/`rest`, for a `dir` already canonical and a `rest` of
 * plain names, at one getattrlist per component of `rest` instead of one per
 * component of the whole path. macOS's realpath is exactly that loop run from
 * the root: ATTR_CMN_NAME is where it gets each component's case as stored.
 * A module file under lib/std is ~14 components deep and two of them are
 * `rest`, so this is ~2 calls where realpath makes ~14.
 *
 * It answers only when the answer cannot differ from realpath's. A symlink,
 * a name the filesystem spells with different bytes than ASCII case allows --
 * another Unicode normalisation, or a volume root reporting its volume name
 * -- or anything unexpected returns false, and the caller runs realpath.
 * JAITHON_RESOLVE_VERIFY=1 runs realpath anyway and reports any difference. */
static bool canonicalUnder(const char *dir, const char *rest, char *out,
                           size_t outSize) {
    size_t at = strlen(dir);
    if (at == 0 || dir[at - 1] == '/' || at + 1 >= outSize) return false;
    memcpy(out, dir, at + 1);

    const char *p = rest;
    while (*p != '\0') {
        const char *slash = strchr(p, '/');
        size_t n = slash != NULL ? (size_t)(slash - p) : strlen(p);
        if (n == 0 || (n == 1 && p[0] == '.') || (n == 2 && p[0] == '.' && p[1] == '.'))
            return false;
        if (at + 1 + n + 1 > outSize) return false;
        out[at] = '/';
        memcpy(out + at + 1, p, n);
        out[at + 1 + n] = '\0';

        struct attrlist request;
        memset(&request, 0, sizeof request);
        request.bitmapcount = ATTR_BIT_MAP_COUNT;
        request.commonattr = ATTR_CMN_NAME | ATTR_CMN_OBJTYPE;
        struct {
            uint32_t        length;
            attrreference_t name;
            fsobj_type_t    type;
            char            bytes[3 * 255 + 1];
        } __attribute__((aligned(4), packed)) reply;
        if (getattrlist(out, &request, &reply, sizeof reply, FSOPT_NOFOLLOW) != 0)
            return false;
        if (reply.type == VLNK) return false;
        const char *name = (const char *)&reply.name + reply.name.attr_dataoffset;
        const char *end = (const char *)&reply + sizeof reply;
        if (reply.name.attr_length == 0 || name < (const char *)&reply ||
            name + reply.name.attr_length > end)
            return false;
        size_t got = strnlen(name, reply.name.attr_length);
        if (got != n || strncasecmp(name, p, n) != 0) return false;
        for (size_t i = 0; i < n; i++)   /* ASCII case only; other bytes must match */
            if ((unsigned char)name[i] >= 0x80 && name[i] != p[i]) return false;
        memcpy(out + at + 1, name, n);
        at += 1 + n;
        p = slash != NULL ? slash + 1 : p + n;
    }
    return true;
}
#endif

/* `candidate` is `dir`/`leaf` and names a file that exists. */
static bool storeResolvedUnder(const char *dir, const char *leaf,
                               const char *candidate, char *out, size_t outSize) {
#ifdef __APPLE__
    static int fast = -1;
    static int verify = -1;
    if (fast < 0) {
        fast = memoEnabled() && envOn("JAITHON_CANONICAL_SUFFIX", true);
        verify = envOn("JAITHON_RESOLVE_VERIFY", false);
    }
    if (fast && isCanonicalDir(dir) && canonicalUnder(dir, leaf, out, outSize)) {
        if (verify) {
            char check[JAI_MAX_PATH];
            if (!jaiPathAbsolute(check, sizeof check, candidate) || strcmp(check, out) != 0) {
                fprintf(stderr, "jaithon: resolve-verify: %s gave %s, realpath %s\n",
                        candidate, out, check);
                return storeResolved(out, outSize, candidate);
            }
        }
        noteCanonicalFile(out);
        return true;
    }
#else
    (void)dir;
    (void)leaf;
#endif
    return storeResolved(out, outSize, candidate);
}

static bool lazyModulePath(void);
static void completePath(void);

void jaiModulePathInit(const char *execDir) {
    memoClear(&sResolved);
    memoClear(&sCanonical);
    dirListClear(&sLibDirs);
    sPathReady = true;

    const char *env = getenv("JAITHON_PATH");
    if (env != NULL) {
        const char *p = env;
        for (;;) {
            const char *sep = strchr(p, ':');
            size_t n = sep != NULL ? (size_t)(sep - p) : strlen(p);
            if (n > 0 && n < JAI_MAX_PATH) {
                char entry[JAI_MAX_PATH];
                memcpy(entry, p, n);
                entry[n] = '\0';
                dirListAdd(&sUserDirs, entry);
            }
            if (sep == NULL) break;
            p = sep + 1;
        }
    }

    char derived[JAI_MAX_PATH];
    if (execDir == NULL || execDir[0] == '\0') {
        const char *exe = jaiExecutablePath();
        if (exe != NULL && exe[0] != '\0') {
            jaiPathDirname(derived, sizeof derived, exe);
            execDir = derived;
        }
    }
    sDeferredExecDir[0] = '\0';
    if (execDir != NULL && execDir[0] != '\0') {
        addLibDirRelative(execDir, "lib");
        addLibDirRelative(execDir, "../lib");
        addLibDirRelative(execDir, "../share/jaithon/lib");
        addLibDirRelative(execDir, "../share/jaithon");
        snprintf(sDeferredExecDir, sizeof sDeferredExecDir, "%s", execDir);
    }
    sDeferredDefaults = getenv("JAITHON_NO_DEFAULT_PATH") == NULL;
    sPathComplete = false;

    if (!lazyModulePath()) completePath();
    else syncModulePathMirror();
}

static bool lazyModulePath(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_LAZY_MODULE_PATH");
        cached = (v != NULL && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached == 1;
}

static void completePath(void) {
    if (sPathComplete) return;
    sPathComplete = true;

    if (sDeferredExecDir[0] != '\0') {
        addPackageDirsRelative(sDeferredExecDir, "packages");
        addPackageDirsRelative(sDeferredExecDir, "../packages");
        addPackageDirsRelative(sDeferredExecDir, "../share/jaithon/packages");
    }

    if (sDeferredDefaults) {
        addLibDir("/usr/local/share/jaithon/lib");
        addLibDir("/usr/local/share/jaithon");
        addLibDir("/opt/homebrew/share/jaithon/lib");
        addLibDir("/opt/homebrew/share/jaithon");
        addPackageSourceDirs("/usr/local/share/jaithon/packages");
        addPackageSourceDirs("/opt/homebrew/share/jaithon/packages");
    }

    syncModulePathMirror();
}

void jaiModulePathComplete(void) {
    ensurePathReady();
    completePath();
}

void jaiModulePathAdd(const char *dir) {
    if (dir == NULL || dir[0] == '\0') return;
    memoClear(&sResolved);
    dirListAdd(&sUserDirs, dir);
    syncModulePathMirror();
}

void ensurePathReady(void) {
    if (!sPathReady) jaiModulePathInit(NULL);
}

/* ------------------------------------------------------------------ */
/* Dotted name -> file                                                  */
/* ------------------------------------------------------------------ */

static bool isNameByte(char c) {
    unsigned char u = (unsigned char)c;
    if (u >= 0x80) return true;
    return (u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') ||
           (u >= '0' && u <= '9') || u == '_';
}

static bool splitModuleName(const char *dotted, int *outDots, char *relative,
                            size_t relSize) {
    *outDots = 0;
    if (relSize > 0) relative[0] = '\0';

    if (dotted == NULL || dotted[0] == '\0') {
        return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                             "empty module path");
    }

    const char *p = dotted;
    while (*p == '.') { (*outDots)++; p++; }
    if (*p == '\0') {
        return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                             "module path '%s' names no module", dotted);
    }

    size_t pos = 0;
    while (*p != '\0') {
        size_t start = pos;
        while (*p != '\0' && *p != '.') {
            if (!isNameByte(*p)) {
                return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                                     "module path '%s' is not a dotted name",
                                     dotted);
            }
            if (pos + 1 >= relSize) {
                return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                                     "module path '%s' is too long", dotted);
            }
            relative[pos++] = *p++;
        }
        if (pos == start) {
            return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                                 "module path '%s' has an empty component",
                                 dotted);
        }
        if (*p == '.') {
            p++;
            if (pos + 1 >= relSize) {
                return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                                     "module path '%s' is too long", dotted);
            }
            relative[pos++] = '/';
        }
    }
    relative[pos] = '\0';
    return true;
}

const char *displayName(const char *dotted) {
    const char *p = dotted;
    while (*p == '.') p++;
    return *p != '\0' ? p : dotted;
}

/* One stat, not jaiPathExists and then jaiPathIsDir: the pair stat'd every
 * module a resolution found twice. */
bool isRegularFile(const char *path) {
    struct stat st;
    return path[0] != '\0' && stat(path, &st) == 0 && !S_ISDIR(st.st_mode);
}

bool storeResolved(char *out, size_t outSize, const char *candidate) {
    /* A relative candidate means something else after a chdir, so only an
     * absolute one is remembered. */
    if (memoEnabled() && candidate[0] == '/') {
        size_t len = strlen(candidate);
        uint32_t hash = memoHash(candidate, len);
        const char *hit = memoFind(&sCanonical, candidate, len, hash);
        if (hit != NULL && strlen(hit) + 1 <= outSize) {
            memcpy(out, hit, strlen(hit) + 1);
            return true;
        }
        if (jaiPathAbsolute(out, outSize, candidate)) {
            memoInsert(&sCanonical, candidate, len, hash, out);
            noteCanonicalFile(out);
            return true;
        }
    } else if (jaiPathAbsolute(out, outSize, candidate)) {
        return true;
    }
    size_t len = strlen(candidate);
    if (len + 1 > outSize) {
        out[0] = '\0';
        return false;
    }
    memcpy(out, candidate, len + 1);
    return true;
}

/* Set when a resolution looked in a directory named relative to the current
 * one: what it found, or did not find, can change with a chdir, so it is not
 * remembered. The REPL imports from ".", and a module path with no directory
 * does the same. */
static bool sSawRelativeDir;

static bool tryDirectory(const char *dir, const char *relative, char *out,
                         size_t outSize) {
    char leaf[JAI_MAX_PATH];
    char candidate[JAI_MAX_PATH];

    if (dir[0] != '/') sSawRelativeDir = true;

    int n = snprintf(leaf, sizeof leaf, "%s%s", relative, JAI_MODULE_EXT);
    if (n > 0 && (size_t)n < sizeof leaf) {
        jaiPathJoin(candidate, sizeof candidate, dir, leaf);
        if (isRegularFile(candidate))
            return storeResolvedUnder(dir, leaf, candidate, out, outSize);
    }

    n = snprintf(leaf, sizeof leaf, "%s/%s", relative, JAI_PACKAGE_FILE);
    if (n > 0 && (size_t)n < sizeof leaf) {
        jaiPathJoin(candidate, sizeof candidate, dir, leaf);
        if (isRegularFile(candidate))
            return storeResolvedUnder(dir, leaf, candidate, out, outSize);
    }
    return false;
}

static void noteSearched(JaiBuf *searched, int *count, const char *dir) {
    (*count)++;
    if (*count > JAI_MAX_SEARCH_REPORTED) return;
    if (searched->count > 0) jaiBufAppendStr(searched, ", ");
    jaiBufAppendStr(searched, dir);
}

static bool relativeBase(const char *fromDir, int dots, char *out,
                         size_t outSize) {
    const char *start = (fromDir != NULL && fromDir[0] != '\0') ? fromDir : ".";
    if (start[0] != '/') sSawRelativeDir = true;
    if (!storeResolved(out, outSize, start)) return false;

    for (int i = 1; i < dots; i++) {
        char parent[JAI_MAX_PATH];
        jaiPathDirname(parent, sizeof parent, out);
        if (parent[0] == '\0' || strcmp(parent, out) == 0) return false;
        if (!storeResolved(out, outSize, parent)) return false;
    }
    return true;
}

static bool resolveUncached(const char *dottedName, const char *fromDir,
                            char *out, size_t outSize);

bool jaiResolveModulePath(const char *dottedName, const char *fromDir,
                          char *out, size_t outSize) {
    if (out == NULL || outSize == 0) return false;
    out[0] = '\0';
    if (dottedName == NULL || !memoEnabled())
        return resolveUncached(dottedName, fromDir, out, outSize);

    ensurePathReady();
    char key[2 * JAI_MAX_PATH];
    size_t len = memoKey(key, sizeof key, fromDir, dottedName);
    if (len == 0) return resolveUncached(dottedName, fromDir, out, outSize);
    uint32_t hash = memoHash(key, len);
    const char *hit = memoFind(&sResolved, key, len, hash);
    if (hit != NULL) {
        size_t n = strlen(hit);
        if (n + 1 <= outSize) {
            memcpy(out, hit, n + 1);
            return true;
        }
    }
    sSawRelativeDir = false;
    if (!resolveUncached(dottedName, fromDir, out, outSize)) return false;
    if (out[0] != '\0' && !sSawRelativeDir)
        memoInsert(&sResolved, key, len, hash, out);
    return true;
}

static bool resolveUncached(const char *dottedName, const char *fromDir,
                            char *out, size_t outSize) {
    if (out == NULL || outSize == 0) return false;
    out[0] = '\0';
    ensurePathReady();

    int dots = 0;
    char relative[JAI_MAX_PATH];
    if (!splitModuleName(dottedName, &dots, relative, sizeof relative)) return false;

    JaiBuf searched;
    jaiBufInit(&searched);
    int searchedCount = 0;
    bool found = false;

    if (dots > 0) {
        char base[JAI_MAX_PATH];
        if (!relativeBase(fromDir, dots, base, sizeof base)) {
            jaiBufFree(&searched);
            return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                                 "relative import '%s' climbs past the root",
                                 dottedName);
        }
        found = tryDirectory(base, relative, out, outSize);
        if (!found) noteSearched(&searched, &searchedCount, base);
    } else {
        if (fromDir != NULL && fromDir[0] != '\0') {
            found = tryDirectory(fromDir, relative, out, outSize);
            if (!found) noteSearched(&searched, &searchedCount, fromDir);
        }
        for (int i = 0; !found && i < sUserDirs.count; i++) {
            found = tryDirectory(sUserDirs.data[i], relative, out, outSize);
            if (!found) noteSearched(&searched, &searchedCount, sUserDirs.data[i]);
        }
        for (int i = 0; !found; i++) {
            if (i == sLibDirs.count) {
                if (sPathComplete) break;
                completePath();   /* appends; entries before i are unchanged */
                if (i == sLibDirs.count) break;
            }
            found = tryDirectory(sLibDirs.data[i], relative, out, outSize);
            if (!found) noteSearched(&searched, &searchedCount, sLibDirs.data[i]);
        }
    }

    if (found && out[0] != '\0') {
        jaiBufFree(&searched);
        return true;
    }
    if (found) {
        /* The file exists but its path does not fit in the caller's buffer. */
        jaiBufFree(&searched);
        return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                             "path of module '%s' is too long", dottedName);
    }

    if (searchedCount > JAI_MAX_SEARCH_REPORTED) {
        jaiBufPrintf(&searched, " and %d more",
                     searchedCount - JAI_MAX_SEARCH_REPORTED);
    }
    jaiBufPush(&searched, '\0');
    const char *dirs = (searchedCount > 0 && searched.data != NULL)
                           ? (const char *)searched.data
                           : "no directories";

    if (vm.frameCount > 0) {
        (void)jaiThrow(vm.cImportError, "%s: cannot find module '%s'; searched %s",
                       jaiDiagCodeString(E0800_MODULE_NOT_FOUND), dottedName, dirs);
    } else {
        JaiDiag *d = jaiDiagError(E0800_MODULE_NOT_FOUND, JAI_SPAN_NONE,
                                  "cannot find module `%s`", dottedName);
        jaiDiagAddNote(d, "searched %s", dirs);
        if (sLibDirs.count == 0) {
            jaiDiagAddHelp(d, "no installed library was found; set JAITHON_PATH "
                              "to the directory holding `std`");
        }
    }
    jaiBufFree(&searched);
    return false;
}

bool jaiResolveModulePathQuiet(const char *dottedName, const char *fromDir,
                               char *out, size_t outSize) {
    JaiDiagBag live = gDiags;
    jaiDiagInit(&gDiags);
    bool hadException = vm.hasException;

    bool found = jaiResolveModulePath(dottedName, fromDir, out, outSize);

    jaiDiagFree(&gDiags);
    gDiags = live;
    if (!hadException && vm.hasException) jaiClearException();
    return found;
}
