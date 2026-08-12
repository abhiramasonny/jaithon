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

#include "runtime/runtime.h"
#include "runtime/modules/module_internal.h"
#include "native/native.h"

#define JAI_PACKAGE_FILE "mod.jai"
#define JAI_PROJECT_MANIFEST "jaithon.package.json"

/* Directories listed in an E0800 note before the list is elided. */
#define JAI_MAX_SEARCH_REPORTED 8

/* ------------------------------------------------------------------ */
/* Search path                                                          */
/* ------------------------------------------------------------------ */

typedef JAI_VEC(char *) DirList;

/* Two lists, searched after the importing file's own directory: `sUserDirs`
 * (JAITHON_PATH plus whatever jaiModulePathAdd was given) and `sLibDirs` (the
 * installed library). They hold C strings rather than ObjStrings so the path
 * survives being configured before the VM is up; vm.modulePath mirrors both
 * for the collector and for tools that want to show the path. */
static DirList sUserDirs;
static DirList sLibDirs;
static bool    sPathReady;

static bool dirListHas(const DirList *list, const char *dir) {
    for (int i = 0; i < list->count; i++) {
        if (strcmp(list->data[i], dir) == 0) return true;
    }
    return false;
}

static void dirListAdd(DirList *list, const char *dir) {
    if (dir == NULL || dir[0] == '\0') return;

    /* Absolute form where possible: it makes the duplicate check meaningful
     * and keeps `..` out of the directory list an error message prints. */
    char absolute[JAI_MAX_PATH];
    const char *entry = jaiPathAbsolute(absolute, sizeof absolute, dir)
                            ? absolute
                            : dir;
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

/* vm.modulePath is a mirror: nothing resolves through it, so it is rebuilt
 * wholesale whenever the path changes. */
static void syncModulePathMirror(void) {
    if (vm.gc == NULL) return;   /* before jaiVMInit there is nothing to intern */

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

/* A workspace package owns one import root at <package>/src. The manifest is
 * the marker that distinguishes a package from an arbitrary directory. JSON
 * validation and dependency checks happen before tests and installation; the
 * runtime only needs the filesystem convention during bootstrap. */
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

void jaiModulePathInit(const char *execDir) {
    /* Idempotent: the derived library directories are recomputed and anything
     * added by hand in the meantime is kept. */
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

    /* `make install` puts the binary in <prefix>/bin and the library in
     * <prefix>/share/jaithon/lib; a build tree has ./jaithon next to ./lib.
     * Both layouts are derived from wherever this executable actually is, so a
     * relocated tree keeps working. */
    char derived[JAI_MAX_PATH];
    if (execDir == NULL || execDir[0] == '\0') {
        const char *exe = jaiExecutablePath();
        if (exe != NULL && exe[0] != '\0') {
            jaiPathDirname(derived, sizeof derived, exe);
            execDir = derived;
        }
    }
    if (execDir != NULL && execDir[0] != '\0') {
        addLibDirRelative(execDir, "lib");
        addLibDirRelative(execDir, "../lib");
        addLibDirRelative(execDir, "../share/jaithon/lib");
        addLibDirRelative(execDir, "../share/jaithon");
        addPackageDirsRelative(execDir, "packages");
        addPackageDirsRelative(execDir, "../packages");
        addPackageDirsRelative(execDir, "../share/jaithon/packages");
    }

    /* The installed library, which a machine with jaithon already on it always
     * has. JAITHON_NO_DEFAULT_PATH exists so that a test can tell the tree it
     * is testing apart from the one that is installed: without it, moving a
     * source aside proves nothing, because the search quietly finds the
     * installed copy and the run succeeds for the wrong reason. */
    if (getenv("JAITHON_NO_DEFAULT_PATH") == NULL) {
        addLibDir("/usr/local/share/jaithon/lib");
        addLibDir("/usr/local/share/jaithon");
        addLibDir("/opt/homebrew/share/jaithon/lib");
        addLibDir("/opt/homebrew/share/jaithon");
        addPackageSourceDirs("/usr/local/share/jaithon/packages");
        addPackageSourceDirs("/opt/homebrew/share/jaithon/packages");
    }

    syncModulePathMirror();
}

void jaiModulePathAdd(const char *dir) {
    if (dir == NULL || dir[0] == '\0') return;
    dirListAdd(&sUserDirs, dir);
    syncModulePathMirror();
}

void ensurePathReady(void) {
    if (!sPathReady) jaiModulePathInit(NULL);
}

/* ------------------------------------------------------------------ */
/* Dotted name -> file                                                  */
/* ------------------------------------------------------------------ */

/* Path components come from source text, so they must not be able to name
 * anything outside a search directory. Spec §2.1 allows non-ASCII identifiers,
 * so the high half is let through and everything else — separators, `..`,
 * control bytes — is not. */
static bool isNameByte(char c) {
    unsigned char u = (unsigned char)c;
    if (u >= 0x80) return true;
    return (u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') ||
           (u >= '0' && u <= '9') || u == '_';
}

/* Split `dotted` into a leading-dot count and a '/'-separated relative path.
 * Reports E0804 and returns false on anything that is not
 * `'.'* ident ('.' ident)*`. */
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
            /* An empty component: "a..b", or a trailing dot. */
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

/* The name a module is known by. Leading dots are an instruction to the
 * resolver, not part of the identity: `.util` is the module `util`. */
const char *displayName(const char *dotted) {
    const char *p = dotted;
    while (*p == '.') p++;
    return *p != '\0' ? p : dotted;
}

bool isRegularFile(const char *path) {
    return path[0] != '\0' && jaiPathExists(path) && !jaiPathIsDir(path);
}

/* Absolute, symlink-resolved form of `candidate`, so that two spellings of the
 * same file share one entry in vm.modules. */
bool storeResolved(char *out, size_t outSize, const char *candidate) {
    if (jaiPathAbsolute(out, outSize, candidate)) return true;
    size_t len = strlen(candidate);
    if (len + 1 > outSize) {
        out[0] = '\0';
        return false;
    }
    memcpy(out, candidate, len + 1);
    return true;
}

/* <dir>/std/math.jai, then the package form <dir>/std/math/mod.jai. */
static bool tryDirectory(const char *dir, const char *relative, char *out,
                         size_t outSize) {
    char leaf[JAI_MAX_PATH];
    char candidate[JAI_MAX_PATH];

    int n = snprintf(leaf, sizeof leaf, "%s%s", relative, JAI_MODULE_EXT);
    if (n > 0 && (size_t)n < sizeof leaf) {
        jaiPathJoin(candidate, sizeof candidate, dir, leaf);
        if (isRegularFile(candidate)) return storeResolved(out, outSize, candidate);
    }

    n = snprintf(leaf, sizeof leaf, "%s/%s", relative, JAI_PACKAGE_FILE);
    if (n > 0 && (size_t)n < sizeof leaf) {
        jaiPathJoin(candidate, sizeof candidate, dir, leaf);
        if (isRegularFile(candidate)) return storeResolved(out, outSize, candidate);
    }
    return false;
}

static void noteSearched(JaiBuf *searched, int *count, const char *dir) {
    (*count)++;
    if (*count > JAI_MAX_SEARCH_REPORTED) return;
    if (searched->count > 0) jaiBufAppendStr(searched, ", ");
    jaiBufAppendStr(searched, dir);
}

/* Base directory of a relative import: one dot is the importer's directory,
 * each further dot climbs one level. */
static bool relativeBase(const char *fromDir, int dots, char *out,
                         size_t outSize) {
    const char *start = (fromDir != NULL && fromDir[0] != '\0') ? fromDir : ".";
    if (!storeResolved(out, outSize, start)) return false;

    for (int i = 1; i < dots; i++) {
        char parent[JAI_MAX_PATH];
        jaiPathDirname(parent, sizeof parent, out);
        if (parent[0] == '\0' || strcmp(parent, out) == 0) return false;
        if (!storeResolved(out, outSize, parent)) return false;
    }
    return true;
}

bool jaiResolveModulePath(const char *dottedName, const char *fromDir,
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
        for (int i = 0; !found && i < sLibDirs.count; i++) {
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

/* The same search, asked speculatively. `jaiResolveModulePath` reports a miss —
 * as a diagnostic before the VM starts and as a raised ImportError once it is
 * running — because every caller so far was an import that has to fail. The
 * type checker is not: it asks whether a module it can see the name of has a
 * readable source, and a "no" is an answer, not an error. Both channels are
 * therefore restored to what they were. */
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
