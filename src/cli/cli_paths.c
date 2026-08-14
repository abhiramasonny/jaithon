/* cli_paths.c — turning the paths on a command line into a list of .jai files.
 *
 * Every subcommand that takes source paths (check, build, disasm, ast,
 * tokens) wants the same thing: each argument that names a file, taken as
 * is; each argument that names a directory, walked recursively for .jai
 * files in a reproducible (sorted) order. collectAllInputs is that shared
 * expansion, kept in one place instead of once per command.
 */
#include "cli/cli_internal.h"

#include <stdlib.h>
#include <sys/stat.h>

#include "native/native.h"

static void freeNameList(char **names, int count) {
    if (names == NULL) return;
    for (int i = 0; i < count; i++) {
        if (names[i] != NULL) JAI_FREE_ARRAY(char, names[i], strlen(names[i]) + 1);
    }
    JAI_FREE_ARRAY(char *, names, count + 1);
}

static int compareNames(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

static inline bool hasJaiExtension(const char *name) {
    const size_t len = strlen(name);
    return len > 4 && memcmp(name + len - 4, ".jai", 4) == 0;
}

static bool collectDirectorySources(const char *path, PathList *out) {
    int count = 0;
    char **names = jaiListDir(path, &count);

    if (names == NULL) {
        cliError("cannot read directory: %s", path);
        return false;
    }

    if (count > 1)
        qsort(names, (size_t)count, sizeof(char *), compareNames);

    bool ok = true;

    for (int i = 0; i < count; ++i) {
        const char *const name = names[i];

        if (name == NULL || name[0] == '.')
            continue;

        if (strcmp(name, "__jaicache__") == 0)
            continue;

        char child[JAI_MAX_PATH];
        jaiPathJoin(child, sizeof child, path, name);

        struct stat st;
        if (stat(child, &st) == 0 && S_ISDIR(st.st_mode)) {
            if (!collectDirectorySources(child, out)) {
                ok = false;
                break;
            }
        } else if (hasJaiExtension(name)) {
            JAI_VEC_PUSH(char *, out, jaiStrdup(child));
        }
    }

    freeNameList(names, count);
    return ok;
}

static bool collectSources(const char *path, PathList *out) {
    struct stat st;

    if (stat(path, &st) != 0) {
        cliError("no such file or directory: %s", path);
        return false;
    }

    if (!S_ISDIR(st.st_mode)) {
        JAI_VEC_PUSH(char *, out, jaiStrdup(path));
        return true;
    }

    return collectDirectorySources(path, out);
}

bool collectAllInputs(const JaiCliOptions *opts, PathList *out,
                      const char *fallback) {
    JAI_VEC_INIT(out);
    if (opts->inputCount == 0 && fallback != NULL) {
        return collectSources(fallback, out);
    }
    for (int i = 0; i < opts->inputCount; i++) {
        if (!collectSources(opts->inputs[i], out)) return false;
    }
    return true;
}

void pathListFree(PathList *list) {
    for (int i = 0; i < list->count; i++) {
        char *p = list->data[i];
        if (p != NULL) JAI_FREE_ARRAY(char, p, strlen(p) + 1);
    }
    JAI_VEC_FREE(char *, list);
}
