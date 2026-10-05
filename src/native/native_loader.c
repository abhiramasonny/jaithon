/* native_loader.c — opens an Apple image the first time anything needs it.
 *
 * See native_image.h for why the images exist. This file owns the slots'
 * contents: the generated trampolines in build/gen/native_bridge.S jump
 * through jaiNativeSlots[i], every slot starts at a binder that calls
 * jaiNativeBind(i), and the first such call for a name opens the image that
 * exports it and points every slot of that image at its definition. Images
 * are opened independently -- a GPU program never loads the window or camera
 * image, nor the frameworks behind them.
 *
 * Where an image is looked for, in order:
 *
 *   $JAITHON_NATIVE_DIR             a directory, instead of the two below
 *   <dir of the real binary>        a build tree, and copies such as
 *                                   ./jaithon-base beside ./jaithon
 *   <dir>/../lib/jaithon            `make install`
 *
 * Each image's file name carries its own LC_UUID
 * (jaithon-native-gpu-<12 hex>.dylib), and the loaded image's UUID is compared
 * in full against the one this binary was linked next to. So two builds of
 * different native sources each find their own images, and a stale one is
 * refused rather than called with the wrong idea of a struct. A refused or
 * missing image is reported once on stderr and its functions run stubs.c's
 * definitions -- exactly what a machine with no GPU, window or camera runs.
 *
 * JAITHON_NATIVE=0 runs the stubs without looking for any image, which is how
 * the fallback can be tested on a Mac. JAITHON_NATIVE_LAZY=0 opens every image
 * at process start: the old cost, paid by dlopen rather than by the static
 * link, as a kill switch if lazy loading ever matters to some framework. */

#include "native/native_image.h"

#ifdef JAI_NATIVE_SPLIT

/* Before native.h, so that its prototypes declare the renamed stub functions
 * (jaiNativeStub_jaiGpuAvailable and so on) that stubs.c defines here. */
#include "gen/native_rename.h"

#include "common/common.h"
#include "jai_native_images.h"   /* build/<type>/native, from the linked images */
#include "native/native.h"
#include "runtime/parallel.h"

#include <dlfcn.h>
#include <mach-o/loader.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Defined in the generated bridge, one per name below and in that order. */
extern void *jaiNativeSlots[];

static const char *const kNames[] = {
#define JAI_NATIVE_FN(name) #name,
#include "gen/native_list.h"
#undef JAI_NATIVE_FN
};

static void *const kStubs[] = {
#define JAI_NATIVE_FN(name) (void *)jaiNativeStub_##name,
#include "gen/native_list.h"
#undef JAI_NATIVE_FN
};

#define NATIVE_COUNT (sizeof kNames / sizeof kNames[0])

static const char *const kImageFiles[JAI_NATIVE_IMAGE_COUNT] = JAI_NATIVE_IMAGE_FILES;
static const uint8_t kImageUuids[JAI_NATIVE_IMAGE_COUNT][16] = JAI_NATIVE_IMAGE_UUIDS;
/* Per name: the image that exports it, or -1 when only the stub exists. */
static const int8_t kImageOf[] = JAI_NATIVE_IMAGE_OF;

_Static_assert(sizeof kImageOf == NATIVE_COUNT,
               "jai_native_images.h was generated from a different name list");

static pthread_mutex_t gBindLock = PTHREAD_MUTEX_INITIALIZER;
static bool gImageBound[JAI_NATIVE_IMAGE_COUNT];

static bool uuidMatches(const void *addressInImage, const uint8_t *expected) {
    Dl_info info;
    if (dladdr(addressInImage, &info) == 0 || info.dli_fbase == NULL) return false;
    const struct mach_header_64 *header = info.dli_fbase;
    if (header->magic != MH_MAGIC_64) return false;
    const uint8_t *cursor = (const uint8_t *)(header + 1);
    for (uint32_t i = 0; i < header->ncmds; i++) {
        struct load_command command;
        memcpy(&command, cursor, sizeof command);
        if (command.cmd == LC_UUID) {
            struct uuid_command uuid;
            memcpy(&uuid, cursor, sizeof uuid);
            return memcmp(uuid.uuid, expected, sizeof uuid.uuid) == 0;
        }
        cursor += command.cmdsize;
    }
    return false;
}

static void *tryOpen(int image, const char *path, char *why, size_t whySize) {
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        const char *err = dlerror();
        snprintf(why, whySize, "%s", err != NULL ? err : path);
        return NULL;
    }
    JaiNativeImageInitFn init = (JaiNativeImageInitFn)dlsym(handle, JAI_NATIVE_IMAGE_INIT);
    if (init == NULL || !uuidMatches((const void *)init, kImageUuids[image])) {
        snprintf(why, whySize, "%s was not built with this binary", path);
        return NULL;
    }
    static const JaiNativeCoreApi api = {
        .abi = JAI_NATIVE_IMAGE_ABI,
        .size = (uint32_t)sizeof(JaiNativeCoreApi),
        .realloc = jaiRealloc,
        .calloc = jaiCalloc,
        .clockMonotonic = jaiClockMonotonic,
        .parallelChunks = jaiParallelChunks,
    };
    if (!init(&api)) {
        snprintf(why, whySize, "%s refused this binary's interface", path);
        return NULL;
    }
    return handle;
}

static bool tryDir(int image, const char *dir, void **handle, char *why, size_t whySize) {
    char path[JAI_MAX_PATH];
    jaiPathJoin(path, sizeof path, dir, kImageFiles[image]);
    if (access(path, R_OK) != 0) return false;
    *handle = tryOpen(image, path, why, whySize);
    return true;
}

static void *openImage(int image) {
    const char *enabled = getenv("JAITHON_NATIVE");
    if (enabled != NULL && strcmp(enabled, "0") == 0) return NULL;
    /* See jaiProcessIsForkedChild: nothing this child has printed is kept. */
    if (jaiProcessIsForkedChild) _exit(JAI_EXIT_NEEDS_EXEC);

    char why[JAI_MAX_PATH + 128];
    why[0] = '\0';
    void *handle = NULL;
    bool found = false;

    const char *override = getenv("JAITHON_NATIVE_DIR");
    if (override != NULL && override[0] != '\0') {
        found = tryDir(image, override, &handle, why, sizeof why);
        if (!found) snprintf(why, sizeof why, "it is not in %s", override);
    } else {
        const char *exe = jaiExecutablePath();
        if (exe != NULL && exe[0] != '\0') {
            char dir[JAI_MAX_PATH];
            char installed[JAI_MAX_PATH];
            jaiPathDirname(dir, sizeof dir, exe);
            jaiPathJoin(installed, sizeof installed, dir, "../lib/jaithon");
            found = tryDir(image, dir, &handle, why, sizeof why) ||
                    tryDir(image, installed, &handle, why, sizeof why);
        }
        if (!found)
            snprintf(why, sizeof why, "it is not next to %s",
                     exe != NULL ? exe : "the jaithon binary");
    }
    if (handle == NULL)
        fprintf(stderr,
                "jaithon: warning: cannot load %s: %s; what it provides acts "
                "as on a machine without it\n", kImageFiles[image], why);
    return handle;
}

/* Caller holds gBindLock. */
static void bindImage(int image) {
    void *handle = openImage(image);
    for (size_t i = 0; i < NATIVE_COUNT; i++) {
        if (kImageOf[i] != image) continue;
        void *target = handle != NULL ? dlsym(handle, kNames[i]) : NULL;
        if (target == NULL) target = kStubs[i];
        __atomic_store_n(&jaiNativeSlots[i], target, __ATOMIC_RELEASE);
    }
    gImageBound[image] = true;
}

void *jaiNativeBind(uint32_t index) {
    int image = kImageOf[index];
    pthread_mutex_lock(&gBindLock);
    if (image < 0)
        __atomic_store_n(&jaiNativeSlots[index], kStubs[index], __ATOMIC_RELEASE);
    else if (!gImageBound[image])
        bindImage(image);
    pthread_mutex_unlock(&gBindLock);
    return __atomic_load_n(&jaiNativeSlots[index], __ATOMIC_ACQUIRE);
}

__attribute__((constructor))
static void bindEagerlyWhenAsked(void) {
    const char *lazy = getenv("JAITHON_NATIVE_LAZY");
    if (lazy == NULL || strcmp(lazy, "0") != 0) return;
    pthread_mutex_lock(&gBindLock);
    for (int image = 0; image < JAI_NATIVE_IMAGE_COUNT; image++)
        if (!gImageBound[image]) bindImage(image);
    pthread_mutex_unlock(&gBindLock);
}

#else

/* Keeps the translation unit non-empty where the images are not split out. */
typedef int JaiNativeLoaderUnit;

#endif /* JAI_NATIVE_SPLIT */
