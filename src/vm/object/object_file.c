/* object_file.c — ObjFile: a handle to an OS file, wrapping the readable/
 * writable/binary flags parsed out of the open mode string.
 *
 * The other half of a file's lifetime, closing it, lives with the rest of
 * per-kind teardown in object.c's jaiFreeObject; jaiFileNew here is only
 * construction.
 */

#include "vm/object/object.h"
#include "vm/object/object_internal.h"   /* pushObjRoot */

#include "vm/gc.h"
#include "vm/table.h"
#include "vm/vm.h"

/* ------------------------------------------------------------------ */
/* Files                                                                */
/* ------------------------------------------------------------------ */

ObjFile *jaiFileNew(FILE *handle, ObjString *path,
                    const char *mode) {
    pushObjRoot(path);
    ObjFile *file = JAI_ALLOCATE_OBJ(ObjFile, OBJ_FILE);
    jaiGCPopRoot();

    const char *p = mode != NULL ? mode : "r";
    bool readable = false;
    bool writable = false;
    bool binary = false;
    bool update = false;

    for (; *p != '\0'; ++p) {
        switch (*p) {
            case '+': update = true; break;
            case 'r': readable = true; break;
            case 'w':
            case 'a':
            case 'x': writable = true; break;
            case 'b': binary = true; break;
            default: break;
        }
    }

    file->handle = handle;
    file->path = path;
    file->readable = readable || update;
    file->writable = writable || update;
    file->binary = binary;
    file->closed = handle == NULL;
    return file;
}
