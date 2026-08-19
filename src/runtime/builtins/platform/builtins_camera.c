/* builtins_camera.c — __prim__.camera_*: video capture for jaicv's VideoCapture.
 *
 * Frames come back as packed BGRA bytes. Converting to BGR here would mean
 * choosing a layout for every caller; handing over what the device produced
 * lets the image type expand it in one pass. */

#include "runtime/builtins/builtins.h"
#include "runtime/runtime.h"
#include "native/native.h"
#include "vm/gc.h"

/* Handles are indices into a small table rather than raw pointers, so a stale
 * handle from Jaithon can be rejected instead of dereferenced. */
#define JAI_MAX_CAMERAS 8

static JaiCamera *sCameras[JAI_MAX_CAMERAS];

static bool cameraFromHandle(Value value, int position, const char *who, JaiCamera **out) {
    int64_t handle;
    if (!jaiArgInt(value, position, who, &handle)) return false;
    if (handle < 0 || handle >= JAI_MAX_CAMERAS || sCameras[handle] == NULL) {
        return jaiThrow(vm.cValueError, "%s(): camera handle %lld is not open", who,
                        (long long)handle);
    }
    *out = sCameras[handle];
    return true;
}

static bool nCameraPermission(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = INT_VAL(jaiCameraPermission());
    return true;
}

static bool nCameraDeviceCount(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = INT_VAL(jaiCameraDeviceCount());
    return true;
}

static bool nCameraDeviceName(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t index;
    if (!jaiArgInt(args[0], 1, "camera_device_name", &index)) return false;
    char name[256];
    if (!jaiCameraDeviceName((int)index, name, sizeof name)) {
        *out = NULL_VAL;
        return true;
    }
    ObjString *text = jaiStringNew(name, strlen(name));
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

static bool nCameraOpen(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t index, width, height;
    double fps;
    if (!jaiArgInt(args[0], 1, "camera_open", &index)) return false;
    if (!jaiArgInt(args[1], 2, "camera_open", &width)) return false;
    if (!jaiArgInt(args[2], 3, "camera_open", &height)) return false;
    if (!jaiArgFloat(args[3], 4, "camera_open", &fps)) return false;

    int slot = -1;
    for (int i = 0; i < JAI_MAX_CAMERAS; i++) {
        if (sCameras[i] == NULL) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        return jaiThrow(vm.cRuntimeError, "camera_open(): %d cameras are already open",
                        JAI_MAX_CAMERAS);
    }

    JaiCamera *camera = jaiCameraOpen((int)index, (int)width, (int)height, fps);
    if (camera == NULL) {
        *out = INT_VAL(-1);
        return true;
    }
    sCameras[slot] = camera;
    *out = INT_VAL(slot);
    return true;
}

static bool nCameraClose(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t handle;
    if (!jaiArgInt(args[0], 1, "camera_close", &handle)) return false;
    if (handle >= 0 && handle < JAI_MAX_CAMERAS && sCameras[handle] != NULL) {
        jaiCameraClose(sCameras[handle]);
        sCameras[handle] = NULL;
    }
    *out = NULL_VAL;
    return true;
}

static bool nCameraSize(int argc, Value *args, Value *out) {
    (void)argc;
    JaiCamera *camera;
    if (!cameraFromHandle(args[0], 1, "camera_size", &camera)) return false;
    int width = 0, height = 0;
    if (!jaiCameraSize(camera, &width, &height)) {
        *out = NULL_VAL;
        return true;
    }
    ObjList *pair = jaiListNew(2);
    if (pair == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(pair));
    jaiListPush(pair, INT_VAL(width));
    jaiListPush(pair, INT_VAL(height));
    jaiGCPopRoot();
    *out = OBJ_VAL(pair);
    return true;
}

static bool nCameraRead(int argc, Value *args, Value *out) {
    (void)argc;
    JaiCamera *camera;
    if (!cameraFromHandle(args[0], 1, "camera_read", &camera)) return false;
    double timeout;
    if (!jaiArgFloat(args[1], 2, "camera_read", &timeout)) return false;

    int width = 0, height = 0;
    if (!jaiCameraSize(camera, &width, &height) || width <= 0 || height <= 0) {
        /* No frame has arrived yet, so there is no size to allocate against;
         * wait for the first one before asking for pixels. */
        if (!jaiCameraRead(camera, NULL, 0, &width, &height, timeout)) {
            *out = NULL_VAL;
            return true;
        }
    }

    size_t needed = (size_t)width * (size_t)height * 4u;
    uint8_t *buffer = JAI_GROW_ARRAY(uint8_t, NULL, 0, needed);
    if (buffer == NULL) return jaiThrow(vm.cRuntimeError, "camera_read(): out of memory");
    if (!jaiCameraRead(camera, buffer, needed, &width, &height, timeout)) {
        JAI_FREE_ARRAY(uint8_t, buffer, needed);
        *out = NULL_VAL;
        return true;
    }

    ObjBytes *pixels = jaiBytesNew(buffer, (size_t)width * (size_t)height * 4u);
    JAI_FREE_ARRAY(uint8_t, buffer, needed);
    if (pixels == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(pixels));
    ObjList *frame = jaiListNew(3);
    if (frame == NULL) {
        jaiGCPopRoot();
        return false;
    }
    jaiGCPushRoot(OBJ_VAL(frame));
    jaiListPush(frame, INT_VAL(width));
    jaiListPush(frame, INT_VAL(height));
    jaiListPush(frame, OBJ_VAL(pixels));
    jaiGCPopRoot();
    jaiGCPopRoot();
    *out = OBJ_VAL(frame);
    return true;
}

void jaiRegisterCameraPrimitives(void) {
    jaiDefineNative("__prim__.camera_permission",   nCameraPermission,  0, 0);
    jaiDefineNative("__prim__.camera_device_count", nCameraDeviceCount, 0, 0);
    jaiDefineNative("__prim__.camera_device_name",  nCameraDeviceName,  1, 1);
    jaiDefineNative("__prim__.camera_open",         nCameraOpen,        4, 4);
    jaiDefineNative("__prim__.camera_close",        nCameraClose,       1, 1);
    jaiDefineNative("__prim__.camera_size",         nCameraSize,        1, 1);
    jaiDefineNative("__prim__.camera_read",         nCameraRead,        2, 2);
}
