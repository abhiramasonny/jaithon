/* camera.m — AVFoundation video capture behind the C ABI in native.h.
 *
 * A capture session runs on its own queue and the delegate keeps only the most
 * recent frame. That is the right trade for computer vision: a reader that
 * falls behind wants the newest frame, not a backlog, and dropping is what a
 * real-time pipeline does anyway.
 *
 * Frames are handed over as packed BGRA bytes, the layout Core Video gives us
 * and the layout jaicv's `Mat.from_bytes` expands in one native pass. */

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include "native/native.h"

#include <string.h>

@interface JaiCameraSink : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property(nonatomic, strong) NSLock *lock;
@property(nonatomic, assign) uint8_t *frame;
@property(nonatomic, assign) size_t frameBytes;
@property(nonatomic, assign) int width;
@property(nonatomic, assign) int height;
@property(nonatomic, assign) uint64_t sequence;
@end

@implementation JaiCameraSink

- (instancetype)init {
    self = [super init];
    if (self != nil) {
        _lock = [[NSLock alloc] init];
        _frame = NULL;
        _frameBytes = 0;
        _width = 0;
        _height = 0;
        _sequence = 0;
    }
    return self;
}

- (void)dealloc {
    if (_frame != NULL) free(_frame);
}

- (void)captureOutput:(AVCaptureOutput *)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection *)connection {
    (void)output;
    (void)connection;
    CVImageBufferRef image = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (image == NULL) return;
    CVPixelBufferLockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
    const uint8_t *source = (const uint8_t *)CVPixelBufferGetBaseAddress(image);
    size_t stride = CVPixelBufferGetBytesPerRow(image);
    size_t width = CVPixelBufferGetWidth(image);
    size_t height = CVPixelBufferGetHeight(image);
    if (source != NULL && width > 0 && height > 0) {
        size_t packed = width * height * 4u;
        [self.lock lock];
        if (self.frameBytes != packed) {
            uint8_t *grown = (uint8_t *)realloc(self.frame, packed);
            if (grown != NULL) {
                self.frame = grown;
                self.frameBytes = packed;
            }
        }
        if (self.frame != NULL && self.frameBytes == packed) {
            /* Core Video pads each row out to its own alignment, so the copy
             * is row by row rather than one memcpy. */
            for (size_t row = 0; row < height; row++) {
                memcpy(self.frame + row * width * 4u, source + row * stride, width * 4u);
            }
            self.width = (int)width;
            self.height = (int)height;
            self.sequence += 1;
        }
        [self.lock unlock];
    }
    CVPixelBufferUnlockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
}

@end

struct JaiCamera {
    AVCaptureSession *session;
    AVCaptureDeviceInput *input;
    AVCaptureVideoDataOutput *output;
    JaiCameraSink *sink;
    dispatch_queue_t queue;
    uint64_t delivered;
};

static NSArray<AVCaptureDevice *> *cameraDevices(void) {
    /* `AVCaptureDeviceTypeExternal` replaced `...ExternalUnknown` in macOS 14;
     * both are listed so the same source builds either side of that. */
    NSMutableArray<AVCaptureDeviceType> *types =
        [@[ AVCaptureDeviceTypeBuiltInWideAngleCamera ] mutableCopy];
    if (@available(macOS 14.0, *)) {
        [types addObject:AVCaptureDeviceTypeExternal];
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        [types addObject:AVCaptureDeviceTypeExternalUnknown];
#pragma clang diagnostic pop
    }
    AVCaptureDeviceDiscoverySession *discovery =
        [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:types
                                                               mediaType:AVMediaTypeVideo
                                                                position:AVCaptureDevicePositionUnspecified];
    return discovery.devices;
}

int jaiCameraPermission(void) {
    @autoreleasepool {
        AVAuthorizationStatus status =
            [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
        if (status == AVAuthorizationStatusAuthorized) return 1;
        if (status == AVAuthorizationStatusNotDetermined) return 0;
        if (status == AVAuthorizationStatusDenied) return -1;
        return -2;
    }
}

int jaiCameraDeviceCount(void) {
    @autoreleasepool {
        return (int)cameraDevices().count;
    }
}

bool jaiCameraDeviceName(int index, char *buffer, size_t capacity) {
    if (buffer == NULL || capacity == 0) return false;
    @autoreleasepool {
        NSArray<AVCaptureDevice *> *devices = cameraDevices();
        if (index < 0 || (NSUInteger)index >= devices.count) return false;
        const char *name = devices[(NSUInteger)index].localizedName.UTF8String;
        if (name == NULL) return false;
        strncpy(buffer, name, capacity - 1);
        buffer[capacity - 1] = '\0';
        return true;
    }
}

/* Authorisation is asked for once and waited on: a capture session that starts
 * before the answer arrives delivers nothing and looks like a broken camera. */
static bool cameraAuthorised(void) {
    AVAuthorizationStatus status =
        [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
    if (status == AVAuthorizationStatusAuthorized) return true;
    if (status != AVAuthorizationStatusNotDetermined) return false;

    __block bool granted = false;
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                             completionHandler:^(BOOL allowed) {
                                 granted = allowed;
                                 dispatch_semaphore_signal(done);
                             }];
    dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 30ll * NSEC_PER_SEC));
    return granted;
}

JaiCamera *jaiCameraOpen(int index, int width, int height, double fps) {
    @autoreleasepool {
        if (!cameraAuthorised()) return NULL;
        NSArray<AVCaptureDevice *> *devices = cameraDevices();
        if (devices.count == 0) return NULL;
        if (index < 0) index = 0;
        if ((NSUInteger)index >= devices.count) return NULL;
        AVCaptureDevice *device = devices[(NSUInteger)index];

        NSError *error = nil;
        AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device
                                                                            error:&error];
        if (input == nil) return NULL;

        AVCaptureSession *session = [[AVCaptureSession alloc] init];
        if (![session canAddInput:input]) return NULL;
        [session addInput:input];

        AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
        output.videoSettings = @{
            (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)
        };
        output.alwaysDiscardsLateVideoFrames = YES;
        if (![session canAddOutput:output]) return NULL;
        [session addOutput:output];

        JaiCamera *camera = (JaiCamera *)calloc(1, sizeof(JaiCamera));
        if (camera == NULL) return NULL;

        JaiCameraSink *sink = [[JaiCameraSink alloc] init];
        dispatch_queue_t queue = dispatch_queue_create("jaithon.camera", DISPATCH_QUEUE_SERIAL);
        [output setSampleBufferDelegate:sink queue:queue];

        /* A requested size is honoured by choosing the device format that
         * matches, since the session presets only offer a handful of shapes. */
        if (width > 0 && height > 0 && [device lockForConfiguration:&error]) {
            AVCaptureDeviceFormat *best = nil;
            for (AVCaptureDeviceFormat *format in device.formats) {
                CMVideoDimensions size =
                    CMVideoFormatDescriptionGetDimensions(format.formatDescription);
                if (size.width == width && size.height == height) {
                    best = format;
                    break;
                }
            }
            if (best != nil) device.activeFormat = best;
            if (fps > 0.0) {
                CMTime duration = CMTimeMake(1, (int32_t)(fps + 0.5));
                device.activeVideoMinFrameDuration = duration;
                device.activeVideoMaxFrameDuration = duration;
            }
            [device unlockForConfiguration];
        }

        camera->session = session;
        camera->input = input;
        camera->output = output;
        camera->sink = sink;
        camera->queue = queue;
        camera->delivered = 0;
        [session startRunning];
        return camera;
    }
}

void jaiCameraClose(JaiCamera *camera) {
    if (camera == NULL) return;
    @autoreleasepool {
        [camera->session stopRunning];
        [camera->output setSampleBufferDelegate:nil queue:NULL];
        camera->session = nil;
        camera->input = nil;
        camera->output = nil;
        camera->sink = nil;
        camera->queue = NULL;
    }
    free(camera);
}

bool jaiCameraSize(JaiCamera *camera, int *width, int *height) {
    if (camera == NULL || camera->sink == NULL) return false;
    [camera->sink.lock lock];
    if (width != NULL) *width = camera->sink.width;
    if (height != NULL) *height = camera->sink.height;
    bool ready = camera->sink.width > 0 && camera->sink.height > 0;
    [camera->sink.lock unlock];
    return ready;
}

bool jaiCameraRead(JaiCamera *camera, uint8_t *destination, size_t capacity,
                   int *width, int *height, double timeoutSeconds) {
    if (camera == NULL || camera->sink == NULL) return false;

    /* Wait for a frame newer than the last one handed out, so a read never
     * returns the same picture twice and a caller polling in a loop sees the
     * frame rate the camera is actually running at. */
    const useconds_t slice = 2000;
    double waited = 0.0;
    for (;;) {
        [camera->sink.lock lock];
        uint64_t sequence = camera->sink.sequence;
        bool fresh = sequence > camera->delivered && camera->sink.frame != NULL;
        if (fresh) {
            size_t needed = (size_t)camera->sink.width * (size_t)camera->sink.height * 4u;
            if (destination != NULL && capacity >= needed) {
                memcpy(destination, camera->sink.frame, needed);
            } else if (destination != NULL) {
                [camera->sink.lock unlock];
                return false;
            }
            if (width != NULL) *width = camera->sink.width;
            if (height != NULL) *height = camera->sink.height;
            camera->delivered = sequence;
            [camera->sink.lock unlock];
            return true;
        }
        [camera->sink.lock unlock];
        if (timeoutSeconds >= 0.0 && waited >= timeoutSeconds) return false;
        usleep(slice);
        waited += (double)slice / 1e6;
    }
}
