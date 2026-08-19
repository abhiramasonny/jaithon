#: `VideoCapture` — frames from a camera or a video file, with OpenCV's method
#: names.
#:
#: ```
#: let camera = VideoCapture(0)
#: if not camera.is_opened() { throw RuntimeError(camera.why()) }
#: let (ok, frame) = camera.read()
#:
#: let clip = VideoCapture("recording.avi")
#: loop {
#:     let (ok, frame) = clip.read()
#:     if not ok { break }
#: }
#: ```
#:
#: A number opens a camera and a string opens a file, which is the distinction
#: OpenCV draws too. The file case reads AVI; see `reader.jai` for what that
#: covers and `writer.jai` for the other direction.
#:
#: macOS gates camera access per application. A command-line binary has no
#: bundle for the system to attribute the request to, so the first `open` may
#: be refused without a prompt; `permission()` says which of those happened and
#: `PERMISSION_HELP` says what to do about it. A file needs none of it.
import std.math
from std.gpu import Kernel
from ..core.mat import Mat, merge
from ..core.types import CV_8U, CV_MAKETYPE, Size
from .reader import AviReader

#: How far into the file the next frame is, in milliseconds.
pub const CAP_PROP_POS_MSEC = 0
#: How far into the file the next frame is, as a fraction between zero and one.
pub const CAP_PROP_POS_AVI_RATIO = 2
#: The four-character code naming the codec, packed into an int.
pub const CAP_PROP_FOURCC = 6
#: How many frames the file holds. Zero for a camera, which has no end.
pub const CAP_PROP_FRAME_COUNT = 7
#: Frame width, in pixels.
pub const CAP_PROP_FRAME_WIDTH = 3
#: Frame height, in pixels.
pub const CAP_PROP_FRAME_HEIGHT = 4
#: Frames per second.
pub const CAP_PROP_FPS = 5
#: Number of frames read so far.
pub const CAP_PROP_POS_FRAMES = 1
#: Channels in the frames this capture produces.
pub const CAP_PROP_CHANNEL = 43

#: Access has been granted.
pub const PERMISSION_GRANTED = 1
#: The system has not asked yet.
pub const PERMISSION_UNDECIDED = 0
#: The user refused.
pub const PERMISSION_DENIED = -1
#: This machine does not offer camera access at all.
pub const PERMISSION_UNAVAILABLE = -2

#: What to tell someone whose camera will not open.
pub let PERMISSION_HELP = "macOS grants camera access per application, and a command-line binary " +
    "has no bundle to attribute the request to. Add the terminal (or the " +
    "jaithon binary) under System Settings > Privacy & Security > Camera, " +
    "then run again."

#: Pack a four-character code back into the int `CAP_PROP_FOURCC` reports,
#: which is the same packing `fourcc` in `writer.jai` does from four strings.
fn fourcc_of(code: str) -> int {
    var packed = 0
    for index in 0..4 {
        let byte = if index < code.len() { ord(code[index]) } else { 32 }
        packed = packed | byte << index * 8
    }
    return packed
}

#: Whether the camera has been granted access, and whether it has been asked.
pub fn permission() -> int { return __prim__.camera_permission() }

#: Number of cameras attached.
pub fn device_count() -> int { return __prim__.camera_device_count() }

#: The system's name for one camera, or null when there is no such device.
pub fn device_name(index: int) -> str? { return __prim__.camera_device_name(index) }

#: Every camera's name, in device order.
pub fn device_names() -> list[str] {
    var names: list[str] = []
    for index in 0..device_count() {
        let name = device_name(index)
        names.push(name ?? f"camera {index}")
    }
    return names
}

#: A camera or a video file, read frame by frame.
pub class VideoCapture {
    var handle: int
    var file: AviReader?
    var requested: Size
    var requested_fps: float
    var frames: int
    var reason: str
    var grabbed: Mat?

    #: Open camera `source`, or the video file `source` names.
    #:
    #: A zero width, height or rate lets a camera pick; a file ignores all
    #: three, since what it holds was decided when it was written.
    pub fn init(self, source: any = 0, width: int = 0, height: int = 0, fps: float = 0.0) {
        self.requested = Size(width, height)
        self.requested_fps = fps
        self.frames = 0
        self.reason = ""
        self.grabbed = null
        self.handle = -1
        self.file = null
        self.open(source, width: width, height: height, fps: fps)
    }

    #: Open, or reopen on a different source. Returns whether it worked.
    pub fn open(self, source: any, width: int = 0, height: int = 0, fps: float = 0.0) -> bool {
        self.release()
        if source is str { return self.open_file(source) }
        if not source is int {
            self.reason = "a capture opens a camera index or a file path"
            return false
        }
        let index = source
        if device_count() == 0 {
            self.reason = "no camera is attached"
            return false
        }
        self.handle = __prim__.camera_open(index, width, height, fps)
        if self.handle >= 0 {
            self.reason = ""
            return true
        }
        let state = permission()
        self.reason = if state == PERMISSION_GRANTED {
            f"camera {index} would not start"
        } elif state == PERMISSION_DENIED {
            "camera access was refused. " + PERMISSION_HELP
        } elif state == PERMISSION_UNAVAILABLE {
            "this machine does not offer camera access"
        } else {
            "camera access has not been granted. " + PERMISSION_HELP
        }
        return false
    }

    #: A video file. Failure is reported the same way a camera's is -- through
    #: `is_opened` and `why` -- rather than by raising, because a caller that
    #: wrote `VideoCapture(path)` in place of `VideoCapture(0)` should not have
    #: to change how it handles the failure.
    fn open_file(self, path: str) -> bool {
        try {
            let reader = AviReader(path)
            self.file = reader
            self.requested = Size(reader.width, reader.height)
            self.requested_fps = reader.fps
            self.reason = ""
            return true
        } catch e: Error {
            self.file = null
            self.reason = str(e)
            return false
        }
    }

    #: Whether frames can be read.
    pub fn is_opened(self) -> bool { return self.handle >= 0 or self.file is not null }

    #: Why the last `open` failed, or the empty string when it did not.
    pub fn why(self) -> str { return self.reason }

    #: Read the next frame as a three-channel `CV_8U` image in BGR order.
    #:
    #: Returns `(false, empty)` when the camera is closed or no frame arrived
    #: within `timeout` seconds.
    pub fn read(self, timeout: float = 5.0) -> tuple[bool, Mat] {
        let clip = self.file
        if clip is not null {
            if self.frames >= clip.frame_count() {
                return (false, Mat(0, 0, CV_MAKETYPE(CV_8U, 3)))
            }
            let frame = clip.frame(self.frames)
            self.frames += 1
            return (true, frame)
        }
        if self.handle < 0 { return (false, Mat(0, 0, CV_MAKETYPE(CV_8U, 3))) }
        let frame = __prim__.camera_read(self.handle, timeout)
        if frame is null { return (false, Mat(0, 0, CV_MAKETYPE(CV_8U, 3))) }
        let width = frame[0]
        let height = frame[1]
        let pixels = frame[2]
        self.frames += 1
        #: The device hands over BGRA; the alpha channel is constant, so the
        #: image is built four-channel and the last one dropped rather than
        #: repacking on the host.
        let bgra = Mat.from_bytes(height, width, CV_MAKETYPE(CV_8U, 4), pixels)
        let planes = bgra.split()
        var colour: list[Mat] = []
        for index in 0..3 { colour.push(planes[index]) }
        return (true, merge(colour))
    }

    #: Read the next frame with its alpha channel left on.
    pub fn read_bgra(self, timeout: float = 5.0) -> tuple[bool, Mat] {
        if self.file is not null {
            throw RuntimeError("a video file has no alpha channel to keep")
        }
        if self.handle < 0 { return (false, Mat(0, 0, CV_MAKETYPE(CV_8U, 4))) }
        let frame = __prim__.camera_read(self.handle, timeout)
        if frame is null { return (false, Mat(0, 0, CV_MAKETYPE(CV_8U, 4))) }
        self.frames += 1
        return (true, Mat.from_bytes(frame[1], frame[0], CV_MAKETYPE(CV_8U, 4), frame[2]))
    }

    #: `grab` then `retrieve`, kept for source compatibility. Reading twice
    #: would drop a frame, so `grab` records the next one and `retrieve`
    #: returns it.
    pub fn grab(self, timeout: float = 5.0) -> bool {
        let (ok, frame) = self.read(timeout: timeout)
        self.grabbed = if ok { frame } else { null }
        return ok
    }

    #: The frame the last `grab` took.
    pub fn retrieve(self) -> tuple[bool, Mat] {
        let frame = self.grabbed
        if frame is null { return (false, Mat(0, 0, CV_MAKETYPE(CV_8U, 3))) }
        return (true, frame)
    }

    #: Read one of the `CAP_PROP_` properties.
    pub fn get(self, property: int) -> float {
        if property == CAP_PROP_POS_FRAMES { return float(self.frames) }
        if property == CAP_PROP_FPS { return self.requested_fps }
        if property == CAP_PROP_CHANNEL { return 3.0 }
        let clip = self.file
        if clip is not null {
            if property == CAP_PROP_FRAME_WIDTH { return float(clip.width) }
            if property == CAP_PROP_FRAME_HEIGHT { return float(clip.height) }
            if property == CAP_PROP_FRAME_COUNT { return float(clip.frame_count()) }
            if property == CAP_PROP_FOURCC { return float(fourcc_of(clip.codec)) }
            if property == CAP_PROP_POS_MSEC {
                if clip.fps <= 0.0 { return 0.0 }
                return float(self.frames) * 1000.0 / clip.fps
            }
            if property == CAP_PROP_POS_AVI_RATIO {
                let total = clip.frame_count()
                if total == 0 { return 0.0 }
                return float(self.frames) / float(total)
            }
            return 0.0
        }
        if property == CAP_PROP_FRAME_COUNT { return 0.0 }
        if self.handle < 0 { return 0.0 }
        let size = __prim__.camera_size(self.handle)
        if size is null { return 0.0 }
        if property == CAP_PROP_FRAME_WIDTH { return float(size[0]) }
        if property == CAP_PROP_FRAME_HEIGHT { return float(size[1]) }
        return 0.0
    }

    #: Set one of the `CAP_PROP_` properties. Size and rate take effect by
    #: reopening the device, which is the only way AVFoundation offers.
    pub fn set(self, property: int, value: float) -> bool {
        let clip = self.file
        if clip is not null {
            #: Seeking a file is setting which frame comes next, and the reader
            #: indexed every one of them when it opened, so it costs nothing.
            #: A camera has no frames to go back to and refuses.
            if property == CAP_PROP_POS_FRAMES {
                let at = int(value)
                if at < 0 or at > clip.frame_count() { return false }
                self.frames = at
                return true
            }
            if property == CAP_PROP_POS_AVI_RATIO {
                if value < 0.0 or value > 1.0 { return false }
                self.frames = int(value * float(clip.frame_count()))
                return true
            }
            return false
        }
        if property == CAP_PROP_FRAME_WIDTH {
            self.requested = Size(int(value), self.requested.height)
        } elif property == CAP_PROP_FRAME_HEIGHT {
            self.requested = Size(self.requested.width, int(value))
        } elif property == CAP_PROP_FPS {
            self.requested_fps = value
        } else {
            return false
        }
        return true
    }

    #: The size of the frames arriving, once one has.
    pub fn frame_size(self) -> Size? {
        let clip = self.file
        if clip is not null { return clip.frame_size() }
        if self.handle < 0 { return null }
        let size = __prim__.camera_size(self.handle)
        if size is null { return null }
        return Size(size[0], size[1])
    }

    #: Stop the capture and free the device or close the file.
    pub fn release(self) -> void {
        let clip = self.file
        if clip is not null {
            clip.release()
            self.file = null
        }
        self.grabbed = null
        self.frames = 0
        if self.handle < 0 { return }
        __prim__.camera_close(self.handle)
        self.handle = -1
    }
}
