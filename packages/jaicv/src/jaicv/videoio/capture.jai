#: `VideoCapture` — frames from a camera, with OpenCV's method names.
#:
#: ```
#: let camera = VideoCapture(0)
#: if not camera.is_opened() { throw RuntimeError(camera.why()) }
#: let (ok, frame) = camera.read()
#: ```
#:
#: macOS gates camera access per application. A command-line binary has no
#: bundle for the system to attribute the request to, so the first `open` may
#: be refused without a prompt; `permission()` says which of those happened and
#: `PERMISSION_HELP` says what to do about it.
import std.math
from std.gpu import Kernel
from ..core.mat import Mat, merge
from ..core.types import CV_8U, CV_MAKETYPE, Size

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
pub let PERMISSION_HELP =
    "macOS grants camera access per application, and a command-line binary " +
    "has no bundle to attribute the request to. Add the terminal (or the " +
    "jaithon binary) under System Settings > Privacy & Security > Camera, " +
    "then run again."

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

#: A live camera, read frame by frame.
pub class VideoCapture {
    var handle: int
    var requested: Size
    var requested_fps: float
    var frames: int
    var reason: str
    var grabbed: Mat?

    #: Open camera `index`. A zero width, height or rate lets the device pick.
    pub fn init(self, index: int = 0, width: int = 0, height: int = 0, fps: float = 0.0) {
        self.requested = Size(width, height)
        self.requested_fps = fps
        self.frames = 0
        self.reason = ""
        self.grabbed = null
        self.handle = -1
        self.open(index, width: width, height: height, fps: fps)
    }

    #: Open, or reopen on a different device. Returns whether it worked.
    pub fn open(self, index: int, width: int = 0, height: int = 0, fps: float = 0.0) -> bool {
        self.release()
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

    #: Whether frames can be read.
    pub fn is_opened(self) -> bool { return self.handle >= 0 }

    #: Why the last `open` failed, or the empty string when it did not.
    pub fn why(self) -> str { return self.reason }

    #: Read the next frame as a three-channel `CV_8U` image in BGR order.
    #:
    #: Returns `(false, empty)` when the camera is closed or no frame arrived
    #: within `timeout` seconds.
    pub fn read(self, timeout: float = 5.0) -> tuple[bool, Mat] {
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
        if self.handle < 0 { return null }
        let size = __prim__.camera_size(self.handle)
        if size is null { return null }
        return Size(size[0], size[1])
    }

    #: Stop the capture and free the device.
    pub fn release(self) -> void {
        if self.handle < 0 { return }
        __prim__.camera_close(self.handle)
        self.handle = -1
        self.grabbed = null
    }
}
