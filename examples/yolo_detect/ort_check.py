"""The reference half of ort_check.jai: yolov8n through onnxruntime.

Prints a shape, a sum, an absolute maximum and a handful of sampled values, in
the same order and format the Jaithon side prints them, so the two outputs can
be read side by side. See ort_check.jai for what the comparison is for.
"""
import numpy as np
import onnxruntime as ort

MODEL = "examples/yolo_detect/yolov8n.onnx"
SIDE = 640


def main() -> int:
    values = (np.arange(3 * SIDE * SIDE, dtype=np.int64) * 37 % 251).astype(np.float32) / 251.0
    x = values.reshape(1, 3, SIDE, SIDE)
    session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    out = session.run(None, {name: x})[0]
    flat = out.reshape(-1)
    print(f"shape {out.shape}")
    print(f"sum {float(flat.sum()):.6f}")
    print(f"absmax {float(np.abs(flat).max()):.6f}")
    for index in (0, 1, 1000, 8400, 100000, flat.size - 1):
        print(f"at {index}: {float(flat[index]):.6f}")
    return 0


raise SystemExit(main())
