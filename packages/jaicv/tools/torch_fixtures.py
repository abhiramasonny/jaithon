#!/usr/bin/env python3
"""Write the .pt checkpoints tests/test_torch_reader.jai reads, and record what
torch says is in them.

Every fixture is deliberately tiny -- these get committed. The oracle file
holds one line per tensor: the dotted key the reader is expected to produce,
the dtype name, the shape, and every element printed with enough digits to
round-trip a float64.

    ~/.venvs/scratch/bin/python packages/jaicv/tools/torch_fixtures.py
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "tests" / "models"


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            flatten(f"{prefix}.{index}" if prefix else str(index), item, out)
    elif isinstance(value, torch.Tensor):
        out.append((prefix, value))
    return out


def render(tensor: torch.Tensor) -> str:
    flat = tensor.detach().reshape(-1)
    if tensor.dtype == torch.bool:
        return " ".join("1" if bool(v) else "0" for v in flat.tolist())
    if tensor.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        return " ".join(f"{float(v):.17g}" for v in flat.tolist())
    return " ".join(str(int(v)) for v in flat.tolist())


def emit(handle, name: str, payload) -> None:
    for key, tensor in flatten("", payload, []):
        dtype = str(tensor.dtype).replace("torch.", "")
        shape = list(tensor.shape)
        handle.write(
            f"tensor {name} {key} {dtype} {len(shape)} "
            f"{' '.join(str(d) for d in shape)} {tensor.numel()} {render(tensor)}\n".replace(
                "  ", " "
            )
        )


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, bias=True)
        self.bn = torch.nn.BatchNorm2d(4)
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.bn(self.conv(x)).mean((2, 3)))


def rewrite_compressed(source: Path, target: Path) -> None:
    """The same archive again, deflated and forced through the zip64 extras.

    torch's own writer stores every member uncompressed and stays under the
    4 GB fields, so nothing it produces exercises method 8 or the zip64
    extension. Real checkpoints past 4 GB do, and this rewrite reaches both
    paths at a few kilobytes.
    """
    import zipfile

    with zipfile.ZipFile(source) as reader:
        members = [(i.filename, reader.read(i.filename)) for i in reader.infolist()]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as writer:
        for name, payload in members:
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            with writer.open(info, "w", force_zip64=True) as handle:
                handle.write(payload)


def rewrite_zip64(source: Path, target: Path) -> None:
    """The same archive again, written the way a >4 GB checkpoint has to be.

    Every central-directory entry parks its sizes and its local-header offset
    in a zip64 extra field behind the 0xFFFFFFFF sentinel, and the end record
    is the zip64 pair. A few kilobytes of fixture then walks exactly the code
    a real multi-gigabyte checkpoint needs, which no torch-written file does.
    """
    import zipfile

    with zipfile.ZipFile(source) as reader:
        members = [(i.filename, reader.read(i.filename)) for i in reader.infolist()]

    body = bytearray()
    directory = bytearray()
    for name, payload in members:
        encoded = name.encode()
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        offset = len(body)
        body += struct.pack(
            "<IHHHHHIIIHH", 0x04034B50, 45, 0, 0, 0, 0, crc, len(payload), len(payload),
            len(encoded), 0,
        )
        body += encoded
        body += payload
        extra = struct.pack("<HHQQQ", 0x0001, 24, len(payload), len(payload), offset)
        directory += struct.pack(
            "<IHHHHHHIIIHHHHHII", 0x02014B50, 45, 45, 0, 0, 0, 0, crc,
            0xFFFFFFFF, 0xFFFFFFFF, len(encoded), len(extra), 0, 0, 0, 0, 0xFFFFFFFF,
        )
        directory += encoded
        directory += extra

    directory_at = len(body)
    end_at = directory_at + len(directory)
    tail = struct.pack(
        "<IQHHIIQQQQ", 0x06064B50, 44, 45, 45, 0, 0,
        len(members), len(members), len(directory), directory_at,
    )
    tail += struct.pack("<IIQI", 0x07064B50, 0, end_at, 1)
    tail += struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    target.write_bytes(bytes(body + directory + tail))


def write_hostile(target: Path) -> None:
    """A checkpoint whose pickle reaches for os.system.

    Written by hand rather than through pickle.dumps so it stays exactly this
    and cannot drift with the host Python. The reader must refuse it by name.
    """
    import zipfile

    body = (
        b"\x80\x02"                          # PROTO 2
        b"cos\nsystem\n"                     # GLOBAL os.system
        b"X\x0b\x00\x00\x00echo pwned"     # BINUNICODE
        b"\x85"                              # TUPLE1
        b"R"                                 # REDUCE
        b"."                                 # STOP
    )
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as writer:
        writer.writestr("torch_hostile/data.pkl", body)
        writer.writestr("torch_hostile/byteorder", "little")
        writer.writestr("torch_hostile/version", "3\n")


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)

    net = Net()
    net.eval()
    with torch.no_grad():
        net(torch.randn(2, 3, 8, 8))
    small_net = net.state_dict()

    base = torch.arange(12, dtype=torch.float32) * 0.5 - 2.0
    wide = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    dtypes = {
        "f32": torch.tensor([[1.5, -2.25], [3.125, 0.0]], dtype=torch.float32),
        "f64": torch.tensor([1e-300, 3.14159265358979, -2.5e300], dtype=torch.float64),
        "i64": torch.tensor([-9223372036854775808, 0, 9223372036854775807], dtype=torch.int64),
        "i32": torch.tensor([-2147483648, 7, 2147483647], dtype=torch.int32),
        "i16": torch.tensor([-32768, 32767], dtype=torch.int16),
        "i8": torch.tensor([-128, 127], dtype=torch.int8),
        "u8": torch.tensor([0, 128, 255], dtype=torch.uint8),
        "bool": torch.tensor([True, False, True, True], dtype=torch.bool),
        "f16": torch.tensor([1.0, -0.5, 65504.0], dtype=torch.float16),
        "bf16": torch.tensor([1.0, -0.5, 100.0], dtype=torch.bfloat16),
        "scalar": torch.tensor(3.5),
        "empty": torch.zeros(0, dtype=torch.float32),
        "empty2d": torch.zeros(0, 5, dtype=torch.float32),
        "transposed": wide.t(),
        "slice": wide[:, 1:3],
        "step": base[::3],
        "shared_head": base[:6],
        "shared_tail": base[6:],
        "nested": {"inner": {"deep": torch.tensor([[7.0, 8.0], [9.0, 10.0]])}},
        "listed": [torch.tensor([1.0]), torch.tensor([2.0, 3.0])],
        "special": torch.tensor([float("inf"), float("-inf"), 0.0, -0.0, 1e-45], dtype=torch.float32),
    }

    #: The shape a training checkpoint actually has: the weights under a key,
    #: an optimizer state whose own keys are ints, and several leaves that are
    #: not tensors at all and must be dropped rather than tripped over.
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss = net(torch.randn(2, 3, 8, 8)).square().mean()
    loss.backward()
    optimizer.step()
    checkpoint = {
        "epoch": 12,
        "notes": "run 7",
        "history": [0.5, 0.25],
        "model": small_net,
        "optimizer": optimizer.state_dict(),
    }

    #: torch writes protocol 2 and nothing else, so the opcodes protocols 3
    #: to 5 added -- the frames, the memo without an explicit slot, the short
    #: string and byte forms, the set opcodes -- appear in no file it produces.
    #: Asking for protocol 5 reaches all of them, and the odd leaves here reach
    #: the singular APPEND and SETITEM, a big integer, and a byte string.
    modern = {
        "weights": small_net,
        "pair": (torch.tensor([4.0, 5.0]), torch.tensor([[6.0]])),
        "single": [torch.tensor([7.0, 8.0, 9.0])],
        "only": {"one": torch.tensor([10.0])},
        "flag": True,
        "nothing": None,
        "big": 2 ** 40,
        "small": -(2 ** 62),
        "blob": b"\x00\x01\xfe",
        "text": "h\u00e9llo",
        "ratio": 0.125,
        "tags": {"a", "b"},
    }

    torch.save(small_net, MODELS / "torch_smallnet.pt")
    torch.save(modern, MODELS / "torch_protocol5.pt", pickle_protocol=5)
    torch.save(checkpoint, MODELS / "torch_checkpoint.pt")
    torch.save(dtypes, MODELS / "torch_dtypes.pt")
    torch.save(net, MODELS / "torch_module.pt")
    torch.save(small_net, MODELS / "torch_legacy.pt", _use_new_zipfile_serialization=False)

    rewrite_compressed(MODELS / "torch_smallnet.pt", MODELS / "torch_deflated.pt")
    rewrite_zip64(MODELS / "torch_smallnet.pt", MODELS / "torch_zip64.pt")
    write_hostile(MODELS / "torch_hostile.pt")

    with (MODELS / "torch_cases.txt").open("w") as handle:
        emit(handle, "torch_smallnet.pt", small_net)
        emit(handle, "torch_dtypes.pt", dtypes)
        emit(handle, "torch_checkpoint.pt", checkpoint)
        emit(handle, "torch_protocol5.pt", modern)

    print("wrote", sorted(p.name for p in MODELS.iterdir()))
    for name in ("torch_smallnet.pt", "torch_checkpoint.pt", "torch_protocol5.pt", "torch_dtypes.pt", "torch_module.pt", "torch_legacy.pt", "torch_deflated.pt", "torch_zip64.pt", "torch_hostile.pt"):
        print(name, (MODELS / name).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
