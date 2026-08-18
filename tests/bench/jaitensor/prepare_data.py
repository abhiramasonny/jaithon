#!/usr/bin/env python3
import gzip
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

IDX = {
    "mnist": {
        "dir": REPO / "data" / "mnist",
        "files": {
            "train-images-idx3-ubyte": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        },
    },
    "fashion": {
        "dir": REPO / "data" / "fashion-mnist",
        "files": {
            "train-images-idx3-ubyte": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
        },
    },
    "kmnist": {
        "dir": REPO / "data" / "kmnist",
        "files": {
            "train-images-idx3-ubyte": "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte": "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte": "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte": "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz",
        },
    },
}

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR_DIR = REPO / "data" / "cifar-10"
RECORD = 1 + 3072
TRAIN_BATCHES = [f"data_batch_{i}.bin" for i in range(1, 6)]
CIFAR100_URLS = (
    "http://datasets.openimaj.org/cifar/cifar-100-binary.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
)
CIFAR100_DIR = REPO / "data" / "cifar-100"
CIFAR100_RECORD = 2 + 3072


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url) as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out)


def ensure_idx(name: str) -> None:
    spec = IDX[name]
    root = spec["dir"]
    for filename, url in spec["files"].items():
        dest = root / filename
        if dest.exists():
            continue
        gz = dest.with_suffix(dest.suffix + ".gz")
        if not gz.exists():
            download(url, gz)
        print(f"decompressing {gz}", flush=True)
        with gzip.open(gz, "rb") as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)


def unpack_cifar_split(batch_paths: list[Path], images_path: Path, labels_path: Path) -> None:
    images = bytearray()
    labels = bytearray()
    for path in batch_paths:
        data = path.read_bytes()
        if len(data) % RECORD != 0:
            raise SystemExit(f"{path} is not CIFAR-10 binary")
        for offset in range(0, len(data), RECORD):
            labels.append(data[offset])
            images.extend(data[offset + 1 : offset + RECORD])
    images_path.write_bytes(images)
    labels_path.write_bytes(labels)


def ensure_cifar() -> None:
    train_x = CIFAR_DIR / "train-images.bin"
    train_y = CIFAR_DIR / "train-labels.bin"
    test_x = CIFAR_DIR / "test-images.bin"
    test_y = CIFAR_DIR / "test-labels.bin"
    if train_x.exists() and train_y.exists() and test_x.exists() and test_y.exists():
        return
    CIFAR_DIR.mkdir(parents=True, exist_ok=True)
    archive = CIFAR_DIR / "cifar-10-binary.tar.gz"
    if not archive.exists():
        download(CIFAR_URL, archive)
    print(f"extracting {archive}", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(CIFAR_DIR, filter="data")
        except TypeError:
            tar.extractall(CIFAR_DIR)
    extracted = CIFAR_DIR / "cifar-10-batches-bin"
    unpack_cifar_split(
        [extracted / name for name in TRAIN_BATCHES],
        train_x,
        train_y,
    )
    unpack_cifar_split([extracted / "test_batch.bin"], test_x, test_y)


def unpack_cifar100(bin_path: Path, images_path: Path, labels_path: Path) -> None:
    data = bin_path.read_bytes()
    if len(data) % CIFAR100_RECORD != 0:
        raise SystemExit(f"{bin_path} is not CIFAR-100 binary")
    images = bytearray()
    labels = bytearray()
    for offset in range(0, len(data), CIFAR100_RECORD):
        labels.append(data[offset + 1])
        images.extend(data[offset + 2 : offset + CIFAR100_RECORD])
    images_path.write_bytes(images)
    labels_path.write_bytes(labels)


def ensure_cifar100() -> None:
    train_x = CIFAR100_DIR / "train-images.bin"
    train_y = CIFAR100_DIR / "train-labels.bin"
    test_x = CIFAR100_DIR / "test-images.bin"
    test_y = CIFAR100_DIR / "test-labels.bin"
    if train_x.exists() and train_y.exists() and test_x.exists() and test_y.exists():
        return
    CIFAR100_DIR.mkdir(parents=True, exist_ok=True)
    archive = CIFAR100_DIR / "cifar-100-binary.tar.gz"
    if not archive.exists():
        last_error = None
        for url in CIFAR100_URLS:
            try:
                download(url, archive)
                last_error = None
                break
            except Exception as error:
                last_error = error
                if archive.exists():
                    archive.unlink()
        if last_error is not None:
            raise last_error
    print(f"extracting {archive}", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(CIFAR100_DIR, filter="data")
        except TypeError:
            tar.extractall(CIFAR100_DIR)
    extracted = CIFAR100_DIR / "cifar-100-binary"
    unpack_cifar100(extracted / "train.bin", train_x, train_y)
    unpack_cifar100(extracted / "test.bin", test_x, test_y)


def main() -> int:
    names = sys.argv[1:] or ["fashion", "mnist", "cifar10"]
    for name in names:
        if name in ("fashion", "fashion-wide", "fashion-deep", "fashion-xl"):
            ensure_idx("fashion")
        elif name in ("mnist", "mnist-deep", "mnist-xl"):
            ensure_idx("mnist")
        elif name == "kmnist":
            ensure_idx("kmnist")
        elif name in ("cifar10", "cifar", "cifar-wide", "cifar-deep", "cifar-bottleneck"):
            ensure_cifar()
        elif name in ("cifar100", "cifar-100", "cifar100-wide", "cifar100-deep"):
            ensure_cifar100()
        else:
            raise SystemExit(f"unknown dataset {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
