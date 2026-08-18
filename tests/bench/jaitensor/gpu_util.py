import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MLP_JAI = REPO / "tests" / "bench" / "jaitensor" / "mlp.jai"
MLP_PY = REPO / "tests" / "bench" / "jaitensor" / "mlp.py"

KEYS = (
    "Device Utilization %",
    "Renderer Utilization %",
    "Tiler Utilization %",
)

WORKLOADS = (
    {
        "name": "fashion",
        "title": "Fashion-MNIST",
        "model": "Dense(784,256,ReLU)+Dense(256,10)",
        "samples": 60000,
    },
    {
        "name": "mnist",
        "title": "MNIST",
        "model": "Dense(784,256,ReLU)+Dense(256,10)",
        "samples": 60000,
    },
    {
        "name": "cifar10",
        "title": "CIFAR-10",
        "model": "Dense(3072,512,ReLU)+Dense(512,10)",
        "samples": 50000,
    },
    {
        "name": "fashion-wide",
        "title": "Fashion-MNIST wide",
        "model": "Dense(784,2048,ReLU)+Dense(2048,10)",
        "samples": 60000,
    },
    {
        "name": "mnist-deep",
        "title": "MNIST deep",
        "model": "Dense(784,256,ReLU)x3+Dense(256,10)",
        "samples": 60000,
    },
    {
        "name": "cifar-wide",
        "title": "CIFAR-10 wide",
        "model": "Dense(3072,1024,ReLU)+Dense(1024,10)",
        "samples": 50000,
    },
    {
        "name": "kmnist",
        "title": "KMNIST",
        "model": "Dense(784,256,ReLU)+Dense(256,10)",
        "samples": 60000,
    },
    {
        "name": "fashion-deep",
        "title": "Fashion-MNIST deep",
        "model": "Dense(784,256,ReLU)x3+Dense(256,10)",
        "samples": 60000,
    },
    {
        "name": "cifar100",
        "title": "CIFAR-100",
        "model": "Dense(3072,512,ReLU)+Dense(512,100)",
        "samples": 50000,
    },
    {
        "name": "fashion-xl",
        "title": "Fashion-MNIST XL",
        "model": "Dense(784,4096,ReLU)+Dense(4096,10)",
        "samples": 60000,
    },
    {
        "name": "cifar-deep",
        "title": "CIFAR-10 deep",
        "model": "Dense(3072,512,ReLU)x3+Dense(512,10)",
        "samples": 50000,
    },
    {
        "name": "cifar100-wide",
        "title": "CIFAR-100 wide",
        "model": "Dense(3072,2048,ReLU)+Dense(2048,100)",
        "samples": 50000,
    },
    {
        "name": "mnist-xl",
        "title": "MNIST XL",
        "model": "Dense(784,8192,ReLU)+Dense(8192,10)",
        "samples": 60000,
    },
    {
        "name": "cifar-bottleneck",
        "title": "CIFAR-10 bottleneck",
        "model": "Dense(3072,1024,256,1024,ReLU)+Dense(1024,10)",
        "samples": 50000,
    },
    {
        "name": "cifar100-deep",
        "title": "CIFAR-100 deep",
        "model": "Dense(3072,512,ReLU)x3+Dense(512,100)",
        "samples": 50000,
    },
)


def read_gpu() -> dict[str, int]:
    out = subprocess.check_output(
        ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    stats = {}
    for key in KEYS:
        match = re.search(rf'"{re.escape(key)}"=(\d+)', out)
        if match:
            stats[key] = int(match.group(1))
    return stats


def sample_for(seconds: float, interval: float) -> list[dict]:
    end = time.perf_counter() + seconds
    rows = []
    while time.perf_counter() < end:
        row = read_gpu()
        row["t"] = time.perf_counter()
        rows.append(row)
        remaining = end - time.perf_counter()
        if remaining > 0:
            time.sleep(min(interval, remaining))
    return rows


class Sampler:
    def __init__(self, interval: float = 0.12):
        self.interval = interval
        self.rows: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        self._thread.join()
        return self.rows

    def _run(self) -> None:
        while not self._stop.is_set():
            row = read_gpu()
            row["t"] = time.perf_counter()
            self.rows.append(row)
            self._stop.wait(self.interval)


def summarize(rows: list[dict], key: str = "Device Utilization %") -> dict:
    values = [row[key] for row in rows if key in row]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 1),
        "median": round(statistics.median(values), 1),
        "p95": round(sorted(values)[max(0, int(0.95 * (len(values) - 1)))], 1),
        "max": max(values),
        "min": min(values),
        "series": values,
    }


def parse_output(text: str) -> dict:
    result = {"epochs": []}
    for line in text.splitlines():
        if line.startswith("epoch "):
            match = re.search(
                r"epoch (\d+)/(\d+): loss=([0-9.]+), accuracy=([0-9.]+)%(?:.*)? \(([0-9.]+)s\)",
                line,
            )
            if match:
                result["epochs"].append(
                    {
                        "epoch": int(match.group(1)),
                        "loss": float(match.group(3)),
                        "accuracy": float(match.group(4)),
                        "seconds": float(match.group(5)),
                    }
                )
        elif line.startswith("test:"):
            match = re.search(r"loss=([0-9.]+), accuracy=([0-9.]+)%", line)
            if match:
                result["test_loss"] = float(match.group(1))
                result["test_accuracy"] = float(match.group(2))
        elif line.startswith("train_seconds:"):
            result["train_seconds"] = float(line.split(":", 1)[1])
        elif line.startswith("samples_per_second:"):
            result["samples_per_second"] = float(line.split(":", 1)[1])
        elif line.startswith("GPU:"):
            result["gpu"] = line.split(":", 1)[1].strip()
        elif line.startswith("framework:"):
            result["framework"] = line.split(":", 1)[1].strip()
        elif line.startswith("workload:"):
            result["workload"] = line.split(":", 1)[1].strip()
        elif line.startswith("features:"):
            result["features"] = int(line.split(":", 1)[1])
        elif line.startswith("hidden:"):
            result["hidden"] = int(line.split(":", 1)[1])
        elif line.startswith("samples:"):
            result["samples"] = int(line.split(":", 1)[1])
    return result


def run_logged(name: str, command: list[str], env: dict[str, str] | None = None) -> dict:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    sampler = Sampler()
    wall_start = time.perf_counter()
    sampler.start()
    proc = subprocess.run(
        command,
        cwd=str(REPO),
        env=merged,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - wall_start
    samples = sampler.stop()
    text = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed ({proc.returncode}):\n{text}")
    parsed = parse_output(proc.stdout)
    parsed["name"] = name
    parsed["wall_seconds"] = round(wall, 4)
    parsed["stdout"] = proc.stdout
    parsed["gpu_util"] = {
        "device": summarize(samples, "Device Utilization %"),
        "renderer": summarize(samples, "Renderer Utilization %"),
        "tiler": summarize(samples, "Tiler Utilization %"),
    }
    return parsed


def run_workload(python: str, epochs: str, spec: dict) -> dict:
    name = spec["name"]
    jaithon_cmd = [str(REPO / "jaithon"), str(MLP_JAI), name]
    pytorch_cmd = [python, str(MLP_PY), name]
    jaithon_env = {"JAITHON_PATH": str(REPO / "lib"), "JAITENSOR_VERBOSE": "1"}
    pytorch_env = {"JAITENSOR_VERBOSE": "1"}
    print(f"warming jaitensor {name} (1 epoch)...", flush=True)
    subprocess.run(
        jaithon_cmd + ["1"],
        cwd=str(REPO),
        env={**os.environ, **jaithon_env},
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"warming pytorch {name} (1 epoch)...", flush=True)
    subprocess.run(
        pytorch_cmd + ["1"],
        cwd=str(REPO),
        env={**os.environ, **pytorch_env},
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"running jaitensor {name}...", flush=True)
    jaithon = run_logged(f"jaitensor-{name}", jaithon_cmd + [epochs], env=jaithon_env)
    time.sleep(1.0)
    print(f"running pytorch {name}...", flush=True)
    pytorch = run_logged(f"pytorch-{name}", pytorch_cmd + [epochs], env=pytorch_env)
    j_sps = jaithon.get("samples_per_second") or 0.0
    p_sps = pytorch.get("samples_per_second") or 0.0
    speedup = round(j_sps / p_sps, 2) if p_sps else None
    print(
        f"{name} jaitensor: {jaithon.get('train_seconds')}s, {j_sps} samples/s, "
        f"GPU mean {jaithon['gpu_util']['device'].get('mean')}%"
    )
    print(
        f"{name} pytorch: {pytorch.get('train_seconds')}s, {p_sps} samples/s, "
        f"GPU mean {pytorch['gpu_util']['device'].get('mean')}%"
    )
    if speedup is not None:
        print(f"{name} speedup: {speedup}×")
    return {
        "title": spec["title"],
        "model": spec["model"],
        "speedup": speedup,
        "jaitensor": jaithon,
        "pytorch": pytorch,
    }


def main() -> int:
    epochs = sys.argv[1] if len(sys.argv) > 1 else "8"
    python = sys.argv[2] if len(sys.argv) > 2 else sys.executable
    names = sys.argv[3:] or [item["name"] for item in WORKLOADS]
    wanted = [item for item in WORKLOADS if item["name"] in names]
    if not wanted:
        raise SystemExit(f"no matching workloads in {names}")
    print("sampling idle GPU...", flush=True)
    idle = sample_for(2.0, 0.12)
    results = {}
    for spec in wanted:
        results[spec["name"]] = run_workload(python, epochs, spec)
        time.sleep(1.0)
    payload = {
        "machine": "Apple M2 Max",
        "hparams": "SGD lr=0.08, batch=512 (2048 when width>=1024, depth>=3, or CIFAR; 1024 at width>=8192), AMP at width>=1024, 8 epochs, no shuffle, CrossEntropy on logits",
        "note": "Each workload uses matching Sequential / nn.Sequential models. ioreg Device Utilization % is whole-chip.",
        "idle": {
            "device": summarize(idle, "Device Utilization %"),
            "renderer": summarize(idle, "Renderer Utilization %"),
            "tiler": summarize(idle, "Tiler Utilization %"),
        },
        "workloads": results,
    }
    out = REPO / "data" / "bench-mlp.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
