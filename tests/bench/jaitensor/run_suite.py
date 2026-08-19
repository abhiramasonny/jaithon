"""Run the jaitensor/PyTorch peers using their in-process timing records."""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


WORKLOADS = (
    "fashion",
    "mnist",
    "kmnist",
    "cifar10",
    "cifar100",
    "fashion-wide",
    "fashion-deep",
    "fashion-xl",
    "mnist-deep",
    "mnist-xl",
    "cifar-wide",
    "cifar-deep",
    "cifar-bottleneck",
    "cifar100-wide",
    "cifar100-deep",
)


class BenchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Record:
    name: str
    seconds: float
    flops: int
    samples: int
    check: str
    detail: str


@dataclass
class Samples:
    name: str
    seconds: list[float]
    flops: int
    samples: int
    detail: str

    @property
    def median(self) -> float:
        return statistics.median(self.seconds)

    @property
    def cv(self) -> float:
        if len(self.seconds) < 2 or self.median <= 0.0:
            return 0.0
        return 100.0 * statistics.stdev(self.seconds) / self.median


def parse_records(stdout: str, command: list[str]) -> list[Record]:
    records: list[Record] = []
    for line in stdout.splitlines():
        if not line.startswith("jtb\t"):
            continue
        parts = line.split("\t", 7)
        if len(parts) != 7:
            raise BenchError(f"malformed benchmark record from {' '.join(command)}: {line!r}")
        _tag, name, seconds, flops, samples, check, detail = parts
        try:
            record = Record(name, float(seconds), int(flops), int(samples), check, detail)
        except ValueError as error:
            raise BenchError(f"invalid benchmark record from {' '.join(command)}: {line!r}") from error
        if record.seconds <= 0.0 or record.flops < 0 or record.samples < 0:
            raise BenchError(f"out-of-range benchmark record from {' '.join(command)}: {line!r}")
        if record.check != "ok":
            raise BenchError(
                f"{record.name} failed its numerical check in {' '.join(command)} "
                f"(detail: {record.detail})"
            )
        records.append(record)
    if not records:
        text = stdout.strip() or "<no stdout>"
        raise BenchError(f"no benchmark records from {' '.join(command)}; output was:\n{text}")
    names = [record.name for record in records]
    if len(set(names)) != len(names):
        raise BenchError(f"duplicate benchmark records from {' '.join(command)}: {names}")
    return records


def progress(text: str) -> None:
    """Say what is running, on stderr, in place.

    The suite takes seven minutes on this machine and used to print nothing at
    all until the table -- so the only thing distinguishing it from a hang was
    knowing that it takes seven minutes. Stderr rather than stdout so the table
    still pipes cleanly, and a carriage return rather than a newline so a
    terminal shows one line rather than three hundred.
    """
    if not sys.stderr.isatty():
        return
    sys.stderr.write("\r\033[K" + text)
    sys.stderr.flush()


def progress_done() -> None:
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()


def sample_command(
    command: list[str],
    runs: int,
    cwd: Path,
    level: str,
    label: str = "",
) -> tuple[list[str], dict[str, Samples]]:
    env = os.environ.copy()
    env["BENCH_LEVEL"] = level
    env["JAITENSOR_VERBOSE"] = "0"
    env.setdefault("PYTHONWARNINGS", "ignore")
    order: list[str] = []
    collected: dict[str, Samples] = {}
    expected_names: list[str] | None = None

    for _run in range(runs):
        if label:
            progress(f"  {label} (run {_run + 1} of {runs})")
        done = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            output = (done.stdout + done.stderr).strip() or "<no output>"
            raise BenchError(f"{' '.join(command)} failed with status {done.returncode}:\n{output}")
        records = parse_records(done.stdout, command)
        names = [record.name for record in records]
        if expected_names is None:
            expected_names = names
            order = names
        elif names != expected_names:
            raise BenchError(
                f"record set changed between runs for {' '.join(command)}: "
                f"expected {expected_names}, got {names}"
            )
        for record in records:
            sample = collected.get(record.name)
            if sample is None:
                collected[record.name] = Samples(
                    record.name, [record.seconds], record.flops, record.samples, record.detail
                )
                continue
            if sample.flops != record.flops or sample.samples != record.samples:
                raise BenchError(f"work count changed between runs for {record.name}")
            sample.seconds.append(record.seconds)
    return order, collected


def merge_samples(
    destination_order: list[str],
    destination: dict[str, Samples],
    order: list[str],
    source: dict[str, Samples],
) -> None:
    for name in order:
        if name in destination:
            raise BenchError(f"duplicate suite benchmark name {name!r}")
        destination_order.append(name)
        destination[name] = source[name]


def duration(seconds: float) -> str:
    milliseconds = seconds * 1000.0
    if milliseconds < 10.0:
        return f"{milliseconds:.2f}ms"
    if milliseconds < 100.0:
        return f"{milliseconds:.1f}ms"
    return f"{milliseconds:.0f}ms"


def rate(work: int, seconds: float, scale: float = 1.0) -> str:
    if work <= 0 or seconds <= 0.0:
        return "—"
    return f"{work / seconds / scale:.1f}"


def render(
    order: list[str],
    jai: dict[str, Samples],
    peer: dict[str, Samples] | None,
    build_kind: str,
    level: str,
    runs: int,
) -> None:
    print(f"{build_kind} build, {level}, median of {runs} (CV shows run-to-run spread)")
    print(
        f"{'benchmark':<24} {'jaithon':>9} {'python3':>9} {'j CV':>7} {'p CV':>7} "
        f"{'j GFLOPS':>10} {'p GFLOPS':>10} {'j samp/s':>10} {'p samp/s':>10} "
        f"{'speedup':>9}   result  route"
    )
    print("─" * 150)

    total_j = 0.0
    total_p = 0.0
    total_flops = 0
    total_samples = 0
    mlp_j = 0.0
    mlp_p = 0.0
    for name in order:
        left = jai[name]
        right = peer.get(name) if peer is not None else None
        if right is not None and (left.flops != right.flops or left.samples != right.samples):
            raise BenchError(
                f"{name} peers report different work: "
                f"jaithon=({left.flops}, {left.samples}), "
                f"python=({right.flops}, {right.samples})"
            )
        jtime = left.median
        ptime = right.median if right is not None else 0.0
        speed = f"{ptime / jtime:.2f}x" if right is not None else "—"
        route = left.detail.rsplit("/", 1)[-1] if "/" in left.detail else (
            "attention" if name.startswith("attn-") else "fused-mlp"
        )
        print(
            f"{name:<24} {duration(jtime):>9} "
            f"{(duration(ptime) if right is not None else '—'):>9} "
            f"{left.cv:>6.1f}% "
            f"{(f'{right.cv:.1f}%' if right is not None else '—'):>7} "
            f"{rate(left.flops, jtime, 1e9):>10} "
            f"{(rate(right.flops, ptime, 1e9) if right is not None else '—'):>10} "
            f"{rate(left.samples, jtime):>10} "
            f"{(rate(right.samples, ptime) if right is not None else '—'):>10} "
            f"{speed:>9}   ok      {route}"
        )
        total_j += jtime
        total_flops += left.flops
        if left.samples > 0:
            total_samples += left.samples
            mlp_j += jtime
        if right is not None:
            total_p += ptime
            if right.samples > 0:
                mlp_p += ptime

    print("─" * 150)
    speed = f"{total_p / total_j:.2f}x" if peer is not None else "—"
    print(
        f"{'total':<24} {duration(total_j):>9} "
        f"{(duration(total_p) if peer is not None else '—'):>9} "
        f"{'—':>7} {'—':>7} "
        f"{rate(total_flops, total_j, 1e9):>10} "
        f"{(rate(total_flops, total_p, 1e9) if peer is not None else '—'):>10} "
        f"{rate(total_samples, mlp_j):>10} "
        f"{(rate(total_samples, mlp_p) if peer is not None else '—'):>10} "
        f"{speed:>9}"
    )


def collect_side(executable: str, root: Path, level: str, runs: int, jaithon: bool):
    order: list[str] = []
    samples: dict[str, Samples] = {}
    bench_root = root / "tests" / "bench" / "jaitensor"
    side = "jaithon" if jaithon else "pytorch"
    for workload in WORKLOADS:
        program = bench_root / ("mlp.jai" if jaithon else "mlp.py")
        command = [executable, "run", str(program), workload] if jaithon else [executable, str(program), workload]
        found_order, found = sample_command(command, runs, root, level, f"{side} {workload}")
        merge_samples(order, samples, found_order, found)
    for stem in ("conv", "gemm", "attn"):
        program = bench_root / f"{stem}.{'jai' if jaithon else 'py'}"
        command = [executable, "run", str(program)] if jaithon else [executable, str(program)]
        found_order, found = sample_command(command, runs, root, level, f"{side} {stem}")
        merge_samples(order, samples, found_order, found)
    progress_done()
    return order, samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jaithon", required=True)
    parser.add_argument("--python", default="")
    parser.add_argument("--level", choices=("easy", "medium", "hard"), required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--build-kind", default="unattributed")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    try:
        order, jai = collect_side(args.jaithon, args.root, args.level, args.runs, True)
        peer = None
        if args.python:
            peer_order, peer = collect_side(args.python, args.root, args.level, args.runs, False)
            if peer_order != order:
                raise BenchError(f"peer benchmark order differs: {order} != {peer_order}")
        render(order, jai, peer, args.build_kind, args.level, args.runs)
    except BenchError as error:
        print(f"jaitensor bench: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
