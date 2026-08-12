#!/usr/bin/env python3
"""Validate Jaithon's workspace packages and their local dependencies."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


NAME = re.compile(r"^[a-z][a-z0-9_]*$")
VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


class PackageError(Exception):
    pass


@dataclass(frozen=True)
class Package:
    name: str
    version: tuple[int, int, int]
    path: Path
    dependencies: dict[str, str]


def fail(message: str) -> None:
    raise PackageError(message)


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing {path}")
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} must contain a JSON object")
    return value


def parse_version(text: object, where: str) -> tuple[int, int, int]:
    if not isinstance(text, str) or (match := VERSION.fullmatch(text)) is None:
        fail(f"{where} must be a semantic version such as 1.2.3")
    return tuple(int(part) for part in match.groups())


def accepts(requirement: str, version: tuple[int, int, int], where: str) -> bool:
    if requirement.startswith("^"):
        lower = parse_version(requirement[1:], where)
        if lower[0] > 0:
            upper = (lower[0] + 1, 0, 0)
        elif lower[1] > 0:
            upper = (0, lower[1] + 1, 0)
        else:
            upper = (0, 0, lower[2] + 1)
        return lower <= version < upper
    return version == parse_version(requirement, where)


def package_from(path: Path) -> Package:
    manifest_path = path / "jaithon.package.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != 1:
        fail(f"{manifest_path}: schema must be 1")

    name = manifest.get("name")
    if not isinstance(name, str) or NAME.fullmatch(name) is None:
        fail(f"{manifest_path}: name must be a lowercase Jaithon identifier")
    version = parse_version(manifest.get("version"), f"{manifest_path}: version")
    if manifest.get("source") != "src":
        fail(f'{manifest_path}: source must be "src"')

    source = path / "src"
    if not source.is_dir():
        fail(f"{manifest_path}: missing source directory {source}")
    public_file = source / f"{name}.jai"
    public_package = source / name / "mod.jai"
    if not public_file.is_file() and not public_package.is_file():
        fail(f"{manifest_path}: src must expose module {name}")

    raw_dependencies = manifest.get("dependencies", {})
    if not isinstance(raw_dependencies, dict):
        fail(f"{manifest_path}: dependencies must be an object")
    dependencies: dict[str, str] = {}
    for dependency, requirement in raw_dependencies.items():
        if not isinstance(dependency, str) or NAME.fullmatch(dependency) is None:
            fail(f"{manifest_path}: invalid dependency name {dependency!r}")
        if not isinstance(requirement, str):
            fail(f"{manifest_path}: dependency {dependency} must have a version string")
        dependencies[dependency] = requirement
    return Package(name, version, path, dependencies)


def load_workspace(workspace_path: Path) -> dict[str, Package]:
    root = workspace_path.parent.resolve()
    workspace = read_json(workspace_path)
    if workspace.get("schema") != 1:
        fail(f"{workspace_path}: schema must be 1")
    members = workspace.get("members")
    if not isinstance(members, list) or not members:
        fail(f"{workspace_path}: members must be a non-empty list")

    packages: dict[str, Package] = {}
    member_paths: set[Path] = set()
    for member in members:
        if not isinstance(member, str):
            fail(f"{workspace_path}: each member must be a path string")
        package_path = (root / member).resolve()
        try:
            package_path.relative_to(root)
        except ValueError:
            fail(f"{workspace_path}: member leaves the workspace: {member}")
        if package_path in member_paths:
            fail(f"{workspace_path}: duplicate member {member}")
        member_paths.add(package_path)
        package = package_from(package_path)
        if package.name in packages:
            fail(f"duplicate package name {package.name}")
        packages[package.name] = package

    package_root = root / "packages"
    unlisted = sorted(
        path.parent
        for path in package_root.glob("*/jaithon.package.json")
        if path.parent.resolve() not in member_paths
    )
    if unlisted:
        fail(f"package is missing from workspace members: {unlisted[0]}")
    return packages


def dependency_order(packages: dict[str, Package]) -> list[str]:
    for package in packages.values():
        for dependency, requirement in package.dependencies.items():
            found = packages.get(dependency)
            if found is None:
                fail(f"{package.name}: dependency {dependency} is not in this workspace")
            where = f"{package.name}: dependency {dependency} requirement"
            if not accepts(requirement, found.version, where):
                actual = ".".join(str(part) for part in found.version)
                fail(f"{package.name}: {dependency} {actual} does not satisfy {requirement}")

    state: dict[str, int] = {}
    order: list[str] = []
    stack: list[str] = []

    def visit(name: str) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            start = stack.index(name)
            fail("dependency cycle: " + " -> ".join(stack[start:] + [name]))
        state[name] = 1
        stack.append(name)
        for dependency in sorted(packages[name].dependencies):
            visit(dependency)
        stack.pop()
        state[name] = 2
        order.append(name)

    for name in sorted(packages):
        visit(name)
    return order


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    workspace_path = Path(argv[1]).resolve() if len(argv) > 1 else root / "jaithon.workspace.json"
    if len(argv) > 2:
        print(f"usage: {Path(argv[0]).name} [WORKSPACE]", file=sys.stderr)
        return 2
    try:
        packages = load_workspace(workspace_path)
        order = dependency_order(packages)
    except PackageError as error:
        print(f"package check failed: {error}", file=sys.stderr)
        return 1
    print(f"packages: {len(packages)} valid ({' -> '.join(order)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
