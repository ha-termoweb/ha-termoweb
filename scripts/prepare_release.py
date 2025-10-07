"""Utility to prepare a tagged release for the project."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
MANIFEST_PATH = REPO_ROOT / "custom_components" / "termoweb" / "manifest.json"

VERSION_PATTERN = re.compile(r"^v(\d+\.\d+\.\d+)$")
PYPROJECT_VERSION_PATTERN = re.compile(r'^(version\s*=\s*")([^\"]+)(")', re.MULTILINE)
MANIFEST_VERSION_PATTERN = re.compile(r'(\"version\"\s*:\s*\")([^\"]+)(\")')


def parse_args(argv: list[str]) -> str:
    """Return the release tag from the command line arguments."""

    parser = argparse.ArgumentParser(description="Prepare a project release")
    parser.add_argument("version", help="Release tag in the form vX.Y.Z")
    parsed = parser.parse_args(argv)
    match = VERSION_PATTERN.fullmatch(parsed.version)
    if not match:
        msg = "Version must be in the form vX.Y.Z"
        raise ValueError(msg)
    return parsed.version


def update_file_version(path: Path, pattern: re.Pattern[str], new_value: str) -> None:
    """Update the version string within *path* using *pattern*."""

    content = path.read_text(encoding="utf-8")
    match = pattern.search(content)
    if not match:
        msg = f"Unable to locate version pattern in {path}"
        raise RuntimeError(msg)
    updated = pattern.sub(lambda m: f"{m.group(1)}{new_value}{m.group(3)}", content, count=1)
    path.write_text(updated, encoding="utf-8")


def run(*args: str) -> None:
    """Execute a subprocess and ensure it completes successfully."""

    subprocess.run(args, check=True, cwd=REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, update files, and perform release automation."""

    args = argv if argv is not None else sys.argv[1:]
    version = parse_args(args)
    semver = version[1:]

    update_file_version(PYPROJECT_PATH, PYPROJECT_VERSION_PATTERN, semver)
    update_file_version(MANIFEST_PATH, MANIFEST_VERSION_PATTERN, semver)

    run("git", "add", str(PYPROJECT_PATH.relative_to(REPO_ROOT)), str(MANIFEST_PATH.relative_to(REPO_ROOT)))
    run("git", "commit", "-m", f"Prepare release {version}")
    run("git", "tag", version)
    run("git", "push")
    run("git", "push", "--tags")
    run("gh", "release", "create", version, "--generate-notes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
