#!/usr/bin/env python3
"""Generate a consolidated map of functions and docstrings for the integration."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass
class FunctionDoc:
    """Container for discovered function docstrings."""

    file: Path
    lineno: int
    qualname: str
    summary: str


class DocExtractor(ast.NodeVisitor):
    """AST visitor that collects function docstrings with qualified names."""

    def __init__(self, file_path: Path) -> None:
        """Initialise the extractor for ``file_path``."""
        self._file_path = file_path
        self._stack: list[str] = []
        self._functions: list[FunctionDoc] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class and traverse into its body."""

        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition and record its docstring."""

        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition and record its docstring."""

        self._handle_function(node)

    def _handle_function(self, node: ast.AST) -> None:
        doc = ast.get_docstring(node)
        qualname_parts = [*self._stack, node.name] if self._stack else [node.name]  # type: ignore[attr-defined]
        qualname = ".".join(qualname_parts)
        if doc:
            summary = doc.strip().splitlines()[0].strip()
            self._functions.append(
                FunctionDoc(self._file_path, node.lineno, qualname, summary)
            )
        self._stack.append(node.name)  # type: ignore[attr-defined]
        self.generic_visit(node)
        self._stack.pop()

    @property
    def functions(self) -> list[FunctionDoc]:
        """Return the collected functions."""

        return self._functions


def iter_python_files(root: Path) -> Iterable[Path]:
    """Yield Python files under ``root`` excluding tests directories."""

    for path in sorted(root.rglob("*.py")):
        if any(part == "tests" for part in path.parts):
            continue
        yield path


def build_map(root: Path) -> list[FunctionDoc]:
    """Parse each Python file and collect function docstrings."""

    functions: list[FunctionDoc] = []
    for file_path in iter_python_files(root):
        try:
            tree = ast.parse(file_path.read_text())
        except SyntaxError as err:  # pragma: no cover - defensive
            sys.stderr.write(f"Skipping {file_path}: {err}\n")
            continue
        extractor = DocExtractor(file_path)
        extractor.visit(tree)
        functions.extend(extractor.functions)
    functions.sort(key=lambda item: (item.file, item.lineno, item.qualname))
    return functions


def format_entries(entries: Iterable[FunctionDoc], root: Path) -> str:
    """Format function docstrings into a human-readable map."""

    lines = ["# TermoWeb Function Map", ""]
    for entry in entries:
        rel_path = entry.file.relative_to(root)
        lines.append(
            f"{rel_path} :: {entry.qualname}\n    {entry.summary}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to repo two levels up).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/function_map.txt"),
        help="Output text file (defaults to docs/function_map.txt).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    entries = build_map(root)
    output_text = format_entries(entries, root)

    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")
    sys.stdout.write(f"Wrote {len(entries)} functions to {output_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
