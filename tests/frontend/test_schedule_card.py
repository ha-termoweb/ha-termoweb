"""Smoke tests for the Lovelace schedule card logic via Node.js."""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="Node.js not installed",
)


def test_schedule_card_js() -> None:
    """Run the JavaScript schedule card checks under Node.js."""

    script_path = pathlib.Path(__file__).resolve().parents[1] / "js" / "test_termoweb_schedule_card.mjs"
    subprocess.run(["node", str(script_path)], check=True)
