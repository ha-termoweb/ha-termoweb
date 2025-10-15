"""Utility for generating translation JSON files using the Google web API."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests

GOOGLE_ENDPOINT = "https://translate.googleapis.com/translate_a/single"
PLACEHOLDER_PATTERN = re.compile(r"\{[^{}]+\}")
BACKTICK_PATTERN = re.compile(r"`[^`]+`")


def parse_args() -> argparse.Namespace:
    """Return the parsed command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Translate a Home Assistant TermoWeb locale file with the Google web API."
        )
    )
    parser.add_argument("source", type=Path, help="Path to the source JSON file.")
    parser.add_argument(
        "target_language",
        help="Language code for the translation output (e.g. 'uk').",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Destination path for the translated JSON file.",
    )
    parser.add_argument(
        "--source-language",
        default="en",
        help="Optional source language override (default: en).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file if present.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Ignore HTTP proxy configuration from the environment.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (use with caution).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds for each translation request (default: 15).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    """Load JSON content from ``path``."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, data: Any) -> None:
    """Write ``data`` as UTF-8 JSON to ``path``."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def protect_tokens(text: str) -> tuple[str, Dict[str, str]]:
    """Replace placeholders and code spans with sentinel tokens."""
    mapping: Dict[str, str] = {}

    def _capture(prefix: str, value: str) -> str:
        token = f"__{prefix}_{len(mapping)}__"
        mapping[token] = value
        return token

    interim = PLACEHOLDER_PATTERN.sub(lambda match: _capture("PH", match.group(0)), text)
    interim = BACKTICK_PATTERN.sub(lambda match: _capture("BT", match.group(0)), interim)
    return interim, mapping


def restore_tokens(text: str, mapping: Dict[str, str]) -> str:
    """Restore placeholder and code span tokens to their originals."""
    for token, value in mapping.items():
        text = text.replace(token, value)
    return text


def create_session(ignore_proxy: bool, insecure: bool) -> requests.Session:
    """Configure and return a reusable Requests session."""
    session = requests.Session()
    if ignore_proxy:
        session.trust_env = False
        session.proxies.update({"http": None, "https": None})
    if insecure:
        session.verify = False
    return session


def _translate_once(
    session: requests.Session,
    text: str,
    source: str,
    target: str,
    timeout: float,
) -> str:
    """Translate ``text`` via the Google web API and return the result."""
    params = {
        "client": "gtx",
        "sl": source,
        "tl": target,
        "dt": "t",
        "q": text,
    }
    response = session.get(GOOGLE_ENDPOINT, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    segments: Iterable[List[Any]] = payload[0]
    return "".join(segment[0] for segment in segments if segment and segment[0])


def translate_text(
    session: requests.Session,
    text: str,
    source: str,
    target: str,
    timeout: float,
) -> str:
    """Translate ``text`` while preserving placeholder tokens and newlines."""
    if not text:
        return text
    protected, mapping = protect_tokens(text)
    parts = protected.split("\n")
    translated_parts = [
        _translate_once(session, part, source, target, timeout) if part else ""
        for part in parts
    ]
    translated = "\n".join(translated_parts)
    return restore_tokens(translated, mapping)


def translate_structure(
    data: Any,
    session: requests.Session,
    source: str,
    target: str,
    timeout: float,
) -> Any:
    """Translate every string inside ``data`` recursively."""
    if isinstance(data, dict):
        return {
            key: translate_structure(value, session, source, target, timeout)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [
            translate_structure(item, session, source, target, timeout)
            for item in data
        ]
    if isinstance(data, str):
        return translate_text(session, data, source, target, timeout)
    return data


def main() -> None:
    """Entrypoint for the translation script."""
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} exists; pass --overwrite to replace the file."
        )

    session = create_session(args.no_proxy, args.insecure)
    source = load_json(args.source)
    translated = translate_structure(
        source,
        session,
        args.source_language,
        args.target_language,
        args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump_json(args.output, translated)


if __name__ == "__main__":
    main()
