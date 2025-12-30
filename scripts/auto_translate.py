#!/usr/bin/env python3
"""
DeepL translation script with aggressive rate control and jitter to avoid hitting rate limits.

Features:
- Single-threaded (no parallelism) or very limited concurrency
- Delay between calls (configurable)
- Exponential backoff + jitter on 429
- Defensive checks and logging
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import time
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

DEEPL_TEXT_ENDPOINT = "https://api-free.deepl.com/v2/translate"
DEEPL_LANGUAGES_ENDPOINT = "https://api-free.deepl.com/v2/languages"
DEEPL_USAGE_ENDPOINT = "https://api-free.deepl.com/v2/usage"

PLACEHOLDER_PATTERN = re.compile(r"\{[^{}]+\}")
BACKTICK_PATTERN = re.compile(r"`[^`]+`")

# Controls
CALL_DELAY = 0.5  # base delay between calls in seconds
JITTER_FACTOR = 0.3  # fraction of CALL_DELAY to jitter (±)
MAX_BATCH_TEXTS = 20
MAX_BATCH_CHARS = 30000
MAX_RETRIES = 6
BACKOFF_BASE = 2.0
INITIAL_DELAY = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeepL auto-translate with aggressive throttling"
    )
    p.add_argument("source", type=Path, help="Source JSON file (in source language)")
    p.add_argument("target_language", help="Target language code (e.g. FI, ET)")
    p.add_argument("output", type=Path, help="Output JSON file")
    p.add_argument(
        "--source-language", default="EN", help="Source language code (default EN)"
    )
    p.add_argument("--api-key", required=True, help="DeepL API key")
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if exists"
    )
    p.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout (secs)")
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def protect_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}

    def cap(pref: str, val: str) -> str:
        token = f"__{pref}_{len(mapping)}__"
        mapping[token] = val
        return token

    s = PLACEHOLDER_PATTERN.sub(lambda m: cap("PH", m.group(0)), text)
    s = BACKTICK_PATTERN.sub(lambda m: cap("BT", m.group(0)), s)
    return s, mapping


def restore_tokens(text: str, mapping: Dict[str, str]) -> str:
    for tok, val in mapping.items():
        text = text.replace(tok, val)
    return text


def count_strings(obj: Any) -> int:
    if obj is None:
        return 0
    if isinstance(obj, dict):
        return sum(count_strings(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(count_strings(v) for v in obj)
    if isinstance(obj, str):
        return 1
    return 0


def collect_leaf_strings(obj: Any, coll: List[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            collect_leaf_strings(v, coll)
    elif isinstance(obj, list):
        for v in obj:
            collect_leaf_strings(v, coll)
    elif isinstance(obj, str):
        coll.append(obj)
    else:
        # ignore
        pass


def batch_texts(texts: List[str]) -> List[List[str]]:
    batches: List[List[str]] = []
    curr: List[str] = []
    curr_chars = 0
    for idx, t in enumerate(texts):
        if t is None:
            continue
        try:
            l = len(t)
        except Exception:
            continue
        if curr and curr_chars + l > MAX_BATCH_CHARS:
            batches.append(curr)
            curr = []
            curr_chars = 0
        if len(curr) >= MAX_BATCH_TEXTS:
            batches.append(curr)
            curr = []
            curr_chars = 0
        curr.append(t)
        curr_chars += l
    if curr:
        batches.append(curr)
    return batches


def get_retry_after_seconds(resp: requests.Response) -> Optional[float]:
    ra = resp.headers.get("Retry-After")
    if ra is None:
        return None
    try:
        return float(ra)
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime

            dt = parsedate_to_datetime(ra)
            return max(dt.timestamp() - time.time(), 0.0)
        except Exception:
            return None


def fetch_with_backoff(
    method: str,
    url: str,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    timeout: float = 15.0,
) -> requests.Response:
    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            if method.upper() == "GET":
                resp = requests.get(
                    url, headers=headers, params=params, timeout=timeout
                )
            else:
                resp = requests.post(url, headers=headers, data=params, timeout=timeout)
            if resp.status_code == 429:
                ra = get_retry_after_seconds(resp)
                wait = ra if ra is not None else delay
                print(
                    f"[fetch_with_backoff] 429 from {url}, waiting {wait:.2f}s (attempt {attempt + 1})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                delay *= BACKOFF_BASE
                continue
            if 500 <= resp.status_code < 600:
                print(
                    f"[fetch_with_backoff] server error {resp.status_code} at {url}, waiting {delay:.2f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
                delay *= BACKOFF_BASE
                continue
            return resp
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(
                f"[fetch_with_backoff] request error {e}, waiting {delay:.2f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= BACKOFF_BASE
    raise RuntimeError(f"{method} {url} failed after retries")


def translate_batch(
    texts: List[str],
    api_key: str,
    src: str,
    tgt: str,
    timeout: float,
) -> List[str]:
    protected: List[Tuple[str, Dict[str, str]]] = []
    for idx, t in enumerate(texts):
        if t is None:
            continue
        try:
            p, mp = protect_tokens(t)
        except Exception as e:
            print(
                f"[translate_batch] error protecting tokens idx {idx}: {e}",
                file=sys.stderr,
            )
            traceback.print_exc()
            continue
        protected.append((p, mp))
    if not protected:
        return []

    payload = {
        "auth_key": api_key,
        "text": [p for (p, _) in protected],
        "source_lang": src.upper(),
        "target_lang": tgt.upper(),
        "preserve_formatting": 1,
    }

    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            # base delay + jitter
            jitter = random.uniform(
                -JITTER_FACTOR * CALL_DELAY, JITTER_FACTOR * CALL_DELAY
            )
            sleep_time = CALL_DELAY + jitter
            if sleep_time > 0:
                time.sleep(sleep_time)

            resp = requests.post(DEEPL_TEXT_ENDPOINT, data=payload, timeout=timeout)
            if resp.status_code == 429:
                ra = get_retry_after_seconds(resp)
                wait = ra if ra is not None else delay
                print(
                    f"[translate_batch] HTTP 429 attempt {attempt + 1}, waiting {wait:.2f}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
                delay *= BACKOFF_BASE
                continue

            resp.raise_for_status()
            resp_json = resp.json()
            translations = resp_json.get("translations")
            if not isinstance(translations, list):
                raise RuntimeError(f"Bad translations value: {translations!r}")
            if len(translations) != len(protected):
                raise RuntimeError(
                    f"Count mismatch: protected {len(protected)}, translations {len(translations)}"
                )
            results: List[str] = []
            for idx2, ((p, mp), tr) in enumerate(zip(protected, translations)):
                if not isinstance(tr, dict):
                    print(
                        f"[translate_batch] warn: translation item idx2={idx2} not dict: {tr!r}",
                        file=sys.stderr,
                    )
                    continue
                txt = tr.get("text")
                if txt is None:
                    print(
                        f"[translate_batch] warn: missing 'text' in translation item idx2={idx2}: {tr!r}",
                        file=sys.stderr,
                    )
                    continue
                try:
                    res = restore_tokens(txt, mp)
                except Exception as e:
                    print(
                        f"[translate_batch] warn restoring tokens idx2={idx2}: {e}",
                        file=sys.stderr,
                    )
                    traceback.print_exc()
                    res = txt
                results.append(res)
            return results

        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code == 400:
                try:
                    body = err.response.json()
                except Exception:
                    body = err.response.text
                sys.exit(f"DeepL 400 Bad Request\nPayload: {payload}\nResponse: {body}")
            if attempt == MAX_RETRIES - 1:
                print(
                    f"[translate_batch] final HTTPError at attempt {attempt + 1}: {err}",
                    file=sys.stderr,
                )
                traceback.print_exc()
                raise
            print(
                f"[translate_batch] HTTPError attempt {attempt + 1}: {err}, waiting {delay:.2f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= BACKOFF_BASE
            continue

        except Exception as ex:
            print(
                f"[translate_batch] unexpected error attempt {attempt + 1}: {ex}",
                file=sys.stderr,
            )
            traceback.print_exc()
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= BACKOFF_BASE
            continue

    raise RuntimeError("translate_batch exhausted retries")


def translate_structure(
    obj: Any,
    api_key: str,
    src: str,
    tgt: str,
    timeout: float,
    cache: Dict[str, str],
    progress: tqdm,
) -> Any:
    if obj is None:
        return obj
    if isinstance(obj, dict):
        return {
            k: translate_structure(v, api_key, src, tgt, timeout, cache, progress)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [
            translate_structure(v, api_key, src, tgt, timeout, cache, progress)
            for v in obj
        ]
    if isinstance(obj, str):
        progress.update(1)
        if obj in cache:
            return cache[obj]
        try:
            out = translate_batch([obj], api_key, src, tgt, timeout)
        except Exception as e:
            print(
                f"[translate_structure] exception for text {obj!r}: {e}",
                file=sys.stderr,
            )
            traceback.print_exc()
            cache[obj] = obj
            return obj
        if not out:
            cache[obj] = obj
            return obj
        cache[obj] = out[0]
        return out[0]
    return obj


def main() -> None:
    args = parse_args()
    src_lang = args.source_language or "EN"
    tgt_lang = args.target_language
    if not tgt_lang:
        sys.exit("Error: target language blank")
    tgt_code = tgt_lang.upper()

    if args.output.exists() and not args.overwrite:
        sys.exit(f"Error: {args.output} exists; use --overwrite")

    # fetch supported target languages (once)
    resp = fetch_with_backoff(
        "GET",
        DEEPL_LANGUAGES_ENDPOINT,
        headers={"Authorization": f"DeepL-Auth-Key {args.api_key}"},
        params={"type": "target"},
        timeout=args.timeout,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        sys.exit(
            f"Failed fetch supported langs: HTTP {resp.status_code}, body: {resp.text}"
        )
    arr = resp.json()
    supported_codes = {
        item["language"].upper()
        for item in arr
        if isinstance(item, dict) and item.get("language")
    }
    if tgt_code not in supported_codes:
        print(f"Skipping '{tgt_code}': not supported", file=sys.stderr)
        sys.exit(0)

    src_data = load_json(args.source)
    total = count_strings(src_data)
    print(f"Total strings: {total}")

    try:
        r2 = fetch_with_backoff(
            "GET",
            DEEPL_USAGE_ENDPOINT,
            headers={"Authorization": f"DeepL-Auth-Key {args.api_key}"},
            params=None,
            timeout=args.timeout,
        )
        r2.raise_for_status()
        print("DeepL usage:", r2.json())
    except Exception as e:
        print(f"Warning: usage fetch failed: {e}", file=sys.stderr)

    all_strings: List[str] = []
    collect_leaf_strings(src_data, all_strings)

    cache: Dict[str, str] = {}
    with tqdm(total=total, desc=f"Translating → {tgt_code}") as prog:
        result = translate_structure(
            src_data, args.api_key, src_lang, tgt_code, args.timeout, cache, prog
        )

    dump_json(args.output, result)
    print("Done →", args.output)


if __name__ == "__main__":
    main()
