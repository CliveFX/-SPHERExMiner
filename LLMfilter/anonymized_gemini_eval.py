#!/usr/bin/env python3
"""Run an anonymized Gemini challenge against compact LLM filter packets."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from llm_filter_experiment import _compact_spectrum


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESPONSE_DIR = ROOT / "responses"


SYSTEM_PROMPT = """You are a spectral sequence triage system.

Classify challenge sequences using only numeric sequence structure and the
labeled examples in the packet. Return strict JSON only.
"""


USER_INSTRUCTIONS = {
    "task": "Classify each anonymized challenge spectrum as signal-like or not signal-like.",
    "constraints": [
        "Use only the numeric arrays, flags, uncertainties, detector ids, and labeled examples.",
        "Treat identifiers as opaque labels with no meaning.",
        "Do not use object names, run names, target names, or outside wavelength associations.",
        "Return exactly one analysis object for every id in challenge_ids.",
        "Keep each reason short and numeric-pattern based.",
    ],
    "return_schema": {
        "packet_id": "string",
        "analyses": [
            {
                "id": "string copied from challenge_ids",
                "signal_like": True,
                "confidence": 0.0,
                "priority": "S|A|B|C|D",
                "center_nm": 0.0,
                "reason": "short numeric-pattern reason",
            }
        ],
        "global_notes": "short string",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_env(env_file: Path, key: str, default: str | None = None) -> str:
    env_values = read_env(env_file)
    value = os.environ.get(key) or env_values.get(key) or default
    if value is None:
        raise SystemExit(f"Missing {key} in environment or {env_file}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def anonymize_spectrum(row: dict[str, Any], anon_id: str, *, labeled: bool) -> dict[str, Any]:
    out = {
        "id": anon_id,
        "summary": row.get("sum"),
        "arrays": row.get("arr"),
    }
    if labeled:
        truth = row.get("truth") or {}
        out["label"] = {
            "signal_like": bool(truth.get("inj")),
            "center_nm": (truth.get("ln") / 1000.0) if truth.get("ln") is not None else None,
            "strength_hint": truth.get("snr"),
        }
    return out


def _extra_examples(
    *,
    jsonl_path: Path,
    excluded_signal_ids: set[str],
    prefix: str,
    limit: int,
) -> list[dict[str, Any]]:
    examples = []
    for row in read_jsonl(jsonl_path):
        signal_id = str(row.get("signal_id"))
        if signal_id in excluded_signal_ids:
            continue
        compact = _compact_spectrum(row, reveal_truth=True)
        examples.append(anonymize_spectrum(compact, f"{prefix}{len(examples):03d}", labeled=True))
        if len(examples) >= limit:
            break
    return examples


def build_anonymized_packet(
    size: int,
    *,
    known_signal_examples: int,
    known_non_signal_examples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    packet = read_json(DATA_DIR / f"compact_challenge_{size}.json")
    challenge_rows = packet.get("challenge_spectra", [])
    challenge_original_ids = {str(row.get("sid")) for row in challenge_rows}
    challenge_ids = [f"c{i:04d}" for i in range(len(challenge_rows))]
    row_map = {
        challenge_ids[i]: {
            "row_index": i,
            "original_signal_id": str(row.get("sid")),
        }
        for i, row in enumerate(challenge_rows)
    }
    signal_examples = _extra_examples(
        jsonl_path=DATA_DIR / "serialized_injected_spectra.jsonl",
        excluded_signal_ids=challenge_original_ids,
        prefix="p",
        limit=known_signal_examples,
    )
    non_signal_examples = _extra_examples(
        jsonl_path=DATA_DIR / "serialized_clean_spectra.jsonl",
        excluded_signal_ids=challenge_original_ids,
        prefix="n",
        limit=known_non_signal_examples,
    )
    anonymized = {
        "packet_id": f"{packet.get('packet_id')}_anonymized_shape_only",
        "format": "compact-shape-only-v1",
        "scale_notes": packet.get("format_notes", {}).get("array_scales", {}),
        "instructions": USER_INSTRUCTIONS,
        "known_signal_examples": signal_examples,
        "known_non_signal_examples": non_signal_examples,
        "challenge_ids": challenge_ids,
        "challenge_spectra": [
            anonymize_spectrum(row, challenge_ids[i], labeled=False) for i, row in enumerate(challenge_rows)
        ],
    }
    return anonymized, row_map


def call_gemini(
    *,
    env_file: Path,
    packet: dict[str, Any],
    model: str,
    base_url: str,
    max_tokens: int,
    timeout_sec: int,
) -> dict[str, Any]:
    key = get_env(env_file, "GEMINI_API_KEY")
    body = {
        "model": model,
        "reasoning_effort": "low",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(packet, separators=(",", ":"))},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        raw = json.loads(response.read().decode("utf-8"))
    raw["_client_elapsed_sec"] = time.time() - started
    raw["_client_payload_chars"] = len(data)
    return raw


def score(parsed: dict[str, Any], truth_path: Path, row_map: dict[str, Any]) -> dict[str, Any]:
    truth_rows = read_json(truth_path)["truth"]
    rows = []
    injected_correctly_found = 0
    clean_incorrectly_flagged = 0
    clean_correctly_ignored = 0
    injected_missed = 0
    for item in parsed.get("analyses", []):
        anon_id = str(item.get("id"))
        mapped = row_map.get(anon_id)
        if mapped is None:
            continue
        row_index = int(mapped["row_index"])
        if row_index >= len(truth_rows):
            continue
        truth = truth_rows[row_index]
        original_id = mapped["original_signal_id"]
        if truth is None:
            continue
        predicted = bool(item.get("signal_like"))
        actual = bool(truth.get("is_injected"))
        if predicted and actual:
            injected_correctly_found += 1
            outcome = "injected_correctly_found"
        elif predicted and not actual:
            clean_incorrectly_flagged += 1
            outcome = "clean_incorrectly_flagged"
        elif not predicted and actual:
            injected_missed += 1
            outcome = "injected_missed"
        else:
            clean_correctly_ignored += 1
            outcome = "clean_correctly_ignored"
        rows.append(
            {
                "id": anon_id,
                "original_signal_id": original_id,
                "outcome": outcome,
                "actual_injected": actual,
                "predicted_signal_like": predicted,
                "confidence": item.get("confidence"),
                "priority": item.get("priority"),
                "reported_center_nm": item.get("center_nm"),
                "actual_center_nm": truth.get("injected_line_nm"),
                "actual_find_me_snr": truth.get("find_me_snr"),
                "reason": item.get("reason"),
            }
        )
    return {
        "summary": {
            "challenge_rows_scored": len(rows),
            "injected_correctly_found": injected_correctly_found,
            "clean_incorrectly_flagged": clean_incorrectly_flagged,
            "clean_correctly_ignored": clean_correctly_ignored,
            "injected_missed": injected_missed,
            "injected_recovery_fraction": (
                injected_correctly_found / (injected_correctly_found + injected_missed)
                if injected_correctly_found + injected_missed
                else None
            ),
            "flagged_clean_fraction": (
                clean_incorrectly_flagged / (clean_incorrectly_flagged + clean_correctly_ignored)
                if clean_incorrectly_flagged + clean_correctly_ignored
                else None
            ),
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--known-signal-examples", type=int, default=8)
    parser.add_argument("--known-non-signal-examples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = get_env(args.env_file, "GEMINI_BASE_URL")
    model = get_env(args.env_file, "GEMINI_MODEL", "gemini-3.5-flash")
    packet, row_map = build_anonymized_packet(
        args.size,
        known_signal_examples=args.known_signal_examples,
        known_non_signal_examples=args.known_non_signal_examples,
    )
    request_path = DATA_DIR / f"gemini_anonymized_challenge_{args.size}.json"
    map_path = DATA_DIR / f"gemini_anonymized_row_map_{args.size}.json"
    write_json(request_path, packet)
    write_json(map_path, row_map)
    print(
        json.dumps(
            {
                "request_path": str(request_path),
                "challenge_count": len(packet["challenge_spectra"]),
                "known_signal_examples": len(packet["known_signal_examples"]),
                "known_non_signal_examples": len(packet["known_non_signal_examples"]),
                "request_chars": len(json.dumps(packet, separators=(",", ":"))),
                "model": model,
                "reasoning_effort": "low",
                "max_tokens": args.max_tokens,
            },
            indent=2,
        ),
        flush=True,
    )
    try:
        raw = call_gemini(
            env_file=args.env_file,
            packet=packet,
            model=model,
            base_url=base_url,
            max_tokens=args.max_tokens,
            timeout_sec=args.timeout_sec,
        )
    except urllib.error.HTTPError as exc:
        print(exc.read().decode(errors="replace"))
        raise
    content = raw["choices"][0]["message"]["content"]
    out = {
        "request": {
            "model": model,
            "base_url": base_url,
            "size": args.size,
            "reasoning_effort": "low",
            "max_tokens": args.max_tokens,
            "elapsed_sec": raw.get("_client_elapsed_sec"),
            "payload_chars": raw.get("_client_payload_chars"),
            "usage": raw.get("usage"),
            "finish_reason": raw["choices"][0].get("finish_reason"),
            "request_path": str(request_path),
            "id_map_path": str(map_path),
        },
        "content": content,
        "raw_response": raw,
    }
    try:
        parsed = json.loads(content)
        out["parse_ok"] = True
        out["parsed"] = parsed
        out["score"] = score(parsed, DATA_DIR / f"truth_{args.size}.json", row_map)
    except Exception as exc:
        out["parse_ok"] = False
        out["parse_error"] = str(exc)
    out_path = RESPONSE_DIR / f"gemini_anonymized_challenge_{args.size}_low_reasoning.json"
    write_json(out_path, out)
    print(
        json.dumps(
            {
                "wrote": str(out_path),
                "elapsed_sec": round(float(out["request"]["elapsed_sec"] or 0), 1),
                "finish_reason": out["request"]["finish_reason"],
                "usage": out["request"]["usage"],
                "parse_ok": out["parse_ok"],
                "score_summary": out.get("score", {}).get("summary"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
