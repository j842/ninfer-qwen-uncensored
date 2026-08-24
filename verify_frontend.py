#!/usr/bin/env python3
"""Verify the grafted frontend resources against a sha256 pin file.

Usage: verify_frontend.py <checkpoint-dir> [pin-file]

Defaults to frontend.sha256 (the Qwen3.8-27B pins) when no pin file is given.
Exits non-zero on the first missing or mismatched file. The converter performs
the same check internally against its built-in pins; running it here fails fast
with a clear message before the ~10-minute quantize step.
"""
import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PIN_FILE = HERE / "frontend.sha256"


def load_pins(pin_file=PIN_FILE):
    pins = {}
    for line in pin_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        want, name = line.split()
        pins[name] = want
    return pins


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(f"usage: {sys.argv[0]} <checkpoint-dir> [pin-file]")
    ckpt = pathlib.Path(sys.argv[1])
    pin_file = pathlib.Path(sys.argv[2]) if len(sys.argv) == 3 else PIN_FILE
    pins = load_pins(pin_file)
    for name, want in pins.items():
        path = ckpt / name
        if not path.exists():
            sys.exit(f"MISSING {name} — re-run the graft step")
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            sys.exit(f"MISMATCH {name}\n  got  {got}\n  want {want}")
        print(f"  {name}: OK")
    print("frontend resources verified")


if __name__ == "__main__":
    main()
