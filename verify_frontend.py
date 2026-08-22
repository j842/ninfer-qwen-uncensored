#!/usr/bin/env python3
"""Verify the grafted frontend resources against frontend.sha256.

Usage: verify_frontend.py <checkpoint-dir>

Exits non-zero on the first missing or mismatched file. The converter performs
the same check internally against its built-in pins; running it here fails fast
with a clear message before the ~10-minute quantize step.
"""
import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PIN_FILE = HERE / "frontend.sha256"


def load_pins():
    pins = {}
    for line in PIN_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        want, name = line.split()
        pins[name] = want
    return pins


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <checkpoint-dir>")
    ckpt = pathlib.Path(sys.argv[1])
    pins = load_pins()
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
