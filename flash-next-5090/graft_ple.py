#!/usr/bin/env python3
"""Swap per_layer_token_embd.weight (IQ4_NL, 26.8 GiB) in the merged
qwen38fn artifact for the Q8_0 copy (50.7 GiB) that unsloth quantized
straight from BF16 — the PLE-Q8_0 discriminating experiment (see README:
t10 19/55 vs home 39/55 with every engine theory eliminated).

  graft_ple.py <src-artifact.gguf> <q8_0-shard3.gguf> <out.gguf>

Pure stdlib + streaming (the build host has no numpy): the KV section is
copied byte-verbatim, the tensor-info table is rewritten with the one new
type + recomputed offsets, and tensor data is streamed in file order.
Tensor sizes are derived from the SOURCE's own offset deltas (not a ggml
type-size table), so any type the artifact contains is handled; the one
size computed from first principles is the incoming Q8_0 tensor's, which
is verified against the donor shard's actual data length.

The donor (Q8_0 shard 3 of unsloth/Qwen3.8-Flash-Next-GGUF) holds exactly
one tensor — per_layer_token_embd.weight, Q8_0, dims [160, 320001536] —
which this script asserts rather than trusts.
"""
import struct
import sys

GGUF_MAGIC = b"GGUF"
PLE = "per_layer_token_embd.weight"
Q8_0 = 8
IQ4_NL = 20
Q8_0_BLCK, Q8_0_BYTES = 32, 34  # 32 weights -> 34 bytes


def parse_header(path):
    """Return (kv_bytes, tensors, header_end, alignment, version).
    tensors: list of dicts {name, dims, type, offset, info_span} in table order."""
    f = open(path, "rb")
    if f.read(4) != GGUF_MAGIC:
        sys.exit(f"{path}: not a GGUF")
    version, = struct.unpack("<I", f.read(4))
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

    def rd_str():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n)

    def skip_val(t):
        if t == 8:
            rd_str()
        elif t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            for _ in range(n):
                skip_val(et)
        else:
            f.read({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[t])

    alignment = 32
    kv_start = f.tell()
    for _ in range(n_kv):
        key = rd_str()
        t, = struct.unpack("<I", f.read(4))
        if key == b"general.alignment":
            pos = f.tell()
            raw = f.read({4: 4, 5: 4, 10: 8, 11: 8}.get(t, 4))
            alignment = int.from_bytes(raw, "little")
            f.seek(pos)
        skip_val(t)
    kv_end = f.tell()
    f.seek(kv_start)
    kv_bytes = f.read(kv_end - kv_start)

    tensors = []
    for _ in range(n_tensors):
        span0 = f.tell()
        name = rd_str().decode()
        nd, = struct.unpack("<I", f.read(4))
        dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
        ttype, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off,
                        "span": (span0, f.tell())})
    header_end = f.tell()
    f.close()
    return kv_bytes, tensors, header_end, alignment, version, n_tensors, n_kv


def align_up(x, a):
    return (x + a - 1) // a * a


def main():
    src_path, donor_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # ── Donor: assert it is exactly the Q8_0 PLE tensor ──────────────────
    _, dt, d_end, d_align, _, d_n, _ = parse_header(donor_path)
    assert d_n == 1 and dt[0]["name"] == PLE and dt[0]["type"] == Q8_0, dt[0]
    d_dims = dt[0]["dims"]
    n_elems = d_dims[0] * d_dims[1]
    assert n_elems % Q8_0_BLCK == 0
    ple_new_size = n_elems // Q8_0_BLCK * Q8_0_BYTES
    d_data_start = align_up(d_end, d_align)
    import os
    d_file = os.path.getsize(donor_path)
    assert d_file - d_data_start == align_up(ple_new_size, d_align), \
        (d_file, d_data_start, ple_new_size)
    print(f"donor OK: {PLE} Q8_0 dims={d_dims} {ple_new_size/2**30:.2f} GiB")

    # ── Source: locate PLE, derive per-tensor padded sizes from offsets ──
    kv_bytes, st, s_end, s_align, s_ver, s_n, s_nkv = parse_header(src_path)
    s_file = os.path.getsize(src_path)
    s_data_start = align_up(s_end, s_align)
    order = sorted(range(len(st)), key=lambda i: st[i]["offset"])
    assert order == list(range(len(st))), "tensor table not in data order"
    for i, t in enumerate(st):
        nxt = st[i + 1]["offset"] if i + 1 < len(st) else s_file - s_data_start
        t["padded"] = nxt - t["offset"]
        assert t["padded"] > 0, t["name"]
    ple = [t for t in st if t["name"] == PLE]
    assert len(ple) == 1, "PLE tensor not found in source"
    ple = ple[0]
    assert ple["type"] == IQ4_NL, f"source PLE type {ple['type']} != IQ4_NL — wrong source?"
    assert ple["dims"] == d_dims, (ple["dims"], d_dims)
    print(f"source OK: {s_n} tensors, PLE at offset {ple['offset']} "
          f"({ple['padded']/2**30:.2f} GiB IQ4_NL, align {s_align})")

    # ── New offsets ──────────────────────────────────────────────────────
    ple["new_padded"] = align_up(ple_new_size, s_align)
    run = 0
    for t in st:
        t["new_offset"] = run
        run += t.get("new_padded", t["padded"])

    # ── Write: header (KVs verbatim, new tensor table), then stream data ─
    with open(out_path, "wb") as out, open(src_path, "rb") as src, open(donor_path, "rb") as don:
        out.write(GGUF_MAGIC)
        out.write(struct.pack("<I", s_ver))
        out.write(struct.pack("<QQ", s_n, s_nkv))
        out.write(kv_bytes)
        for t in st:
            nb = t["name"].encode()
            out.write(struct.pack("<Q", len(nb)))
            out.write(nb)
            out.write(struct.pack("<I", len(t["dims"])))
            out.write(struct.pack(f"<{len(t['dims'])}Q", *t["dims"]))
            out.write(struct.pack("<I", Q8_0 if t["name"] == PLE else t["type"]))
            out.write(struct.pack("<Q", t["new_offset"]))
        hdr_end = out.tell()
        assert hdr_end == s_end, (hdr_end, s_end)  # same-size table by construction
        out.write(b"\x00" * (align_up(hdr_end, s_align) - hdr_end))

        CHUNK = 64 * 2**20

        def stream(fh, start, count):
            fh.seek(start)
            left = count
            while left:
                b = fh.read(min(CHUNK, left))
                assert b, "short read"
                out.write(b)
                left -= len(b)

        done = 0
        for t in st:
            if t["name"] == PLE:
                stream(don, d_data_start, ple_new_size)
                out.write(b"\x00" * (t["new_padded"] - ple_new_size))
            else:
                stream(src, s_data_start + t["offset"], t["padded"])
            done += 1
            if done % 200 == 0 or t["name"] == PLE:
                print(f"  {done}/{s_n} tensors written ({out.tell()/2**30:.1f} GiB)", flush=True)
        print(f"wrote {out.tell()/2**30:.2f} GiB -> {out_path}")


if __name__ == "__main__":
    main()
