#!/usr/bin/env python3
"""Graft a converter-produced qwen4exp MTP head (blk.48.*) into the 4-shard
UD-Q4_K_XL target, writing ONE merged GGUF:
  - KVs from shard 1, minus split.*, with block_count 48->49,
    attention.compress_ratios += [0], + nextn_predict_layers = 1
  - all tensors from the 4 shards, then every blk.48.* tensor from the sidecar

Usage: graft_mtp.py <shard1> <shard2> <shard3> <shard4> <mtp-sidecar.gguf> <out.gguf>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tree" / "gguf-py"))
import gguf  # noqa: E402

shards = [gguf.GGUFReader(p) for p in sys.argv[1:5]]
side = gguf.GGUFReader(sys.argv[5])
out = sys.argv[6]

ARCH = "qwen4exp"
writer = gguf.GGUFWriter(out, ARCH)

n_mod = 0
for field in shards[0].fields.values():
    name = field.name
    if name == gguf.Keys.General.ARCHITECTURE or name.startswith("GGUF."):
        continue
    if name.startswith("split."):
        print(f"  drop {name}")
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    if name == f"{ARCH}.block_count":
        assert val == 48, val
        val = 49
        n_mod += 1
        print(f"  {name}: 48 -> 49")
    elif name == f"{ARCH}.attention.compress_ratios":
        assert len(val) == 48, len(val)
        val = list(val) + [0]
        n_mod += 1
        print(f"  {name}: len 48 -> 49 (MTP block dense)")
    writer.add_key_value(name, val, val_type, sub_type=sub_type)
assert n_mod == 2, f"expected to modify 2 KVs, modified {n_mod}"
writer.add_key_value(f"{ARCH}.nextn_predict_layers", 1, gguf.GGUFValueType.UINT32)
print(f"  + {ARCH}.nextn_predict_layers = 1")

mtp_tensors = [t for t in side.tensors if t.name.startswith("blk.48.")]
assert mtp_tensors, "sidecar has no blk.48.* tensors"
print(f"sidecar contributes {len(mtp_tensors)} blk.48 tensors "
      f"({sum(t.n_bytes for t in mtp_tensors)/2**30:.2f} GiB):")
for t in sorted(mtp_tensors, key=lambda t: t.name):
    print(f"    {t.name:44s} {t.tensor_type.name:8s} {list(t.shape)}")

seen = set()
all_tensors = []
for r in shards:
    for t in r.tensors:
        assert t.name not in seen, t.name
        seen.add(t.name)
        all_tensors.append((r, t))
for t in mtp_tensors:
    assert t.name not in seen, t.name
    all_tensors.append((side, t))

total = sum(t.n_bytes for _, t in all_tensors)
print(f"writing {len(all_tensors)} tensors, {total/2**30:.1f} GiB -> {out}")
for _, t in all_tensors:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
done = 0
for r, t in all_tensors:
    writer.write_tensor_data(t.data, tensor_endianess=r.endianess)
    done += t.n_bytes
    if done % (10 * 2**30) < t.n_bytes:
        print(f"  ... {done/2**30:.0f} GiB", flush=True)
writer.close()
print("DONE")
