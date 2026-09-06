#!/usr/bin/env python3
"""Range-fetch the mtp.* tensors of Qwen/Qwen3.8-Flash-Next from HF and write
them as a single model.safetensors in --out (a fake HF checkpoint dir for
convert_hf_to_gguf.py --mtp). ~10 GB instead of the 360 GB checkpoint."""
import json, os, struct, sys, urllib.request

REPO = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main"
OUT = sys.argv[sys.argv.index("--out") + 1]

def get(url, rng=None, tries=4):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mtp-fetch/1"})
            if rng:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]-1}")
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if a == tries - 1:
                raise
            print(f"  retry {a+1} for {url.split('/')[-1]} ({e})", flush=True)

idx = json.loads(get(f"{REPO}/model.safetensors.index.json"))
wm = idx["weight_map"]
mtp = sorted((k, v) for k, v in wm.items() if k.startswith("mtp."))
byshard = {}
for name, shard in mtp:
    byshard.setdefault(shard, []).append(name)
print(f"{len(mtp)} tensors across {len(byshard)} shards", flush=True)

tensors = {}  # name -> (dtype, shape, bytes)
total = 0
for shard, names in sorted(byshard.items()):
    url = f"{REPO}/{shard}"
    n = struct.unpack("<Q", get(url, (0, 8)))[0]
    hdr = json.loads(get(url, (8, 8 + n)))
    base = 8 + n
    for name in names:
        e = hdr[name]
        s0, s1 = e["data_offsets"]
        print(f"  {shard} {name} {e['dtype']} {e['shape']} {(s1-s0)/1e6:.1f} MB", flush=True)
        data = get(url, (base + s0, base + s1))
        assert len(data) == s1 - s0, name
        tensors[name] = (e["dtype"], e["shape"], data)
        total += s1 - s0
print(f"fetched {total/1e9:.2f} GB", flush=True)

# manual safetensors writer (keeps bf16 raw; numpy has no bf16)
os.makedirs(OUT, exist_ok=True)
header, off = {}, 0
order = sorted(tensors)
for name in order:
    dt, shape, data = tensors[name]
    header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
    off += len(data)
hj = json.dumps(header).encode()
pad = (8 - len(hj) % 8) % 8
hj += b" " * pad
with open(os.path.join(OUT, "model.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(hj)))
    f.write(hj)
    for name in order:
        f.write(tensors[name][2])
print("wrote", os.path.join(OUT, "model.safetensors"), flush=True)

# minimal checkpoint metadata
for fn in ("config.json", "tokenizer.json", "tokenizer_config.json",
           "generation_config.json", "special_tokens_map.json"):
    try:
        open(os.path.join(OUT, fn), "wb").write(get(f"{REPO}/{fn}"))
        print("fetched", fn, flush=True)
    except Exception as e:
        print("skip", fn, e, flush=True)
