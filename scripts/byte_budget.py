import json, os, struct, re, collections, sys
M = os.path.expanduser("~/.cache/huggingface/hub/models--<org>--Qwen3.8-Flash-Next-NVFP4/snapshots/<revision>")
DT = {"BF16":2,"F16":2,"F32":4,"F8_E4M3":1,"F8_E5M2":1,"U8":1,"I8":1,"I32":4,"I64":8,"BOOL":1,"F64":8,"U32":4,"U16":2}
tensors = {}
for fn in sorted(os.listdir(M)):
    if not fn.endswith(".safetensors"): continue
    p = os.path.join(M, fn)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    for k, v in hdr.items():
        if k == "__metadata__": continue
        shape = v["shape"]; dt = v["dtype"]
        numel = 1
        for s in shape: numel *= s
        tensors[k] = (dt, shape, numel * DT.get(dt, 0), fn)
print("total tensors", len(tensors))
def cls(name):
    if "plefp8" in tensors[name][3] or ".ple." in name: return "PLE table/ple"
    if "experts" in name and "shared" not in name: return "MoE experts"
    if "shared_expert" in name: return "shared_expert"
    if "mtp" in name: return "MTP"
    if "visual" in name: return "vision"
    if "linear_attn" in name: return "GDN linear_attn"
    if "self_attn" in name: return "full self_attn"
    if "lm_head" in name: return "lm_head"
    if "embed_tokens" in name: return "embed_tokens"
    if "hyper" in name or "hc" in name.split(".")[-2:] or ".hc" in name: return "hyper_connection"
    if "mlp.gate" in name: return "router gate"
    if "norm" in name: return "norms"
    return "other:" + ".".join(name.split(".")[:4])
agg = collections.defaultdict(lambda: collections.defaultdict(int))
for k,(dt,shape,b,fn) in tensors.items():
    agg[cls(k)][dt] += b
tot = 0
print(f"{'class':<22}{'dtype':<10}{'GB':>8}")
for c in sorted(agg, key=lambda c: -sum(agg[c].values())):
    for dt, b in sorted(agg[c].items(), key=lambda x:-x[1]):
        print(f"{c:<22}{dt:<10}{b/1e9:>8.2f}")
        tot += b
print("TOTAL GB", round(tot/1e9,2))
# per-layer detail for one GDN layer, one attn layer, one expert set, MTP, lm_head
print("\n--- layer 0 (linear_attn) tensors ---")
for k,(dt,shape,b,fn) in sorted(tensors.items()):
    if re.match(r"model\.language_model\.layers\.0\.", k) or re.match(r"language_model\.model\.layers\.0\.", k) or re.match(r"model\.layers\.0\.", k):
        if "experts." in k and not "experts.0." in k: continue
        print(f"  {k:<90}{dt:<8}{str(shape):<28}{b/1e6:>8.2f} MB")
print("\n--- layer 3 (full_attention) non-expert tensors ---")
for k,(dt,shape,b,fn) in sorted(tensors.items()):
    if re.search(r"layers\.3\.", k) and "experts." not in k:
        print(f"  {k:<90}{dt:<8}{str(shape):<28}{b/1e6:>8.2f} MB")
print("\n--- MTP / lm_head / embed / PLE (non-table) ---")
for k,(dt,shape,b,fn) in sorted(tensors.items()):
    if ("mtp" in k and "experts." not in k) or "lm_head" in k or "embed_tokens" in k or (".ple." in k and "plefp8" not in fn):
        print(f"  {k:<90}{dt:<8}{str(shape):<28}{b/1e6:>8.2f} MB")
print("\n--- expert tensor example ---")
for k,(dt,shape,b,fn) in sorted(tensors.items()):
    if re.search(r"layers\.0\..*experts\.0\.", k) or re.search(r"layers\.0\..*experts\.(w13|w2|gate_up|down)", k):
        print(f"  {k:<90}{dt:<8}{str(shape):<28}{b/1e6:>8.2f} MB")
n_exp = len([k for k in tensors if re.search(r"layers\.0\..*experts\.\d+\.", k)])
print("expert tensors in layer 0:", n_exp)
