#!/usr/bin/env python3
"""RUNG 8.75: real-image multi-token GREEDY DECODE through ZLUDA on gfx1151.

Rung 7 proved token-for-token greedy decode on the real Qwen3-VL-2B TEXT tower. Rung 8/8.5 proved a single
full real-IMAGE forward through ZLUDA (projections + attention). This rung closes the gap between them: it runs
HF's `model.generate(do_sample=False)` for several steps on a REAL image+prompt with `F.linear` routed through
ZLUDA->rocBLAS, exercising the incremental KV-cache decode path *with image tokens in context*, and checks the
generated token-id sequence is identical to the same model+inputs decoding unpatched on torch-CPU fp32.

Run: LD_LIBRARY_PATH=<zluda>:<nvrtc>:/opt/rocm/lib HSA_OVERRIDE_GFX_VERSION=11.5.1 python3 zluda_e2e_decode.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zluda_e2e import install_zluda_linear, uninstall_zluda_linear, build_inputs, MODEL
import zluda_e2e

NEW = int(os.environ.get("E2E_NEW_TOKENS", "8"))

def main():
    import torch
    from transformers import AutoModelForImageTextToText
    torch.manual_seed(0)
    print("loading", MODEL, "CPU fp32 ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    inputs = build_inputs()
    seq = inputs["input_ids"].shape[1]
    print("seq_len:", seq, "| grid_thw:", inputs.get("image_grid_thw", None),
          "| new_tokens:", NEW, flush=True)
    gen_kw = dict(max_new_tokens=NEW, do_sample=False, num_beams=1, use_cache=True)

    with torch.no_grad():
        ref_ids = model.generate(**inputs, **gen_kw)[0].tolist()
    ref_new = ref_ids[seq:]
    print("CPU oracle decode done. new ids:", ref_new, flush=True)

    orig = install_zluda_linear()
    try:
        with torch.no_grad():
            zl_ids = model.generate(**inputs, **gen_kw)[0].tolist()
    finally:
        uninstall_zluda_linear(orig)
    zl_new = zl_ids[seq:]
    print(f"ZLUDA decode done. F.linear GEMMs: {zluda_e2e._Z.calls} | new ids:", zl_new, flush=True)

    match = zl_new == ref_new
    npref = 0
    for a, b in zip(zl_new, ref_new):
        if a == b:
            npref += 1
        else:
            break
    print(f"token_for_token_match={match} matched_prefix={npref}/{len(ref_new)}", flush=True)
    print("VERDICT:", "PASS" if match else (f"PARTIAL(prefix {npref}/{len(ref_new)})"), flush=True)
    return 0 if match else 1

if __name__ == "__main__":
    sys.exit(main())
