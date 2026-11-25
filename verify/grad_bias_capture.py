"""Gradient-bias diagnostic: capture REAL q/k/v from a craftsman finetune
main decode (baseline path, SAM3D_ATTN_IMPL unset), first masked_sdpa
call per shift, then abort the segment run.

Protocol mirrors ab_driver.py / flash_audit_driver.py: appearance_init_2
cache, seed 12345, ckpt=true, 1 iteration. Output:
~/alt3d/speed/diag_qkv.pt with per-shift {q,k,v} fp16 CPU [M,H,C],
seq_lens, and value statistics.
"""
import os
import sys
import time

os.chdir("/Users/fhijaz/alt3d/GenIA")
sys.path.insert(0, "/Users/fhijaz/alt3d/GenIA")
assert "SAM3D_ATTN_IMPL" not in os.environ, "flash gate must be OFF for capture"

import torch  # noqa: E402
import importlib  # noqa: E402

_ATT = "sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention"
wa = importlib.import_module(_ATT + ".windowed_attn")

fix = torch.load("/Users/fhijaz/alt3d/speed/flash_fixture.pt", weights_only=False)
EXPECT = {len(fix["seq_lens_shift0"]): 0, len(fix["seq_lens_shift4"]): 4}

STATE = {"in_decode": [], "captured": {}}
_orig_masked = wa.masked_sdpa


class CaptureDone(Exception):
    pass


def masked_spy(q, k, v, q_seqlen, kv_seqlen):
    if STATE["in_decode"] and STATE["in_decode"][-1] == "main":
        nw = len(q_seqlen)
        if nw in EXPECT:
            shift = EXPECT[nw]
            if shift not in STATE["captured"]:
                assert list(q_seqlen) == list(fix[f"seq_lens_shift{shift}"]), \
                    f"seq_lens mismatch for shift {shift}"
                rec = {}
                for name, t in (("q", q), ("k", k), ("v", v)):
                    tt = t[0].detach().to("cpu", torch.float16).clone()
                    rec[name] = tt
                    rec[name + "_stats"] = {
                        "mean": float(tt.float().mean()),
                        "std": float(tt.float().std()),
                        "absmean": float(tt.float().abs().mean()),
                        "max": float(tt.float().abs().max()),
                    }
                rec["seq_lens"] = list(q_seqlen)
                STATE["captured"][shift] = rec
                print(f"CAPTURED shift {shift}: q {tuple(rec['q'].shape)} "
                      f"qstd {rec['q_stats']['std']:.4f} "
                      f"kstd {rec['k_stats']['std']:.4f} "
                      f"vstd {rec['v_stats']['std']:.4f}", flush=True)
                if len(STATE["captured"]) == 2:
                    torch.save(STATE["captured"],
                               "/Users/fhijaz/alt3d/speed/diag_qkv.pt")
                    print("CAPTURE_SAVED", flush=True)
                    raise CaptureDone()
    return _orig_masked(q, k, v, q_seqlen, kv_seqlen)


wa.masked_sdpa = masked_spy

import genia.core.finetuning as ft  # noqa: E402

_orig_decode = ft.decode_tokens


def decode_spy(decoder, feats, coords, *a, **k):
    label = "main" if getattr(feats, "requires_grad", False) else "anchor"
    STATE["in_decode"].append(label)
    try:
        return _orig_decode(decoder, feats, coords, *a, **k)
    finally:
        STATE["in_decode"].pop()


ft.decode_tokens = decode_spy

from hydra import compose, initialize_config_dir  # noqa: E402

overrides = [
    "+experiment=mono_static", "dataset=image", "pipeline=mono_ours",
    "processing=map_anything",
    "dataset.path=/Users/fhijaz/alt3d/GenIA/demo/data/image",
    "dataset.scene_name=craftsman",
    "shape_init.shape_cache_dir=null",
    "appearance_init=perframe",
    "global_pose_refine@global_pose_refine_1=low_rgb",
    "processing.resume_from_cache=true",
    "output.save_cache=true",
    "output.output_dir=/Users/fhijaz/alt3d/GenIA/results/speedprof/image/craftsman/gradbias_capture",
    "finetuning.num_iterations=1",
    "finetuning.exit_after=true",
    "finetuning.random_background_seed=12345",
    "finetuning.decoder_checkpoint=true",
]
with initialize_config_dir(config_dir="/Users/fhijaz/alt3d/GenIA/core/configs",
                           version_base=None):
    cfg = compose(config_name="genia", overrides=overrides)

from genia.core.run import run  # noqa: E402

t0 = time.perf_counter()
try:
    run(cfg)
    print("RUN_COMPLETED_WITHOUT_CAPTURE", flush=True)
except CaptureDone:
    print(f"CAPTURE_DONE wall={time.perf_counter() - t0:.1f}", flush=True)
