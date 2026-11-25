"""Gradient-bias diagnostic: mtlattn varlen flash vs float64 ground truth.

Arms (identical fp16 inputs, identical upstream gradient values):
  (a) mtl_fp16    mtlattn.varlen_attention, fp16 in/out (production flash)
  (b) mtl_fp32    mtlattn.varlen_attention, fp32 upcast in, fp16 out
                  (production flash32 wiring)
  (c) baseline    masked_sdpa -> masked_chunked_sdpa (production baseline)
  (d) sdpa_fp32   per-window F.scaled_dot_product_attention, fp32 leaves
                  (sanity reference for MPS fp32 math)
Truth: float64 CPU dense per-window attention via autograd (exact
softmax, scale 1/sqrt(64)), inputs = the exact fp16 values in float64,
upstream gradient = the exact fp16 gout values in float64.

Stats per arm/tensor (out, dq, dk, dv): max abs, mean abs, SIGNED mean,
err-vs-truth slope <e,t>/<t,t> and Pearson corr; per-window signed/mean
errors; window-length and position breakdowns; boundary-row analysis;
second gout seed for sign/structure consistency; determinism re-run;
neighbor-perturbation and window-permutation leakage tests (shift 0);
uniform-length synthetic controls (128 / 256, randn and scale-matched).

Results -> ~/alt3d/speed/grad_bias_results.json (saved incrementally).
"""
import json
import math
import os
import sys
import time

os.environ["SAM3D_MPS_USE_SDPA"] = "1"
os.environ["SPARSE_BACKEND"] = "native"
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ["MTLATTN_FORCE_KERNEL"] = "1"
sys.path.insert(0, "/Users/fhijaz/alt3d/GenIA")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.masked_sdpa import masked_sdpa  # noqa: E402
import mtlattn  # noqa: E402

DEV = "mps"
SCALE = 1.0 / math.sqrt(64)
OUT_JSON = "/Users/fhijaz/alt3d/speed/grad_bias_results.json"
RESULTS = {"meta": {"scale": SCALE, "started": time.time()}}


def save():
    RESULTS["meta"]["updated"] = time.time()
    with open(OUT_JSON, "w") as f:
        json.dump(RESULTS, f)


def log(*a):
    print(*a, flush=True)


def sync():
    torch.mps.synchronize()


# ---------------------------------------------------------------- arms
def make_ctx(seq):
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq), 0).tolist()),
                      dtype=torch.int32, device=DEV)
    return {"seq": list(seq), "cu": cu, "maxlen": max(seq)}


def arm_mtl_fp16(q, k, v, ctx):
    return mtlattn.varlen_attention(q, k, v, ctx["cu"], ctx["cu"], ctx["maxlen"])


def arm_mtl_fp32(q, k, v, ctx):
    out = mtlattn.varlen_attention(q.float(), k.float(), v.float(),
                                   ctx["cu"], ctx["cu"], ctx["maxlen"])
    return out.to(torch.float16)


def arm_baseline(q, k, v, ctx):
    return masked_sdpa(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                       ctx["seq"], ctx["seq"])


def arm_sdpa_fp32(q, k, v, ctx):
    outs = []
    s = 0
    for ln in ctx["seq"]:
        qw = q[s:s + ln].permute(1, 0, 2).unsqueeze(0)
        kw = k[s:s + ln].permute(1, 0, 2).unsqueeze(0)
        vw = v[s:s + ln].permute(1, 0, 2).unsqueeze(0)
        o = F.scaled_dot_product_attention(qw, kw, vw)
        outs.append(o.squeeze(0).permute(1, 0, 2))
        s += ln
    return torch.cat(outs, dim=0)


ARMS = [("mtl_fp16", arm_mtl_fp16), ("mtl_fp32", arm_mtl_fp32),
        ("baseline", arm_baseline), ("sdpa_fp32", arm_sdpa_fp32)]


def run_arm(fn, qcpu, kcpu, vcpu, ctx, gout_mps, fp32_leaves=False):
    dt = torch.float32 if fp32_leaves else torch.float16
    q = qcpu.to(DEV, dt).detach().clone().requires_grad_(True)
    k = kcpu.to(DEV, dt).detach().clone().requires_grad_(True)
    v = vcpu.to(DEV, dt).detach().clone().requires_grad_(True)
    out = fn(q, k, v, ctx)
    loss = (out.float() * gout_mps.float()).sum()
    loss.backward()
    res = {"out": out.detach().float().cpu(),
           "dq": q.grad.float().cpu(), "dk": k.grad.float().cpu(),
           "dv": v.grad.float().cpu()}
    del out, loss, q, k, v
    torch.mps.empty_cache()
    return res


# ---------------------------------------------------------------- truth
def compute_truth(qcpu, kcpu, vcpu, seq, gout64):
    """float64 CPU per-window autograd. qcpu etc: fp16 CPU [M,H,C].
    gout64: float64 CPU [M,H,C]. Returns float32 CPU packed tensors."""
    M, H, C = qcpu.shape
    out = torch.zeros(M, H, C, dtype=torch.float32)
    dq = torch.zeros(M, H, C, dtype=torch.float32)
    dk = torch.zeros(M, H, C, dtype=torch.float32)
    dv = torch.zeros(M, H, C, dtype=torch.float32)
    s = 0
    for ln in seq:
        e = s + ln
        q = qcpu[s:e].double().permute(1, 0, 2).clone().requires_grad_(True)
        k = kcpu[s:e].double().permute(1, 0, 2).clone().requires_grad_(True)
        v = vcpu[s:e].double().permute(1, 0, 2).clone().requires_grad_(True)
        sc = torch.matmul(q, k.transpose(-1, -2)) * SCALE
        p = torch.softmax(sc, dim=-1)
        o = torch.matmul(p, v)
        o.backward(gout64[s:e].permute(1, 0, 2))
        out[s:e] = o.detach().permute(1, 0, 2).float()
        dq[s:e] = q.grad.permute(1, 0, 2).float()
        dk[s:e] = k.grad.permute(1, 0, 2).float()
        dv[s:e] = v.grad.permute(1, 0, 2).float()
        s = e
    return {"out": out, "dq": dq, "dk": dk, "dv": dv}


# ---------------------------------------------------------------- stats
def pearson(x, y):
    x = x.double()
    y = y.double()
    xm = x - x.mean()
    ym = y - y.mean()
    d = (xm.norm() * ym.norm()).item()
    if d == 0:
        return 0.0
    return float((xm * ym).sum().item() / d)


def tensor_stats(err, tru):
    e = err.double()
    t = tru.double()
    tt = float((t * t).sum())
    return {
        "max_abs": float(e.abs().max()),
        "mean_abs": float(e.abs().mean()),
        "signed_mean": float(e.mean()),
        "mean_abs_truth": float(t.abs().mean()),
        "slope_err_on_truth": float((e * t).sum()) / tt if tt else 0.0,
        "corr_err_truth": pearson(e.flatten(), t.flatten()),
    }


def full_stats(res, tru, seq):
    """Global stats + per-window + position/length/boundary breakdowns."""
    out = {"global": {}, "per_window": {}, "groups": {}, "boundary": {}}
    bounds = []
    s = 0
    for ln in seq:
        bounds.append((s, s + ln))
        s += ln
    for name in ("out", "dq", "dk", "dv"):
        err = res[name] - tru[name]
        out["global"][name] = tensor_stats(err, tru[name])
        pw_signed, pw_abs = [], []
        # pooled group accumulators: [sum_err, sum_abs, count]
        len_groups = {"len>=256": [0.0, 0.0, 0], "64<=len<256": [0.0, 0.0, 0],
                      "len<64": [0.0, 0.0, 0]}
        pos_groups = {"first_third": [0.0, 0.0, 0], "mid_third": [0.0, 0.0, 0],
                      "last_third": [0.0, 0.0, 0]}
        bnd = {"row_first": [0.0, 0.0, 0], "row_last": [0.0, 0.0, 0],
               "row_interior": [0.0, 0.0, 0]}
        nw = len(seq)
        for wi, (a, b) in enumerate(bounds):
            ln = b - a
            es = err[a:b].double()
            pw_signed.append(float(es.mean()))
            pw_abs.append(float(es.abs().mean()))
            se, sa, cnt = float(es.sum()), float(es.abs().sum()), es.numel()
            gk = "len>=256" if ln >= 256 else ("64<=len<256" if ln >= 64 else "len<64")
            len_groups[gk][0] += se
            len_groups[gk][1] += sa
            len_groups[gk][2] += cnt
            pk = ("first_third" if wi < nw / 3 else
                  "mid_third" if wi < 2 * nw / 3 else "last_third")
            pos_groups[pk][0] += se
            pos_groups[pk][1] += sa
            pos_groups[pk][2] += cnt
            if name in ("out", "dq"):
                for key, sl in (("row_first", slice(a, a + 1)),
                                ("row_last", slice(b - 1, b)),
                                ("row_interior", slice(a + 1, b - 1))):
                    if sl.stop > sl.start:
                        eb = err[sl].double()
                        bnd[key][0] += float(eb.sum())
                        bnd[key][1] += float(eb.abs().sum())
                        bnd[key][2] += eb.numel()
        out["per_window"][name] = {"signed_mean": pw_signed,
                                   "mean_abs": pw_abs}
        lens = torch.tensor(seq, dtype=torch.float64)
        out["groups"][name] = {
            "by_length": {k: {"signed_mean": v[0] / v[2], "mean_abs": v[1] / v[2],
                              "count": v[2]} for k, v in len_groups.items() if v[2]},
            "by_position": {k: {"signed_mean": v[0] / v[2], "mean_abs": v[1] / v[2],
                                "count": v[2]} for k, v in pos_groups.items() if v[2]},
            "corr_meanabs_vs_len": pearson(torch.tensor(pw_abs), lens),
            "corr_signed_vs_len": pearson(torch.tensor(pw_signed), lens),
        }
        if name in ("out", "dq"):
            out["boundary"][name] = {
                k: {"signed_mean": v[0] / v[2], "mean_abs": v[1] / v[2],
                    "count": v[2]} for k, v in bnd.items() if v[2]}
    return out


def make_gout(M, H, C, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(M, H, C, generator=g).to(torch.float16)


# ---------------------------------------------------------------- batches
def real_batches():
    cap = torch.load("/Users/fhijaz/alt3d/speed/diag_qkv.pt",
                     weights_only=False)
    batches = {}
    for shift in (0, 4):
        rec = cap[shift]
        batches[f"shift{shift}"] = {
            "q": rec["q"], "k": rec["k"], "v": rec["v"],
            "seq": list(rec["seq_lens"]),
            "provenance": "captured real q/k/v, craftsman finetune main decode",
            "qkv_stats": {t: rec[t + "_stats"] for t in ("q", "k", "v")},
        }
    return batches


def synth_batch(seq, seed, scale_q=1.0, scale_k=1.0, scale_v=1.0):
    M = sum(seq)
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(M, 12, 64, generator=g) * scale_q).to(torch.float16)
    k = (torch.randn(M, 12, 64, generator=g) * scale_k).to(torch.float16)
    v = (torch.randn(M, 12, 64, generator=g) * scale_v).to(torch.float16)
    return {"q": q, "k": k, "v": v, "seq": list(seq)}


# ---------------------------------------------------------------- main
def process_batch(tag, batch, do_leakage=False):
    seq = batch["seq"]
    M = sum(seq)
    H, C = batch["q"].shape[1], batch["q"].shape[2]
    ctx = make_ctx(seq)
    entry = {"provenance": batch.get("provenance", "synthetic"),
             "n_windows": len(seq), "M": M, "max_len": max(seq),
             "arms": {}}
    if "qkv_stats" in batch:
        entry["qkv_stats"] = batch["qkv_stats"]
    RESULTS[tag] = entry
    ref_mtl = {}
    for seed_name, seed in (("A", 20251125), ("B", 20251126)):
        gout_cpu = make_gout(M, H, C, seed)
        gout_mps = gout_cpu.to(DEV)
        t0 = time.perf_counter()
        tru = compute_truth(batch["q"], batch["k"], batch["v"], seq,
                            gout_cpu.double())
        log(f"[{tag}] truth seed {seed_name} done {time.perf_counter()-t0:.1f}s")
        for arm_name, fn in ARMS:
            res = run_arm(fn, batch["q"], batch["k"], batch["v"], ctx,
                          gout_mps, fp32_leaves=(arm_name == "sdpa_fp32"))
            st = full_stats(res, tru, seq)
            entry["arms"].setdefault(arm_name, {})[seed_name] = st
            g = st["global"]
            log(f"[{tag}] {arm_name} seed {seed_name}: "
                f"out sm {g['out']['signed_mean']:+.3e} max {g['out']['max_abs']:.3e} | "
                f"dq sm {g['dq']['signed_mean']:+.3e} max {g['dq']['max_abs']:.3e} slope {g['dq']['slope_err_on_truth']:+.3e} | "
                f"dk sm {g['dk']['signed_mean']:+.3e} | dv sm {g['dv']['signed_mean']:+.3e}")
            if arm_name == "mtl_fp16":
                ref_mtl[seed_name] = res
            else:
                del res
            save()
        # cross-seed per-window consistency is computed at the end from JSON
        del tru
        if seed_name == "A" and do_leakage:
            entry["determinism"], entry["leakage"] = leakage_block(
                batch, ctx, gout_mps, ref_mtl["A"])
            save()
    # cross-seed consistency of per-window signed dq errors
    cons = {}
    for arm_name in entry["arms"]:
        a = entry["arms"][arm_name]["A"]["per_window"]["dq"]["signed_mean"]
        b = entry["arms"][arm_name]["B"]["per_window"]["dq"]["signed_mean"]
        cons[arm_name] = pearson(torch.tensor(a), torch.tensor(b))
    entry["cross_seed_pw_dq_corr"] = cons
    save()
    log(f"[{tag}] done. cross-seed pw dq corr: {cons}")
    return entry


def leakage_block(batch, ctx, gout_mps, ref):
    seq = batch["seq"]
    det = {}
    # determinism: identical re-run of mtl_fp16
    res2 = run_arm(arm_mtl_fp16, batch["q"], batch["k"], batch["v"], ctx,
                   gout_mps)
    det = {n: float((res2[n] - ref[n]).abs().max()) for n in ("out", "dq", "dk", "dv")}
    del res2
    log(f"[leak] determinism max diffs: {det}")

    # window-order permutation (reverse + shuffle), fwd+bwd, unpermute
    nw = len(seq)
    gperm = torch.Generator().manual_seed(99)
    perm = torch.randperm(nw, generator=gperm).tolist()
    bounds = []
    s = 0
    for ln in seq:
        bounds.append((s, s + ln))
        s += ln

    def permute_packed(t, order):
        return torch.cat([t[bounds[i][0]:bounds[i][1]] for i in order], dim=0)

    pseq = [seq[i] for i in perm]
    pctx = make_ctx(pseq)
    gout_p = permute_packed(gout_mps.cpu(), perm).to(DEV)
    resp = run_arm(arm_mtl_fp16, permute_packed(batch["q"], perm),
                   permute_packed(batch["k"], perm),
                   permute_packed(batch["v"], perm), pctx, gout_p)
    # unpermute: resp rows are in permuted window order; map back
    inv = {}
    pos = 0
    for new_i, old_i in enumerate(perm):
        ln = seq[old_i]
        inv[old_i] = (pos, pos + ln)
        pos += ln
    perm_diff = {}
    for n in ("out", "dq"):
        back = torch.empty_like(ref[n])
        for old_i, (a, b) in inv.items():
            oa, ob = bounds[old_i]
            back[oa:ob] = resp[n][a:b]
        perm_diff[n] = float((back - ref[n]).abs().max())
    del resp
    log(f"[leak] permutation max diffs (out/dq): {perm_diff}")

    # neighbor perturbation: targets keep own q/k/v; all other windows'
    # k,v scaled x100; gout zero outside targets. Compare target slices.
    targets = sorted({0, nw // 2, nw - 1})
    k2 = batch["k"].clone()
    v2 = batch["v"].clone()
    gout2 = torch.zeros_like(gout_mps)
    for wi, (a, b) in enumerate(bounds):
        if wi in targets:
            gout2[a:b] = gout_mps[a:b]
        else:
            k2[a:b] = (k2[a:b].float() * 100.0).to(torch.float16)
            v2[a:b] = (v2[a:b].float() * 100.0).to(torch.float16)
    pert = {}
    for arm_name, fn in (("mtl_fp16", arm_mtl_fp16), ("baseline", arm_baseline)):
        res = run_arm(fn, batch["q"], k2, v2, ctx, gout2)
        ref_res = ref if arm_name == "mtl_fp16" else None
        if ref_res is None:
            ref_res = run_arm(fn, batch["q"], batch["k"], batch["v"], ctx,
                              gout_mps)
        d = {}
        for n in ("out", "dq", "dk", "dv"):
            mx = 0.0
            for wi in targets:
                a, b = bounds[wi]
                mx = max(mx, float((res[n][a:b] - ref_res[n][a:b]).abs().max()))
            d[n] = mx
        pert[arm_name] = d
        del res
        if arm_name != "mtl_fp16":
            del ref_res
        log(f"[leak] neighbor-perturbation {arm_name}: {d}")
    return det, {"permutation": perm_diff, "neighbor_perturbation": pert,
                 "targets": targets}


def main():
    batches = {}
    if os.path.exists("/Users/fhijaz/alt3d/speed/diag_qkv.pt"):
        batches.update(real_batches())
        log("using CAPTURED real q/k/v")
    else:
        fix = torch.load("/Users/fhijaz/alt3d/speed/flash_fixture.pt",
                         weights_only=False)
        for shift in (0, 4):
            b = synth_batch(fix[f"seq_lens_shift{shift}"], 777)
            b["provenance"] = "FALLBACK synthetic randn seed 777, real seq_lens"
            batches[f"shift{shift}"] = b
        log("capture missing - FALLBACK synthetic batches")
    for tag, batch in batches.items():
        process_batch(tag, batch, do_leakage=(tag == "shift0"))

    # synthetic controls: uniform window lengths
    scales = None
    if "shift0" in batches and "qkv_stats" in batches["shift0"]:
        st = batches["shift0"]["qkv_stats"]
        scales = (st["q"]["std"], st["k"]["std"], st["v"]["std"])
    for L, nw in ((128, 256), (256, 128)):
        seq = [L] * nw
        b = synth_batch(seq, 777)
        b["provenance"] = f"control uniform len {L}, randn seed 777"
        process_batch(f"control_uniform{L}_randn", b)
        if scales:
            b2 = synth_batch(seq, 777, *scales)
            b2["provenance"] = (f"control uniform len {L}, randn scaled to "
                                f"captured q/k/v std {scales}")
            process_batch(f"control_uniform{L}_scaled", b2)
    log("DIAG_DONE")


if __name__ == "__main__":
    main()
