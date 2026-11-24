import math
import torch
import torch.nn.functional as F
from genia.core.visibility_attn import build_footprint_bias

torch.manual_seed(7)

def reference(bias, q, k, v):
    """Dense reference: upstream to_dense() bias + F.scaled_dot_product_attention (fp32)."""
    dense = bias.to_dense().float()  # (N, 1, maxL, P)
    N, P = len(bias.q_seqlen), bias.kv_len
    outs, qo = [], 0
    for n, L in enumerate(bias.q_seqlen):
        qh = q[0, qo:qo + L].permute(1, 0, 2).unsqueeze(0).float()
        kh = k[0, n * P:(n + 1) * P].permute(1, 0, 2).unsqueeze(0).float()
        vh = v[0, n * P:(n + 1) * P].permute(1, 0, 2).unsqueeze(0).float()
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=dense[n, :, :L])
        outs.append(o[0].permute(1, 0, 2))
        qo += L
    return torch.cat(outs, dim=0)

def run_case(name, streams, L_list, P, alpha, compensate, device, dtype=torch.float32):
    N = len(L_list)
    T = sum(L_list)
    H, D = 4, 16
    q = torch.randn(1, T, H, D, device=device, dtype=dtype)
    k = torch.randn(1, N * P, H, D, device=device, dtype=dtype)
    v = torch.randn(1, N * P, H, D, device=device, dtype=dtype)
    bias = build_footprint_bias(streams, L_list, P, alpha, compensate)
    out = bias.attend(q, k, v, L_list, [P] * N)
    ref = reference(bias, q.float(), k.float(), v.float())
    d = (out.float().cpu() - ref.cpu()).abs().max().item()
    print(f"{name} [{device}/{dtype}]: max abs diff vs dense ref = {d:.3e}")
    return d

def make_masks(L_list, width, density=0.08, device="cpu"):
    return [torch.rand(L, width, device=device) < density for L in L_list]

for device in ["cpu", "mps"]:
    # Case 1: two streams, first active, second passive, no compensation
    L_list, P = [7, 5], 48
    masks = make_masks(L_list, 24, device=device)
    streams = [(0, 24, masks, True), (24, 48, None, False)]
    d = run_case("active+passive, compensate=off", streams, L_list, P, 3.0, False, device)
    assert d < 1e-5, d
    # Case 2: compensation on (c_i active)
    d = run_case("active+passive, compensate=on", streams, L_list, P, 3.0, True, device)
    assert d < 1e-5, d
    # Case 3: two active streams + one passive, compensation on
    masks2 = make_masks(L_list, 16, device=device)
    streams3 = [(0, 24, masks, True), (24, 40, masks2, True), (40, 48, None, False)]
    d = run_case("2 active + passive, compensate=on", streams3, L_list, P, 5.0, True, device)
    assert d < 1e-5, d
    # Case 4: single view
    masks1 = make_masks([9], 24, device=device)
    streams4 = [(0, 24, masks1, True), (24, 48, None, False)]
    d = run_case("single view, compensate=on", streams4, [9], P, 4.0, True, device)
    assert d < 1e-5, d
    # Case 5: alpha=0 with compensation -> c should be 0, plain attention
    d = run_case("alpha=0, compensate=on", streams, L_list, P, 0.0, True, device)
    assert d < 1e-5, d

# Case 6: fp16 on MPS (the model's dtype) vs fp32 CPU reference
L_list, P = [7, 5], 48
masks = make_masks(L_list, 24, device="mps")
streams = [(0, 24, masks, True), (24, 48, None, False)]
d = run_case("fp16 inputs, compensate=on", streams, L_list, P, 3.0, True, "mps", torch.float16)
assert d < 2e-2, d
print("ALL STAGE-3 ATTENTION A/B CHECKS PASSED")
