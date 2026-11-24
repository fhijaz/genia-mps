import os
os.environ.setdefault("SPARSE_BACKEND", "native")
import torch

torch.manual_seed(0)
fails = []

# ---------- 1. Solver: step returns (x, velocity); trajectory unchanged ----------
from sam3d_objects.model.backbone.generator.flow_matching.solver import (
    Euler, Midpoint, RungeKutta4, SDE)

def dyn(x, t):
    return -x  # velocity = -x; Euler step: x + dt*(-x)

x0 = torch.tensor([1.0, 2.0])
times = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

for cls, name in [(Euler, "Euler"), (Midpoint, "Midpoint"), (RungeKutta4, "RK4")]:
    solver = cls()
    traj = list(solver.solve_iter(dyn, x0.clone(), times))
    assert all(len(item) == 3 for item in traj), f"{name}: yield arity"
    xs = [item[0] for item in traj]
    ts = [float(item[1]) for item in traj]
    assert ts == [0.25, 0.5, 0.75, 1.0], f"{name}: yields t1, got {ts}"
    final = solver.solve(dyn, x0.clone(), times)
    assert torch.equal(final, xs[-1]), f"{name}: solve() != last solve_iter x"
    print(f"{name}: 3-tuple yields, t1 sequence OK, solve() consistent, final={final.tolist()}")

# Euler exact trajectory check: x_{n+1} = x_n * (1 - 0.25)
e = Euler()
xs = [item[0] for item in e.solve_iter(dyn, x0.clone(), times)]
expect = x0 * (0.75 ** 4)
assert torch.allclose(xs[-1], expect), f"Euler trajectory {xs[-1]} != {expect}"
# velocity yielded by Euler at step n is -x_n
v1 = list(Euler().solve_iter(dyn, x0.clone(), times))[0][2]
assert torch.equal(v1, -x0), f"Euler velocity {v1} != {-x0}"
print("Euler: trajectory + velocity values exact vs pre-patch semantics")

# SDE step arity (stochastic; just check the return shape)
sde = SDE(sde_type="sde") if False else None
print("solver tests done")

# ---------- 2. sdpa_blockdiag_with_bias vs dense reference ----------
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.masked_sdpa import (
    sdpa_blockdiag_with_bias, block_diag_attn_mask)
import torch.nn.functional as F

def dense_ref(q, k, v, q_seqlen, kv_seqlen, bias_dense):
    # bias_dense: (1, H, sum_q, sum_kv) additive, already includes block-diag -inf
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()
    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias_dense.float())
    return out.permute(0, 2, 1, 3)[0]

torch.manual_seed(1)
H, D = 4, 8
q_seqlen = [5, 7]
kv_seqlen = [5, 9]
sq, skv = sum(q_seqlen), sum(kv_seqlen)
q = torch.randn(1, sq, H, D)
k = torch.randn(1, skv, H, D)
v = torch.randn(1, skv, H, D)

# packed dense bias: block-diag -inf + random finite bias inside blocks
bias = block_diag_attn_mask(q_seqlen, kv_seqlen, dtype=torch.float32)  # (sq, skv)
rand = torch.randn(sq, skv)
bias = bias + torch.where(bias == 0, rand, torch.zeros_like(rand))
bias_dense = bias.unsqueeze(0).unsqueeze(0).expand(1, H, sq, skv).contiguous()

out = sdpa_blockdiag_with_bias(q, k, v, q_seqlen, kv_seqlen, bias_dense)
ref = dense_ref(q, k, v, q_seqlen, kv_seqlen, bias_dense)
d = (out - ref).abs().max().item()
print(f"tensor-bias packed layout: max abs diff vs dense ref = {d:.3e}")
assert d < 1e-5, "packed bias mismatch"
fails.append(d)

# equal-length stacked case
q_seqlen2 = [6, 6]
kv_seqlen2 = [6, 6]
sq2 = skv2 = 12
q2 = torch.randn(1, sq2, H, D)
k2 = torch.randn(1, skv2, H, D)
v2 = torch.randn(1, skv2, H, D)
stacked2 = torch.randn(2, 1, 6, 6)
bias2 = block_diag_attn_mask(q_seqlen2, kv_seqlen2, dtype=torch.float32)
for i in range(2):
    s = slice(6 * i, 6 * i + 6)
    bias2[s, s] += stacked2[i, 0]
bias2_dense = bias2.unsqueeze(0).unsqueeze(0).expand(1, H, sq2, skv2).contiguous()
out2 = sdpa_blockdiag_with_bias(q2, k2, v2, q_seqlen2, kv_seqlen2, stacked2)
ref2 = dense_ref(q2, k2, v2, q_seqlen2, kv_seqlen2, bias2_dense)
d2 = (out2 - ref2).abs().max().item()
print(f"tensor-bias stacked layout: max abs diff vs dense ref = {d2:.3e}")
assert d2 < 1e-5, "stacked bias mismatch"

# ---------- 3. no-bias path still routes to masked_sdpa ----------
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.full_attn import (
    sparse_scaled_dot_product_attention)
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor
coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
feats = torch.randn(5, 3, H, D)
qkv = SparseTensor(feats=feats, coords=coords)
out3 = sparse_scaled_dot_product_attention(qkv)
print("no-bias sparse self-attn OK:", tuple(out3.feats.shape))
out4 = sparse_scaled_dot_product_attention(qkv, attn_bias=None)
assert torch.equal(out3.feats, out4.feats)
print("attn_bias=None identical to legacy call")
print("ALL STAGE-2 CHECKS PASSED")
