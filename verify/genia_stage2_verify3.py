import os
os.environ.setdefault("SPARSE_BACKEND", "native")
import torch
import torch.nn.functional as F
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.masked_sdpa import (
    sdpa_blockdiag_with_bias, block_diag_attn_mask)

torch.manual_seed(3)
H, D = 4, 8
q_seqlen = kv_seqlen = [6, 6]
sq = 12
q = torch.randn(1, sq, H, D)
k = torch.randn(1, sq, H, D)
v = torch.randn(1, sq, H, D)
stacked = torch.randn(2, 1, 6, 6)
bias = block_diag_attn_mask(q_seqlen, kv_seqlen, dtype=torch.float32)
for i in range(2):
    s = slice(6 * i, 6 * i + 6)
    bias[s, s] += stacked[i, 0]
bias_dense = bias.unsqueeze(0).unsqueeze(0).expand(1, H, sq, sq).contiguous()
out = sdpa_blockdiag_with_bias(q, k, v, q_seqlen, kv_seqlen, stacked)
ref = F.scaled_dot_product_attention(
    q.permute(0, 2, 1, 3).float(), k.permute(0, 2, 1, 3).float(),
    v.permute(0, 2, 1, 3).float(), attn_mask=bias_dense.float()).permute(0, 2, 1, 3)[0]
d = (out - ref).abs().max().item()
print(f"stacked layout: max abs diff vs dense ref = {d:.3e}")
assert d < 1e-5
print("STACKED OK")
