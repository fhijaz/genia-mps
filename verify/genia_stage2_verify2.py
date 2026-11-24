import os
os.environ.setdefault("SPARSE_BACKEND", "native")
import torch
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.full_attn import (
    sparse_scaled_dot_product_attention)

torch.manual_seed(2)
H, D = 4, 8
coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]])

# --- mimic _fused_pre: feats [T, 3*H*D] -> replace with [T, 3, H, D] ---
base = SparseTensor(feats=torch.randn(5, 3 * H * D), coords=coords)
qkv = base.replace(base.feats.reshape(5, 3, H, D))
print("qkv feats:", qkv.feats.shape, "shape:", qkv.shape)
out = sparse_scaled_dot_product_attention(qkv)
out_none = sparse_scaled_dot_product_attention(qkv, attn_bias=None)
assert torch.equal(out.feats, out_none.feats)
print("1-arg self-attn no-bias OK, attn_bias=None identical:", tuple(out.feats.shape))

# --- 2-arg cross-attn with a dummy .attend bias object (GenIA FootprintBias route) ---
q_base = SparseTensor(feats=torch.randn(5, H * D), coords=coords)
q = q_base.replace(q_base.feats.reshape(5, H, D))
kv = torch.randn(2, 6, 2, H, D)  # [N, L, 2, H, C] dense context

class DummyBias:
    def attend(self, qq, kk, vv, q_seqlen, kv_seqlen):
        assert qq.shape[0] == 1 and qq.dim() == 4, qq.shape
        self.seen = (tuple(qq.shape), q_seqlen, kv_seqlen)
        # attend = plain per-view SDPA without bias
        import torch.nn.functional as F
        outs, qo, kvo = [], 0, 0
        for ql, kvl in zip(q_seqlen, kv_seqlen):
            o = F.scaled_dot_product_attention(
                qq[:, qo:qo + ql].permute(0, 2, 1, 3),
                kk[:, kvo:kvo + kvl].permute(0, 2, 1, 3),
                vv[:, kvo:kvo + kvl].permute(0, 2, 1, 3))
            outs.append(o.permute(0, 2, 1, 3)[0])
            qo += ql; kvo += kvl
        return torch.cat(outs, dim=0)

dummy = DummyBias()
out2 = sparse_scaled_dot_product_attention(q, kv, attn_bias=dummy)
print("attend-object route OK:", tuple(out2.feats.shape), "hook saw:", dummy.seen)

# --- 2-arg cross-attn with tensor bias vs manual reference ---
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.masked_sdpa import (
    block_diag_attn_mask)
import torch.nn.functional as F
q_seqlen = [3, 2]
kv_seqlen = [6, 6]
bias = block_diag_attn_mask(q_seqlen, kv_seqlen, dtype=torch.float32)
rand = torch.randn(5, 12)
bias = bias + torch.where(bias == 0, rand, torch.zeros_like(rand))
bias = bias.unsqueeze(0).unsqueeze(0).expand(1, H, 5, 12).contiguous()
out3 = sparse_scaled_dot_product_attention(q, kv, attn_bias=bias)
# manual reference
qf = q.feats.reshape(5, H, D).unsqueeze(0)
kf = kv.reshape(12, 2, H, D).unbind(dim=1)
kk = kf[0].unsqueeze(0); vv = kf[1].unsqueeze(0)
ref = F.scaled_dot_product_attention(
    qf.permute(0, 2, 1, 3).float(), kk.permute(0, 2, 1, 3).float(),
    vv.permute(0, 2, 1, 3).float(), attn_mask=bias.float()).permute(0, 2, 1, 3)[0]
d = (out3.feats.float() - ref).abs().max().item()
print(f"2-arg tensor-bias vs manual ref: max abs diff = {d:.3e}")
assert d < 1e-5
print("ALL STAGE-2 ATTENTION CHECKS PASSED")
