"""Stage 4 verification: differentiable rasterizer.

1. Gradient verification by central finite differences (float64, CPU).
   RGB+alpha are continuous in all parameters -> checked over the full image.
   Expected depth is smooth only where alpha is comfortably > 0 (it steps
   from 0 to ~z at the culled 3-sigma footprint boundary, exactly as gsplat's
   RGB+ED does), so depth is checked on a fixed interior mask (alpha > 0.3).
   A naive full-output gradcheck fails on that step for ANY implementation
   of these semantics; that is a property of the semantics, not the code.
2. Brute-force per-pixel reference A/B (rgb, alpha; depth on alpha > 0.05).
3. Forward A/B vs the fork's gsplat_silicon (should be bit-exact).
4. MPS forward+backward smoke.
5. Benchmarks at 100k / 1M gaussians, 512x512.
"""
import itertools
import sys, time
import torch

sys.path.insert(0, "/Users/fhijaz/alt3d/GenIA")
sys.path.insert(0, "/Users/fhijaz/alt3d/sam3d-objects-mac")

from genia.core.utils.diff_rasterizer import rasterization as diff_rast
from sam3d_objects.model.backbone.tdfy_dit.renderers.gsplat_silicon import (
    rasterization as silicon_rast)

torch.manual_seed(7)


def make_scene(N, dev, dtype, sh_degree=0, spread=1.2, zbase=4.0):
    K = (sh_degree + 1) ** 2
    means = torch.cat([
        (torch.rand(N, 2, device=dev, dtype=dtype) - 0.5) * 2 * spread,
        zbase + torch.rand(N, 1, device=dev, dtype=dtype) * 1.5 - 0.75,
    ], dim=1)
    quats = torch.randn(N, 4, device=dev, dtype=dtype)
    scales = 0.05 + 0.13 * torch.rand(N, 3, device=dev, dtype=dtype)
    opac = 0.25 + 0.6 * torch.rand(N, device=dev, dtype=dtype)
    colors = torch.randn(N, K, 3, device=dev, dtype=dtype) * 0.4
    if sh_degree == 0:
        colors[:, 0, :] = torch.rand(N, 3, device=dev, dtype=dtype) * 1.4 - 0.2
    return [means, quats, scales, opac, colors]


def cam(W, H, dev, dtype, f=0.9):
    K = torch.tensor([[f * W, 0, W / 2.0], [0, f * W, H / 2.0], [0, 0, 1]],
                     device=dev, dtype=dtype)
    return torch.eye(4, device=dev, dtype=dtype)[None], K[None]


NAMES = ["means", "quats", "scales", "opac", "colors"]
fails = []

# ---------------- 1. finite-difference gradient check ----------------
for deg in (0, 1):
    W = H = 32
    base = [p.double() for p in make_scene(8, "cpu", torch.float64, sh_degree=deg)]
    viewmats, Ks = cam(W, H, "cpu", torch.float64)
    wgt_rgb = torch.randn(W * H * 3, dtype=torch.float64)
    wgt_alp = torch.randn(W * H, dtype=torch.float64)
    wgt_dep = torch.randn(W * H, dtype=torch.float64)
    with torch.no_grad():
        _, alpha0, _ = diff_rast(*base, viewmats, Ks, W, H, sh_degree=deg,
                                 render_mode="RGB+ED", tile_size=64)
    interior = (alpha0[0, ..., 0] > 0.3).reshape(-1)

    def loss_rgb_alpha(params):
        rgbd, alpha, _ = diff_rast(*params, viewmats, Ks, W, H, sh_degree=deg,
                                   render_mode="RGB+ED", tile_size=64)
        return (rgbd[0, ..., :3].reshape(-1) * wgt_rgb).sum() + \
               (alpha.reshape(-1) * wgt_alp).sum()

    def loss_depth(params):
        rgbd, _, _ = diff_rast(*params, viewmats, Ks, W, H, sh_degree=deg,
                               render_mode="RGB+ED", tile_size=64)
        dep = rgbd[0, ..., 3].reshape(-1)
        return (dep[interior] * wgt_dep[interior]).sum()

    eps = 1e-6
    worst_all = 0.0
    for loss_fn in (loss_rgb_alpha, loss_depth):
        params = [p.clone().requires_grad_(True) for p in base]
        analytic = torch.autograd.grad(loss_fn(params), params)
        for i in range(5):
            for idx in itertools.product(*[range(s) for s in base[i].shape]):
                pp = [p.clone() for p in base]
                pm = [p.clone() for p in base]
                pp[i][idx] += eps
                pm[i][idx] -= eps
                fd = float((loss_fn(pp) - loss_fn(pm)) / (2 * eps))
                worst_all = max(worst_all, abs(fd - float(analytic[i][idx])))
    print(f"GRADIENT FD CHECK deg={deg}: max |analytic - fd| = {worst_all:.3e} "
          f"(rgb+alpha full frame; depth on {int(interior.sum())} interior px)")
    if worst_all > 1e-5:
        fails.append(f"grad deg {deg}")

# ---------------- 2. brute-force reference ----------------
def brute_force(means, quats, scales, opac, colors_rgb, viewmat, K, W, H, bg):
    from genia.core.utils.diff_rasterizer import _quat_to_rotmat
    Rv, tv = viewmat[:3, :3], viewmat[:3, 3]
    p = means @ Rv.T + tv
    Rq = _quat_to_rotmat(quats)
    M = Rq * scales[:, None, :]
    S = M @ M.transpose(1, 2)
    Sc = Rv[None] @ S @ Rv.T[None]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    img = bg[None, None, :].expand(H, W, 3).clone()
    dep = torch.zeros(H, W)
    alp = torch.zeros(H, W)
    order = torch.argsort(p[:, 2])
    for yy in range(H):
        for xx in range(W):
            T = 1.0
            col = torch.zeros(3)
            dz = 0.0
            for gi in order.tolist():
                z = p[gi, 2]
                if z <= 0.1:
                    continue
                u = fx * p[gi, 0] / z + cx
                v = fy * p[gi, 1] / z + cy
                J = torch.tensor([[fx / z, 0, -fx * p[gi, 0] / z**2],
                                  [0, fy / z, -fy * p[gi, 1] / z**2]])
                c2 = J @ Sc[gi] @ J.T + 0.3 * torch.eye(2)
                inv = torch.inverse(c2)
                d = torch.tensor([xx + 0.5 - u, yy + 0.5 - v])
                pw = -0.5 * d @ inv @ d
                a = min(float(opac[gi]) * float(torch.exp(pw.clamp(max=0.0))), 0.999)
                w = a * T
                col = col + w * colors_rgb[gi]
                dz = dz + w * float(z)
                T = T * (1 - a)
                if T < 1e-6:
                    break
            img[yy, xx] = col + T * bg
            alp[yy, xx] = 1 - T
            dep[yy, xx] = dz / (1 - T) if (1 - T) > 0 else 0.0
    return img, alp, dep


W = H = 24
params = make_scene(20, "cpu", torch.float32, sh_degree=0)
means, quats, scales, opac, colors = params
viewmats, Ks = cam(W, H, "cpu", torch.float32)
bg = torch.tensor([0.1, 0.2, 0.3])
from genia.core.utils.diff_rasterizer import _compute_colors
rgb_pre = _compute_colors(means, colors, 0, viewmats[0])
img_r, alp_r, dep_r = brute_force(means, quats, scales, opac, rgb_pre,
                                  viewmats[0], Ks[0], W, H, bg)
rgbd, alpha, _ = diff_rast(means, quats, scales, opac, colors, viewmats, Ks,
                           W, H, sh_degree=0, render_mode="RGB+ED",
                           backgrounds=bg[None], tile_size=16)
d_rgb = (rgbd[0, ..., :3] - img_r).abs().max().item()
d_alp = (alpha[0, ..., 0] - alp_r).abs().max().item()
mask = alp_r > 0.05
d_dep = (rgbd[0, ..., 3] - dep_r).abs()[mask].max().item()
print(f"BRUTE-FORCE A/B: rgb {d_rgb:.2e}  alpha {d_alp:.2e}  "
      f"depth(alpha>0.05) {d_dep:.2e}")
if max(d_rgb, d_alp, d_dep) > 1e-3:
    fails.append("brute force")

# ---------------- 3. forward A/B vs gsplat_silicon ----------------
W = H = 128
for deg in (0, 2):
    params = [p.float() for p in make_scene(3000, "cpu", torch.float32, sh_degree=deg)]
    means, quats, scales, opac, colors = params
    viewmats, Ks = cam(W, H, "cpu", torch.float32)
    bg = torch.zeros(1, 3)
    with torch.no_grad():
        rgbd, alpha, _ = diff_rast(means, quats, scales, opac, colors, viewmats,
                                   Ks, W, H, sh_degree=deg, render_mode="RGB+ED",
                                   backgrounds=bg)
        scol, salp, _ = silicon_rast(means, quats, scales, opac, colors, viewmats,
                                     Ks, W, H, sh_degree=deg, backgrounds=bg)
    d_rgb = (rgbd[0, ..., :3] - scol[0]).abs().max().item()
    d_alp = (alpha[0, ..., 0] - salp[0, ..., 0]).abs().max().item()
    print(f"SILICON A/B deg={deg}: rgb {d_rgb:.2e}  alpha {d_alp:.2e}")
    if max(d_rgb, d_alp) > 5e-4:
        fails.append(f"silicon deg {deg}")

# ---------------- 4. MPS smoke ----------------
dev = "mps"
W = H = 128
params = [p.float().to(dev).requires_grad_(True)
          for p in make_scene(5000, "cpu", torch.float32, sh_degree=1)]
viewmats, Ks = cam(W, H, dev, torch.float32)
rgbd, alpha, _ = diff_rast(*params, viewmats, Ks, W, H, sh_degree=1,
                           render_mode="RGB+ED")
loss = rgbd.square().mean() + alpha.mean()
loss.backward()
gn = [float(p.grad.norm()) if p.grad is not None else -1 for p in params]
finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
print(f"MPS SMOKE: loss {float(loss):.4f}  grad norms {[round(g, 3) for g in gn]}  finite {finite}")
if not finite or any(g <= 0 for g in gn):
    fails.append("mps smoke")

# ---------------- 5. benchmarks ----------------
for N in (100_000, 1_000_000):
    W = H = 512
    params = [p.float().to(dev) for p in
              make_scene(N, "cpu", torch.float32, sh_degree=0, spread=1.6)]
    viewmats, Ks = cam(W, H, dev, torch.float32)
    with torch.no_grad():
        diff_rast(*params, viewmats, Ks, W, H, sh_degree=0, render_mode="RGB+ED")
    torch.mps.synchronize()
    t0 = time.time()
    with torch.no_grad():
        rgbd, alpha, _ = diff_rast(*params, viewmats, Ks, W, H, sh_degree=0,
                                   render_mode="RGB+ED")
    torch.mps.synchronize()
    t_fwd = time.time() - t0
    params = [p.requires_grad_(True) for p in params]
    torch.mps.empty_cache()
    t0 = time.time()
    rgbd, alpha, _ = diff_rast(*params, viewmats, Ks, W, H, sh_degree=0,
                               render_mode="RGB+ED")
    (rgbd.square().mean() + alpha.mean()).backward()
    torch.mps.synchronize()
    t_fb = time.time() - t0
    mem_fb = torch.mps.current_allocated_memory() / 1e9
    print(f"BENCH N={N}: fwd {t_fwd:.2f}s  fwd+bwd {t_fb:.2f}s  "
          f"(resident after bwd {mem_fb:.1f} GB = params+grads+outputs)")
    del params, rgbd, alpha
    torch.mps.empty_cache()

print("FAILS:", fails if fails else "none")
