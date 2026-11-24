"""Stage 5 verification: silicon-backed mesh rendering vs brute-force raycast."""
import sys
import torch

sys.path.insert(0, "/Users/fhijaz/alt3d/GenIA")
sys.path.insert(0, "/Users/fhijaz/alt3d/sam3d-objects-mac")

from genia.core.utils.mesh_rendering import (
    render_depth_and_alpha, render_rgba_and_depth)

torch.manual_seed(5)
H = W = 100
fx = fy = 100.0
cx = cy = 50.0

# Slanted quad (2 triangles) spanning x,y in [-0.4, 0.4], z from 1.6 to 2.4,
# plus a small occluding triangle in front at z=1.2.
verts = torch.tensor([
    [-0.4, -0.4, 1.6], [0.4, -0.4, 2.0], [0.4, 0.4, 2.4], [-0.4, 0.4, 2.0],
    [-0.1, -0.1, 1.2], [0.15, -0.05, 1.2], [0.0, 0.15, 1.2],
], dtype=torch.float32)
faces = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=torch.int32)
vert_rgb = torch.tensor([
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
    [0.2, 0.9, 0.9], [0.9, 0.2, 0.9], [0.9, 0.9, 0.2],
], dtype=torch.float32)


def raycast_reference(verts, faces, vert_rgb, fx, fy, cx, cy, H, W):
    """Per-pixel ray through pixel centre, Moller-Trumbore, nearest hit."""
    img = torch.zeros(H, W, 3)
    dep = torch.zeros(H, W)
    alp = torch.zeros(H, W)
    V = verts
    F = faces.long()
    for yy in range(H):
        for xx in range(W):
            d = torch.tensor([(xx + 0.5 - cx) / fx, (yy + 0.5 - cy) / fy, 1.0])
            best_t = float("inf")
            best = None
            for fi in range(F.shape[0]):
                a, b, c = V[F[fi, 0]], V[F[fi, 1]], V[F[fi, 2]]
                e1, e2 = b - a, c - a
                pv = torch.cross(d, e2)
                det = float(e1 @ pv)
                if abs(det) < 1e-12:
                    continue
                inv = 1.0 / det
                tv = -a  # origin at 0
                u = float(tv @ pv) * inv
                if u < -1e-9 or u > 1 + 1e-9:
                    continue
                qv = torch.cross(tv, e1)
                v = float(d @ qv) * inv
                if v < -1e-9 or u + v > 1 + 1e-9:
                    continue
                t = float(e2 @ qv) * inv
                if t > 0 and t < best_t:
                    best_t = t
                    best = (fi, u, v)
            if best is not None:
                fi, u, v = best
                w0 = 1 - u - v
                hit = best_t * d
                dep[yy, xx] = hit[2]
                alp[yy, xx] = 1.0
                img[yy, xx] = (w0 * vert_rgb[F[fi, 0]] + u * vert_rgb[F[fi, 1]]
                               + v * vert_rgb[F[fi, 2]])
    return img, alp, dep


img_r, alp_r, dep_r = raycast_reference(verts, faces, vert_rgb, fx, fy, cx, cy, H, W)
depth, alpha = render_depth_and_alpha(verts, faces, None, fx, fy, cx, cy, H, W)
rgb, alpha2, depth2 = render_rgba_and_depth(verts, faces, vert_rgb, None, fx, fy, cx, cy, H, W)

both = (alp_r > 0) & (alpha > 0)
only_ref = (alp_r > 0) & (alpha == 0)
only_new = (alp_r == 0) & (alpha > 0)
print(f"coverage: ref {int(alp_r.sum())} px, silicon {int(alpha.sum())} px, "
      f"ref-only {int(only_ref.sum())}, silicon-only {int(only_new.sum())} "
      f"(edge pixels may differ by rasterisation rule)")
d_dep = (depth - dep_r).abs()[both].max().item()
d_dep2 = (depth2 - dep_r).abs()[both].max().item()
d_rgb = (rgb - img_r).abs()[both].max().item()
d_alp = (alpha - alpha2).abs().max().item()
print(f"depth max diff (covered by both): {d_dep:.2e} / rgba path {d_dep2:.2e}")
print(f"rgb max diff: {d_rgb:.2e};  alpha consistency between the two fns: {d_alp:.2e}")

# texture path: quad facing camera at z=2 covering the centre, UV = full [0,1]
tex = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]])  # glTF rows: top row first
verts2 = torch.tensor([[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0],
                       [0.5, 0.5, 2.0], [-0.5, 0.5, 2.0]])
faces2 = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
uv = torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])  # glTF: v down
# caller passes tex pre-flipped to nvdiffrast bottom-up order:
tex_flipped = tex.flip(0).contiguous()
rgb_t, alp_t, dep_t = render_rgba_and_depth(
    verts2, faces2, vert_rgb[:4], None, fx, fy, cx, cy, H, W,
    uv=uv, tex=tex_flipped)
# projected quad spans pixels ~25..75 in both axes. With a 2x2 texture and
# align_corners=True, quadrant-centre pixels sit at uv (0.25|0.75, 0.25|0.75)
# and bilinearly blend all four texels; expectations below are the exact
# blends for the UVs assigned above (v=1 at the top edge), computed by hand.
samples = {(37, 37): (0.375, 0.25, 0.75), (37, 62): (0.625, 0.75, 0.75),
           (62, 37): (0.625, 0.25, 0.25), (62, 62): (0.375, 0.75, 0.25)}
ok = True
for (yy, xx), want in samples.items():
    got = tuple(round(float(c), 3) for c in rgb_t[yy, xx])
    match = all(abs(g - w) < 0.02 for g, w in zip(got, want))
    ok = ok and match
    print(f"tex sample (y={yy}, x={xx}): got {got} want {want} {'OK' if match else 'MISMATCH'}")
print("STAGE5:", "PASS" if (d_dep < 1e-3 and d_rgb < 1e-3 and ok) else "CHECK ABOVE")
