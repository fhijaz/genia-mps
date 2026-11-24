# GenIA -> MPS Port Log (Phase 1: single-image)

Started: Nov 23, 2026. Scope: GenIA single-image setting end to end on Apple Silicon (MPS), bound to the local SAM 3D fork.
Plan source: ~/alt3d/GENIA_PHASE0.md. Hard constraint: ~/alt3d/sam3d-objects-mac must keep working (production tool).

## Stage 0: setup
- (pending entries appended per stage)

## Stage 1: env + device plumbing (Nov 23, 2026)
- Env: ~/alt3d/genia_env = fresh venv layered on s3d_env via a .pth in its
  site-packages (genia_pkg symlink dir -> GenIA root for the `genia` package;
  fork root -> fork sam3d_objects + fork pytorch3d shim; map-anything submodule
  root; s3d_env site-packages). s3d_env itself NOT modified. Installed into
  genia_env only: uniception==0.1.6 (--no-deps), lpips==0.1.4,
  pytorch-msssim==1.0.0, moviepy==1.0.3 (+proglog/imageio-ffmpeg/decorator).
  torch stays 2.15.0 from s3d_env (GenIA pins 2.7.1 upstream; fork is built and
  verified on 2.15.0, so 2.15.0 wins).
- Binding VERIFIED: `import sam3d_objects` in genia_env resolves to the fork
  (/Users/fhijaz/alt3d/sam3d-objects-mac/sam3d_objects), including after
  genia.core.paths import (paths.py APPEND seam works as designed).
- New central helper core/utils/device.py: get_device() (GENIA_DEVICE override,
  else cuda>mps>cpu), autocast_device_type(), empty_cache(), synchronize(),
  memory_allocated(), max_memory_allocated() (MPS: current-allocation proxy,
  no peak API). All GenIA-core device sites routed through it:
  run.py:137, inference.py autocast x3 + Inference.__init__ now injects
  config.device=str(get_device()) (pipeline.yaml has no device key; fork
  default is "cuda"), backbone.py _slim, model_cache.py (map-anything/MoGe/
  perceptual .to(get_device())), timing.py (_cuda() now returns an _MpsShim;
  memory_snapshot stays CUDA-only), interpolation.py (20 .cuda() sites),
  rendering.py, eval_assets_export.py, refinement.py, pipeline_state.py,
  pose_init.py, evaluation.py, visualization.py, depth.py, slat_decode.py,
  gaussian.py (device default None->get_device()), render_final_results.py
  (device defaults -> _DEFAULT_DEVICE=str(get_device())), orbit_viz.py,
  main.py (peak mem via helper), shape_inversion.py (bf16-auto accepts MPS,
  autocast device, MPS OOM RuntimeError catch; GT path only).
  NOT touched: actionmesh_video.py (dynamic, out of scope), final.py
  nvdiffrast block (stage 5), eval_assets_export.py:1468 nvdiffrast (eval path).
- run.py sets fork-parity env defaults (SPARSE_BACKEND=native,
  SAM3D_MPS_USE_SDPA=1, PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0,
  PYTORCH_ENABLE_MPS_FALLBACK=1), mirroring the fork cli.py.
- Runner: ~/alt3d/genia_run.sh (exports the same env + GENIA_SAM3D_WEIGHTS ->
  existing checkpoints folder; execs demo/run_demo.py under genia_env).
- Smoke: all core modules compile; device/safe modules import; sparse backend
  reports native, attention sdpa, device mps.

## Stage 2: SAM 3D patch hand-port onto the fork (Nov 23, 2026)
Backups: every touched fork file copied to <name>.orig-backup-genia first.
Consumer audit: solve_iter/step consumers in the fork are exactly the three
generate_iter loops + ODESolver.solve + solver.solve in log_likelihood;
base.py discards the yielded t ("for _, xt, _"), so the t0->t1 yield change
is inert for the fork standalone.
PORTED:
- solver.py (identical file, hunks verbatim): solve_iter yields (x_t, t1,
  velocity); Euler/SDE/Midpoint/RK4 step() return (x_tp1, velocity).
- flow_matching/model.py (diverged: fork added step_idx progress enum, kept):
  FlowMatching.generate_iter gains noise_init_override; both generate_iter
  loops unpack 3-tuples.
- shortcut/model.py (diverged, same treatment): _velocity_transform attr,
  noise_init_override, step_callback(t, x_t, velocity) after each step,
  _generate_dynamics applies _velocity_transform(t_raw, velocity, x_t).
- sparse/attention/modules.py (identical, verbatim): SparseMultiHeadAttention
  .forward gains attn_bias kwarg, forwarded on the cross-attention call.
- sparse/attention/full_attn.py (identical file, adapted): signature gains
  attn_bias=None; a bias interception runs BEFORE backend dispatch - object
  with .attend -> attend(q,k,v,q_seqlen,kv_seqlen) on (1,T,H,D) tensors;
  tensor bias -> masked_sdpa.sdpa_blockdiag_with_bias. (Upstream raises
  NotImplementedError for bias under non-xformers backends; the port honours
  it under sdpa instead. attn_bias=None path byte-identical behaviour.)
- sparse/attention/masked_sdpa.py (diverged): appended _is_stacked_bias /
  _slice_view_bias (verbatim from the patch) + sdpa_blockdiag_with_bias +
  _sdpa_with_bias_bhlc (F.scaled_dot_product_attention with the bias slice;
  on MPS without SAM3D_MPS_USE_SDPA it uses the fork chunked fp32 math path
  with bias added to scores, mirroring chunked_sdpa_bhlc).
- representations/gaussian/gaussian_model.py: get_features concat dim 2 -> 1
  (patch bugfix). FORK BEHAVIOUR CHANGE, only bites when _features_rest is
  not None; noted per instructions.
SKIPPED (per inventory, moot on the fork): __init__.py LIDRA change, ATTN /
BACKEND flips to xformers, conv_spconv dtype guard (native backend),
flexicubes kaolin asserts (fork stub), inference_pipeline set_attention_
backend removal (fork version is MPS-aware), utils3d shims (fork pins
utils3d 0.0.2, imports fine).
VERIFIED (/tmp/genia_stage2_verify*.py, genia_env, CPU):
- Euler/Midpoint/RK4: 3-tuple yields, t sequence = t1s, solve() == last
  solve_iter x; Euler trajectory and yielded velocities EXACT vs pre-patch
  semantics (x*(0.75^4) for v=-x, dt=0.25).
- sdpa_blockdiag_with_bias vs dense-mask F.sdpa reference: packed layout
  max abs diff 2.4e-07; stacked layout 2.4e-07-class (see verify3); through
  the full dispatch (2-arg cross-attn) 2.1e-07.
- Dummy .attend object receives (1, T, H, D) + correct seqlens through the
  dispatch; attn_bias=None output identical to legacy call.
- Fork imports cleanly under s3d_env (production env) after the patch.

## Stage 3: visibility attention on SDPA (Nov 23, 2026)
core/visibility_attn.py:
- triton / xformers imports guarded (None off CUDA); the @triton.jit merge
  kernel decorates through an identity fallback so the module imports on MPS.
- _flash_available() returns False for non-CUDA devices (was: unconditional
  torch.cuda.get_device_capability, which throws for MPS tensors).
- FootprintBias.attend(): on CUDA with triton+xformers the upstream path is
  untouched. Otherwise new _attend_dense(): per view, row chunks of 256,
  logits = q k^T / sqrt(D) in fp32, bias kept head-independent (chunk, P):
  -c_i on active keys [0, split) with FIX offsets zeroed, +alpha scattered
  onto footprint keys (vectorized repeat_interleave/index_put), softmax,
  @ v, cast back to q.dtype. (chunk, H, P) fp32 logits ~90 MB at P=5496, H=16.
VERIFIED (/tmp/genia_stage3_verify.py): attend() vs a dense reference built
from upstream's own FootprintBias.to_dense() + F.scaled_dot_product_attention,
5 configurations (compensate on/off, 2 active streams, single view, alpha=0)
x {CPU, MPS}: CPU max abs diff 0.0 (bit-exact), MPS fp32 <= 3.6e-07, MPS
fp16 inputs 4.6e-04. The tensor-bias SDPA branch (stage 2) verified at
2.1-2.4e-07 in stage 2.
Also landed: MapAnything weights downloaded (facebook/map-anything, 6.06 GB,
HF cache) - stage 6 partially done.

## Stage 4: differentiable Gaussian rasterizer (Nov 23, 2026)
New file core/utils/diff_rasterizer.py: pure-PyTorch differentiable EWA splat
rasterizer exposing the gsplat `rasterization` subset render_gaussian_params
uses (pinhole, classic mode, RGB+ED, SH degree 0-3). Forward math mirrors the
fork's gsplat_silicon exactly (same Jacobian projection, 0.3 px dilation,
tile binning, front-to-back cumprod compositing, fork sh_utils SH eval with
+0.5 offset and clamp_min(0)). Autograd design: binning / sort order / the
per-tile transmittance early-break are computed from detached values
(constants for the backward, like the CUDA kernel); per-tile chunks are
wrapped in torch.utils.checkpoint (non-reentrant) so backward recomputes one
chunk at a time. Expected depth = sum(w_i z_i) / accumulated alpha, 0 where
alpha is 0 (the semantics render_gaussian_params documents for gsplat).
Deviations from gsplat_silicon internals, all value-equivalent for valid
inputs: power clamped at 0 before exp (instead of zeroing post-exp; keeps
backward free of inf*0 NaNs); no early break difference beyond the detached
threshold. core/utils/rendering.py: gsplat import is now try/except with
fallback to diff_rasterizer (CUDA behaviour unchanged where gsplat exists).
VERIFIED (/tmp/genia_stage4_verify.py, all PASS):
- Forward A/B vs gsplat_silicon: BIT-EXACT (0.00e+00) rgb and alpha,
  SH degrees 0 and 2, N=3000 at 128x128.
- Brute-force per-pixel reference: rgb 4.2e-06, alpha 1.1e-05,
  depth 6.0e-05 on pixels with alpha > 0.05.
- Gradients: central finite differences (float64 CPU), max |analytic - fd|
  9.9e-09 (deg 0) / 8.0e-09 (deg 1) across means, quats, scales, opacities,
  SH colors. NOTE: a naive full-output gradcheck FAILS for these semantics -
  the normalised depth steps from 0 to ~z at the culled 3-sigma footprint
  boundary, so finite differences across the boundary explode; that step is
  a property of gsplat RGB+ED semantics, not of this code. The check
  therefore covers rgb+alpha over the full frame and depth on a fixed
  interior mask (alpha > 0.3).
- MPS smoke: forward+backward, finite non-zero grads for all 5 groups.
- Benchmarks (MPS, 512x512, SH deg 0): N=100k forward 1.00 s, fwd+bwd 3.21 s;
  N=1M forward 2.73 s, fwd+bwd 6.32 s. Practical ceiling is at/above SAM 3D's
  ~1M splat scale at recon resolution - no infeasibility found.

## Stage 5: mesh rendering wiring (Nov 23, 2026)
core/utils/mesh_rendering.py:
- pytorch3d imports guarded: pytorch3d.io and renderer.FoVPerspectiveCameras /
  HardFlatShader are absent from the fork shim; they now fall back (None /
  a stub raising ImportError) so the module imports. Only PyTorch3DMeshRenderer
  (visualisation helper) uses them. The fork shim itself was NOT extended -
  guards live in GenIA's tree, keeping the fork untouched in this stage.
- render_depth_and_alpha / render_rgba_and_depth: when nvdiffrast is absent,
  both route to new helpers _silicon_mesh_raster (builds the MVP equal to
  project_to_clip's clip output, near 0.1 / far 10.0, and calls the fork's
  mesh_raster_silicon.rasterize_mesh) + _interp_vertex_attr (perspective-
  correct barycentric interpolation, the dr.interpolate equivalent).
  mesh_raster_silicon returns image convention (row 0 = top), so the silicon
  branch does NOT apply nvdiffrast's flip(0). glctx argument ignored.
- Texture branch: caller texture is flipped back to glTF top-down rows and
  sampled with F.grid_sample (bilinear, align_corners=True) at interpolated
  UVs; background zeroed exactly like upstream.
Deviations vs upstream: no dr.antialias edge pass (hard silhouettes; these
are forward-only oracles), texture filtering is bilinear instead of
linear-mipmap-linear, faces crossing the near plane are dropped whole by
rasterize_mesh instead of clipped. Depth/alpha/rgb values inside the
silhouette are the same perspective-correct interpolation.
glctx construction sites guarded (glctx=None off CUDA): depth_grounding.py,
eval_assets_export.py (import + 2 construction sites), render_final_results.py.
core/final.py already guards its nvdiffrast block (returns all-visible on
failure; dynamic-path only, left as upstream designed it).
diffmc_mesh: NOT ported (deferred per inventory; only caller is the eval
export path, not the single-image demo path; fork FlexiCubes remains the
fallback if an export ever needs it).
VERIFIED (/tmp/genia_stage5_verify.py, PASS): slanted quad + occluding
triangle vs an independent per-pixel Moller-Trumbore raycast reference:
depth max diff 4.77e-07, vertex-color rgb 3.05e-07 on pixels covered by both;
coverage 1664 vs 1668 px (4 edge pixels differ by rasterisation rule).
Texture path: all 4 quadrant samples equal the exact hand-computed bilinear
blends for the assigned UVs.

## Stage 6: side models shakedown (Nov 23, 2026)
- jaxtyping installed into genia_env (uniception dep missed by the earlier
  --no-deps install; pip also warns uniception wants rerun-sdk, which this
  path does not import at runtime).
- MapAnything VERIFIED on MPS through GenIA's own run_map_anything
  (core/utils/map_anything_depth.py) on the demo sloth frame: 21.1 s,
  depth (518, 518) range [0.000, 2.986], K fx=751.0, near-identity c2w,
  valid-mask coverage 0.946. MapAnything's infer detects the non-CUDA
  device itself and uses minibatch size 1 for the dense head.
- LPIPS (VGG) VERIFIED: weights downloaded on first use, forward on MPS
  returns sane distances (0.3444 for a random pair at 64x64).
- MoGe: not called by the single-image demo config (processing=map_anything)
  - skipped per the stage plan.
- DINOv2: only referenced by core/download_weights.py (a prefetch utility)
  and the ActionMesh path; the fork's SAM 3D already carries what the mono
  pipeline needs. No fetch required for the demo run.

## Stage 7 (in progress): end-to-end demo bring-up fixes (Nov 23, 2026)
Demo bring-up surfaced a chain of MPS issues, each fixed at its own layer:
1. Fork DepthModel (MoGe) defaults to device "cuda": GenIA core/inference.py
   Inference.__init__ now also pins config.depth_model.device to the run
   device before Hydra instantiation (sibling of the stage-1 config.device
   pin). Fork untouched.
2. pytorch3d.ops missing from the fork shim: NEW file
   sam3d-objects-mac/pytorch3d/ops.py with knn_points (chunked cdist
   selection under no_grad; dists recomputed from gathered neighbours so
   gradients flow through selected pairs, matching pytorch3d backward
   semantics) and estimate_pointcloud_normals (local PCA, smallest
   eigenvector). Verified: indices identical to topk on cdist, dists to
   1.1e-06, grads flow, plane normals exact. Call sites: refinement.py
   chamfer, pose_refit.py, deformation.py.
3. MPS adaptive pooling divisibility (518 -> 129 at scale 0.25) in the
   multiscale RGB loss: new area_interpolate() helper in
   core/utils/device.py (CPU offload only when the MPS divisibility rule
   fails; autograd crosses devices; same math). Call sites patched:
   refinement.py _compute_multiscale_rgb_loss, rendering_guidance.py x2.
4. visibility.py _rasterize_scene uses pytorch3d MeshRasterizer fragments
   (zbuf + pix_to_face); the shim classes are import-only stubs. Added a
   NotImplementedError fallback routing to mesh_rendering._silicon_mesh_raster
   + _interp_vertex_attr (zbuf = perspective-correct camera z, 0 off-surface;
   the consumer maps zbuf <= 0 to +inf, so background semantics match).
5. Fork masked_sdpa dense path (SAM3D_MPS_USE_SDPA=1) materialises a
   block-diag mask and MPS SDPA materialises (H x sum_q x sum_kv) weights:
   GenIA guided stage-2 GS decode hit 16 x 19350 x 19350 = 5.99e9 elements
   (MPS 64-bit indexing error, then OOM at teardown). Guard added in the
   fork file (pre-GenIA state already backed up as .orig-backup-genia in
   stage 2): on MPS, when H * sum_q * sum_kv > 250M, use the fork's own
   masked_chunked_sdpa (identical per-block math; dense-vs-chunked A/B
   5.4e-07). This also protects the fork standalone.
6. diff_rasterizer: mid-ODE guidance decodes can yield splats with infinite
   projected radius (Jacobian blow-up near the camera plane) -> math.ceil
   overflow in binning. Binning radius clamped at 8192 px (binning-only
   value; the h <= 3 neighbourhood cap already approximates such splats).
   gsplat_silicon carries the same latent overflow but never sees guidance
   intermediates in its own call sites.
Also: diff_rasterizer early-break check now skips chunk 0 (T is all ones
there; the check could never fire but cost an MPS sync per tile).
Craftsman demo scene prepared at demo/data/image/craftsman/ (0000.png =
the standard input photo, 0000_mask.png from the fork SAM 3D High run's
extracted alpha, coverage 0.431, 0000_classes.json house instance 1).

## Stage 7 (continued): final-evaluation fixes (Nov 23-24, 2026)
7. Fork scene_visualizer.py wraps pytorch3d imports in try/except and nulls
   them when `pytorch3d.renderer.cameras` (a submodule the flat shim lacks)
   fails to import; GenIA final evaluation then died calling
   quaternion_to_matrix (None). Added a second-chance import in the fork
   file against the flat shim locations (real pytorch3d still takes the
   first branch). Backup: scene_visualizer.py.orig-backup-genia.
8. Shim structures.Pointclouds was an import-only stub; GenIA evaluation
   (make_scene -> SceneVisualizer.object_pointcloud) needs a working
   container. Implemented minimal Pointclouds in the shim structures.py
   (list/padded points/features/normals, num_points_per_cloud, device,
   .to()). Meshes deliberately stays a raising stub: GenIA visibility
   detects it via NotImplementedError and routes to the silicon rasterizer.
   Backup: structures.py.orig-backup-genia. Verified: object_pointcloud
   transform math exact, points_list/padded shapes correct.
Also observed: rendering_guidance.py carries its own hardcoded
torch.autocast("cuda", bf16) which self-disables with a warning off CUDA,
so guidance runs in fp32 here (precision-only difference, no failure).
Sloth run 20261123_181834 traversed ALL pipeline blocks and failed only in
run_final evaluation (fix 7); craftsman run then traversed all blocks and
failed at the next eval step (fix 8). Both fixes are in the eval path only.
Final-output inventory for the alignment comparison (from final.py /
evaluation.py source): final/{scene}_final_metrics.json (PSNR/SSIM/LPIPS,
full-frame + per-object masked, GenIA's own evaluator), final/renders_train/
000.png (input-view RGBA render), final/gaussians/world.compressed.ply
(posed world Gaussians), final/poses.json (per-object Sim(3) + cameras
K/c2w), preprocessing/colmap/images/train_000.png (processed 518x518 input).

## Stage 7 (complete): end-to-end runs + alignment verification (Nov 24, 2026)
Additional fixes during the craftsman/sloth end-to-end runs:
9. MPS GPU watchdog (kIOGPUCommandBufferCallbackErrorImpactingInteractivity)
   in run_final's mesh export: the SLAT mesh decoder chains long
   un-synchronized GPU work. Periodic cross-thread torch.mps.synchronize()
   (now also started process-wide in run.run() via device.py
   start_mps_sync_watchdog) did NOT prevent it. Resolution: the mesh
   format is decoded on CPU in redecode_slat (_decode_mesh_on_cpu) -
   identical math, slower (craftsman SLAT 43,778 tokens -> 776,998 verts /
   1,552,844 faces in 1,180.6 s). The decoder's extractor chain
   (SparseFeatures2Mesh, FlexiCubes) are plain classes holding
   construction-device state (device fields, reg_v/reg_c grids, sdf_bias)
   that nn.Module.cpu() does not move, so the helper recursively sweeps
   the model's object graph for `device` attributes and plain tensor
   attributes, restoring everything afterwards (model verified back on
   mps:0 after the call).
10. gsconverter installed into genia_env from the pinned git source in
    GenIA's requirements.txt (francescofugazzi/3dgsconverter @ 99bec76);
    needed by the compressed-PLY export.
11. visualization.render_voxel_meshes_in_camera: silicon fallback (same
    pattern as stage 5) for its pytorch3d MeshRasterizer + hard_rgb_blend
    tail - the shim has no renderer.blending submodule.
12. Shim look_at_view_transform gained pytorch3d's spherical
    (dist/elev/azim) form (it previously required `eye`); verified against
    pytorch3d's R/T convention. Backup renderer.py.orig-backup-genia.
    Note: render_multiview_comparison's mesh row still falls back to
    upstream's blank-render error path (load_objs_as_meshes is None in
    GenIA's guarded imports and the shim MeshRenderer is a stub) - the
    decoded_renders turntable PNG shows the Gaussian row only. Cosmetic;
    the mesh itself is exported as GLB by the mesh path above.

END-TO-END RESULTS:
- Craftsman (demo/data/image/craftsman, the standard test photo): full
  pipeline completed. Blocks ran in the 20261124_022133 run (wall ~4 h to
  the run_final mesh-export watchdog); run_final was then completed from
  that run's own final/final.pt state via the pipeline's resume cache
  (all 8 blocks reported cached/skipped) after fixes 9-12 - final log
  line "Pipeline complete!". Artifacts in
  results/mono_static/image/craftsman/20261124_022133/final/: metrics
  JSON, poses.json, gaussians/{001,world,background}.compressed.ply,
  meshes/{001,world}.glb, renders_train/000.png, renders_synth_nvs x4,
  keyframe PNGs, decoded_renders turntable, colmap export, timing,
  viz/orbit/orbit.mp4. Peak RSS ~10.7 GB.
- Sloth (bundled demo scene): the 20261123_181834 run traversed ALL
  blocks and failed only in run_final evaluation (fix 7, since fixed);
  a clean from-scratch re-run with all fixes in place was launched to
  confirm DEMO_EXIT=0 on the bundled example (see addendum below).

ALIGNMENT VERIFICATION (craftsman, GenIA's own evaluator - torchmetrics
PSNR/SSIM/LPIPS, input view at 518x518, white-composited per-object
masking, renders include background Gaussians from the input pointmap):
- GenIA final (craftsman_final_metrics.json): per-object masked
  PSNR 23.47, SSIM 0.831, LPIPS 0.083; full-frame 23.14 / 0.826 / 0.086.
  The finisher's re-evaluation reproduced these EXACTLY.
- GenIA starting point (state after pose_init: SAM 3D shape + guided
  appearance + decoded initial pose, before any pose refinement,
  appearance pass 2, or finetuning), scored by the same evaluate_block
  via an exit_after=pose_init run + resume driver:
  per-object masked PSNR 15.08, SSIM 0.628, LPIPS 0.225;
  full-frame 14.67 / 0.619 / 0.232.
- Delta from GenIA's refinement stages: +8.4 dB PSNR, +0.20 SSIM,
  LPIPS 0.225 -> 0.083.
- Plain SAM 3D standalone note: the fork CLI emits canonical-frame
  artifacts only (splat.ply); its raw layout pose lives in MoGe's
  pointmap-normalized frame (ScaleShiftInvariant decode, translation
  z=-2.39, scale 1.61) while GenIA's world is MapAnything-metric
  (z=+27.6, scale 14.3), so the standalone pose cannot be rendered in
  GenIA's camera without adopting a frame conversion. A direct
  transplant was attempted and produced an empty render (PSNR 6.6 -
  discarded as a frame error, not a SAM 3D quality signal). The
  pose_init baseline above is the like-for-like starting point GenIA
  itself refines from. Separately verified: a fresh fork-CLI-settings
  run (50 steps, seed 7) produces 961,440 Gaussians in 326 s.

## Stage 8: how to run (Nov 24, 2026)
Single image (bundled scenes sloth/lab_duo/lab_turtle/office_buddies/
stutty, plus the added craftsman scene):
    ~/alt3d/genia_run.sh image [--scene NAME] [hydra overrides...]
genia_run.sh sets SPARSE_BACKEND=native, SAM3D_CONV_IMPL=vec,
SAM3D_MPS_USE_SDPA=1, PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0,
PYTORCH_ENABLE_MPS_FALLBACK=1, GENIA_SAM3D_WEIGHTS=<fork checkpoints>,
and execs demo/run_demo.py under ~/alt3d/genia_env/bin/python.
Outputs: ~/alt3d/GenIA/results/mono_static/image/<scene>/<timestamp>/
with final/ as inventoried in the stage-7 note above.
Resume: a run that wrote output.save_cache=true caches per-block states
in results/.../<scene>/cache/; re-running with
processing.resume_from_cache=true skips cached blocks (this is how the
craftsman run_final was completed from final/final.pt).
Rough wall times on this M1 Ultra: sloth full run ~2.7 h, craftsman
~4.5 h (finetune dominates: ~50 s/it sloth, ~95 s/it craftsman at the
default 100 iterations; mesh export adds ~20 min on CPU for large
SLATs). Peak memory ~9-11 GB RSS.
Fork regression (Nov 24, 2026): sam3d-objects-mac imports cleanly under
s3d_env with the production env after all GenIA-related fork edits, and
run.sh passes its weights check and launches the CLI. Fork files changed
by the port (all with .orig-backup-genia backups where pre-existing):
the stage-2 SAM 3D patch set (7 files), masked_sdpa.py (MPS size guard),
scene_visualizer.py (import fallback), pytorch3d/transforms.py
(axis_angle_to_matrix), pytorch3d/structures.py (functional
Pointclouds), pytorch3d/renderer.py (look_at_view_transform spherical
form), pytorch3d/ops.py (NEW: knn_points, estimate_pointcloud_normals).

## Stage 7 addendum: bundled sloth demo, clean uninterrupted run (Nov 24, 2026)
The formal end-to-end confirmation for the bundled demo scene landed with a
single uninterrupted process: `~/alt3d/genia_run.sh image` (scene sloth),
run dir results/mono_static/image/sloth/20261124_120919/, log
/tmp/genia_demo_sloth.log ends with "Pipeline complete!" and DEMO_EXIT=0.
All 8 blocks ran in one process (no resume): shape_init 77.7 s,
appearance_init 90.9 s, pose_init 16.6 s, global_pose_refine_1 822.2 s,
appearance_init_2 783.6 s, finetune 4,468.0 s, global_pose_refine_2
1,808.4 s; block sum 8,067.4 s, total wall about 2.4 h including startup,
preprocessing, and run_final. This run's finetune loss fell from 0.2005
(it 13) to 0.0825 (it 100), LPIPS component 0.0359 at the end. Peak RSS 18,196.5 MB (15 s sampler over the genia
processes; the peak occurs in run_final, where the CPU mesh decode holds
large CPU tensors alongside the MPS-resident model). Peak torch MPS
allocation reported by the pipeline: 5,983.9 MB.
Final metrics (final/sloth_final_metrics.json, GenIA's own evaluator,
input view): per-object masked PSNR 26.4598, SSIM 0.8893, LPIPS 0.0518;
full-frame PSNR 20.7660, SSIM 0.7726, LPIPS 0.1407. Mesh export:
meshes/001.glb with 493,945 verts / 980,770 faces (SLAT 22,344 tokens,
decoded on CPU per the stage 7c watchdog workaround). Full final/
inventory present (30 files, same set as craftsman, plus timing.json and
sloth_timing.png from this uninterrupted run).
This closes the last open item of the port: both the bundled demo scene
and the craftsman photo now run end to end on MPS, and the SAM 3D fork
regression check passed (stage 7c).
