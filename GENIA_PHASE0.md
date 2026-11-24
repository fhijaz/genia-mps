# GenIA -> Apple Silicon (MPS) Port: Phase 0 Inventory

Date: Nov 23, 2026. Scope: inventory only. Nothing was installed beyond the clone; nothing was executed.
Method note: every claim below is marked **(verified)** when it comes from reading the code in this clone or the local fork, or **(inferred)** when it is an extrapolation. Effort figures are estimates.

## 1. Clone facts (verified)

- Repo: https://github.com/facebookresearch/GenIA, cloned to `~/alt3d/GenIA` on the Mac Studio with `--recurse-submodules`.
- Commit: `f6155f906fa1ed671cf70e7f4c91b946fae97e62`, branch `release`. Repo created 2026-10-08, 2 commits, 0 releases, 136 stars at read time.
- Size on disk: 694 MB (includes demo data and submodules).
- License: CC BY-NC 4.0 (non-commercial), plus third-party licenses, several non-commercial.
- Submodules (`.gitmodules` + `git submodule status --recursive`):
  - `submodules/sam-3d-objects` @ f91db411 - facebookresearch/sam-3d-objects. The SAM 3D backbone. GenIA imports it as the `sam3d_objects` package and patches it (see section 3).
  - `submodules/map-anything` @ f7ebafb4 (tag v1.1.1) - facebookresearch/map-anything. Default preprocessing backend: predicts depth + cameras.
  - `submodules/actionmesh` @ be4c2905 - facebookresearch/ActionMesh. Dynamic (video) setting only: per-frame deforming shapes.
    - nested: `submodules/actionmesh/third_party/TripoSG` @ fc5c4099 - VAST-AI-Research/TripoSG, ActionMesh's shape backbone (put on sys.path, not installed).
- Demo data ships in-repo: `demo/data/image/` (sloth, lab_duo, lab_turtle, office_buddies, stutty), `demo/data/multiview/` (pablo, stuffed_toy, lab_duck), `demo/data/dynamic/` (camel, dinosaur).
- Stack pins (install.sh / requirements.txt): torch 2.7.1+cu128, python 3.11 (conda), open3d==0.18.0. NOTE: the local SAM port found open3d 0.18.0 segfaults next to torch on this Mac and uses 0.19.0; a GenIA env must resolve that conflict.

## 2. CUDA touchpoint inventory

Classification: **(a)** hard blocker = compiled CUDA extension in the compute path; **(b)** device-string patch = mechanical; **(c)** already has a CPU/MPS fallback or is pure PyTorch.

### 2.1 GenIA core (`core/` + `demo/`, *.py only, grep-verified counts)

| Touchpoint | Hits | Files | Class | Notes |
|---|---|---|---|---|
| `.cuda(` | 37 | 6 | (b) | utils/interpolation.py, utils/rendering.py, utils/eval_assets_export.py, utils/refinement.py, utils/pipeline_state.py, pose_init.py |
| `torch.cuda` | 45 | 16 | (b) | Mostly empty_cache / memory stats / synchronize. `torch.cuda.memory_allocated()` (e.g. backbone `_slim`) raises on a CUDA-less run and needs a device abstraction, not just a rename. |
| Device selection | 1 site | core/run.py:137 | (b) | `torch.device("cuda" if torch.cuda.is_available() else "cpu")` - on the Mac this silently picks CPU, never MPS. One-line fix, global effect. |
| `torch.autocast(device_type="cuda")` | 3+ sites | core/inference.py (lines ~469, ~900, ~1331) | (b) | Must become `"mps"` (or follow the run device). |
| `gsplat` | 33 | 13 | (a) | Single direct import verified: `from gsplat.rendering import rasterization` at core/utils/rendering.py:20, wrapped by `render_gaussian_params`. Other files mention gsplat in comments or call GenIA's own render helpers. See section 4. |
| `nvdiffrast` | 22 | 7 | (a) | Lazy imports inside functions in core/final.py, core/utils/depth_grounding.py, core/utils/eval_assets_export.py, core/utils/render_final_results.py, and the choke point core/utils/mesh_rendering.py (`render_depth_and_alpha`, `render_rgba_and_depth`). See section 4. |
| `diso` (DiffMC) | 1-2 | core/utils/mesh_rendering.py | (a) | `diffmc_mesh`: occupancy grid -> iso-surface mesh. Used by core/utils/slat_decode.py and eval export. Export-path, deferrable (section 4). |
| `fused_ssim` | 7 | core/utils/refinement.py | (c) | Guarded import (line ~43) with automatic fallback to `pytorch_msssim`, which is already in requirements.txt. |
| `xformers` | 5 | core/visibility_attn.py | (a) | `FootprintBias.attend` calls `xops.memory_efficient_attention` with a `BlockDiagonalGappyKeysMask`. Plus the SAM 3D patch's xformers path (section 3). See mismatch #1. |
| `triton` | 3 | 1 | ? | Not isolated to a call site in this pass; flagged for Phase 1 confirmation. Suspected comment or optional path. |
| `pytorch3d` | 76 | 26 | mixed | Breakdown by submodule string: transforms ~18 (+5 named functions: quaternion_to_matrix, quaternion_multiply, quaternion_invert, matrix_to_quaternion, matrix_to_euler_angles), renderer 7 (+blending 3), structures 5, ops 4 (knn_points 3), io 1, bare mentions 15, `pytorch3d_convention` comment strings 14. Transforms are pure Python math (class c, portable). `ops.knn_points` needs a pure-torch KNN (easy). Renderer/structures usage concentrates in eval/final/viz paths and in GenIA's own `PyTorch3DMeshRenderer` (core/utils/mesh_rendering.py, used from core/utils/rendering.py). Real pytorch3d is a compiled CUDA extension (class a as shipped). |
| `flash_attn` | 0 | 0 | - | Not used in GenIA core. |
| `spconv` | 0 | 0 | - | Not used in GenIA core (only inside SAM 3D). |
| `kaolin` | 0 | 0 | - | Not used in GenIA core. |

### 2.2 install.sh compiled dependencies (all class (a) as shipped)

diso==0.1.4, pytorch3d v0.7.9 (git build), gsplat v1.5.3, nvdiffrast v0.4.0, fused-ssim (git), xformers==0.0.31, spconv-cu126==2.3.8 + cumm-cu126==0.7.11 (for SAM 3D), torch==2.7.1+cu128. Pure-Python/git deps that are MPS-safe in principle: MoGe (microsoft/MoGe @0744441), utils3d (@3fab839), map-anything + uniception==0.1.6, ActionMesh.

### 2.3 Vendored SAM 3D submodule (reference only - the port strategy reuses the local fork instead)

grep hits in `sam3d_objects/`: spconv 31, pytorch3d 40, gsplat 18, nvdiffrast 6, kaolin 1. The local fork already substitutes every one of these (section 3), which is the whole reason the fork-reuse strategy exists.

### 2.4 map-anything submodule

**0 hits** for `.cuda(`, `torch.cuda`, `flash_attn`, `xformers`, `triton` in the `map_anything/` package (verified). Pure PyTorch ViT stack. Class (c), pending a runtime shakedown.

## 3. SAM 3D API map

### 3.0 The strategy seam (verified)

GenIA does NOT require its vendored copy. `core/paths.py` documents it verbatim: SAM3D_OBJECTS_ROOT is "APPEND[ed] to sys.path, never insert(0): an already-importable `sam3d_objects` must keep winning." `core/run.py` and `core/inference.py` repeat the same comment. GenIA is designed to run against whichever `sam3d_objects` package is importable first. The local MPS fork provides the same package name, so the port strategy is: make the fork importable, let GenIA bind to it, and port GenIA's SAM 3D patch (below) onto the fork.

Fork vs vendored tree diff (verified, `diff -rq`, excluding pycache): of 106 vendored .py files, **zero are missing in the fork**; 26 files differ; the fork only ADDS `_kaolin_stub.py`, `experimental_retopo/`, `modules/sparse/conv/conv_native.py` (+ our .orig backup), `modules/sparse/conv/native_backend.py`, `renderers/gsplat_silicon.py`, `renderers/mesh_raster_silicon.py`. The fork's git history does not contain the vendored commit f91db41 (separate repo lineage), so version comparison is by file content, not merge-base.

### 3.1 What GenIA's patch does (submodules/sam-3d-objects.patch, 536 lines, 16 files, read in full)

Hooks GenIA cannot run without:
1. **Solver velocity exposure**: `ODESolver.solve_iter` yields `(x_t, t1, velocity)` instead of `(x_t, t0)`; every solver `step()` returns `(x_tp1, velocity)`.
2. **`noise_init_override`** kwarg in `FlowMatching.sample` and `ShortCut.sample`: replaces the initial noise x_0.
3. **`step_callback`** kwarg in `ShortCut.sample`: `callable(t, x_t, velocity)` after each ODE step.
4. **`ShortCut._velocity_transform`**: optional `callback(t_raw, velocity_dict, x_t) -> velocity_dict` applied inside `_generate_dynamics`. This is the denoising-guidance hook (rendering guidance rewrites the velocity).
5. **Visibility attention bias plumbing**: `SparseMultiHeadAttention.forward` gains an `attn_bias` kwarg, forwarded to `sparse_scaled_dot_product_attention`, which gains the same kwarg. A bias object with an `.attend(q, k, v, q_seqlen, kv_seqlen)` method (GenIA's `FootprintBias`) runs attention itself; a tensor bias goes through a new `xops_blockdiag_with_bias` in masked_sdpa.py (per-view xformers MEA loop). The patch also flips ATTN/BACKEND defaults from "sdpa" to "xformers", and the patched code **raises NotImplementedError if a tensor attn_bias is passed under a non-xformers backend**.
6. Housekeeping: spconv dtype guard under autocast; `Gaussian.get_features` concat dim 2 -> 1 (bugfix); FlexiCubes kaolin `check_tensor` asserts removed; `inference_pipeline.set_attention_backend()` (auto flash_attn env) removed; utils3d >= 1.x name shims (pointmap, layout_post_optimization_utils, image_mesh); broken `sam3d_objects.init` import removed from `__init__.py`.

### 3.2 Mapping table (GenIA need -> local fork status)

| GenIA uses | Fork status |
|---|---|
| `sam3d_objects` package (paths.py APPEND seam) | Same name/layout. **Verified.** |
| `sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap` | Exists, same name. File differs from vendored (fork's MPS edits). **Verified present.** |
| Pipeline attributes GenIA's `Inference`/`backbone` reach into: `models["ss_generator"]`, `models["slat_generator"]`, `condition_embedders`, `get_condition_input`, `compute_pointmap`, `preprocess_image`, `ss_preprocessor`, `slat_preprocessor`, `pose_decoder`, `ss_cfg_strength(_pm)`, `slat_cfg_strength`, `slat_mean`, `slat_std`, `dtype`, `shape_model_dtype`, `depth_model` | Grep parity: every probed symbol is present in the fork's pipeline files or absent from BOTH trees' pointmap file (the missing-in-file ones live in the parent `InferencePipeline` in both trees). **Grep-verified parity; runtime verification is Phase 1 work.** |
| `model.backbone.tdfy_dit.modules.sparse as sp` (SparseTensor isinstance/feats/coords in pipeline_state.py; used in 8+ GenIA files) | Same module; `modules/sparse/basic.py` differs (fork's MPS SparseTensor, class present). **Same name, fork behavior - which is what the port wants.** |
| `models.structured_latent_flow.SLatFlowModel` (core/inference.py) | File **identical** in fork; class present. **Verified.** |
| `representations.gaussian.Gaussian` / `gaussian_model` (gaussian.py, rendering.py, refinement.py, evaluation.py, pipeline_state.py) | Exists; file differs. **One real mismatch**: fork line 96 still has upstream `torch.cat((self._features_dc, self._features_rest), dim=2)`; GenIA's patch fixes it to `dim=1`. One-line port, only bites when `_features_rest is not None`. |
| `modules.norm.ChannelLayerNorm32, LayerNorm32` (shape_inversion.py) | File identical in fork. **Verified.** |
| `data.dataset.tdfy.pose_target.InstancePose, PoseTarget, ScaleShiftInvariant` (refinement.py, pose_token_gt.py) | File differs in fork; all three classes present (lines 16, 92, 286). **Same names; internal diffs not audited.** |
| `utils.visualization.SceneVisualizer` (slat_decode.py) | Exists in fork (file differs). **Same name.** |
| Patch hunk: solver.py velocity yield | solver.py **identical** in fork -> hunk applies cleanly. |
| Patch hunks: flow_matching/model.py, shortcut/model.py (noise_init_override, step_callback, _velocity_transform) | Both files **differ** in fork -> hand-port required. Hunks are small and well-delimited. Must also check for other `solve_iter` consumers in the fork affected by the yield signature change. |
| Patch hunks: sparse/attention/modules.py, sparse/attention/full_attn.py (attn_bias plumbing) | Both files **identical** in fork -> apply cleanly. BUT the runtime branch they feed is xformers-only (see mismatch #1); the fork keeps ATTN="sdpa", so an MPS attention-bias branch must be added. |
| Patch hunk: masked_sdpa.py (`xops_blockdiag_with_bias`) | File differs in fork; the added function is xformers-based and cannot run on MPS as written. Superseded by the MPS attend() plan (section 7). |
| Patch hunks: backend flips to xformers (modules/attention/__init__.py identical file; modules/sparse/__init__.py differs) | **Do not port.** The fork's ATTN="sdpa" is the MPS-correct setting. |
| Patch hunk: spconv dtype guard (conv_spconv.py) | File identical, but the fork routes sparse conv through its native backend (conv/__init__.py differs; conv_native.py added). **Moot.** |
| Patch hunk: flexicubes kaolin removal | flexicubes.py identical; fork already stubs kaolin. **Moot.** |
| Patch hunks: utils3d shims (pointmap.py identical; layout_post_optimization_utils.py, image_mesh.py differ) | GenIA core never imports utils3d directly (0 hits); these matter only inside SAM 3D internals, which already run on the fork. **Moot for the port.** |
| Patch hunk: inference_pipeline.py set_attention_backend removal; __init__.py LIDRA fix | Both files differ in fork; fork already handles backend selection and init its own way. **Moot / verify at runtime.** |
| GenIA `core/inference.py` `_install_vis_bias_hooks` (line ~72) | Installs `SparseVisibilityBiasHook` (core/visibility_attn.py) into the SLAT generator's attention modules; depends on the attn_bias plumbing above. |
| GenIA `core/backbone.py` `_slim` | Reaches into `pipeline.condition_embedders[*].module_list[*].backbone` (DINOv2 sharing), halves Linear/Conv weights, wraps `pipeline.depth_model`. Attribute structure grep-verified present in fork; its `torch.cuda.*` calls are class (b). |

## 4. Renderer call sites and substitution candidates

| Call site | Library | Purpose | Differentiable? | Frequency | MPS substitute |
|---|---|---|---|---|---|
| core/utils/rendering.py `render_gaussian_params` (sole direct gsplat import) | gsplat `rasterization`, render_mode "RGB+ED" | THE Gaussian renderer: returns (rgb, alpha, expected-depth). Feeds rendering guidance, refinement, finetuning, pose refine, eval renders. | **Yes** - losses backprop through it (core/finetuning.py: "Gaussian attrs -> apply_pose -> gsplat render -> loss -> backward"; rendering_guidance.py rescales the render gradient into the SLAT velocity). | Per ODE step during guided appearance steps; per iteration in finetune/refine. At recon resolution (~504-518 px crop) by default. | **None existing.** Fork's `gsplat_silicon.rasterization` is explicitly "forward-only (no autograd)" (its docstring) and returns colors+alphas, no expected depth. A new differentiable pure-PyTorch EWA rasterizer is the critical-path build (section 7). gsplat_silicon can cover forward-only eval renders meanwhile. |
| core/utils/mesh_rendering.py `render_depth_and_alpha`, `render_rgba_and_depth` | nvdiffrast | Mesh depth/alpha buffers: anchor visibility oracle (core/final.py), depth-grounding silhouettes, eval export. | No - used as a depth oracle / forward buffer. | Per frame, at FINAL/export and grounding. | Fork's `renderers/mesh_raster_silicon.rasterize_mesh` (pure-PyTorch tile z-buffer, MPS) - good fit. final.py's use is additionally guarded (returns all-visible on any failure, and its inputs are dynamic-scene data), so the static path degrades gracefully even before substitution. |
| core/utils/mesh_rendering.py `diffmc_mesh` + `PyTorch3DMeshRenderer` | diso (DiffMC) + nvdiffrast / pytorch3d | Occupancy -> iso-surface mesh; differentiable mesh rendering. Used from slat_decode.py (mesh decode) and eval_assets_export.py. | Render: yes in refinement contexts. | Mesh decode / export path. | Partial: mesh decode on MPS is already proven in the fork via its FlexiCubes path (the craftsman runs produced mesh.glb). Differentiable MPS mesh rendering has no ready substitute: the Hunyuan `torch_rasterizer.py` detaches and moves geometry to CPU (forward-only, verified lines ~224-225). Deferrable for Phase 1 (export-path concern). |
| SAM 3D-internal rendering (`sam3d_objects/.../renderers/gaussian_render.py`) | gsplat upstream | SAM 3D's own layout/viz rendering. | n/a | Inside backbone calls. | Already substituted by the fork (gsplat_silicon drop-in). No work. |

Speed warning (verified from the SAM port record): the forward-only gsplat_silicon already strained at high view counts during texture baking (macOS GPU watchdog at 100 views @ 1024; ran at 40 views @ 768). A differentiable version in an inner loop will be slower per render; render resolution and iteration counts are the pressure valves, and GenIA's defaults already render at the recon crop, not full frame.

## 5. Side models

- **MapAnything** (submodule, v1.1.1): the DEFAULT preprocessing backend (depth + cameras) - on the single-image critical path. Zero CUDA touchpoints in its package (verified). Standard ViT stack; runtime deps include uniception==0.1.6 (installed --no-deps by install.sh). Weights: HF `facebook/map-anything` (section 6). MPS risk: low-moderate (inferred - standard transformer; watch for the usual fp64/positional quirks during shakedown).
- **MoGe** (HF `Ruicheng/moge-vitl`; code via pip git microsoft/MoGe @0744441): two roles, both off the default path. (a) Inside SAM 3D as its DepthModel - GenIA wraps it in `_OnDemandGPU` (core/backbone.py) and notes GenIA passes its own pointmap, "so normally never" calls it. (b) Optional `processing=moge` backend. Not vendored here, so its stack was not audited; it is a standard PyTorch ViT (inferred). Low priority.
- **ActionMesh** (submodule + nested TripoSG): imported ONLY by core/actionmesh_video.py (grep-verified), used ONLY for the dynamic setting's shape injection. **Confirmed scoped out** for single-image and multiview. Its ~17 GB of weights are likewise unnecessary for static runs.
- **DINOv2** (torch hub `dinov2_vitl14_reg`) and **LPIPS VGG**: small support downloads; DINOv2 also lives inside SAM 3D's encoders (already local). LPIPS feeds the perceptual loss in guidance/finetuning - pure PyTorch (inferred standard).

## 6. Weights (core/download_weights.py, read in full)

| Item | Source | Needed for single-image? | Status |
|---|---|---|---|
| SAM 3D Objects `checkpoints/*` (pipeline.yaml + ss/slat generators, encoders, decoders) | HF `facebook/sam-3d-objects` (gated; user already accepted the license) | Yes | **Already on disk**: `~/alt3d/sam3d-objects-mac/checkpoints/hf/checkpoints/` contains pipeline.yaml and all 17 ckpt/yaml files (verified by listing). Point `GENIA_SAM3D_WEIGHTS` at that folder; no re-download. GenIA has no weights/adapters of its own - its LoRA adapters are created per-run during FINETUNE. |
| MapAnything | HF `facebook/map-anything` (*.json/*.safetensors/*.pt) | Yes (default preprocessing) | New download. Size not stated in code (unverified; large-ViT class). |
| MoGe | HF `Ruicheng/moge-vitl` | Only for the optional moge backend | New download, deferrable. |
| DINOv2 vitl14_reg | torch hub (facebookresearch/dinov2) | Yes (support) | Small new download. |
| LPIPS VGG backbone | torchvision via lpips package | Yes (perceptual loss) | Small new download. |
| ActionMesh set: VAST-AI/TripoSG, facebook/dinov2-large, briaai/RMBG-1.4, facebook/ActionMesh | HF | **No** - dynamic only | Skip. Docstring sizes the set at ~17 GB total. |

## 7. Phase 1 GO/NO-GO: single-image setting

**Verdict: GO.** The architecture cooperates: GenIA was designed to bind to an external `sam3d_objects`, the fork it will bind to is upstream + MPS substitutions with zero missing files, the SAM 3D weights are already local, and every hard blocker is confined to a small number of choke points (one gsplat import, one mesh-render helper module, one attention-bias mechanism, one patch). The demo's single-image default path is: preprocessing (map-anything) -> shape_init -> appearance_init=perframe -> pose_init -> global_pose_refine_1 (render-based, low_rgb) -> appearance_init_2 (canonical) -> finetune (canonical TTR) -> global_pose_refine_2 (photometric) -> FINAL. Every stage after preprocessing touches at least one of the blockers below, so there is no free partial run - but the blockers are independent and can be built in parallel order.

### Three hardest mismatches (ranked)

1. **Differentiable Gaussian rasterizer (build-from-scratch).** GenIA needs autograd through splatting, including expected depth (RGB+ED), inside two inner loops (guidance per ODE step; finetune per iteration). All three local rasterizers are forward-only (gsplat_silicon by its own docstring; mesh_raster_silicon by construction; Hunyuan's by detach-to-CPU). This is the critical path and the schedule risk.
2. **Visibility attention bias without xformers.** GenIA's signature mechanism reaches attention two ways, both xformers: tensor bias -> the patch's `xops_blockdiag_with_bias`; `FootprintBias.attend` -> `xops.memory_efficient_attention` with `BlockDiagonalGappyKeysMask` (core/visibility_attn.py, ~1145 lines). The fork's ATTN="sdpa" makes the patched code raise NotImplementedError for tensor bias. The MPS route: reimplement `attend()` as per-view key selection + `F.scaled_dot_product_attention` (the gappy mask is an index-selection structure, so this is mathematically direct), and add an SDPA branch for tensor bias (per-view loop with additive bias). Numeric A/B against a dense-mask CPU reference is mandatory - this is the SAM-conv-style verification job.
3. **Patch hand-port onto diverged fork files.** 9 of the patch's 16 files differ in the fork (incl. flow_matching/model.py, shortcut/model.py, gaussian_model.py, masked_sdpa.py, sparse/__init__.py). The hunks are small but semantic: the solver yield signature change ripples to any other solve_iter consumer; `_velocity_transform` interacts with the fork's SDPA attention default; the Gaussian get_features dim fix changes a feature layout the fork's own export path may rely on. Each hunk needs an A/B against the fork's standalone craftsman behavior before and after.

### Per-component effort estimates (estimates, agent working time)

| Component | Estimate |
|---|---|
| Env assembly + device plumbing (MPS device selection, .cuda()/torch.cuda sweep in GenIA core, autocast device_type, open3d pin conflict, map-anything install + MPS shakedown) | 0.5-1 day |
| SAM 3D patch hand-port to the fork (7 clean-apply files trivial; 9 diverged files by hand; solve_iter consumer audit; craftsman A/B) | 0.5-1.5 days |
| FootprintBias.attend MPS reimplementation + SDPA tensor-bias branch, with dense-reference numeric A/B | 1-2 days |
| Differentiable MPS Gaussian rasterizer (RGB+ED, autograd EWA splatting, gradcheck, speed work) | 2-4 days |
| pytorch3d shim extension (knn_points via cdist, transforms completeness, structures/io stubs for eval paths) | 0.5-1 day |
| mesh_raster_silicon wiring into mesh_rendering depth/alpha helpers | 0.25-0.5 day |
| End-to-end single-image demo run + verification (render from input view vs photo, compared against plain fork SAM 3D output - the GenIA alignment claim, measured) | 0.5-1 day |
| **Total** | **~5-10 days** |

### Ranked risks

1. Differentiable rasterizer speed on MPS inside guidance/finetune loops (could make runs hours-long; mitigations: recon-crop render res is already the default, iteration caps are config keys).
2. Visibility-attention numeric drift vs the xformers reference semantics (gappy-mask edge cases; mitigated by dense CPU reference A/B on small cases).
3. Patch/fork semantic drift silently changing SAM 3D standalone behavior (mitigated by before/after craftsman A/B per hunk group).
4. Memory on the 64 GB Mac: GenIA caches per-step states and renders; GenIA's own `_slim` cuts the backbone from 13.0 to 5.9 GB on CUDA and should transfer; MPS watermark discipline from the Wan work applies.
5. Research-code brittleness: 2 commits, no releases, a large Hydra config surface, and at least one known-unisolated triton reference. Expect small surprises; none identified that change the verdict.

### Substitution plan Phase 1 would execute

1. Bind GenIA to the fork's `sam3d_objects` via the designed sys.path APPEND seam; `GENIA_SAM3D_WEIGHTS` -> the existing checkpoints folder.
2. Hand-port the patch: port solver velocity yield, noise_init_override, step_callback, _velocity_transform, attn_bias plumbing, Gaussian dim fix (each with A/B); skip the xformers backend flips, the spconv guard, and the utils3d shims (moot on the fork).
3. New MPS module: SDPA implementation of `FootprintBias.attend` + tensor-bias block-diagonal SDPA branch, selected on MPS.
4. New MPS module: differentiable pure-PyTorch Gaussian rasterizer exposing the subset of gsplat's `rasterization` signature that `render_gaussian_params` uses (RGB+ED first, SH degree 0 path first), swapped in at rendering.py's single import.
5. Wire the fork's mesh_raster_silicon into mesh_rendering's depth/alpha helpers for MPS.
6. Device sweep in GenIA core: central device = MPS; `.cuda()` -> `.to(device)` in the 6 files; guard `torch.cuda.*`; autocast device_type follows the run device.
7. Extend the local pytorch3d shim (knn_points, transforms, minimal structures/io) to cover GenIA's imports on the single-image path.
8. Explicitly out of scope for Phase 1: ActionMesh/dynamic, DiffMC differentiable mesh rendering, fused_ssim (fallback exists), MoGe backend.
