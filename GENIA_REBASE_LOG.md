# GenIA rebase log: MPS port f6155f9 -> upstream release head

Started Nov 25, 2026. User approval: "Yes, do the rebase and merge properly"
(chat, Nov 25, 2026). Map: GENIA_REBASE_SCOPING.md (Nov 24, pinned to
01ea014). This log is appended disk-first as work proceeds.

## Phase A - groundwork

- Upstream release head at fetch time: 60979b8eb420f64401303f5ebcb7679001d276d0
  ("November 25, 2026 second update: the demo 3D view on 3DView 2.0, a clear
  error for a shadowed SAM 3D Objects"). Head moved past the scoping pin
  01ea014 by one commit (60979b8). Commits f6155f9..60979b8: e50ccd0,
  01ea014, 60979b8 (3 commits).
- WATCH ITEM: 60979b8 title mentions a new error for a shadowed SAM 3D
  Objects - our install binds GenIA to the fork via an already-importable
  sam3d_objects package (core/paths.py), i.e. exactly a "shadowed" layout.
  Must inspect that commit before merging.

### Phase A findings (Nov 25, 2026)

- Refreshed upstream diff f6155f9..60979b8: 64 files, +4283/-317. The bulk
  over the scoping count (39 files at 01ea014) is demo-app surface
  (demo/ui/*, demo/space/*, demo/app.py, demo/record_examples.py,
  .gitmodules, new submodule gitlinks). Core files changed: 34 (2 added:
  core/utils/progress.py, core/utils/live_poses.py).
- SUBMODULE PIN (the scoping unknown, now read from the fetched tree):
  sam-3d-objects pin at 60979b8 is f91db411c50efee93d8db7aeb323885650f6f722 -
  IDENTICAL to the pin at f6155f9. map-anything (f7ebafb) and actionmesh
  (be4c290) pins and the sam-3d-objects.patch blob are also identical.
  New submodules sam3 (875ed6f) and 3DView (16376be) are demo-app-only and
  are skipped. (The scoping inferred the pin had moved from the CHANGELOG
  mechanisms; the gitlink says it did not.)
- sam3d_objects IMPORT SURFACE: `git grep` of `from/import sam3d_objects`
  over core+demo is byte-identical between f6155f9 and 60979b8. No new call
  sites, no changed attribute references. GenIA at head is designed against
  the same sam-3d-objects commit the port was built on.
- FORK COMPAT LIST (small; the fork itself needs NO changes):
  1. core/run.py check_backbone() (new in 60979b8): the Hydra CLI entry
     (reached via demo/run_demo.py -> python -m genia.core) sys.exits when
     the importable sam3d_objects is not the checkout submodule. Our fork
     binding is exactly that case, deliberately (core/paths.py appends the
     submodule; an already-importable copy wins). Adapt during the run.py
     hand-merge: accept the fork root (sibling sam3d-objects-mac, or
     GENIA_SAM3D_FORK_ROOT override) and verify patched-ness functionally -
     the imported masked_sdpa module must provide masked_chunked_sdpa,
     which only exists via the GenIA patch lineage.
  2. LIDRA_SKIP_INIT: upstream at 60979b8 sets it nowhere (their Oct 28
     patch drops the gated startup import in the submodule copy). The fork
     still gates on it (sam3d_objects/__init__.py line 9). Keep the port
     existing set in core/run.py (process-wide, set at import before any
     pipeline module loads); accept upstream removals in datavis.py,
     inference.py, main.py, render_final_results.py.
  No other compat items. Gap is small -> proceeding to Phase B without a
  stop, per the task terms.
- OVERLAP: port touches 28 paths (f6155f9..mps-port). 11 carried whole
  (upstream never touched): core/backbone.py, core/pose_init.py,
  core/utils/device.py, core/utils/diff_rasterizer.py,
  core/utils/evaluation.py, core/utils/gaussian.py,
  core/utils/model_cache.py, core/utils/orbit_viz.py, core/utils/rendering.py,
  core/utils/timing.py, core/visibility_attn.py.
  17 files changed by both sides -> hand-merge: core/finetuning.py,
  core/inference.py, core/main.py, core/rendering_guidance.py, core/run.py,
  core/shape_inversion.py, core/utils/depth.py, core/utils/depth_grounding.py,
  core/utils/eval_assets_export.py, core/utils/interpolation.py,
  core/utils/mesh_rendering.py, core/utils/pipeline_state.py,
  core/utils/refinement.py, core/utils/render_final_results.py,
  core/utils/slat_decode.py, core/utils/visualization.py, core/visibility.py.
  All other upstream changes taken as-is.

## Phase B - merge (working copy ~/alt3d/GenIA-rebase, branch rebase-60979b8)

Method: cp -a of the live tree, then a true `git merge origin/release`
(60979b8) into mps-port (395ef25). Merge commit 85364cb (local only).

- CARRIED WHOLE (11, verified 0 diff vs mps-port): backbone.py, pose_init.py,
  utils/device.py, utils/diff_rasterizer.py, utils/evaluation.py,
  utils/gaussian.py, utils/model_cache.py, utils/orbit_viz.py,
  utils/rendering.py, utils/timing.py, visibility_attn.py.
- UPSTREAM-ONLY files taken as-is (verified 0 diff vs 60979b8): configs,
  datavis, download_weights, final, pipeline, preprocessing,
  sam3d_shape_cache, utils/config, utils/gt_data, utils/live_poses (new),
  utils/map_anything_depth, utils/pose_refit, utils/progress (new),
  utils/recon_cache, utils/sequence, demo/run_demo.py, demo app files,
  docs/install/requirements.
- AUTO-MERGED, semantically verified by markers + three-way churn
  accounting (14): finetuning.py (metrics flag + upstream floor occlusion
  both present), inference.py (solver hooks intact), shape_inversion.py,
  utils/depth.py, utils/depth_grounding.py, utils/eval_assets_export.py
  (wireframe function deleted upstream; port guard hunks kept),
  utils/interpolation.py, utils/mesh_rendering.py (diffmc_mesh + diso
  deleted upstream; silicon raster fallbacks intact),
  utils/pipeline_state.py, utils/render_final_results.py,
  utils/slat_decode.py (+0/-42 vs port: only the decode_shape_to_occ
  deletion; +190/-3 vs upstream: MPS decode integration preserved),
  utils/visualization.py, visibility.py (silicon routing + floor binning),
  rendering_guidance.py (_prepare_frame_targets: upstream nearest-exact
  centre-aligned resize/K math AND the port area_interpolate RGB path
  both present and correctly combined - the scoping predicted conflict
  auto-merged; verified by reading the merged function).
- CONFLICTS (3, all import/env blocks):
  * core/main.py - both sides added an import at the same anchor; kept
    both (device.max_memory_allocated + utils.progress).
  * core/run.py - port env block kept verbatim (LIDRA_SKIP_INIT set +
    MPS env setdefaults); upstream LIDRA removal declined here because
    the fork still gates on the variable (fork __init__.py:9). Upstream
    removals in datavis/inference/main/render_final_results accepted
    (run.py sets it process-wide before pipeline imports).
  * core/utils/refinement.py - kept port device import alongside
    upstream live_poses/progress imports and progress.tqdm.
- ADAPTATION (run.py check_backbone, the Phase A compat item): fork root
  (SAM3D_OBJECTS_ROOT.parents[2]/sam3d-objects-mac, overridable via
  GENIA_SAM3D_FORK_ROOT) accepted before the submodule check; patched-ness
  probed functionally (imported masked_sdpa must provide
  masked_chunked_sdpa - present in the fork at line 71). Upstream
  submodule path + git patch check unchanged for non-fork bindings.
- No dangling references to decode_shape_to_occ / diffmc_mesh /
  export_coarse_shape_wireframe in merged code (only in untracked
  .orig-backup files); no diso imports remain.
- Compile sweep: genia_env python -m compileall over core + run_demo.py:
  clean (exit 0), so the merged tree is valid Python 3.11.

## Phase C - verification (working copy)

Cheap gates (Nov 25, 2026), all PASS:
- Imports + config: GENIA_ROOT resolves to GenIA-rebase via the
  genia_pkg_rebase shim; 15 merged/new modules import cleanly
  (main, inference, finetuning, rendering_guidance, visibility,
  shape_inversion, refinement, slat_decode, mesh_rendering,
  eval_assets_export, pipeline_state, depth, sequence, progress,
  live_poses); hydra config composes for the craftsman image setting.
- check_backbone: fork accepted silently in the production binding.
  Negative test (GENIA_SAM3D_FORK_ROOT=/nonexistent/bogus): the guard
  fires with upstream exit message - the check still protects against
  a genuinely wrong binding. One probe bug found and fixed during the
  gates (follow-up commit d9f61f3): the fork attention package
  __init__ re-exports the masked_sdpa FUNCTION, shadowing the
  submodule, so the first probe form (from-import + hasattr) bound
  the function and false-fired; the probe now imports
  masked_chunked_sdpa itself from the module path.
- Fork regression (fork untouched): test_default_device.py PASS
  (subm 9.537e-07, strided coords equal 3.338e-06, inverse 4.768e-07).
- Finetune segment (seed 12345 harness adapted to the rebase tree,
  appearance_init_2 cache resume, SAM3D_ATTN_IMPL unset -> launcher
  flash32): VERIFY_SPY varlen=120, varlen_fp32=120, varlen_fp16=0,
  masked_sdpa=0; pace 24.4-24.6 s/it (flash32 class); wall 138.4 s.
  Losses it1 0.4226 / it2 0.4203: same class as every recorded arm
  (0.4230-0.4265) but identical to none. Expected, not a failure:
  upstream changed this segment code path - the Nov 17 finetune
  occlusion test (floor vs round) sits inside the loss, and vmask
  reads 0.80 here vs 0.81 in the fp32 validation run. Per the task
  terms the segment is a smoke + dtype proof only; the verdict
  rests on the full runs.

- Full runs (from scratch, launcher defaults via genia_run_rebase.sh:
  flash32 + MPS mesh decode, check_backbone exercised in the real
  python -m genia.core subprocess): sloth launched Nov 25, 2026,
  run dir results/mono_static/image/sloth/20261125_155249; cleared
  startup and preprocessing, into SHAPE_INIT at first check.

- SLOTH FULL RUN (20261125_155249): PASS. Exit 0, "Pipeline complete!",
  wall 3,490 s (58 min; pre-flash baseline run took ~2.4 h), core
  3,239.9 s, peak RSS 10.6 GiB. Finetune 100/100, final loss 0.0856,
  block 1,043.3 s. Masked metrics (obj_1) vs pre-rebase sloth
  baselines: PSNR 26.9593 vs 26.4598 (+0.50 dB), SSIM 0.8971 vs
  0.8893 (+0.0078), LPIPS 0.0465 vs 0.0518 (-0.0053) - better on all
  three. New sloth baselines proposed: 26.9593 / 0.8971 / 0.0465.

- CRAFTSMAN FULL RUN (20261125_165124): launched Nov 25, 2026.
  Infra note: the monitoring ssh session dropped ~29 min in (client
  exit 255); the run itself was unaffected (remote process tree
  intact, writing to rebase_craftsman.log) and was in
  GLOBAL_POSE_REFINE_1 at the check. Monitoring re-armed as a
  log-waiter.

- CRAFTSMAN FULL RUN (20261125_165124): PASS. Exit 0, "Pipeline
  complete!", wall ~6,880 s (1 h 55 min), core 6,427.7 s, peak RSS
  10.6 GiB. Finetune 100/100 at 24.03 s/it (flash32 class), final
  loss 0.1380 (flash32 reference run: 0.1379; healthy band, nowhere
  near the fp16 arms 0.2354). MPS mesh decode exported 001.glb /
  world.glb. Masked metrics (obj_1):
    PSNR  25.9867  vs original baseline 23.4736 (+2.5131)
                   vs flash32 reference 26.0353 (-0.0486)
    SSIM  0.8598   vs 0.8315 (+0.0283)  vs 0.8639 (-0.0041)
    LPIPS 0.0664   vs 0.0834 (-0.0170)  vs 0.0666 (-0.0002)
  The rebased tree reproduces the validated flash32 configuration
  essentially exactly on this scene, and sits well above the
  original baselines; the half-pixel fixs numeric effect here is
  smaller than run-to-run starting-point variance.
  New craftsman baselines proposed: 25.9867 / 0.8598 / 0.0664.

## Live swap

Both full runs passed sanity, so the working copy goes live:
~/alt3d/GenIA -> ~/alt3d/GenIA-pre-rebase-20261125 (preserved),
~/alt3d/GenIA-rebase -> ~/alt3d/GenIA. The genia_pkg/genia symlink
targets the GenIA path, so the standard binding serves the rebased
tree with no launcher change.

- POST-SWAP SMOKE (production path, no shim): GENIA_ROOT via the
  standard genia_pkg binding resolves to the swapped tree; launcher
  override matrix intact (unset -> flash32, off -> off, flash ->
  flash, flash32 -> flash32); dtype-spy segment through
  genia_run.sh exports with PYTHONPATH unset: varlen=120,
  varlen_fp32=120, varlen_fp16=0, masked_sdpa=0, wall 141.4 s,
  it2 loss 0.4203 - identical to the working-copy segment, so the
  swapped tree behaves exactly as the verified copy.

## Outcome

Rebase complete and live. ~/alt3d/GenIA is upstream release
60979b8 + the full MPS port (branch rebase-60979b8: merge 85364cb
+ probe fix d9f61f3, local commits only). The pre-rebase tree is
preserved at ~/alt3d/GenIA-pre-rebase-20261125. New standing
baselines: craftsman 25.9867 / 0.8598 / 0.0664, sloth 26.9593 /
0.8971 / 0.0465 (masked, obj_1). Nothing committed or pushed to
any remote in this task; publishing awaits the users word.
