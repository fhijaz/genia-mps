# GenIA MPS speed sprint log (iso-quality)

Started: Nov 24, 2026. Approved by Farrukh in chat ("Run the GenIA speed
sprint: profile first, then optimize iso-quality"). Standard: bit-exact or
within the pipeline's own run-to-run noise, proven by A/B. No reduced
iteration/step counts, no precision cuts without a noise-floor A/B, no
approximate math swaps without equivalence evidence.

## Baselines (from GENIA_PORT_LOG.md final report)
- Sloth clean run (20261124_120919): shape_init 77.7 s, appearance_init
  90.9 s, pose_init 16.6 s, global_pose_refine_1 822.2 s,
  appearance_init_2 783.6 s, finetune 4,468.0 s, global_pose_refine_2
  1,808.4 s; block sum 8,067.4 s; wall ~2.4 h. Peak RSS 18.2 GB (in
  run_final CPU mesh decode), peak MPS alloc 5,983.9 MB.
- Craftsman (20261124_022133): wall ~4.5 h; finetune 100 it at ~95 s/it
  (loss 0.397 -> ~0.17); CPU mesh decode 1,180.6 s (43,778 SLAT tokens ->
  776,998 verts / 1,552,844 faces); final masked metrics PSNR 23.4736,
  SSIM 0.8315, LPIPS 0.0834; peak RSS ~10.7 GB.
- diff_rasterizer standalone bench (512x512, SH deg 0): N=100k fwd 1.00 s,
  fwd+bwd 3.21 s; N=1M fwd 2.73 s, fwd+bwd 6.32 s.

## Phase 1: profile
(pending)

## Phase 2: optimizations
(pending)

## Phase 3: end-to-end validation
(pending)

## Phase 2 candidate verified early: MPS-native area_interpolate
Standalone A/B (~/alt3d/speed/test_area_interp.py) of a separable
pooling-matrix implementation (PyTorch adaptive-pool window rule:
start=floor(o*in/out), end=ceil((o+1)*in/out); out = Wy @ x @ Wx^T,
matrices cached per shape) vs the current helper (CPU offload path):
- 11 shape/scale combos incl. the real call sites (1,3,518,518) @0.5
  and @0.25, batched, non-square, size= forms, non-contiguous inputs,
  on CPU and MPS.
- Forward max abs diff <= 3.0e-07; gradient max abs diff <= 1.8e-07
  (fp32 rounding level, 1-2 ulp). ALL PASS.
Not yet integrated - integration waits for the profile ranking, and the
helper will keep delegating non-4D inputs and upsampling to the old path.

## Phase 1 profile run 1 (partial, process died during profiler export)
Run 1 (torch.profiler + wall timers) completed the pipeline segments:
gpr1 (10 it) saved cache 17:14, appearance_init_2 re-ran in ~29 min,
finetune ran 6 iterations at 166.7 s/it UNDER the profiler (baseline
unprofiled ~95 s/it), losses 0.4237 -> 0.4014. After the finetune block
the process died silently (no traceback) during
prof.key_averages(group_by_input_shape=True) export - consistent with
jetsam OOM on millions of profiler events; a mid-hang stack sample
showed the main thread inside KinetoEvent->Python conversion. All
in-memory wall-timer data was lost. Lesson: no key_averages on long MPS
profiles; wall timers with an incremental on-disk sink instead.
Recovery: profile_driver2.py (timers only, JSONL sink flushed per
record), run separately for gpr (10 it from pose_init cache) and
finetune (6 it from the appearance_init_2 cache run 1 saved).

## Phase 1 results (profile_driver2.py, wall timers, MPS-synced)

### Finetune iteration, craftsman (43,778 voxels), ~93 s/it profiled
(baseline unprofiled ~95 s/it; block total 593.0 s for setup+6 it+tail)
| section | s/it | % |
|---|---|---|
| decoder anchors (total) | 43.9 | 47 |
|   anchor decode forward | 11.4 | 12 |
|   anchor backward + loss | ~32.5 | 35 |
| main backward + optimizer + grad-norm/drift metrics (remainder) | ~36 | 39 |
| main decode forward | 11.4 | 12 |
| loss forward (render 1.46 + LPIPS/SSIM/depth/normals ~0.1) | 1.5 | 2 |
| area_interpolate (4 calls) | 0.005 | ~0 |
Every decode_tokens call costs ~11.4 s (14 calls accounted exactly:
initial + 6 main + 6 anchor + final). The SLAT decoder runs 2 forwards
and 2 backwards per iteration; gradient checkpointing is ON for
craftsman (43,778 >= 30,000 voxel threshold), so each backward
recomputes the torso forward. Loss-side terms are all trivial; LPIPS
forward is inside the 0.05 s frame_loss (its backward is in the main
backward remainder). area_interpolate is a NON-issue for speed
(1 ms/call, CPU offload included) - hypothesis 1 refuted by measurement.

### global_pose_refine iteration, craftsman, 5.44 s/it (10 iters, 1 frame)
| section | s/it | % |
|---|---|---|
| render forward (render_gaussian_params) | 1.56 | 29 |
| remainder: rasterizer backward + pose optimizer + bookkeeping | ~3.87 | 71 |
| frame loss total | 0.012 | ~0 |
| area_interpolate (4 calls) | 0.004 | ~0 |
Extrapolated: sloth gpr2 1,808 s/500 it = 3.6 s/it is consistent with
scene scaling. GPR is rasterizer fwd+bwd bound.

### Ranking for Phase 2
1. Decoder gradient checkpointing on MPS (finetune: both backwards pay
   recompute; peak RSS baseline only ~10.7 GB on a 64 GB machine, the
   checkpoint is a CUDA OOM guard) - A/B next.
2. Mesh decode on CPU: 1,180.6 s one-off per run - MPS chunked test next.
3. Rasterizer backward recompute (gpr 71% remainder; finetune shares
   the rasterizer) - chunk-granularity A/B.
4. area_interpolate MPS-native: verified equivalent, ~0 measured
   effect; integrate as cleanup only.

## Phase 2 attempt: decoder_checkpoint=false on MPS - REJECTED (4.3x SLOWER)
A/B from the appearance_init_2 cache, 4 iterations, background seed
12345 fixed. Noise floor first (two identical checkpoint-ON runs):
losses bit-identical for iterations 1-2, then max |dloss| 1.13e-03 by
iteration 4; final feats max|diff| 8.0e-05, mean 1.9e-06. That is the
pipeline run-to-run noise envelope.
Checkpoint ON: 91.1 s/it, run wall 465 s, peak RSS 10.6 GiB.
Checkpoint OFF: ~396 s/it (2 iterations completed, then killed).
Loss trajectory matched (it2 0.4230 vs 0.423043) - the math is
equivalent, but on MPS the retained un-checkpointed sparse graph makes
backward pathologically slow (stack samples: deep Python-level churn,
MPSGraph specialization-cache teardown). Gradient checkpointing is
load-bearing on MPS; do not disable it. Candidate closed.

## Phase 2 attempt: MPS mesh decode with main-thread sync chunking - FAILED
Craftsman SLAT (43,778 tokens) through slat_decoder_mesh on MPS,
reference = baseline CPU-decoded final/meshes/001.glb (776,998 verts /
1,552,844 faces). Four attempts, all killed by the MPS watchdog
(kIOGPUCommandBufferCallbackErrorImpactingInteractivity):
1. plain forward - died, surfaced at SparseTensor layout .item().
2. sync hooks on 5 top-level children (50 ms throttle) - died.
3. sync hooks on all 159 submodules (50 ms throttle) - died.
4. sync hooks on all 159 submodules, unthrottled - died.
Conclusion: some indivisible submission inside one module (sparse
subdivision/conv at the highest resolution, or the FlexiCubes
extraction) exceeds the watchdog on its own; no module-boundary sync
placement can split it. Per the sprint protocol, CPU mesh decode
(1,180.6 s) stays. Scripts: ~/alt3d/speed/mesh_decode_test.py,
compare_mesh.py (unused - no MPS mesh to compare).

## Phase 2 attempt: rasterizer checkpoint chunk - REJECTED (changes math)
Bench (~/alt3d/speed/bench_raster.py): 800k synthetic Gaussians, 518x518,
_rasterize_one_view fwd+bwd. Production chunk=4096: fwd 1.68 s, bwd 2.20 s,
fwd+bwd reps 3.28 s. chunk=16384/65536/unbounded: fwd faster (1.09 s) but
bwd slower (2.55 s), fwd+bwd reps 3.67 s - NOT faster overall. And NOT
math-equivalent: vs chunk=4096, color max diff 2.4e-05, depth 1.9e-04,
grad_means max diff 5.8e-01. Diagnosis: with min_transmittance=0 (early
exit disabled) the diffs collapse to fp-grouping level (color 2.4e-07,
alpha 0.0, depth 3.6e-07), proving the chunk parameter also gates the
per-chunk early-transmittance break - it is not a pure checkpointing
knob. Changing it changes rendered results: a quality change, forbidden.
No code change to diff_rasterizer.py. Production chunk=4096 is already
the measured fwd+bwd optimum.

## Phase 2: bf16 LPIPS - not attempted
Profile shows the whole frame_loss (incl. LPIPS forward) is 0.05 s/it;
VGG backward is a small share of the main backward next to the decoder.
Upside is minor, precision risk nonzero; skipped per the sprint rules.

## Phase 2: area_interpolate MPS-native - INTEGRATED
core/utils/device.py (backup: device.py.orig-backup-speed). Only the
branch that previously offloaded to CPU changed: MPS + 4D +
non-divisible downsample now pools on-device via a cached separable
pooling-matrix matmul (out = Wy @ x @ Wx^T, PyTorch area window rule).
GENIA_AREA_INTERP_NATIVE=0 restores the CPU-offload path exactly; all
other inputs take the original code paths untouched.
Verified against the shipped function
(~/alt3d/speed/test_area_interp_integrated.py): 9 cases x CPU/MPS,
worst forward diff 2.98e-07, worst gradient diff 1.49e-07 (CPU cases
bit-exact 0.0 - direct path). Measured speed effect in the profile:
~0 (1 ms/call) - this lands as a device-crossing/sync-hazard removal,
not a speed win. Honest framing kept.

## Phase 3: end-to-end validation launched
Fresh craftsman run via ~/alt3d/genia_run.sh image --scene craftsman
(run_demo starts from scratch: no cache resume, no cache save) with
the integrated area_interpolate as the only code change vs baseline.
Targets: wall ~4.5 h; per-block vs sloth/craftsman baselines; final
masked metrics PSNR 23.4736 / SSIM 0.8315 / LPIPS 0.0834 (the port
reproduced these EXACTLY on re-evaluation, so any drift is a red flag).

## Phase 3 validation CANCELLED by user (Nov 24, ~22:15 PST)
The fresh craftsman validation run (20261124_202132, started 20:21 PST)
was killed at the user direction during finetune (28/100 iterations,
46:15 into the block, ~1 h 51 min into the run; last loss 0.2877,
95.26 s/it - on baseline pace). Process tree (time wrapper, run_demo,
and the genia.core worker) confirmed fully stopped. Partial run dir
left on disk. Consequence: the area_interpolate integration is
UNVALIDATED end to end.

## area_interpolate integration REVERTED (user direction)
core/utils/device.py restored from device.py.orig-backup-speed;
verified: diff clean, MD5 137e0af42530a2b245e2998c907a5b49 for both,
no GENIA_AREA_INTERP_NATIVE/_area_pool_mps symbols remain, git status
shows no modified tracked files. The change is shelved, not lost: the
A/B evidence above stands and it can be re-integrated later as part of
a validated optimization batch. Backup file and ~/alt3d/speed scripts
kept in place.

## POSTSCRIPT: mesh-decode retest with AGX relaxation (Nov 24, ~22:20 PST)
After the four failed attempts above, new evidence: the watchdog error
kIOGPUCommandBufferCallbackErrorImpactingInteractivity has a documented
driver-level relaxation - env var AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1, set
before the process creates its Metal device (MLX issue #3267 confirmed
workaround; llama.cpp sets it by default in its Metal backend).
Retest (~/alt3d/speed/mesh_decode_test.py, craftsman SLAT 43,778 tokens,
env var exported before Python start):
- PLAIN variant (no sync hooks): COMPLETED. MPS wall 162.7 s vs CPU
  baseline 1,180.6 s = 7.3x faster (~17 min saved per run).
- Mesh comparison vs the CPU-decoded reference (001.glb): NOT
  bit-identical. MPS: 776,920 verts / 1,552,684 faces vs CPU: 776,998 /
  1,552,844 (-78 verts, -160 faces = 0.010%). Bounding boxes identical.
  Nearest-neighbor surface agreement (50k samples, both directions):
  median 5.9e-06, p99 ~2.4-2.9e-04, ~0.11% of points >1e-03 from the
  other surface, max 0.019. Interpretation: same geometry to fp
  tolerance; the count delta is borderline FlexiCubes cells flipping
  topology on device-level fp differences - the mesh is an export
  artifact (evaluation metrics are render-based, unaffected).
- NOT integrated (per user instruction): candidate first optimization
  for the next batch - integrate into run_final redecode_slat behind
  the env var, then re-validate end to end including final metrics.

## Fork regression check (after revert, Nov 24)
s3d_env, SPARSE_BACKEND=native, SAM3D_MPS_USE_SDPA=1: sam3d_objects
imports cleanly; run.sh weights check passes (all 6 checkpoint files
present). PASS. No fork files were modified during this sprint.

## Sprint end state
GenIA tree at pre-sprint baseline (only core/utils/device.py was ever
modified, now reverted and verified). No end-to-end validation of any
optimization has been run - per user direction, validation happens
once a speedup worth validating exists. Candidate queue for the next
batch, in measured-value order: (1) MPS mesh decode behind
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 (7.3x on a 1,180.6 s stage, mesh
fp-equivalent but not bit-identical - needs user sign-off on that
class of difference plus metric re-validation); (2) area_interpolate
MPS-native (verified equivalent, ~0 speed effect - hygiene only).

## MPS mesh decode integration + full validation (Nov 24, 2026, ~22:27 PST)
Approved by Farrukh in chat ("Integrate the MPS mesh decode, then run
the full validation"). His approval covers the known non-bit-exact class
of the MPS mesh (~0.010% FlexiCubes topology flips vs CPU decode,
surface agreement to fp tolerance) - settled, not re-litigated.
Plan: (1) integrate - genia_run.sh exports
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=${AGX_RELAX_CDM_CTXSTORE_TIMEOUT:-1};
core/utils/slat_decode.py gains _decode_mesh_on_mps next to the
untouched _decode_mesh_on_cpu, selected in redecode_slat by
GENIA_MESH_DECODE_DEVICE (default mps, =cpu restores the original path
exactly). Backups: genia_run.sh.orig-backup-mpsdecode,
slat_decode.py.orig-backup-mpsdecode. (2) Standalone verification
through the INTEGRATED path on the baseline craftsman SLAT
(final/final.pt of run 20261124_022133). Acceptance: no watchdog kill,
wall < 300 s, topology delta <= ~0.02%, NN p99 <= ~1e-03. On failure:
restore backups, no validation launch. (3) Fresh from-scratch craftsman
validation via genia_run.sh (no cache resume, no cache save), compared
against baseline wall ~4.5 h and masked metrics PSNR 23.4736 / SSIM
0.8315 / LPIPS 0.0834 (baseline JSON: 23.473628997802734 /
0.8314967751502991 / 0.08344723284244537), plus mesh comparison vs
baseline 001.glb (776,998 verts / 1,552,844 faces) and the fork
regression check.
STATUS: backups made; integration in progress.
INTEGRATION DONE (~22:29 PST): git diff on slat_decode.py = 53
insertions / 4 deletions (import os; _decode_mesh_on_mps added after
_decode_mesh_on_cpu, which is byte-untouched; redecode_slat MPS branch
now dispatches on GENIA_MESH_DECODE_DEVICE, default mps). genia_run.sh
gained the AGX export block after PYTORCH_ENABLE_MPS_FALLBACK. bash -n
and py_compile pass. _decode_mesh_on_mps: ensures slat_decoder_mesh,
moves the SLAT to the model device, plain no_grad forward (the proven
AGX configuration - no sync hooks), then sweeps the returned
MeshExtractResult tensor attributes back to CPU so callers see the CPU
path output contract. Next: standalone verification through
redecode_slat on the baseline craftsman SLAT.
STEP 2 STANDALONE VERIFICATION: PASS (~22:31 PST).
Driver ~/alt3d/speed/mesh_decode_integrated_test.py called the
INTEGRATED redecode_slat(formats=["mesh"]) on the baseline craftsman
SLAT from 20261124_022133/final/final.pt (43,778 tokens), env =
production set with AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 exported before
Python start and GENIA_MESH_DECODE_DEVICE unset (default MPS path).
Result: completed, no watchdog kill; wall 164.3 s (sprint standalone:
162.7 s; CPU baseline 1,180.6 s); returned mesh tensors on CPU (output
contract asserted); mesh 776,920 verts / 1,552,684 faces - counts
identical to the sprint MPS decode. Test-process peak RSS 11.31 GB.
Comparison vs baseline CPU 001.glb via ~/alt3d/speed/compare_mesh_nn.py
(point-to-surface distance, trimesh.proximity, 50k samples/direction -
NOTE: an earlier draft of that script measured point-to-point-set NN,
whose floor is the sample spacing (~6e-03) even for identical surfaces;
discarded and rewritten): bbox identical; delta -78 verts (-0.0100%),
-160 faces (-0.0103%); cand->ref median 2.95e-06, p99 4.19e-05, max
1.78e-03, 0.01% of samples >1e-03; ref->cand median 2.93e-06, p99
4.20e-05, max 1.69e-03, 0.01% >1e-03. All acceptance criteria met
(wall < 300 s, topology <= ~0.02%, p99 <= ~1e-03).
STEP 3: launching the fresh from-scratch craftsman validation now.
VALIDATION CANCELLED BY USER (~22:48 PST, change of plan: "Cancel
validation. Do rebase scoping."). The fresh craftsman run
(20261124_223550, launched 22:35:49 PST) was killed 12 min in, during
global_pose_refine_1 (block dirs through 03_global_pose_refine_1
written). Process tree (time wrapper, run_demo, genia.core worker,
resource tracker) terminated with SIGTERM and verified gone - no
genia_env python processes remain. Partial run dir left on disk.
STATE KEPT, per user direction: the MPS mesh decode INTEGRATION stays
in place exactly as made (genia_run.sh AGX export default;
_decode_mesh_on_mps in slat_decode.py with GENIA_MESH_DECODE_DEVICE=cpu
fallback to the untouched CPU path). Step 2 standalone verification
COMPLETED and passed before the launch (see entry above) - the
integration is standalone-verified but NOT end-to-end validated.
End-to-end validation is deferred until after the rebase decision
(scoping: ~/alt3d/GENIA_REBASE_SCOPING.md).
REBASE SCOPING DONE (~23:05 PST): ~/alt3d/GENIA_REBASE_SCOPING.md.
Upstream release head is now 01ea014 (advanced past e50ccd0; f6155f9
verified as ancestor). Upstream delta is small (39 files, ~1,081 churn
lines); the Nov 12 speed and Nov 3 memory mechanisms live in the
sam-3d-objects submodule, not GenIA, so a GenIA-only rebase buys an
estimated ~0-3% end to end; the substantive GenIA-side change is the
Nov 17 half-pixel correctness fix, which re-baselines metrics.
Verdict: not worth it for speed; worth it only for correctness/currency
or as step 1 before a fork rebase. Temp clone deleted unprocessed
(runtime execution-risk flag; analysis done by plain-text diff -
details in the scoping doc method note).

## Follow-up leads (Nov 24, ~23:40 PST): logging check + neighbor-map cache
User direction: "Explore the neighbor-map caching lead next, logging check
first." Iso-quality standard unchanged (bit-exact or within the measured
noise floor: seeded segment losses bit-identical 2 it, max |dloss| 1.13e-03
by it 4; feats max|diff| 8.0e-05). No end-to-end validation in this task.

### Phase 0 plan (logging / deferred metric reads)
Live per-step .item() sites in the finetune loop (finetuning.py, canonical
config: drift 0.01, sh_reg 0.1, sh_consistency 0.1, opacity anchor 0.1,
refine_poses=true, microbatch_size=1 but n_total_valid=1 so grad accum is
OFF - compute_total_loss takes the plain path, per-term metrics are
reporting-only, no loss-based best-tracking exists):
- compute_total_loss: 6 per-term .item() per frame + total .item()
- main loop: drift/dc_reg/dc_cons/sh_reg/sh_cons .item(), vis_masked_frac,
  total .item()
- _apply_decoder_anchors: opacity_anchor .item() (rotation anchor off)
- grad norms: one .norm(2).item() per optimizer param with a grad
- parameter drift: token_l2, canonical norm, per-frame pose deltas, scale
Design: GENIA_FINETUNE_METRICS_EVERY=N (default 1 = current behavior).
N>1 stashes detached scalar tensors and flushes every N iterations and at
loop exit, reproducing the identical float arithmetic in the identical
order at flush time, so loss_history values must be BIT-IDENTICAL; only
the tqdm postfix shows last-flushed values between flushes. Grad-accum
mode keeps eager reads (its reported total is rebuilt from the floats).
A/B: sprint segment protocol (appearance_init_2 cache, seed 12345,
ckpt=true, 4 it): A1 (N=1), B1 (N=1000), A2 (N=1).

### Phase 0 RESULT: deferred metric reads - NO WIN (lead dead)
Implementation: finetuning.py (backup finetuning.py.orig-backup-mapcache),
GENIA_FINETUNE_METRICS_EVERY (default 1 = original eager reads). N>1
stashes detached scalar tensors for ALL per-step metric reads (loss terms
in compute_total_loss, reg values, vis_masked_frac, anchors, per-param
grad norms, parameter-drift block) and flushes every N iterations and at
loop exit, replaying the identical float arithmetic in the identical
order. Grad-accum mode keeps eager reads (not exercised here: accum is
off for this scene). tqdm postfix shows last-flushed values when N>1.
Equivalence (seed 12345, ckpt=true, 4 it, appearance_init_2 cache):
- A1 vs B1: losses bit-identical it1-2, max |dloss| 1.73e-03 by it4;
  feats max|diff| 7.77e-05, mean 1.94e-06.
- A2 vs B2: bit-identical it1-2, max |dloss| 1.10e-03; feats 8.06e-05 /
  1.95e-06. Baseline-vs-baseline (A1 vs A2): 9.6e-04 / 7.82e-05.
  => B sits ON the pipeline noise floor in both windows; the flush
  reproduces reported losses exactly (it1-2 bit-identical despite being
  materialized only at the loop-end flush). No metric feeds control flow
  (no loss-based best-tracking exists in this loop - confirmed in code).
Timing (finetune block seconds, 4 it):
- Slow window (transient external load; load avg 15-min 3.92 decaying,
  page faults identical ~301k in all runs): A1 590.7, B1 599.2 (B +1.4%).
- Fast window: A2 405.3 (101.3 s/it), B2 433.3 (108.3 s/it) (B +6.9%).
Both windows agree in sign: deferring the reads wins NOTHING measurable;
the MPS queue is evidently not deep when the reads fire (CPU-side Python
dominates), so the sync stalls cost ~0. Code stays in the tree behind the
default-OFF flag as an available knob; genia_run.sh untouched. Lead dead.

### Phase 1 (neighbor-map lead): round 1 finding - WRONG LAYER
Instrumented run (2 it, map_driver.py): the fork conv_native pair cache
saw ZERO activity during finetune decodes (gets/hits/builds/key all 0),
while 6 decode_tokens calls ran at ~11.3 s each. Root cause: the GS SLAT
decoder (SLatGaussianDecoder) is a SparseTransformerBase - windowed /
serialized sparse ATTENTION + SparseLinear. It contains no sparse
convolutions at all (grep: no SparseConv3d/SparseInverseConv3d in the
decoder, its base, or the transformer blocks). The task briefs premise
(conv-bound decode) is wrong for this decoder; conv_native IS live and
content-cached for other stages (probe: synthetic SparseConv3d forward
populates _PAIR_CACHE, 2nd call hits), just not on the finetune path.
Also: NativeSparseData.indice_dict is never read for compute in the
native backend (only propagated by replace_feature).
The true analog of "neighbor maps" here: attention serialization layouts
(calc_serialization / calc_window_partition), cached per-SparseTensor in
_spatial_cache - and decode_tokens builds a FRESH SparseTensor per call,
so they rebuild on every main and anchor decode. Round 2 instruments
those builders + a full register_spatial_cache inventory.

### Phase 1 RESULT (round 2): the real per-call build is 0.6% of the iteration - lead DEAD, Phase 2 not warranted
Instrumented run (2 it, map_driver2.py; builders timed, plus a global
register_spatial_cache inventory on the SparseTensor class):
Per decode_tokens forward (~11.3 s): exactly TWO layout builds, both
calc_window_partition (window_size 8, shifts 0 and 4), 0.26-0.29 s
total = ~2.5% of the forward. calc_serialization: never called (this
decoder uses windowed attention). Registrations inventory for the whole
run: only window_partition_8_0 and window_partition_8_4, once per
decode, 12 builds / 1.70 s total across 6 decode forwards - and ZERO
additional builds during backwards: gradient-checkpoint recompute hits
the within-lineage _spatial_cache (the cache dict is shared through
replace()), so the backward does not repay the build.
Per finetune iteration: 2 decode forwards x ~0.29 s = ~0.58 s of layout
building on a ~95-101 s iteration = ~0.6%. A perfect cross-call cache
(content-keyed, like the conv pair cache) could recover at most that
~0.6% - under the 3% stop rule, and 8x under the 5%-of-iteration
integration bar. PHASE 2 NOT RUN. The other ~97.5% of each decode
forward is the attention/MLP math itself (windowed SDPA, linears,
norms, small-op Python overhead), not index construction.
Final state of this lead: fork untouched (no Phase 2 code written);
the conv pair cache remains as-is (correct and used by other stages).

## Flash-attention scoped test (Nov 25, 2026)
Approved by Farrukh in chat ("Run the scoped flash-attention test: audit,
micro-benchmark, segment A/B"). Three steps with kill criteria between
them. Standard: a kernel swap may differ at fp-noise level but must stay
within the measured segment noise floor (losses bit-identical it1-2,
max |dloss| ~1e-03 by it4, feats max|diff| ~8e-05); a candidate that
only works by computing in LOWER precision than the current path fails
the standard outright. Candidates: (a) mtlattn (varlen, differentiable,
fp32 claimed; wheels pinned to torch 2.12 vs our 2.15.0), (b)
mps-flash-attn (drop-in SDPA / direct API, fp32 + backward claimed).

### Step 1 audit - code findings (read-only)
Dispatch chain for the GS SLAT decoder windowed self-attention in
GenIA finetune (fork sam3d_objects, ATTN backend = "sdpa" by default in
modules/sparse/__init__.py; genia_run.sh sets SAM3D_MPS_USE_SDPA=1,
SPARSE_BACKEND=native):
- Decoder: SLatGaussianDecoder (slat_decoder_gs.yaml): resolution 64,
  model_channels 768, num_blocks 12, num_heads 12 -> head_dim 64,
  attn_mode "swin", window_size 8, latent_channels 8.
- Block attention module (modules/sparse/attention/modules.py:152)
  calls sparse_windowed_scaled_dot_product_self_attention with
  shift_window alternating 0 / 4 across blocks.
- windowed_attn.py: the uniform branch requires EVERY window to hold
  exactly window_size (=8) tokens - never true here - so the varlen
  branch runs: q,k,v = [1, M, H, C] packed -> masked_sdpa(q,k,v,
  seq_lens, seq_lens).
- masked_sdpa (masked_sdpa.py): mps_should_chunk() is False
  (SAM3D_MPS_USE_SDPA=1), BUT the GenIA port guard diverts to
  masked_chunked_sdpa whenever H * sum_q * sum_kv > 250M. Here
  12 * 43778 * 43778 = 2.30e10 >> 2.5e8, so finetune windowed
  attention NEVER reaches F.scaled_dot_product_attention.
- ACTUAL PATH: masked_chunked_sdpa -> per-window Python loop ->
  chunked_sdpa_bhlc: manual matmul + softmax, q/k/v upcast to fp32,
  scores/softmax/PV in fp32, query-chunked at 128, output cast back to
  the input dtype. I.e. manual matmul+softmax, not SDPA at all.
- Dtype context: finetuning/canonical.yaml sets
  decoder_autocast_bf16: true, so decoder linears run bf16 autocast
  and qkv feats ENTER attention as bf16; the chunked kernel upcasts to
  fp32 for the attention math itself and returns bf16. "Current fp32
  path" for the standard = fp32 attention compute (scores, softmax,
  PV) with bf16 in/out. A candidate must match that precision class.
- Kernel contract a replacement must serve: packed varlen self-
  attention, M=43,778 tokens, H=12, C=64, per-window seq_lens from
  calc_window_partition (window 8, shifts 0 and 4), scale 1/sqrt(64),
  no mask/bias/causal/dropout, bf16 inputs, fp32 accumulation, full
  autograd backward (training path, gradient checkpointing active).
(Occupancy stats + profiler share: measured next, appended below.)

### Step 1 audit - measurements (flash_audit_driver.py, 1-it segment)
Dispatch CONFIRMED empirically (spies on the real segment run):
- Per decode forward: 12 masked_sdpa calls (12 blocks), ALL diverted
  by the guard to masked_chunked_sdpa; F.scaled_dot_product_attention
  calls during decode = 0. 3,582 window calls per decode forward
  (6 layers x 279 windows @shift0 + 6 x 318 @shift4), each a
  chunked_sdpa_bhlc call looping ceil(len/128) query chunks in Python.
- qkv at attention entry: [43778, 3, 12, 64], dtype float16 (empirical
  correction to the code-reading note above: inputs arrive fp16, not
  bf16 - decoder weights are fp16; the chunked kernel upcasts to fp32
  for scores/softmax/PV and returns fp16). opt_feats fp32 [43778, 8],
  coords int32 [43778, 4], autocast_bf16=true on decode_tokens.
Window occupancy (craftsman SLAT, 43,778 tokens, window 8):
- shift 0: 279 windows; min 1, median 139, mean 156.9, p95 320,
  max 335; sum len^2 = 9,630,390.
- shift 4: 318 windows; min 2, median 128, mean 137.7, p95 256,
  max 345; sum len^2 = 8,083,996.
- Padding waste if windows were padded to max: 93,465 / 109,710
  token slots vs 43,778 real = +113% / +151%. Dense global attention
  would need M^2 = 1.92e9 pairs vs ~8.9e6 actually computed (0.46%).
Attention share of decode (standalone captured triple, ckpt on):
- Forward wall 12.94-13.61 s; masked_sdpa synced wall 9.91-10.54 s
  = ~77%. Profiler device time: attention-core bmm/matmul 9.31 s +
  softmax 0.07 s of 11.63 s total = ~81%.
- Backward wall 32.5 s (incl. ~13 s checkpoint recompute of forward).
  Profiler fwd+bwd session (45.11 s attributed vs 45.4 s wall):
  attention-core 37.12 s (82.3%), attention softmax fwd+bwd 1.55 s
  (3.4%) -> attention ~86% of decode fwd+bwd. aten::bmm alone 36.9 s.
  (Caveat: MPS Kineto under-attributes some dense-linear time -
  aten::mm/addmm totals look low - but the bracketed wall timers are
  direct: attention is 77% of forward by wall clock, and the bmm
  mass sits entirely inside the attention calls.)
- Peak MPS memory, standalone fwd+bwd: ~19.0 GB.
VERDICT: attention share ~77% fwd / ~86% fwd+bwd, far above the 25%
ceiling-check. Step 2 justified. Replacement contract: packed varlen
q,k,v [43778, 12, 64] fp16, cu_seqlens int32 (280 / 319 entries),
max_seqlen 335 / 345, scale 1/sqrt(64), no mask/bias/causal/dropout,
fp32 accumulation, differentiable, out [43778, 12, 64] fp16.

### Step 2 - BLOCKED at candidate install (execution-risk check)
mtlattn 0.5.0 wheel (cp311 universal2) was downloaded into the
scratch dir ~/alt3d/speed/flash_scratch/mtlattn_pkg with --no-deps
(genia_env untouched). At that point the runtime flagged the
downloaded third-party package content with an execution-risk check:
importing/running it is prohibited without the user explicitly
agreeing to run that third-party native code. Per instructions the
test was NOT proceeded with: mtlattn was never imported, the
from-source build was not attempted, and mps-flash-attn (same class:
third-party native extension) was not installed either. Step 3 not
reached. Unblocking requires Farrukh to explicitly approve executing
the mtlattn (and/or mps-flash-attn) package code on the Mac.
Step 1 stands complete (above): attention ~77% of decode forward,
~86% of fwd+bwd - the prize is real if the test is approved.

### Steps 2+3 - resumed Nov 25, 2026 after explicit approval
Farrukh approved in chat: "Approve both mtlattn and mps-flash-attn"
(explicit go-ahead to execute both third-party native packages on the
Mac for this benchmark). Step 2 proceeds on that authority.
- Bench fixture regenerated exactly: appearance_init_2 cache SLAT
  coords [43778,4] -> calc_window_partition gives shift0 279 windows
  (max 335, min 1), shift4 318 (max 345, min 2). Saved
  ~/alt3d/speed/flash_fixture.pt.
- mtlattn 0.5.0 LOAD TEST: PASSED against torch 2.15.0 as-is (wheel
  Requires-Dist torch>=2.14; the 2.12 pin was the 0.4.x line). No
  from-source build needed. _C.mpp_available() = True on this M1
  Ultra / macOS 26.7.3. Smoke (2 windows, fp16, grad on): default
  routing and MTLATTN_FORCE_KERNEL=1 agree exactly, grads finite.
  Note: varlen_attention default routing falls back to a per-seq
  native-SDPA loop when _kernel_wins() is False (non-MPP machines);
  here MPP is available and training mode routes to the kernel.
- mps-flash-attn 0.6.3 installed --no-deps --target into
  flash_scratch/mfa_pkg (genia_env untouched). API: dense batched
  flash_attention(q,k,v [B,H,N,D], attn_mask bool True=masked) with
  autograd; ships _C_next build for torch 2.13+, imports fine on
  2.15. Smoke: masked vs unmasked diff vs fp32 ref both ~1e-3
  (fp16 class); mask honored on the smoke shape.
- Bench design (flash_bench.py): unit = one production masked_sdpa
  call, real seq_lens from the fixture, shifts 0 and 4. Arms:
  baseline masked_chunked_sdpa (via masked_sdpa, SAM3D_MPS_USE_SDPA=1
  guard path); mtlattn varlen default routing; mtlattn
  MTLATTN_FORCE_KERNEL=1; mtlattn fp32 inputs + forced kernel
  (informational: production inputs are fp16); mps-flash-attn padded
  batch (pad to max len, bool key mask prebuilt from seq_lens,
  per-call pad/slice timed). Fixed-seed qkv, warmup 2, median of 5,
  fwd and fwd+bwd separate, correctness = out/dq/dk/dv max-abs-diff
  vs baseline on identical inputs; NaN-leak test on the padded arm.
STEP 2 RESULTS (flash_bench.py; median of 5, warmup 2; per single
masked_sdpa call; peak = MPS driver allocated):
shift 0 (279 win, max 335):        fwd s   fwd+bwd s  peak GB  maxdiff out/dq/dk/dv
  baseline masked_chunked_sdpa     0.774     2.838     2.78    (reference)
  mtlattn varlen default routing   0.0266    0.0867    2.16    0.002/0.002/0.002/0.002
  mtlattn forced kernel            0.0266    0.0873    2.16    0.002 all (routing already picks kernel)
  mtlattn fp32 in, forced kernel   0.0754    0.2103    2.16    0.0005/0.0005/0.0005/0.001
  mps-flash-attn padded+mask       0.0728    0.2238    5.38    3.34/60.2/47.0/42.0  FAIL
shift 4 (318 win, max 345):
  baseline                         0.787     2.868     2.78    (reference)
  mtlattn default                  0.0231    0.0825    2.16    0.002 all
  mtlattn fp32 forced              0.0687    0.1798    2.16    0.001 all
  mps-flash-attn padded+mask       0.0794    0.2790    6.45    2.97/158/133/248  FAIL
- mtlattn fp16 diffs are fp16-ulp class (out mean|.| ~0.095, ulp
  ~0.001); fp32 arm halves them. Both far inside any fp-noise band;
  the segment A/B envelope is the arbiter for the fp16 arm.
- mps-flash-attn failure diagnosed (flash_mfa_diag*.py): kernel is
  correct unmasked (3e-4) and honors the bool mask at B=1, but at
  production batch B=279 the mask has no effect - identical wrong
  outputs (3.3428) with (B,1,L,L) and (B,H,L,L) masks - and
  NaN-filled padding leaks into real outputs (33.4M nonfinite).
  Cannot serve the contract as-is; also 2.6x slower than mtlattn
  and ~2.5x the memory (padding cost +113-151% slots is real).
KILL CRITERION: PASSED by mtlattn - 32.7x (shift0) / 34.8x (shift4)
on fwd+bwd vs the ~20% bar. Winner: mtlattn varlen_attention, fp16
production form, default routing. Proceeding to Step 3.
STEP 3 WIRING: winner (mtlattn varlen_attention, fp16 production
form) wired into the fork at the exact audited call site:
sam3d_objects/.../sparse/attention/windowed_attn.py, varlen branch,
ATTN=="sdpa" case. Gate: SAM3D_ATTN_IMPL=flash AND device mps AND
dtype fp16; otherwise the original masked_sdpa path runs untouched.
No global replace_sdpa; masked_sdpa itself unmodified (visibility /
guidance attention cannot be rerouted by this change). Inside the
branch MTLATTN_FORCE_KERNEL=1 is set so anchor (inference-mode)
decodes use the kernel instead of mtlattn per-seq native routing.
Backup: windowed_attn.py.orig-backup-flashattn. mtlattn stays in
the scratch dir (PYTHONPATH for the test); genia_env untouched.
Segment A/B next: arms fbase1, fbase2 (flag off), fflash (flag on),
protocol seed 12345 / ckpt=true / 4 it from appearance_init_2 cache.
STEP 3 RESULTS (segment A/B, seed 12345, ckpt=true, 4 it):
  arm     wall s   finetune block s (core)   losses it1-4
  fbase1  505.1    439.9 (429.4)   0.424916 0.423043 0.425680 0.415920
  fbase2  466.9    403.5 (393.1)   0.424916 0.423043 0.426287 0.417236
  fflash  166.4    103.6 (93.3)    0.426364 0.424503 0.426458 0.417771
Same-session floor (fbase1 vs fbase2): losses bit-identical it1-2,
max |dloss| 1.315e-03; feats max|diff| 8.154e-05, mean 1.974e-06.
Flash vs bases: max |dloss| 1.851e-03 (vs fbase1) / 1.460e-03 (vs
fbase2); feats max|diff| 1.199e-04, mean ~8.0e-06. Structure:
flash carries a ~1.45e-03 offset from it1 (kernel accumulation
order), then tracks without divergence - by it4 flash-vs-fbase2
(5.4e-04) is below the base-vs-base it4 gap (1.32e-03), and the
final loss 0.417771 sits next to the baseline pair (0.415920 /
0.417236). Verdict: same order as the noise floor but ~1.4x the
same-session point envelope on max stats - a marginal pass, not a
clean one; recorded as such for Farrukhs decision.
Speed: finetune block 421.7 s (base mean) -> 103.6 s = 4.1x;
per-iteration ~105 s -> ~26 s (block total / 4). Segment wall
486 s -> 166 s. Peak RSS unchanged (~11.33 GB all arms).
PROJECTION (labeled): at ~79 s saved per iteration, a 100-it
craftsman finetune drops ~2.9 h -> ~0.72 h; a ~4.5 h end-to-end
run drops to ~2.3 h (about half). Assumes the per-iteration
saving is constant across iterations and scenes.
Flag SAM3D_ATTN_IMPL stays DEFAULT OFF. Nothing pushed.
Files: fork windowed_attn.py +29/-4 (published repo file,
fhijaz/sam3d-objects-mac main; backup .orig-backup-flashattn);
GenIA untouched; genia_env untouched; all bench/diag scripts and
both candidate packages live in ~/alt3d/speed + flash_scratch/.

### fp32 mtlattn arm - segment test (Nov 25, 2026, follow-up)
Farrukhs pick after the fp16 marginal pass: "Segment-test the fp32
mtlattn arm first, then decide." Same segment protocol (seed 12345,
ckpt=true, 4 it, appearance_init_2 cache). Arms this session:
fbase3 (fresh baseline, flag off, same-session floor anchor) and
fflash32 (SAM3D_ATTN_IMPL=flash32). Wiring: flash32 added as a
distinct elif in the same gated varlen branch of windowed_attn.py -
q/k/v upcast to fp32 for mtlattn.varlen_attention, output cast back
to fp16, MTLATTN_FORCE_KERNEL=1, mirroring flash_bench.py arm_mtl_fp32
exactly. Pre-edit backup: windowed_attn.py.orig-backup-flash32;
.orig-backup-flashattn left intact. Flag stays DEFAULT OFF.
Wiring verified: diff vs .orig-backup-flash32 is a pure 25-line
insertion (the flash32 elif); flash and off paths byte-for-byte
unchanged; py_compile OK; .orig-backup-flashattn untouched
(md5 0d6ee91bffdea1ac8b9998ea2f2a704b). Runs starting: fbase3 then
fflash32 (run_flash32_ab.sh mirrors run_flash_ab.sh).
FP32 ARM RESULTS (segment A/B, seed 12345, ckpt=true, 4 it):
  arm       wall s  block s (core)   s/it(core)  losses it1-4
  fbase1    505.1   439.9 (429.4)    107.4       0.424916 0.423043 0.425680 0.415920
  fbase2    466.9   403.5 (393.1)     98.3       0.424916 0.423043 0.426287 0.417236
  fbase3    474.2   409.3 (398.9)     99.7       0.424916 0.423043 0.426648 0.417280
  fflash    166.4   103.6 (93.3)      23.3       0.426364 0.424503 0.426458 0.417771
  fflash32  181.0   118.7 (108.6)     27.1       0.426145 0.424251 0.424496 0.412744
Fresh baseline fbase3 reproduces the recorded session: it1-2
bit-identical to fbase1/fbase2; vs fbase1 max |dloss| 1.359e-03,
feats max 7.710e-05; vs fbase2 max 3.612e-04, feats 7.844e-05.
Three-baseline envelope: max |dloss| 1.359e-03, feats max
8.154e-05, feats mean ~1.97e-06. Both floors used.
fp32 arm deviations: max |dloss| 3.177e-03 (vs fbase1) /
4.492e-03 (fbase2) / 4.536e-03 (fbase3) = 2.3-3.3x the envelope;
feats max 1.186-1.197e-04 (~1.46x envelope, same as fp16 arm),
feats mean 4.03-4.07e-06 (~2.1x floor mean, HALF the fp16 arms
~8.0e-06). Structure differs from fp16: it1-2 offsets ~1.21-1.23e-03
(slightly under fp16s ~1.45e-03) but the gap GROWS at it4 -
fflash32 final loss 0.412744 sits 3.2-4.5e-03 BELOW the tight
3-baseline it4 cluster (0.415920-0.417280), where fp16s it4 gap
had closed to 5.4e-04. No NaN / structural breakage; loss stays
in band and final feats are as close as fp16s. VERDICT: MARGINAL,
and strictly worse than the fp16 arm on the headline loss
statistic - the fp32 upcast does NOT convert the marginal pass
into a clean one. Its only numeric win is feats mean deviation.
Speed: block 118.7 s vs baseline mean 417.6 s = 3.5x (fp16:
103.6 s = 4.0x); fp32 block is 15% slower than fp16 despite the
2.4x slower unit call (0.2103 vs 0.0867 s fwd+bwd) because
attention no longer dominates the accelerated block. Peak RSS
unchanged (~11.33-11.35 GB all arms).
PROJECTION (labeled): 100-it finetune ~2.90 h -> ~0.83 h (fp32)
vs ~0.72 h (fp16); end-to-end ~4.5 h -> ~2.4 h (fp32) vs ~2.3 h
(fp16). Assumes constant per-iteration saving.
Head-to-head for the adoption decision: fp16 is faster AND
closer on max |dloss| (1.46-1.85e-03, 1.1-1.4x envelope); fp32
only wins feats mean (4.1e-06 vs 8.0e-06, both tiny absolute).
If adopting, the fp16 arm is the better-evidenced one; the fp32
test removes the tighter-numerics rationale for the upcast cost.
SAM3D_ATTN_IMPL stays DEFAULT OFF (flash and flash32). Nothing
pushed; genia_env untouched (mtlattn via PYTHONPATH only).
Compare script: ~/alt3d/speed/flash32_compare.py; runner
run_flash32_ab.sh; logs ab_logfbase3.log / ab_logfflash32.log.

### ADOPTION - fp16 mtlattn arm as production default (Nov 25, 2026)
Farrukh decided in chat: "Adopt the fp16 arm: flag on for GenIA runs
plus a real install". Scope: (1) real pip install of mtlattn 0.5.0
into genia_env; (2) genia_run.sh defaults SAM3D_ATTN_IMPL=flash with
caller override respected (same pattern as AGX_RELAX_CDM_CTXSTORE_TIMEOUT);
(3) verification through the launcher env (2-it segment, losses vs the
recorded fflash arm) + fork regression smoke with flag off. NO full
end-to-end run; validation stays deferred until Farrukh calls it.
Record appended below as steps complete.
ADOPTION STEP 1 - REAL INSTALL: DONE.
- pip install --no-deps mtlattn==0.5.0 into genia_env (same wheel as
  the scratch copy: mtlattn-0.5.0-cp311-cp311-macosx_13_0_universal2,
  438 kB, from the local pip cache). pip output: "Installing collected
  packages: mtlattn / Successfully installed mtlattn-0.5.0" - nothing
  else collected or changed.
- pip freeze before/after diff: exactly ONE added line
  (mtlattn==0.5.0); no existing package version changed.
- torch still 2.15.0 after install.
- Clean-process verification (env -u PYTHONPATH, no path additions):
  import mtlattn resolves to genia_env/lib/python3.11/site-packages;
  mtlattn._C.mpp_available() = True; varlen fp16 smoke (2 windows,
  H=12, C=64, MTLATTN_FORCE_KERNEL=1, grads on): out [90,12,64] fp16
  finite, dq/dk/dv all finite. SMOKE OK.
ADOPTION STEP 2 - LAUNCHER DEFAULT: DONE.
- Backup: ~/alt3d/genia_run.sh.orig-backup-flashdefault.
- Diff vs backup is a pure 7-line insertion after the AGX export:
  a comment block plus
    export SAM3D_ATTN_IMPL="${SAM3D_ATTN_IMPL:-flash}"
  mirroring the AGX caller-respecting pattern. bash -n syntax OK.
- Semantics verified in isolation: unset -> flash; caller "off" ->
  off; caller "flash32" -> flash32.
ADOPTION STEP 3a - PRODUCTION-PATH CHECK: running. Driver
~/alt3d/speed/ab_driver_verify2.py = ab_driver.py protocol with
num_iterations=2 and call-count spies on mtlattn.varlen_attention
and windowed_attn.masked_sdpa; launched via run_adopt_verify.sh,
which evals genia_run.sh own export lines (so SAM3D_ATTN_IMPL comes
from the new launcher default) with PYTHONPATH explicitly unset -
no scratch-dir dependency. Results appended when done.
ADOPTION STEP 3a - PRODUCTION-PATH CHECK: PASSED.
- Run: ~/alt3d/speed/run_adopt_verify.sh -> ab_driver_verify2.py,
  2 iterations, seed 12345, ckpt=true, appearance_init_2 cache,
  output ab_adopt_verify. Env came from genia_run.sh own export
  lines; header confirms SAM3D_ATTN_IMPL=flash (the new launcher
  default) and PYTHONPATH=None - the installed genia_env mtlattn is
  the only copy used; no scratch path involved.
- Spy counts (counting wrappers on mtlattn.varlen_attention and
  windowed_attn.masked_sdpa): varlen_calls=120, varlen_fp16=120,
  masked_sdpa_calls=0. The flash branch fired for every windowed
  attention call in the segment; the original path never ran.
- Losses (loss-history JSON, full precision): it1
  0.4263641834259033, it2 0.4245033860206604 - bit-identical across
  the two verification runs and equal to the recorded fflash arm
  (0.426364 / 0.424503) at its recorded precision; clearly distinct
  from the baseline arms (0.424916 / 0.423043). Segment run wall
  125.6 s (2 it, incl. in-process startup after compose).
- Tooling note: the first verify launch crashed at compose on a
  shell-quoting defect in the driver heredoc (stripped quotes in
  three os.environ lines) and the second completed the run but
  crashed in the final spy print for the same reason; both fixed,
  third run clean end to end. No model code was affected.
ADOPTION STEP 3b - FORK REGRESSION, FLAG OFF: PASSED.
- cd sam3d-objects-mac && env -u SAM3D_ATTN_IMPL s3d_env python
  test_default_device.py -> "DEFAULT-DEVICE REGRESSION: PASS"
  (subm 9.537e-07, strided 3.338e-06 coords equal, inverse
  4.768e-07 coords equal; same class as the recorded baseline).
  Standalone SAM 3D behavior unchanged when the gate is off.
SCRATCH STATE: ~/alt3d/speed/flash_scratch/ is LEFT IN PLACE (it
holds the rejected mps-flash-attn package, the mtlattn scratch
copy, and the bench harness history), but production no longer
depends on it: genia_env has its own mtlattn 0.5.0 install and the
verification above ran with PYTHONPATH unset.
ADOPTION STATE: COMPLETE. Production GenIA runs via genia_run.sh
now default to SAM3D_ATTN_IMPL=flash (fp16 mtlattn varlen kernel);
caller override intact (flash32 / off / any other value skips the
fp16 gate per windowed_attn.py). NO end-to-end run was performed;
end-to-end validation remains deferred until Farrukh calls it.
UNPUBLISHED CHANGES LEFT BY THE WHOLE FLASH PROGRAM (for a later
commit decision; nothing pushed):
- fhijaz/sam3d-objects-mac lineage: windowed_attn.py gated flash /
  flash32 branches (backups .orig-backup-flashattn,
  .orig-backup-flash32 alongside).
- fhijaz/genia-mps repo content: genia_run.sh SAM3D_ATTN_IMPL
  default (backup genia_run.sh.orig-backup-flashdefault) and this
  log. genia_env site-packages is machine state, not repo content.

## End-to-end validation (Nov 25, 2026)
Approved by Farrukh in chat ("Run the end-to-end validation with the
tolerance standard"). Fresh from-scratch craftsman run via the
production launcher (genia_run.sh image --scene craftsman; run_demo
starts from scratch: no cache resume, no cache save), production
configuration exactly as adopted - launcher defaults apply
(SAM3D_ATTN_IMPL=flash, AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1, MPS mesh
decode integrated, mtlattn 0.5.0 in genia_env). No code changes.
BASELINES (run 20261124_022133, pre-flash, CPU mesh decode): masked
PSNR 23.4736 / SSIM 0.8315 / LPIPS 0.0834 (full precision
23.473628997802734 / 0.8314967751502991 / 0.08344723284244537);
wall ~4.5 h; finetune ~95 s/it; CPU mesh decode 1,180.6 s, mesh
776,998 verts / 1,552,844 faces.
TOLERANCE STANDARD (user-approved): PASS requires run completes end
to end with final mesh + evaluator metrics; masked metrics within
PSNR 0.5 dB / SSIM 0.02 / LPIPS 0.02 of baselines; finetune pace in
the flash class (~20-35 s/it); mesh decode on the MPS path (minutes
class). On PASS the run metrics become the new standing baselines.
Projection: wall ~2.3 h, finetune ~26 s/it, MPS decode ~164 s, mesh
776,920 / 1,552,684.
LAUNCH: pending.
LAUNCHED Nov 25, 2026 08:53:19 PST (first attempt 08:52:42 died
instantly: setsid does not exist on macOS - launch mechanics only,
no pipeline code involved; relaunched with plain nohup + disown).
Command: /usr/bin/time -l ~/alt3d/genia_run.sh image --scene
craftsman, log /tmp/genia_validation_craftsman.log, run_demo PID
66661. Caller env verified clean (SAM3D_ATTN_IMPL / AGX /
GENIA_MESH_DECODE_DEVICE all unset -> launcher defaults apply:
SAM3D_ATTN_IMPL=flash, AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1, mesh
decode default mps). Composed config confirms from-scratch:
processing.resume_from_cache=false, output.save_cache=false,
seed 42 (same as baseline run).
PROGRESS: blocks 00-02 done by ~09:03; global_pose_refine_1 done
~09:42 (silent inner loop - liveness confirmed by worker CPU-time
advance, GPU-bound); appearance_init_2 done ~09:48. FINETUNE PACE
CHECK PASSED: at 09:57 finetune is at it 25/100 running
21.9-22.0 s/it (flash class 20-35 s/it; baseline was ~95 s/it) -
the launcher flash default is effective in the real pipeline.
Loss 0.297 at it 25, descending normally.
PROGRESS: finetune completed ~10:24 (100 it, ~22 s/it avg, final
loss ~0.24). global_pose_refine_2 running: Composite opt 500 it
at ~5.6 s/it (matches the 5.44 s/it profile), ETA ~46 min for the
block, then run_final.
RESULT: FAIL (tolerance standard). Run dir
results/mono_static/image/craftsman/20261125_085320. The run
completed end to end in ONE uninterrupted process ("Pipeline
complete!", DEMO exit clean under /usr/bin/time -l): wall
7,127.23 s = 1.98 h (baseline ~4.5 h); total core 6,680.34 s;
peak RSS 10.56 GiB (baseline ~10.7 GB); peak MPS alloc
6,039.6 MB.
BLOCKS (timing.json total_seconds; baseline craftsman per-block
was not preserved except finetune - its timing.json was
overwritten by the finisher pass):
  gt_shapes_inversion 0.0 (status ran)
  shape_init 77.7 | appearance_init 191.0 | pose_init 16.6
  global_pose_refine_1 2,349.0 | appearance_init_2 579.7
  finetune 2,222.0 (22.2 s/it; baseline 9,435 s bar total,
  ~94-95 s/it) | global_pose_refine_2 1,245.1
CRITERIA:
- Completion + final mesh + metrics: MET.
- Finetune pace: MET (22.2 s/it, flash class).
- MPS mesh decode: MET. run_final (decode + exports +
  evaluation) fits inside wall-minus-core = 447 s total
  including startup/preprocessing; the baseline CPU decode
  alone took 1,180.6 s, so the MPS path is confirmed by
  arithmetic and by the default config. Mesh from THIS runs
  SLAT (43,778 tokens, same count as baseline): 804,444 verts /
  1,607,372 faces (001.glb and world.glb) - counts differ from
  both references because the final SLAT state differs.
- METRICS: FAILED on all three. Masked (obj_1): PSNR 21.7348
  vs 23.4736 (delta -1.739 dB, tol 0.5); SSIM 0.7590 vs 0.8315
  (delta -0.0725, tol 0.02); LPIPS 0.1158 vs 0.0834 (delta
  +0.0323, tol 0.02). Full-frame: 20.9970 / 0.7495 / 0.1223
  (baseline 23.1410 / 0.8262 / 0.0865).
DIAGNOSIS: the finetune trajectory under flash attention lands
in a materially worse optimum. Final finetune loss 0.2354 vs
baseline 0.1729; the gap is present by it49 (0.2752 vs 0.2417)
and widens (it87: 0.2374 vs 0.1797); gradient norms differ
systematically (tokens 1.9e-02 vs 8.3e-03, lora 2.9e-01 vs
1.3e-01 at it99) - a different basin, not noise around the
baseline path. The 4-iteration segment A/B showed a small fixed
offset at it1 and judged it non-divergent over 4 iterations
(marginal pass, ~1.4x the noise envelope); over 100 iterations
that offset compounds. vmask also differs slightly pre-finetune
(0.82 vs 0.81), consistent with flash being active in the
backbone SLAT decodes during appearance stages too. The MPS
mesh decode is exonerated: metrics are computed from Gaussian
renders, independent of the mesh artifact. Pose-refine blocks
do not use the gated attention. All other configuration was
identical (seed 42, from-scratch, same launcher env).
CONSEQUENCE: baselines DO NOT change - the original craftsman
baselines (23.4736 / 0.8315 / 0.0834) stand. The production
flash default is NOT validated end to end; this is recorded as
a failed validation, not tuned or re-run.

## Gradient-bias diagnostic (Nov 25, 2026)
Approved by Farrukh in chat ("Go ahead: run the diagnostic code on my
Mac") after the first attempt stopped at a platform execution-risk
flag on the ~/alt3d/speed/ harness. QUESTION: does the mtlattn varlen
flash kernel introduce a SYSTEMATIC gradient bias (segment-boundary
leakage / per-segment LSE handling in backward), or is its error
unbiased noise comparable to the baseline path own error vs ground
truth (trajectory chaos -> dead end for finetune)? Supersedes the
Step-2 micro-benchmark (synthetic randn seed 777, max-diff only).
METHOD: real q/k/v captured from a craftsman finetune decode (spy on
masked_sdpa, baseline path, appearance_init_2 cache protocol);
float64 CPU dense per-window attention as ground truth (exact
softmax, scale 1/sqrt(64), forward + backward, fixed seeded upstream
gradient identical across arms); arms (a) mtlattn varlen fp16,
(b) mtlattn varlen fp32 inputs, (c) masked_chunked_sdpa baseline,
(d) MPS SDPA fp32 sanity. Stats per arm for out/dQ/dK/dV: max abs,
mean abs, SIGNED mean, err-vs-truth slope/corr; per-window breakdown
(error vs window length; first/middle/last segments; boundary rows);
repeat with a second upstream-gradient seed (sign/structure
consistency); neighbor-perturbation + window-permutation leakage
tests; uniform-length synthetic control (128 / 256).
STATUS: running. Results appended below as they land.
STATUS: COMPLETE (Nov 25, 2026). Scripts: grad_bias_capture.py,
grad_bias_diag.py, grad_bias_analyze.py; raw results
grad_bias_results.json; capture grad_bias_capture.log; run log
grad_bias_diag.log - all in ~/alt3d/speed/.

BATCH PROVENANCE: REAL captured q/k/v (preferred route succeeded).
Spy on masked_sdpa (baseline path, SAM3D_ATTN_IMPL unset) during a
1-iteration craftsman finetune segment (appearance_init_2 cache,
seed 12345, ckpt=true); first main-decode call per shift saved to
diag_qkv.pt in 82 s. Shapes [43778, 12, 64] fp16 per tensor.
Activation scales (std): shift0 q 1.6208 / k 2.5936 / v 1.0736;
shift4 q 1.6001 / k 8.7263 / v 0.6754 - far from the randn(0,1) of
the old micro-benchmark, esp. shift-4 keys; this matters below.
TRUTH: float64 CPU dense per-window attention via autograd (exact
softmax, scale 1/sqrt(64)), inputs = exact fp16 values in float64,
upstream gradient = exact fp16 gout values in float64; two gout
seeds (A 20251125, B 20251126) per batch. Sanity arm (per-window
MPS SDPA fp32) sits ~100x below baseline error (dq mean-abs
4.7e-08 vs 4.9e-06, shift0), confirming the yardstick.

GLOBAL ERRORS vs float64 truth (mean-abs; signed mean; seed A):
shift0 (279 win):        out            dq             dk             dv
  mtl_fp16   ma 6.34e-05  1.52e-04      2.21e-05       2.31e-05
             sm -3.1e-08  -1.9e-07      -1.4e-07       -6.5e-09
  mtl_fp32   ma 6.25e-05  4.90e-06      4.00e-06       1.45e-05
  baseline   ma 6.25e-05  4.88e-06      4.00e-06       1.45e-05
shift4 (318 win):
  mtl_fp16   ma 6.26e-05  5.85e-04      4.82e-05       6.73e-05
             sm +1.3e-08  +8.0e-08      -2.3e-08       +1.7e-08
  mtl_fp32   ma 6.06e-05  9.11e-06      1.26e-05       4.44e-05
  baseline   ma 6.06e-05  8.52e-06      1.26e-05       4.44e-05
mean|truth| dq: 2.77e-02 (shift0), 4.83e-02 (shift4).
RATIOS mtl_fp16 / baseline (mean-abs, seed A): out x1.01 / x1.03,
dq x31.2 / x68.7, dk x5.5 / x3.8, dv x1.6 / x1.5 (shift0 / shift4).
mtl_fp32 / baseline: x1.00-x1.07 on EVERY tensor, both shifts, both
seeds - frequently digit-identical max errors (the shared fp16
in/out rounding floor). Uniform controls (128/256, randn and
scale-matched): mtl_fp16 dq ratio only x2.0-x2.9 - the blowup is a
real-activation effect (large keys), not a uniform-window effect.

BIAS TESTS - all negative for mtl_fp16:
- Signed means flip sign across gout seeds (dq shift0 -1.9e-07 ->
  -1.9e-08; dk -1.4e-07 -> +2.3e-08; shift4 dq +8.0e-08 ->
  -9.3e-09). Relative signed bias ~1e-5 of mean|truth|.
- Err-vs-truth slope (multiplicative bias): |slope| <= ~1e-5,
  signs inconsistent across seeds/tensors.
- Cross-seed per-window dq signed-error correlation: -0.13
  (shift0), +0.20 (shift4) - no persistent per-window structure
  (baseline +0.04 / -0.02, same class).
- No localization: dq mean-abs flat across window-length groups
  (shift0: 1.52e-04 / 1.52e-04 / 1.54e-04 for len >=256 / mid /
  <64; corr vs len +0.05) and across boundary rows (first row
  1.51e-04, last row 1.52e-04, interior 1.52e-04). Group signed
  means all ~1e-7 with mixed signs, both shifts.

LEAKAGE TESTS (shift0, mtl_fp16) - BIT-EXACT ZERO:
- Determinism: identical re-run diff 0.0 on out/dq/dk/dv.
- Window-order permutation (fwd+bwd, unpermuted): diff 0.0.
- Neighbor perturbation: other windows k/v x100, gout zeroed
  outside target windows {0, 139, 278}: target out/dq/dk/dv diff
  0.0 (baseline control also 0.0). Cross-segment contamination
  and LSE cross-talk are structurally excluded.

VERDICT: NO systematic gradient bias - the bug class is excluded.
The flash kernel errors are UNBIASED (signed stats ~1e-7 abs /
~1e-5 relative, sign-unstable across gradient seeds, no per-window,
length, position, or boundary structure) and segment-isolated
(bit-exact). What the fp16 arm does have is a materially larger
error MAGNITUDE than the baseline fp32 chunked math on real
activations: dq mean-abs x31 (shift0) / x69 (shift4), dk x4-6,
dv x1.5, out x1.0 - the ratio ordering dq >> dk > dv tracks the
operand magnitudes in each backward path (dq multiplies by the
large keys, std 2.6 / 8.7), the signature of fp16 accumulation
inside the kernel backward, not of segmentation or softmax/LSE
mishandling. The fp32 arm of the same kernel is baseline-exact
(x1.00-1.07 everywhere), proving the kernel algorithm itself is
correct; the excess is purely the fp16 precision class. So the
end-to-end finetune divergence (loss 0.2354 vs 0.1729) is
noise-driven trajectory divergence: per-step dQ noise ~0.5-1.2%
of signal (vs baseline ~0.02%), amplified over 100 iterations of
a finetune already shown to amplify even baseline-class pointwise
differences. Classification: unbiased noise / trajectory chaos,
with the precision caveat - a dead end for the fp16 arm as a
drop-in replacement at production precision, NOT a fixable bias
bug. (The fp32 arm remains pointwise baseline-class; its earlier
4-it segment loss deviation is consistent with the same trajectory
sensitivity, at ~3.5x speed instead of ~4.1x.)

## FP32 arm end-to-end validation (Nov 25, 2026)
Approved by Farrukh in chat ("Validate the fp32 arm end to end, then
decide the default"). Fresh from-scratch craftsman run, identical to
the fp16 validation run (20261125_085320) EXCEPT the attention
implementation: SAM3D_ATTN_IMPL=flash32 set explicitly in the caller
env (launcher default is flash; caller value respected). The fork's
flash32 branch (windowed_attn.py) upcasts q/k/v to fp32 for
mtlattn.varlen_attention, casts the output back to fp16, and sets
MTLATTN_FORCE_KERNEL=1 itself - no other companion setting needed.
Everything else production: AGX relaxation + MPS mesh decode via
launcher defaults, seed 42, resume_from_cache=false,
save_cache=false, no code changes anywhere.
WHY: the gradient-bias diagnostic proved the fp32 arm pointwise
baseline-exact vs float64 truth (x1.00-1.07 on out/dQ/dK/dV, both
shifts, both seeds) while the fp16 arm failed end to end on
backward dQ noise magnitude (x31-69). This run decides whether
fp32 holds over the full 100-iteration trajectory.
ACCEPTANCE (same tolerance standard): completes end to end with
final mesh + evaluator metrics; masked metrics within PSNR 0.5 dB /
SSIM 0.02 / LPIPS 0.02 of the standing baselines (23.4736 /
0.8315 / 0.0834); MPS mesh decode in the minutes class. On PASS the
fp32 metrics become candidates for the new baselines (Farrukh
makes the final default decision in chat; launcher NOT changed by
this run). Segment projection: finetune ~29.7 s/it, wall ~2.4 h.
EARLY VERIFICATION: observational spy (sitecustomize pass-through
wrapper on mtlattn.varlen_attention, ~/alt3d/speed/fp32val_spy/,
inert unless FP32VAL_SPY=1) counts varlen calls and records the q
input dtype per process - math untouched, returns the original
result unchanged. Spy smoke-tested standalone before launch
(intercepted 1 fp32 call correctly). STOP conditions at finetune:
pace ~95 s/it means the override did not take; ~22 s/it means the
fp16 arm is somehow active - either way stop and diagnose.
STATUS: launching.
LAUNCHED Nov 25, 2026 11:25:53 PST. Command: env -u
AGX_RELAX_CDM_CTXSTORE_TIMEOUT -u GENIA_MESH_DECODE_DEVICE
SAM3D_ATTN_IMPL=flash32 FP32VAL_SPY=1 PYTHONPATH=<spy dir>
/usr/bin/time -l ~/alt3d/genia_run.sh image --scene craftsman,
log /tmp/genia_validation_fp32.log, run_demo PID 72762, worker
PID 72763. Composed config verified from-scratch: seed 42,
resume_from_cache=false, save_cache=false. Spy files exist for
run_demo, worker, and one child PID - the wrapper is loaded in
the worker. PROGRESS: blocks 00-02 done ~11:39; in
global_pose_refine_1 at 11:54 (silent loop, worker CPU advancing).
Spy still 0 varlen calls pre-finetune - the backbone stages do
not take the varlen branch; the decisive branch check is at
finetune (expected ~12:21 PST).
EARLY VERIFICATION PASSED (12:22 PST, finetune it 2/100):
- Spy (worker PID 72763): 400 varlen calls recorded, ALL with
  q dtype torch.float32 (400/400) - the flash32 branch is active
  in the real pipeline; zero fp16-dtype calls.
- Finetune pace 25.3 / 25.2 s/it - fp32 class (segment core was
  27.1 s/it); NOT baseline (~95) and NOT fp16 (~22). No stop
  condition triggered; run continues.
- vmask=0.81 matches the baseline run (fp16 run had 0.82).
- Initial finetune loss 0.2559 (it0), 0.2628 (it1); early grad
  norms baseline-class (tokens 5.6e-03, lora 1.2e-01 at it0;
  cf. fp16 run it0 1.6e-02 / 2.6e-01). Note the starting loss
  differs across all three full runs (baseline 0.397, fp16
  0.3441, fp32 0.2559) - the stochastic appearance stages vary
  run to run on MPS; trajectory shape and final values are the
  comparators, per the earlier noise-floor findings.
RESULT: PASS (no quality degradation - all three masked metrics
BETTER than the standing baselines; see the band note below).
Run dir results/mono_static/image/craftsman/20261125_112555.
One uninterrupted process, "Pipeline complete!", clean exit
under /usr/bin/time -l. No crashes, no retries, no code changes.
EFFECTIVE ENVIRONMENT: caller env SAM3D_ATTN_IMPL=flash32
(explicit override of the launcher's flash default);
AGX_RELAX_CDM_CTXSTORE_TIMEOUT and GENIA_MESH_DECODE_DEVICE
unset -> launcher defaults (AGX relaxation on, MPS mesh decode).
Spy active via FP32VAL_SPY=1 + PYTHONPATH=<spy dir> only.
BRANCH VERIFICATION (mandatory early check, passed at finetune
it 2 and re-confirmed at completion): worker spy total 5,160
mtlattn.varlen_attention calls, EVERY one with q dtype
torch.float32 (5,160/5,160) - the flash32 branch served the
whole run; zero fp16-dtype calls. Finetune pace 25.16 s/it
(tqdm average; timing.json 25.3 s/it) - fp32 class, not
baseline (~95) and not fp16 (~22). vmask=0.81, matching the
baseline run (fp16 run: 0.82).
WALL: 7,347.61 s = 2.04 h (fp16 validation 7,127.23 s = 1.98 h;
original baseline ~4.5 h). Peak RSS 10.56 GiB (same as fp16 run).
BLOCKS (timing.json total_seconds; fp16 validation in parens):
  gt_shapes_inversion 0.0 (0.0) | shape_init 77.7 (77.7)
  appearance_init 191.5 (191.0) | pose_init 16.6 (16.6)
  global_pose_refine_1 2,333.7 (2,349.0)
  appearance_init_2 638.6 (579.7)
  finetune 2,528.8 (2,222.0) | global_pose_refine_2 1,079.5
  (1,245.1) | core sum 6,865.5 (6,680.3)
MPS MESH DECODE: MET (minutes class). Wall-minus-core = 482 s
covers startup + preprocessing + all of run_final (MPS decode,
exports, evaluation); the baseline CPU decode alone took
1,180.6 s. Mesh from this run's SLAT (43,778 tokens, same count
as baseline and fp16 runs): 798,107 verts / 1,594,358 faces
(001.glb and world.glb) - counts differ from both references
because the final SLAT state differs.
FINETUNE TRAJECTORY (loss; this run / baseline / fp16 run):
  it0   0.2559 / 0.397  / 0.3441
  it25  0.2063 / -      / 0.297 (it25)
  it49  0.1782 / 0.2417 / 0.2752
  it87  0.1397 / 0.1797 / 0.2374
  final 0.1379 / 0.1729 / 0.2354
Smooth monotonic descent, baseline-class gradient norms
throughout; the fp16 run's late plateau/divergence is absent.
METRICS (masked, obj_1) vs standing baselines:
  PSNR  26.0353 vs 23.4736  delta +2.5617 dB
  SSIM  0.8639  vs 0.8315   delta +0.0324
  LPIPS 0.0666  vs 0.0834   delta -0.0168
Full-frame: 25.7477 / 0.8602 / 0.0687 (baseline 23.1410 /
0.8262 / 0.0865) - better on all three there too.
BAND NOTE: the tolerance standard was written as a symmetric
band (PSNR 0.5 dB / SSIM 0.02 / LPIPS 0.02). LPIPS is inside
the band; PSNR and SSIM are outside it on the FAVORABLE side.
Nothing degraded on any metric, so the standard's purpose
(iso-quality: no loss beyond noise) is met. The overshoot is
NOT a systematic fp32 effect - the diagnostic proved the fp32
arm pointwise identical to the baseline path (x1.00-1.07), so
it cannot manufacture +2.56 dB. The credible mechanism is the
one already documented for this pipeline: the stochastic
appearance stages handed this run a better starting point
(initial finetune loss 0.2559 vs the baseline run's 0.397) and
the trajectory-sensitive finetune amplified it into a better
optimum (final loss 0.1379 vs 0.1729). Configuration
equivalence is established independently of the metrics by
the spy record, the pace, and the identical seed/config.
CONSEQUENCE: the fp32 configuration's metrics (26.0353 /
0.8639 / 0.0666) become CANDIDATES for the new standing
baselines. The launcher default was NOT changed by this run -
Farrukh makes the final default decision in chat.

## Launcher default: flash32 (Nov 25, 2026)
Farrukh decided in chat: "Make flash32 the launcher default",
following the fp32 arm's end-to-end validation PASS (craftsman
run 20261125_112555: 2.04 h, metrics better than baselines on
all three, 5,160/5,160 varlen calls verified fp32). The fp16
arm stays available as an explicit caller override only.
CHANGE: ~/alt3d/genia_run.sh, one line only (line 25):
-  export SAM3D_ATTN_IMPL="${SAM3D_ATTN_IMPL:-flash}"
+  export SAM3D_ATTN_IMPL="${SAM3D_ATTN_IMPL:-flash32}"
Caller-override semantics preserved exactly; the AGX export
and everything else in the file untouched (diff vs backup is
the single line). Backup taken before the edit:
~/alt3d/genia_run.sh.orig-backup-flash32default. bash -n OK.
OVERRIDE MATRIX (clean shells, launcher's own export lines
eval'd, the run_adopt_verify.sh technique):
  unset -> flash32; caller off -> off; caller flash -> flash;
  caller flash32 -> flash32. All as designed.
PRODUCTION-PATH SMOKE (flag unset, pure launcher default):
~/alt3d/speed/run_flash32default_verify.sh ->
ab_driver_verify3.py (verify2 protocol plus an fp32 dtype
counter), 2 iterations, seed 12345, ckpt=true,
appearance_init_2 cache, output ab_flash32default_verify.
Spy counts: varlen=120, varlen_fp32=120, varlen_fp16=0,
varlen_other=0, masked_sdpa=0 - every windowed attention
call in the segment took the flash32 branch under the new
default; the original path never ran. Segment wall 134.8 s.
Losses (full precision): it1 0.4261449873447418, it2
0.4242507517337799 - matching the recorded fflash32 segment
arm (0.426145 / 0.424251) at its recorded precision, and
distinct from the baseline arms (0.424916 / 0.423043) and
the fp16 arm (0.426364 / 0.424503).
FORK REGRESSION, FLAG OFF: PASSED. cd sam3d-objects-mac &&
env -u SAM3D_ATTN_IMPL s3d_env python test_default_device.py
-> "DEFAULT-DEVICE REGRESSION: PASS" (subm 9.537e-07,
strided 2.861e-06 coords equal, inverse 4.768e-07 coords
equal; same class as the adoption run). Standalone SAM 3D
behavior unchanged when the gate is off.
No git commits or pushes in this task.
