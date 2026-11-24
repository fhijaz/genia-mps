# GenIA on Apple Silicon: the MPS port

GenIA ([facebookresearch/GenIA](https://github.com/facebookresearch/GenIA)) is Meta's
test-time input-alignment layer over SAM 3D Objects: it takes the SAM 3D reconstruction
of a single photo and refines pose and appearance against the input image. Upstream
GenIA is CUDA-only. This project ports its **single-image setting** to Apple Silicon
(MPS) so it runs end to end on a Mac Studio (M1 Ultra, 64 GB), with every replacement
component verified against the original it substitutes for.

This repository holds the pieces that do not belong in either code fork: the runner
script, the full port log, the Phase 0 inventory, and the A/B verification scripts.

## The three repositories

| Repo | Branch | What is in it |
| --- | --- | --- |
| [fhijaz/GenIA](https://github.com/fhijaz/GenIA) | `mps-port` | GenIA core port: central device plumbing (`core/utils/device.py`), a differentiable pure-PyTorch Gaussian rasterizer (`core/utils/diff_rasterizer.py`), SDPA visibility attention, and silicon mesh-render wiring. Based on upstream `release` @ f6155f9, the commit the port was built and verified against. |
| [fhijaz/sam3d-objects-mac](https://github.com/fhijaz/sam3d-objects-mac) | `mps-port` | The community SAM 3D MPS fork plus everything the GenIA port needed from it: GenIA compatibility hooks, the `get_features` and masked-SDPA fixes, a minimal pytorch3d shim, and the vectorized native sparse convolution. |
| fhijaz/genia-mps (this repo) | `main` | Runner, port log, Phase 0 inventory, verification scripts. |

The SAM 3D fork changes are also offered upstream to
[ankitmahala07/sam3d-objects-mac](https://github.com/ankitmahala07/sam3d-objects-mac)
as three theme branches (open a pull request from each compare view):

- Bugfixes: [fix-get-features-and-sdpa-guard](https://github.com/ankitmahala07/sam3d-objects-mac/compare/main...fhijaz:sam3d-objects-mac:fix-get-features-and-sdpa-guard?expand=1)
- GenIA compatibility hooks: [genia-compat-hooks](https://github.com/ankitmahala07/sam3d-objects-mac/compare/main...fhijaz:sam3d-objects-mac:genia-compat-hooks?expand=1)
- Sparse-conv vectorization: [sparse-conv-vectorization](https://github.com/ankitmahala07/sam3d-objects-mac/compare/main...fhijaz:sam3d-objects-mac:sparse-conv-vectorization?expand=1)

## Verified results

Measured on the M1 Ultra during the port (full detail in `GENIA_PORT_LOG.md`):

- Both single-image scenes run the complete 8-block pipeline end to end on MPS and
  export metrics, Gaussian PLYs, GLB meshes, renders, and an orbit video: the bundled
  sloth demo (clean run, exit 0, about 2.4 h wall) and the craftsman photo (about
  4.5 h wall; the finetune stage dominates; mesh decode runs on CPU because the
  macOS GPU watchdog kills the long unsynchronized MPS decode).
- Craftsman alignment, scored with GenIA's own evaluator on the input view: GenIA
  final PSNR 23.5 / SSIM 0.8 / LPIPS 0.1 versus its SAM 3D starting point at
  PSNR 15.1 / SSIM 0.6 / LPIPS 0.2, a +8.4 dB refinement gain. Peak memory about
  10.7 GB RSS on the craftsman run.
- Component-level verification: the differentiable rasterizer is bit-exact against
  the fork's forward gsplat path and its gradients match finite differences to
  about 1e-8; the SDPA visibility attention matches a dense reference to about
  1e-7; the vectorized sparse conv passes 24/24 A/B checks against the original
  implementation (worst diff 3.8e-06).
- The standalone SAM 3D fork still passes its regression check after all port
  edits, and reconstructs the same craftsman photo in about 326 s.

## Reproduce the setup

1. Clone both forks on their `mps-port` branches:

   ```sh
   git clone -b mps-port git@github.com:fhijaz/GenIA.git
   git clone -b mps-port git@github.com:fhijaz/sam3d-objects-mac.git
   ```

2. Get the SAM 3D checkpoints separately from Hugging Face
   ([facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects);
   access requires accepting the SAM license there) and place them under
   `sam3d-objects-mac/checkpoints/hf/checkpoints/`. **No model weights are
   included in any of these three repositories.** GenIA ships no weights of its
   own; MapAnything and LPIPS weights download at setup time as described in the
   port log.
3. Python environment: `genia_env` layers on top of the fork's `s3d_env` through a
   `.pth` file so `import sam3d_objects` binds to the fork checkout, and
   `GENIA_SAM3D_WEIGHTS` points at the fork checkpoints. The exact environment
   variables are set by the runner script.
4. Run (adjust the paths at the top of `genia_run.sh` to your checkout layout):

   ```sh
   ./genia_run.sh image [--scene NAME] [hydra overrides...]
   ```

   Outputs land in `GenIA/results/mono_static/image/<scene>/<timestamp>/`.

## Files in this repo

- `genia_run.sh` - the runner: sets `SPARSE_BACKEND=native`,
  `SAM3D_CONV_IMPL=vec`, `SAM3D_MPS_USE_SDPA=1`,
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, `PYTORCH_ENABLE_MPS_FALLBACK=1`, and
  `GENIA_SAM3D_WEIGHTS`, then runs GenIA's demo entry point.
- `GENIA_PORT_LOG.md` - the complete port log: per-stage changes, every file
  modified, A/B verification numbers, deviations from upstream semantics, and the
  stage-8 run instructions.
- `GENIA_PHASE0.md` - the Phase 0 inventory that scoped the port (binding
  analysis, blocker list, weights audit).
- `verify/` - the A/B verification scripts for stages 2-5 (solver and attention
  bias, visibility attention, rasterizer forward/gradient checks, mesh wiring).

## Scope and deferred work

Single-image setting only. The multiview and dynamic settings are untested, and
ActionMesh is untouched. DiffMC mesh export is deferred; the FlexiCubes GLB path
is what runs. CUDA branches in the ported code are preserved unchanged but cannot
be executed on this hardware, so only the MPS and CPU paths are verified.

## License

This port is a derivative of GenIA and is shared under the same terms. The
following is GenIA's license statement, reproduced verbatim from the GenIA
repository (the complete license text is in `LICENSE` in this repository):

This project's own source code is licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

Copyright (c) 2026 the authors.

This covers the code authored by this project: core/ (except where a file states
otherwise, e.g. core/inference.py, which derives from SAM 3D Objects), demo/
(except the input data under demo/data/, where each scene's LICENSE.md applies),
docs/ and the setup files at the repository root.

It does NOT cover the third-party components under submodules/, nor any model weights
downloaded at setup time. Each of those keeps its own licence, some of which impose
further restrictions. See THIRD_PARTY_LICENSES.md before use.

