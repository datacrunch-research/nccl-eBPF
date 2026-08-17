# Agent Note: CUDA 13.4 preview stack upgrade for the evaluation rack

Status: proposed

## Problem

The evaluation rack runs CUDA 13.1 toolkits, driver stack R595 (595.71.05:
nvidia-driver-open + fabricmanager + imex in lockstep), and a heterogeneous system
NCCL (16x 2.27.7, 1x 2.29.3; all benchmarks pin a source-built 2.29.7). The owner
wants the most experimental capable configuration: CUDA 13.4 developer preview,
latest NCCL (2.31.2, 2026-08-11), latest driver. Complication: **since 13.4 the Linux
driver is decoupled from the toolkit** — and no 13.4-feature driver exists for Linux
datacenter yet (R616+ is the 13.4 driver; Linux tops out at R610 — our sbsa repo
candidate is 610.57.04). Everything CUDA-13.x runs on drivers >=580 via minor-version
compatibility, so the DP toolkit runs on today's R595 minus GA-gated features.
Reference recipe: scaling-inference `supercomputing/install_nv_driver.md` +
`install_cuda.sh` (written for x86 HGX; the GB300 deltas are sbsa arch, the
Canonical nvidia-64k kernel, and **nvidia-imex in the version lockstep** — the x86
doc does not cover IMEX, and IMEX is the MNNVL keystone).

## Proposal

Three stages, each independently valuable, ordered by risk:

1. **P0 — toolkit + NCCL, no reboot, no fabric risk (any time).** Add
   `nvidia-preview-keyring` (verify the noble/preview channel serves sbsa), pdsh
   `cuda-toolkit-13-4` side-by-side on all trays; keep `/usr/local/cuda -> 13.1` as
   default. Build NCCL 2.31.2 from source (against 13.1 and 13.4), rebuild the
   NCCLbpf plugin against its headers (check whether the tuner ABI moved past v5),
   and A/B the evaluation vs NCCL 2.29.7 — this directly answers the open
   version-sensitivity question (does 2.31's tuner model close the w8 window
   natively?) and re-tests the tray03-class discovery fix that landed in 2.31.2.
2. **P1 — driver guinea pigs: trays 03 and 14.** They are idle, excluded from all
   workloads, and need driver/host resets anyway (degraded-GPU maintenance). Full
   lockstep upgrade {nvidia-open, nvidia-driver-open, nvidia-fabricmanager,
   libnvidia-nscq, nvidia-imex} = 610.57.04 + reboot. Answers three questions at
   once: do the dead NVLinks recover after reset (re-validate with the fused-topo
   recipe until `count="18"` returns); does an R610-IMEX tray join the R595 MNNVL
   clique (mixed-version fabric behavior — the thing we must know before any
   rack-wide move); does the plugin stack run unchanged on R610.
3. **P2 — rack-wide, one maintenance window.** Coordinated drain (serving ablations
   paused), pdsh lockstep upgrade to the newest branch (R610 now; R616+ if 13.4 GA
   has landed), `apt-mark hold` to prevent branch drift, reboot all, fabric/IMEX
   verification, then the w60 evaluation as the acceptance gate (it exists and is
   exactly an acceptance suite now). Optionally homogenize system NCCL in the same
   window, though pinned source builds already make it moot for benchmarks.

Also: extend `supercomputing/install_nv_driver.md` with the GB300/sbsa section
(IMEX lockstep, 64k-page Canonical kernel, NVL72 fabric verification steps).

## Alternatives considered

- **Upgrade the driver now to "match" 13.4** — impossible/mismatched: no Linux
  13.4-feature driver exists; jumping to R610 rack-wide buys little for 13.4 (>=580
  compat already covers it) while taking full fabric-stack risk; do drivers on their
  own cadence (P1/P2), not coupled to the toolkit.
- **Wait for 13.4 GA for everything** — foregoes months of NCCL 2.31 + DP toolkit
  data; P0 has no meaningful risk.
- **Docker images with 13.4** — heavier for the plugin/bpftime source builds and
  MNNVL device plumbing; bare side-by-side toolkits are the established pattern here.

## Open questions

- Does packages.nvidia.com noble/preview serve sbsa/arm64 for cuda-toolkit-13-4?
  (Verify at P0 start; the x86 recipe is confirmed.)
- NCCL 2.31.2 tuner/profiler ABI versions vs our plugin (v5/v6) — rebuild vs port.
- Mixed R595/R610 IMEX clique behavior (P1 answers empirically).

## References

- Tracking PR: [datacrunch-research/nccl-eBPF#1](https://github.com/datacrunch-research/nccl-eBPF/pull/1); roadmap entry: [findings comment](https://github.com/datacrunch-research/nccl-eBPF/pull/1#issuecomment-5315577928).
