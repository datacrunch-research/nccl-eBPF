# Agent Note: rackscale bandwidth policy exploration

Status: proposed

## Problem

The completed evaluation ([docs/gb300/mnnvl_eval.md](../../../../docs/gb300/mnnvl_eval.md))
settled the algorithm/protocol question and left the bandwidth question open.

Findings that frame it:

1. Algo/proto forcing has **no headroom above w8**: the default tuner (NVLS across the
   range) is within run-to-run noise of the best measured arm at every size at
   w16/w32/w60; every forced deviation loses, up to -75%. The measured exploit is a
   small-clique phenomenon (+43% @w4 protocol-only, +42% @w8 algo+proto, gone by w16).
   Corollary: above w8 a static policy's correct action is no-override
   ([nvl72_size_aware](../../implemented/feature/2026-08-14-nvl72-size-aware-policy.md)).
2. **The unexplained rackscale bandwidth plateau**: large-message AllReduce busbw is
   836 GB/s at w8, then 645 (w16), 680 (w32), 689 (w60) — scale-up costs ~18% of
   per-GPU bus bandwidth and never recovers. Nothing in the algo/proto space closes it.
   Whether this is fundamental (NVLS tree depth / switch hops) or tunable is unknown.
3. The evaluation never exercised three dimensions that plausibly move bandwidth:
   **registered buffers** (nccl-tests ran unregistered; serving stacks register, NVLS
   has zero-copy paths for registered buffers, and the tuner context already carries
   `reg_buff`), **channel-count actions** (the tuner ABI can set `nChannels` today;
   only algo/proto cells were used), and **NVLS chunk size** (`NCCL_NVLS_CHUNKSIZE`,
   env-only — invisible to the tuner ABI, an interface gap of the same class as the
   NVL-domain info we already exposed).
4. AlltoAll (MoE dispatch) plateaus at ~468 GB/s from w16 and is outside the tuner's
   reach (grouped send/recv); ReduceScatter — half of TP's traffic — was never measured.

Open questions, in decreasing immediacy:

- Q1 Can the w16+ large-message plateau be closed with registration, channels, or
  chunk size — or is it a fabric property? (The bandwidth question.)
- Q2 Does buffer registration change the *whole* landscape (windows, crossovers), not
  just the peak? Policies may need a `reg_buff` branch.
- Q3 ReduceScatter windows: does the w8 exploit exist for RS, TP serving's other half?
- Q4 Are channel-count policies busbw-neutral but SM-footprint positive — i.e. an
  overlap-aware serving policy (fewer comm SMs during decode) rather than an idle-rack
  busbw policy? Needs a serving workload and profiler feedback; divergence-safety
  design required.
- Q5 Do the window edges move on a finer-than-f2 size grid, and across NCCL versions
  (2.31.2's tuner model may close the w8 window natively)?

## Proposal

**Highlighted first step (most immediate and relevant): attack Q1+Q2 with one
registered-buffer + channels + chunk-size evaluation at w60/w8.** Concretely:

1. Runner feature (small): `REG=1` arm support in `nccl_bench.sh` (nccl-tests `-R 1`),
   so every existing arm can run registered vs unregistered.
2. Experiment A (no code): at w8 and w60, large sizes (256M-8G), sweep
   {unregistered, registered} x {default channels, tuner-forced 8/16/24/32 via the
   existing SET_CHANNELS action} x {default, NCCL_NVLS_CHUNKSIZE in 64K..1M}.
   ~120 runs, one exclusive allocation, interleaved reps.
3. Decision gate: if any combination recovers a meaningful part of the 836-vs-689 gap,
   codify it — channels via the tuner (policy-actionable today, add a large-message
   branch to `nvl72_size_aware` keyed on `reg_buff`/size); chunk size via env pinning
   in the serving launchers, plus an upstream proposal to expose chunk-size-class
   knobs through the tuner ABI (same argument as the NVL-domain-info exposure).
4. Then Q3 (add `reduce_scatter_perf` to the evaluation blocks — pure configuration)
   and Q4 (design note first: overlap-aware policies need the profiler loop and a
   rank-uniformity story before any implementation).

## Alternatives considered

- Chasing AlltoAll bandwidth here — rejected: it bypasses the tuner; that plane
  belongs to DeepEP/transfer-engine work, not NCCLbpf.
- Jumping straight to overlap-aware serving policies (Q4) — deferred: highest ceiling
  but needs serving integration and divergence-safe adaptive design; the Q1/Q2
  evaluation is days-scale and informs whether static bandwidth policies exist at all.

## References

- Tracking PR: [datacrunch-research/nccl-eBPF#1](https://github.com/datacrunch-research/nccl-eBPF/pull/1); roadmap entry: [findings comment](https://github.com/datacrunch-research/nccl-eBPF/pull/1#issuecomment-5315577928).
