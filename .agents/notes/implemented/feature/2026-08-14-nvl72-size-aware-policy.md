# Agent Note: nvl72_size_aware policy

Status: implemented

## Problem

Tuner policies do not transfer across topology or scale. Measured on GB300 NVL72
(NCCL 2.29.7, exclusive allocation): the repo's `size_aware_v5` (tuned on 8x B300)
loses 35-50% at 4-8M on one tray; `nvlink_ring_mid_v2` gains up to +42% at w8 but
loses up to -55% (AllReduce, w32) and -47% (AllGather @64M, w32) at scale because it
has neither a scale guard nor a collective-type guard. A policy safe to leave enabled
fleet-wide must know where its evidence ends.

## Decision

`nvl72_size_aware.bpf.c`: overrides only when **all** guards pass —
`n_nvl_domains == 1` (single NVL domain; zero/other → no-op), `coll_type ==
ALLREDUCE`, and `n_ranks` inside a measured window:

- r<=4: Ring/LL128 at 4-32M (+22% @4M, +43% @16M, +8% @32M).
- r<=8: Ring/LL128 at 4-32M, Ring/Simple at 64-192M (+19..+42% at 4-32M, +9/+6% at
  64/128M).
- r>8: **no override.** Clean w16/w32 sweeps show the default tuner within noise of
  the best arm at every size and Ring losing everywhere below 512M; the earlier
  w16 (8-32M) and w32 (32M) candidate windows — fit on data later found contaminated
  by a concurrent serving job — are refuted (-3..-10%).

Only rank-uniform inputs (`n_bytes`, `n_ranks`, `n_nvl_domains`, `coll_type`) so all
ranks compute identical actions and cannot diverge-hang. Evidence tables:
`docs/gb300/mnnvl_eval.md`.

## Alternatives considered

- **Deploy `nvlink_ring_mid_v2` as-is** — rejected by measurement (scale and
  collective-type damage above).
- **Telemetry-adaptive policy** (`adaptive_channels` family) — rejected for fleet use:
  rank-local telemetry can diverge across ranks; benign only while the profiler is
  detached (verified: without telemetry all ranks emit identical actions).
- **Windows for w16/w32** — tried, refuted by clean data; the exploit is a
  narrow-world phenomenon (proto-only at w4, algo+proto at w8, gone by w16).

## Consequences

- Zero measured cost where the policy stands down (tracks baseline within noise at
  w16/w32 at every size; noop delta <=1 us on 8B-256K).
- w64 behavior is no-override by construction; validation run pending.
- Re-derivation procedure when hardware/NCCL changes: rerun the arm sweeps
  (`scripts/nccl_bench.sh`), refit windows from the per-size table, keep the guards.
- Depends on the
  [NVL-domain context note](../architecture/2026-08-14-nvl-domain-info-in-policy-context.md).
