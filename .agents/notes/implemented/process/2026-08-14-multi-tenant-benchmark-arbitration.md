# Agent Note: multi-tenant benchmark arbitration

Status: implemented

## Problem

Benchmarks on a shared NVL72 rack raced a colleague's serving jobs twice in one day.
First a 16-tray job started mid-sweep and silently turned two scale ladders into
plausible-looking contention noise (bimodal "instability", fake crossovers, a fake
+78% policy win) — caught only by a post-hoc timestamp audit against `scontrol`;
109 runs quarantined. Then the reverse: our own evaluation blocked a fresh window and
aborted 100+ runs. Direct `mpirun -H` bypasses slurm, so neither side gets warned.
The failure mode is not lost time; it is **wrong numbers that look right**.

## Decision

Two layers:

1. `scripts/nccl_bench.sh` `preflight_busy_check`: before every run, ssh each target
   tray and refuse to launch if any GPU has compute processes. Retries for
   `PREFLIGHT_WAIT` (60 s default) before failing — back-to-back runs briefly see
   their own predecessor's draining ranks (this self-collision aborted the first
   evaluation's w64 block). `FORCE=1` overrides.
2. Evaluations run inside an **exclusive slurm allocation**
   (`eval_sbatch.sh` in the scaling-inference repo: `-N16 --exclusive`, tray03/16
   excluded, tray list derived from `$SLURM_JOB_NODELIST`), so slurm arbitrates
   between benchmarks and serving jobs; the per-run guard stays as belt-and-braces.

## Alternatives considered

- **Idle-gap watchers** (poll `squeue`, fire when quiet) — rejected: ablation cadences
  reopen the race; we lost a w64 block to exactly this.
- **Guard only, no slurm** — rejected: the guard prevents contamination but converts
  collisions into lost windows; arbitration needs a queue.
- **Trusting run output sanity checks** — rejected: contaminated numbers passed
  eyeball review; only the timestamp audit caught them.

## Consequences

- Every result directory pairs logs with `.meta` sidecars (timestamp, trays, NCCL
  version, plugin SHA) so a timestamp audit is always possible after the fact.
- Quarantined data is retained (`results/contaminated_*/` in the scaling-inference
  repo), never deleted: refuted conclusions stay traceable to their bad inputs.

## References

- Landed via [datacrunch-research/nccl-eBPF#1](https://github.com/datacrunch-research/nccl-eBPF/pull/1).
