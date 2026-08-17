# Journal: NCCLbpf rackscale campaign (M0-M3)

Session: 2026-08-14, agent: Claude (Fable 5) + SOL (codex gpt-5.6-sol) as parallel
debugger. Written retrospectively 2026-08-17 when the journal convention was adopted;
times UTC, from run metadata and slurm accounting.

## Outcome in one line

Paper reproduced and scaled: the tuner exploit is +43% at w4, +42% at w8 over MNNVL,
and **gone from w16 up**; the guarded `nvl72_size_aware` policy captures the window and
is measured-inert everywhere else; two silently degraded GPUs found on the way.

## Timeline

- **~11:20** M0 build. bpftime had never built on arm64. Four dependency fixes
  (boost, boost-program-options, yaml-cpp, bare `clang` meta-pkg); Frida devkit
  resolves linux-arm64; 64K pages a non-issue. Paper harness passes end-to-end
  (verifier 15/15, hot-reload 1.12us/0 lost, contention curve identical).
- **~11:30** Plugin loads on GPU against system NCCL 2.29.3 (`TUNER/Plugin: Using
  eBPFPolicy (v5)`); 590 GB/s @128M with checks on.
- **~11:45** First 2-tray run: `ncclCommQueryProperties` missing on remote trays —
  **the rack's system NCCL is heterogeneous** (16x 2.27.7, tray01 2.29.3). Decision:
  pin every run to the shared-FS source-built 2.29.7 (paper's version); restart M1.
- **~11:50** 2-tray mpirun hangs in init: OMPI OOB/BTL crossing VLANs. Fix: `--mca
  {oob,btl}_tcp_if_include bond0.225`, baked into the runner.
- **~12:00** 2-tray = 51 GB/s, `2 coll / 0 nvls channels`. Topology dump shows tray03
  GPU1 with zero `<nvlink>` entries. SOL launched in parallel on the source trace;
  healthy trays 06+07 give 685 GB/s -> hardware, not NCCL. Dossier:
  [../debug/2026-08-14-mnnvl-channel-collapse/report.md](../debug/2026-08-14-mnnvl-channel-collapse/report.md).
- **12:09** A colleague's 16-tray serving job starts mid-sweep. Not noticed until
  garbage w16/w32 "findings" (fake instability, fake +78%) prompted a timestamp audit
  against `scontrol`. **109 runs quarantined; two conclusions retracted before
  publication.** Guard added (per-run GPU-occupancy preflight), then the structural
  fix: campaigns run as exclusive slurm allocations. Note:
  [../notes/implemented/process/2026-08-14-multi-tenant-benchmark-arbitration.md](../notes/implemented/process/2026-08-14-multi-tenant-benchmark-arbitration.md).
  (Also: an overly broad `pkill -f` killed SOL's first process — match exact binaries
  on the specific host only.)
- **14:18-15:07** Clean campaign (slurm 47899, 179 runs): w8/w16/w32 x AR/AG/A2A +
  small-message + variance + divergence. Headline: **the window closes with scale**;
  w16/w32 candidate policy windows (fit on the contaminated data) formally refuted.
  Policy refit: AllReduce-only + single-NVL-domain + r<=8. Note:
  [../notes/implemented/feature/2026-08-14-nvl72-size-aware-policy.md](../notes/implemented/feature/2026-08-14-nvl72-size-aware-policy.md).
  Divergence hazard precisified: adaptive policies are rank-uniform until the
  profiler feeds asymmetric telemetry — no hang under benchmark conditions.
  w64 block self-aborted: preflight tripped on our own draining w32 ranks ->
  settle-retry added.
- **15:58-16:46** w64 rerun (47925): pathological (208 GB/s @8G, 0 nvls). Diagnostic
  job 47967: pair tests + instrumented topo dump + 12-tray control -> **tray14 GPU3,
  zero NVLinks, quieter variant** (no smi hang, 12ch instead of 1x2 fallback); no
  NCCL-at-16-hosts issue. Rack: trays 03/14/16 excluded; max = w60.
- **17:30** w60 (47970, 45 runs): 689 GB/s @8G, default==NVLS everywhere, policy
  inert-by-design and measured inert; AG/A2A unperturbed; env-NVLS AllGather invalid
  at 60 ranks (non-power-of-2). M2/M3 closed. Final analysis:
  [../../docs/gb300/mnnvl_eval.md](../../docs/gb300/mnnvl_eval.md).
- **Security pass** (same evening): the dossier's logs/XMLs had gone to this public
  fork with internal IPs, NIC GUIDs, fabric UUIDs, host hashes, one username. Scrubbed
  and the branch **squash-recreated + force-pushed** so no reachable commit ever
  carried them. Note:
  [../notes/implemented/process/2026-08-14-public-fork-redaction-and-history-rewrite.md](../notes/implemented/process/2026-08-14-public-fork-redaction-and-history-rewrite.md).

## Dead ends / retracted

- w16 "default tuner instability" and "+78% @32M w32": contamination artifacts,
  retracted. Never trust mid-campaign anomalies without a scheduler-log timestamp audit.
- Idle-gap watchers for rack time: lost the race twice; slurm arbitration is the answer.
- git-lfs for logs: GitHub rejects LFS uploads to public forks; plain git (~2 MB).
- My nvidia-smi nvlink grep as a health probe: wrong pattern, and tray14 proves smi
  behavior is not a reliable signal anyway — use the fused-topo/channel-line recipe.

## Where everything lives

Fork PR #1 (datacrunch-research/nccl-eBPF, branch `gb300-aarch64`): code, policies,
scripts mirror, docs, this workspace. Raw benchmark data + report.html: scaling-inference
(private) `gb300/scripts/collectives/results/`. Campaign orchestration:
`campaign_sbatch.sh` / `w60_sbatch.sh` / `w64_diag_sbatch.sh` there too.

## Open threads (not started)

Upstream PRs to eunomia-bpf (missing policies, NVL-domain ctx, arm64 CI); NCCL WARN
patch to NVIDIA; tray03/tray14 maintenance + re-validation; profiler-mode deep dive
(M4 stretch); Mooncake gap sketch expansion.
