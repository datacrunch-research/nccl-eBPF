# Agent Note: upstream contributions from the GB300 campaign

Status: proposed

## Problem

The rackscale campaign produced fixes and findings whose natural home is upstream, not
this fork. Left only here, they rot as the fork drifts: eunomia-bpf/nccl-eBPF keeps
shipping a CMake that omits its own paper's headline policy, NCCL keeps degrading 12x
silently when a clique GPU loses NVLink, and the arm64 build path stays unvalidated
upstream while this fork quietly depends on it.

## Proposal

Three tracks, independent, in priority order:

1. **eunomia-bpf/nccl-eBPF PRs** (small, mechanical, low review risk):
   a. register the 8 missing policy objects in CMake (incl. `nvlink_ring_mid_v2`);
   b. expose `ncclNvlDomainInfo_v5_t` in the policy context (ABI-append, see the
      [implemented note](../../implemented/architecture/2026-08-14-nvl-domain-info-in-policy-context.md));
   c. `llvm-config-18` lookup fix + the four arm64 dependency notes, ideally with an
      arm64 CI job;
   d. offer the multi-tray benchmark runner (`scripts/`) and the w8-window /
      window-closes measurements as evaluation material — this is the multi-node
      validation their paper's Discussion names as missing.
2. **NCCL (NVIDIA) issue + patch**: the silent NVLS/channel collapse when a fused-clique
   GPU has zero NVLinks; draft WARN patch and reproducible fault injection in the
   [dossier](../../../debug/2026-08-14-mnnvl-channel-collapse/report.md). Reference
   2.31.2's remote-device-type discovery as the related upstream change.
3. **Rack operations**: maintenance ticket for tray03 GPU1 / tray14 GPU3 (driver/host
   reset, then re-validate with the fused-topo dump recipe until `count="18"` returns);
   re-run the w60 block as w64/w72 when trays return.

## Open questions

- Whether to upstream the `nvl72_size_aware` policy itself: it is site-derived; the
  mechanism (guards + measured windows) generalizes, the thresholds do not. Likely
  ship as an example with a "derive your own windows" README rather than as defaults.
- Timing of (1d) relative to the owners' eBPF '26 camera-ready.
