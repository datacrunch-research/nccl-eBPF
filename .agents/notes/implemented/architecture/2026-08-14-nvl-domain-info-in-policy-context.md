# Agent Note: NVL domain info in the policy context

Status: implemented

## Problem

The tuner-v5 init hands the plugin `ncclNvlDomainInfo_v5_t` (`nNvlDomains`,
`min/maxRanksPerNvlDomain`) — exactly the topology a rackscale policy needs to know it
is running inside one NVL72 fabric domain rather than a multi-domain (IB-connected)
job. `pluginInitImpl` discarded it (`(void)nvl_domain_info`), so no eBPF policy could
condition on domain shape, and a policy written for the single-domain rack could
misfire on any other deployment.

## Decision

Snapshot the three fields in `TunerContext` at init and expose them to policies as
`n_nvl_domains` / `min_ranks_per_nvl_domain` / `max_ranks_per_nvl_domain` in
`struct nccl_policy_ctx`, appended **after** the existing `reserved` field, plus a new
`reserved2`. All-zero means "not provided" (pre-v5 NCCL), so absence is distinguishable
from a real domain count. Verified live on GB300: NCCL 2.29.7 populates
`nNvlDomains=1` for an 8-rank cross-tray MNNVL communicator.

## Alternatives considered

- **New context struct / ABI version bump** — rejected: every existing `.bpf.o` would
  need recompiling; appending after `reserved` keeps all prior field offsets, and the
  PREVAIL context bounds admit the larger struct, so old programs run unmodified
  (harness stayed 24/24 after the change).
- **Passing a pointer to the NCCL struct into the program** — rejected: the verifier
  would need a new pointer type and bounds; copying three u32s is simpler and safer.

## Consequences

- Policies can guard on domain shape; see the
  [nvl72_size_aware note](../feature/2026-08-14-nvl72-size-aware-policy.md).
- The ABI append pattern is now the precedent for future context growth: extend after
  the last field, never reorder, zero means unknown.

## References

- Landed via [datacrunch-research/nccl-eBPF#1](https://github.com/datacrunch-research/nccl-eBPF/pull/1).
