# NCCLbpf on GB300: M0/M1 technical report

2026-08-14 · milestones M0 (aarch64 port + CPU harness) and M1 (single-tray GPU repro) of the NCCLbpf reproduction plan.
Work branch: [datacrunch-research/nccl-eBPF#1](https://github.com/datacrunch-research/nccl-eBPF/pull/1) (`gb300-aarch64`). Living doc: [mnnvl_eval.md](mnnvl_eval.md). Paper: arXiv:2603.11438 [1].

## 1. Context

We are evaluating eBPF as an extension and observability mechanism for GPU communication. Vehicle: reproduce the NCCLbpf paper's evaluation, then scale it from the authors' single node (8x B300, NVL18) to our GB300 NVL72 rack over MNNVL. The paper names multi-node validation as its missing piece. M0/M1 establish the port and the single-tray baseline; M2/M3 (rackscale + NVL72-aware policy) are queued behind rack availability.

## 2. What NCCLbpf actually is

Not kernel eBPF, not tracing. It is a NCCL **tuner v5 + profiler v6 plugin** embedding **bpftime** (userspace eBPF: LLVM JIT, PREVAIL verifier, maps). NCCL calls `getCollInfo` before each collective; a verified eBPF program runs synchronously and can force algo/proto (by zeroing cost-table entries) and channel counts. `SEC("uprobe")` is cosmetic — nothing is attached. No root needed; per-process load via `NCCL_TUNER_PLUGIN`. It sits above the transport, so it sees every MNNVL collective — where kernel-side eBPF sees nothing, because NVLink data movement never enters the kernel. Limits found in code review: overrides silently no-op on IGNORE-marked algo/proto cells; rank-local adaptive policies can diverge across ranks (hang risk; the paper's taint analysis is not implemented); no telemetry export path exists.

## 3. Testbed

| | paper | ours |
|---|---|---|
| GPUs | 8x B300 SXM6, single node, NVL18 | GB300 NVL72: 18 trays x 4, one MNNVL domain |
| CPU | EPYC 9575F (x86-64) | Grace (aarch64, 144c, 64 KB pages) |
| CUDA / NCCL | 13.0 / 2.29.7 | 13.1 / 2.29.7 (source build, sm_103) |
| Launch | local mpirun | mpirun -H over ssh, 4 ranks/tray, OMPI iface pinned |

All runs pin `LD_LIBRARY_PATH` to the source-built NCCL: the rack's system NCCL is heterogeneous (16 trays at 2.27.7, tray01 at 2.29.3) and 2.27 predates the tuner-v5 ABI.

## 4. M0 — first aarch64 build, paper harness on Grace

The stack had never built on arm64 (CI is amd64-only). It works. Four mundane fixes: `libboost-dev` + `libboost-program-options-dev` (bpftime shm / PREVAIL conformance), `libyaml-cpp-dev` (PREVAIL), and the bare `clang` meta-package (bpftime's daemon target). Frida devkit auto-resolves linux-arm64; 64 KB pages caused zero issues. Two repo bugs fixed on the branch: 8 of 24 policies missing from CMake (including the paper's headline `nvlink_ring_mid_v2`), and `llvm-config-15`-only lookup.

`test_ebpf_plugin` — the harness behind the paper's Tables 1, safety, and hot-reload sections — passes end to end:

| metric | paper (EPYC) | ours (Grace) |
|---|---|---|
| verifier matrix | 14/14 | 15/15 |
| hot reload: swap / full load | 1.07 us / 9.4 ms | 1.12 us / 10.4 ms |
| hot reload lost calls | 0 / 400k | 0 / 400k |
| adaptive contention curve | 12→2→12 channels | identical |
| getCollInfo Δ P50: noop | +80 ns | +128 ns |
| map lookup / lookup+update | +110 / +120 ns | +192 / +224 ns |
| slo_enforcer | +130 ns | +256 ns |

Grace dispatch is ~1.6-2x EPYC, quantized to the 32 ns generic-timer step. Same cost structure (base + per-map-op). At collective timescales (>=30 us) both are noise.

## 5. M1 — single tray, 4x GB300

AllReduce 8B-8GiB, 20 iters, 5 reps/arm, plus one `-c 1` correctness run (passed). Bus BW GB/s, out-of-place means:

| size | default | noop | env Ring | env Tree | env NVLS | nvlink_ring_mid_v2 | size_aware_v5 |
|---|---|---|---|---|---|---|---|
| 4M | 153.9 | 153.4 | 153.7 | 108.3 | 103.4 | **188.3 (+22%)** | 100.6 (-35%) |
| 8M | 280.8 | 280.1 | 281.0 | 172.4 | 172.4 | 280.2 | 140.6 (-50%) |
| 16M | 250.3 | 250.7 | 250.5 | 240.8 | 212.0 | **359.0 (+43%)** | 248.7 |
| 32M | 407.2 | 406.9 | 407.3 | 274.3 | 247.4 | **439.9 (+8%)** | 407.1 |
| 128M | 594.6 | 594.6 | 594.7 | 395.2 | 415.6 | 593.4 | 595.0 |
| 8G | 683.9 | 683.8 | 684.2 | 544.1 | 678.1 | 683.4 | 684.1 |

Findings:

1. **Noop overhead is zero at every size.** At >=4M, within noise. On the small-message sweep (8B-256K, 200 iters, 5 reps): delta <=0.8 us, sign-alternating — statistically zero. The paper reported +1.3 us (+4%) fixed cost at small sizes on 8x B300; on 4x GB300 (base ~14-17 us) no fixed cost is resolvable.
2. **Peak parity**: 684 GB/s @8G vs the paper's 4-GPU 682.
3. **The paper's exploit does not transfer to w4**: their default rode NVLS everywhere; ours already picks Ring mid-range, and NVLS never wins below 1G at 4 GPUs.
4. **A different exploit exists and the same policy catches it**: forcing Ring/**LL128** at 4-32M beats the default protocol choice by +22% @4M and +43% @16M. The win survives; the mechanism moved from algorithm to protocol.
5. **Policies do not transfer across topologies**: the authors' `size_aware_v5` loses 35-50% here. Verified-safe ≠ performant — the case for measurement-derived, scale-guarded policies (M3's `nvl72_size_aware`).
6. **Tuner forcing == env forcing** where comparable (within ~0.3%): the cost-table mechanism is faithful.

## 6. Early M2 signal (clean data)

First multi-tray validation of the plugin anywhere: 8 GPUs across 2 trays over MNNVL (still one NCCL "node" — the whole NVL72 rack is a single NVLink domain), all ranks load and run it. The paper's headline **returns at w8, larger than the original** (+42% vs their +27% peak). Side by side, AllReduce busbw GB/s, policy = `nvl72_size_aware` (w8) / `nvlink_ring_mid_v2` (w4):

| size | w4 default | w4 policy | Δ | w8 default | w8 policy | Δ | regime |
|---|---|---|---|---|---|---|---|
| 4M | 153.9 | 188.3 | +22% | 128.1 | 153.5 | +20% | proto window opens |
| 8M | 280.8 | 280.2 | 0% | 192.3 | 272.4 | +42% | w4 default guesses LL128 right; w8 does not |
| 16M | 250.3 | 359.0 | +43% | 276.3 | 372.2 | +35% | heart of the window |
| 32M | 407.2 | 439.9 | +8% | 344.7 | 449.5 | +30% | still open |
| 64M | 558.3 | 557.4 | 0% | 420.9 | 458.6 | +9% | closing at w4, open at w8 |
| 128M | 594.6 | 593.4 | 0% | 595.1 | 627.9 | +5.5% | tail |
| >=256M | = | = | 0% | = | = | 0% | NVLS wins; policy stands down |

Reading the default columns vertically shows the physics: 4M busbw drops w4→w8 (latency-bound), 8G rises 684→836 (bandwidth-bound); the exploit lives between the regimes. At w4 the default's algorithm is right and only its protocol choice is wrong; at w8 its cost model swings to NVLS ~64x too early in message size (measured crossover ~256M). Hypothesis for w16-w64: the window narrows as Ring's per-hop latency grows — no static policy is right twice, which is exactly the case for measured, per-scale, hot-swappable eBPF policies.

## 7. Operational findings along the way

- **tray03 GPU1 has NVLink down and NCCL degrades 12x silently** (2 channels, 0 NVLS, 51 vs 685 GB/s, zero warnings). Full root cause with file:line chain, a fault-injection repro on healthy trays, and a draft upstream WARN patch: [`.agents/debug/2026-08-14-mnnvl-channel-collapse/report.md`](../../.agents/debug/2026-08-14-mnnvl-channel-collapse/report.md). NCCL 2.31.2's remote-device-type discovery is the related upstream fix.
- **Multi-tenant collision**: a colleague's 16-tray job started mid-sweep and silently turned our w16/w32 numbers into plausible garbage; caught by timestamp audit, 109 runs quarantined, and the runner now refuses busy trays (`preflight_busy_check`).
- `NCCL_ALGO=NVLSTree` is invalid inside a single NVL domain — the rack is one clique (`nNodes=1`), inter-domain algorithms never apply.

## 8. Reproduce

```bash
git clone -b gb300-aarch64 https://github.com/datacrunch-research/nccl-eBPF.git
# build: see docs/gb300/mnnvl_eval.md "Build on aarch64" (apt deps -> bpftime -> NCCL -> plugin)
# CPU harness: src/nccl-policy-plugin/build/test_ebpf_plugin
# GPU arm:
TRAYS=1 ARM=policy:nvlink_ring_mid_v2 NCCL_LIB_DIR=$HOME/nccl-eBPF/nccl/build/lib \
  scripts/nccl_bench.sh run   # in this repo
```

Raw logs + meta sidecars live in the scaling-inference repo: `gb300/scripts/collectives/results/` (quarantined runs kept in `contaminated_20260814_job47646/`); its `report.py` renders `results/report.html`. The runner/parser are mirrored in this repo under `scripts/`.

## 9. Next

M2: clean w8-env/w16/w32/w64 ladders, small-message overhead vs world size, AllGather variance, divergence-hazard demo (campaign scripted, waiting on a rack window behind a colleague's wideEP ablations). M3: validate `nvl72_size_aware` per-scale windows on clean data; hot-swap demo at max world.

## References

[1] Zheng et al. "NCCLbpf: Verified, Composable Policy Execution for GPU Collective Communication." eBPF '26 Workshop, 2026. arXiv:2603.11438.
[2] Zheng et al. "bpftime: Userspace eBPF Runtime for Fast Uprobe and Syscall Hook." 2025.
[3] Gershuni et al. "Simple and Precise Static Analysis of Untrusted Linux Kernel Extensions." PLDI, 2019.
