# NCCLbpf on GB300 NVL72: eBPF policies for NCCL over MNNVL

Status: complete (2026-08-14). M0-M3 done: aarch64 port + paper harness, single-tray repro, rackscale ladder w8/w16/w32/w60 (AllReduce + AllGather + AlltoAll), and the guarded `nvl72_size_aware` policy validated at every scale.

We reproduce the evaluation of NCCLbpf [1] and scale it from the paper's single-node 8x B300 to our GB300 NVL72 rack. Goal: measure the value of eBPF as an extension and observability mechanism for GPU communication, focused on the MNNVL scale-up domain. The paper names our exact setting as its missing validation: "multi-node experiments with larger rank counts are needed."

## What NCCLbpf actually is

Not kernel eBPF. Not tracing. It is a policy plane: a NCCL tuner (v5) + profiler (v6) plugin that embeds bpftime [2] as a userspace eBPF runtime (LLVM JIT + PREVAIL verifier [3] + maps). NCCL calls the plugin's `getCollInfo` before each collective. The plugin runs a verified eBPF program synchronously. The program can zero cost-table entries (forcing algo/proto) and set channel counts. `SEC("uprobe")` in the policies is only a section name the verifier accepts; no uprobe is ever attached.

Consequences for us:

- No root needed. Loading is per-process via `NCCL_TUNER_PLUGIN` (dlopen by NCCL itself).
- It sits above the transport. It sees every collective on MNNVL, where kernel-side eBPF (tc/XDP/kprobes on the net path) sees nothing: NVLink data movement never enters the kernel.
- The tuner override is best-effort. It only zeroes a cost cell if NCCL did not mark that algo/proto IGNORE (`plugin.cpp:1588`). Overrides can silently no-op.
- Policies that branch on rank-local state (`adaptive_*`, `slo_enforcer`, `cpu_aware`) can diverge across ranks and hang a collective. Rank-uniform-safe set: `noop`, `size_aware*`, `ring_simple_all`, `nvlink_ring_mid*`. The taint analysis the paper describes for this is not implemented in the repo.
- There is no telemetry export path (no ringbuf/CSV/prometheus). Maps live in per-PID POSIX shm.
- No NIXL code exists in the repo, despite GPU-comms framing (see gap note at the end).

## Testbed

| | paper [1] | ours |
|---|---|---|
| GPUs | 8x B300 SXM6 (NVL18, single node) | GB300 NVL72 rack: 18 trays x 4 GPUs, one MNNVL fabric domain |
| CPU | AMD EPYC 9575F (x86-64) | Grace (aarch64, 144 cores/tray, 64 KB pages) |
| CUDA / NCCL | 13.0 / 2.29.7 | 13.1 / 2.29.7 (built from source, sm_103) |
| eBPF runtime | bpftime @8a359fc | same, first arm64 build |
| Launch | local mpirun | OpenMPI 4.1.6 over ssh, `-H trayXX:4`, 4 ranks/tray, `-g 1` |

Excluded trays: 16 (down), 03 (GPU1 NVLink degraded, found during this work — see incidents). Max healthy world: 64 GPUs.

All runs pin `LD_LIBRARY_PATH` to the source-built NCCL 2.29.7. Reason: the rack's system NCCL is heterogeneous (16 trays at 2.27.7, tray01 at 2.29.3), and 2.27 predates the tuner-v5 ABI.

## Build on aarch64 (M0)

The stack had never been built on arm64 (CI is amd64-only). It works. Four fixes, all mundane:

```bash
sudo apt install clang-18 llvm-18-dev libclang-18-dev libbpf-dev \
                 libboost-dev libboost-program-options-dev libboost-filesystem-dev \
                 libyaml-cpp-dev clang llvm   # bare `clang` needed by bpftime's daemon target

cd ~/nccl-eBPF && git checkout gb300-aarch64   # our branch; submodules pinned
git submodule update --init --recursive

# bpftime static libs (two passes, per docs/tmp/bpftime-migration-results.md)
mkdir -p build-bpftime && cd build-bpftime
cmake ../bpftime -DCMAKE_BUILD_TYPE=Release -DBPFTIME_ENABLE_UNIT_TESTING=OFF && make -j32
cmake ../bpftime -DCMAKE_BUILD_TYPE=Release -DBPFTIME_ENABLE_UNIT_TESTING=OFF \
      -DENABLE_EBPF_VERIFIER=ON && make -j32 bpftime-verifier runtime

# NCCL 2.29.7 (also provides plugin headers not shipped in libnccl-dev)
cd ../nccl && make -j32 src.build NVCC_GENCODE="-gencode=arch=compute_103,code=sm_103" \
      CUDA_HOME=/usr/local/cuda-13.1

# plugin + policies
cd ../src/nccl-policy-plugin && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DLLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config-18 \
      -DCUDAToolkit_ROOT=/usr/local/cuda-13.1 && make -j16
```

Notes. Frida devkit auto-downloads a linux-arm64 build. 64 KB pages caused no issues. The repo's CMake only registers 16 of 24 policies; we added the missing 8 on our branch (including the paper's headline `nvlink_ring_mid_v2`). CMake looks for `llvm-config-15|llvm-config`, hence the explicit `-DLLVM_CONFIG_EXECUTABLE`. On "never built on arm64" vs the shipped `vmlinux/arm64/*.h`: those template-inherited headers serve only the optional kernel_observer (kernel-side CO-RE types), and even there the default include path resolves to the x86 header (`vmlinux/vmlinux.h -> x86/vmlinux.h`, Makefile fallback ignores its own `$(ARCH)`); the userspace plugin path never uses vmlinux.h at all. The arch risk was the bpftime JIT/FridaGum/shm stack — amd64-only in CI — which is what M0 validated, with zero plugin source changes.

### M0 results: paper harness on Grace

`test_ebpf_plugin` (the actual harness behind the paper's Tables 1 and safety/hot-reload sections) passes end to end. exit 0.

| metric | paper (EPYC 9575F) | ours (Grace) |
|---|---|---|
| verifier matrix | 14/14 | 15/15 (repo gained one case) |
| hot reload swap / load | 1.07 us / 9.4 ms | 1.12 us / 10.4 ms |
| hot reload lost calls | 0 / 400k | 0 / 400k |
| adaptive contention curve | 12 -> 2 -> 12 channels | identical |
| getCollInfo delta P50: noop | +80 ns | +128 ns |
| lookup_only / lookup_update | +110 / +120 ns | +192 / +224 ns |
| slo_enforcer | +130 ns | +256 ns |

Grace plugin-dispatch overhead is ~1.6-2x EPYC. All Grace timings quantize to 32 ns (generic timer resolution). Structure matches the paper: fixed base + per-map-op increments. At GPU collective timescales (>=30 us) both are noise.

## M1: single tray, 4x GB300

AllReduce 8B-8GiB, 20 iters, 5 reps per arm, `-c 0` (one `-c 1` sanity run passed). Bus BW GB/s, out-of-place, mean over reps:

| size | baseline | noop | env Ring | env Tree | env NVLS | nvlink_ring_mid_v2 | size_aware_v5 |
|---|---|---|---|---|---|---|---|
| 4M | 153.9 | 153.4 | 153.7 | 108.3 | 103.4 | **188.3 (+22%)** | 100.6 (-35%) |
| 8M | 280.8 | 280.1 | 281.0 | 172.4 | 172.4 | 280.2 | 140.6 (-50%) |
| 16M | 250.3 | 250.7 | 250.5 | 240.8 | 212.0 | **359.0 (+43%)** | 248.7 |
| 32M | 407.2 | 406.9 | 407.3 | 274.3 | 247.4 | **439.9 (+8%)** | 407.1 |
| 64M | 558.3 | 558.4 | 558.9 | 301.1 | 389.3 | 557.4 | 556.3 |
| 128M | 594.6 | 594.6 | 594.7 | 395.2 | 415.6 | 593.4 | 595.0 |
| 512M | 638.2 | 638.3 | 638.3 | 495.0 | 462.1 | 639.1 | 639.1 |
| 8G | 683.9 | 683.8 | 684.2 | 544.1 | 678.1 | 683.4 | 684.1 |

Findings:

1. **Noop plugin overhead is zero — including at small messages.** Baseline and noop agree within run-to-run noise at every size >=4M. On the dedicated small-message sweep (8B-256K, 200 iters, 5 reps), the delta is <=0.8 us and alternates sign across sizes — statistically indistinguishable from zero. The paper measured +1.3 us (+4%) fixed cost on its 8x B300; on 4x GB300 (base small-message latency ~14-17 us) no fixed cost is resolvable.
2. **Peak parity with the paper's 4-GPU baseline.** 684 GB/s @8G vs their 682. Their 128M number was 577; ours 595.
3. **The paper's headline exploit does not transfer.** On 8x B300, NCCL's default rode NVLS everywhere and Ring beat it by 5-27% at 4-128M. On 4x GB300, the default already selects Ring in that range, and NVLS never wins below 1G. The specific inefficiency the paper's policy exploited is absent at this scale/topology.
4. **A different exploit exists, and the same policy catches part of it.** `nvlink_ring_mid_v2` forces Ring/LL128 at 4-32M. That protocol override beats the default protocol choice by +22% @4M and +43% @16M. So the win survives, but the mechanism shifted from algorithm choice (Ring vs NVLS) to protocol choice (LL128 vs default) — the default algo was already Ring.
5. **Policies do not transfer across topologies.** The repo's `size_aware_v5` (tuned for their box) costs -35% @4M and -50% @8M here. Verified-safe does not mean performant. This is the case for deriving policies from measurements per topology (M3).
6. **Tuner forcing == env forcing** where comparable (Ring/Simple at 64-192M: 557.4 vs 558.9 within noise). The cost-table mechanism works.

## Incidents found on the way (observability case studies)

These are the most instructive outputs so far for the "eBPF for GPU observability" question.

### tray03 GPU1: silent 12x collective slowdown

First 2-tray run (trays 02+03): 51 GB/s bus BW, "2 coll channels, 2 collnet channels, 0 nvls channels". Same lib, same shape on trays 06+07: 685 GB/s, "32 coll, 24 nvls". Root cause: in NCCL's fused MNNVL clique topology, tray03 GPU1 (busid 0009:01:00.0) contributes zero `<nvlink>` entries — a real hardware degradation (`nvidia-smi nvlink -s` hangs on that tray). **NCCL emits zero warnings while delivering 8% of expected bandwidth.**

The full mechanism (root-caused in a parallel SOL debugging session; complete report with per-step file:line in [`.agents/debug/2026-08-14-mnnvl-channel-collapse/report.md`](../../.agents/debug/2026-08-14-mnnvl-channel-collapse/report.md)):

1. Per-link NVML discovery drops links silently (`xml.cc:766-823`); tray03 GPU1 loses all 18.
2. The NVLS graph search requires NVS capacity to every GPU (`search.cc:387-417`); it returns zero channels, and the generic fallback excludes NVLS (`search.cc:1216`).
3. `init.cc:1255-1273` then clears the provisional `nvlsSupport` without a message.
4. The ring search cannot close a high-bandwidth ring; it takes the emergency fallback — one search channel at bwIntra=0.1, PATH_SYS (`search.cc:1216-1237`). Postset duplication (`connect.cc:436-453`) turns 1 into the observed 2 coll channels. The "2" is not derived from any bandwidth; it is the 1-channel fallback doubled.

Fault injection reproduces the exact control-plane signature on healthy trays (per-host `NCCL_TOPO_FILE` templates, one with `count="0"` for GPU1): `2 coll / 0 nvls`, `isAllDirectP2p 0`. Repro commands and retained topo/graph XMLs live next to the report. Two upstream artifacts came out of it: a draft NCCL WARN patch for both silent points (fused-clique NVLink audit + NVLS-clear message), and the observation that NCCL 2.31.2's discovery (`nvmlDeviceGetNvLinkRemoteDeviceType`, commits `aed770bd`/`716dce1c`/`598c836f`) removes the remote-PCI dependency for NVSwitch endpoints — a backport candidate, though tray03 itself is sick hardware, so a topology override that models it healthy is *not* an acceptable workaround there.

Detection today requires reading `NCCL_DEBUG=INFO` channel lines or dumping topo XML and counting nvlink entries per GPU. A rack health probe doing exactly that (fused-clique nvlink audit) would catch this class in seconds.

### Rack NCCL heterogeneity

16 trays at system NCCL 2.27.7; tray01 at 2.29.3. A binary built on tray01 dies on other trays (`ncclCommQueryProperties` missing), and 2.27 cannot load tuner-v5 plugins at all. Pinning a shared-FS source build is the fix, and should be the default for any rack-wide benchmark.

### OpenMPI on multi-homed trays

`mpirun` across trays hangs in init unless OOB/BTL TCP is pinned: `--mca oob_tcp_if_include bond0.225 --mca btl_tcp_if_include bond0.225`. The many interfaces (bonds, IB, docker) otherwise cross-connect and drop peers.

### Coda: the guard worked, and slurm is the real answer

The rerun evaluation hit the same pattern in reverse: a fresh ablation job claimed the rack four minutes in. This time `preflight_busy_check` aborted every affected run — 32 clean runs kept, zero polluted, versus 109 quarantined the first time. Permanent fix: evaluations now go through `evaluation_sbatch.sh` — an exclusive slurm allocation (tray03/16 excluded) that derives the tray list from `$SLURM_JOB_NODELIST`, so slurm arbitrates between our benchmarks and serving jobs instead of both sides racing free gaps.

### Multi-tenant collision: benchmarks silently invalidated

Mid-evaluation, a colleague's 16-tray serving job (slurm `glm52-ep`) started at 12:09:00 while our direct-mpirun sweeps were still running. Everything measured after that instant — w8 env-forced arms, the 40 AllGather variance runs, all w16/w32 sweeps — was contention noise, and it *looked* like signal: bimodal "instability", fake algo crossovers, a fake +78% policy win. A timestamp audit against `scontrol show job` caught it; 109 runs were quarantined (`results/contaminated_20260814_job47646/`) and two early conclusions retracted before publication. Direct mpirun bypasses slurm arbitration, so nothing warned either party. Fixes: `nccl_bench.sh` now refuses to start if any target tray has compute processes on its GPUs (`preflight_busy_check`, `FORCE=1` to override), and rack-idle is verified via `squeue` before evaluations. Lesson for the observability thesis: the failure mode wasn't slow runs, it was *plausible wrong numbers*.

### NVLSTree is invalid inside one NVL domain

Env-forcing `NCCL_ALGO=NVLSTree` fails with "invalid usage" at every multi-tray shape: the whole NVL72 rack is a single NVLink domain, NCCL models the clique as one node (`nNodes=1`), and inter-domain algorithms never apply. Worth knowing before sweeping algo matrices on NVL72.

## M2: rackscale ladder

Status: complete for w8/w16/w32 (exclusive slurm allocation, job 47899, 179 runs, 2026-08-14 14:18-15:07 UTC). The first 16-tray block (job 47925) produced a rack-scale repeat of the silent-collapse family — root-caused the same day to **a second degraded GPU: tray14 GPU3, zero NVLinks in the fused topology** (`rank=51, busid 0019:06:00.0`; pair test 82 GB/s and `12 coll / 0 nvls` vs ~700 GB/s healthy pairs; 12-tray control without it fully healthy at 48 ranks, so no NCCL-at-scale issue). Quieter variant than tray03: `nvidia-smi` does not hang and the graph keeps 12 channels instead of the 1x2 fallback; the invariant signature is `0 nvls channels` + channel count far below 32. Affected w64 data quarantined (`results/w64_tray14_incident/`); healthy maximum is now 15 trays and the max-scale block reran as **w60** (job 47970, 45 runs, clean). Full analysis in the [incident dossier addendum](../../.agents/debug/2026-08-14-mnnvl-channel-collapse/report.md).

### w60 (15 trays, max healthy scale): everything confirms

AllReduce: baseline == env NVLS at every size (86 GB/s @4M rising to **689 GB/s @8G**); Ring loses everywhere (down to -58% mid-range); the un-guarded `nvlink_ring_mid_v2` costs -66% @4M and -75% @64M where it fires. **`nvl72_size_aware` tracks baseline within run-to-run spread at every size** — the r>8 no-override branch is now validated at w16, w32, and w60. AllGather: default == Ring == policy (no headroom, ~496 GB/s @8G); env-forced NVLS AllGather is rejected as invalid usage at 60 ranks (non-power-of-2; it ran at 8/16/32 — the default simply never selects it there). AlltoAll: baseline == noop within noise (455 GB/s @8G). Small-message noop delta: mean -0.2 us over 16 sizes (single-size spread grows to ~3 us at 60 ranks, sign-alternating — still no resolvable fixed cost).

### Headline: the exploit closes with scale

AllReduce, default tuner vs best arm (busbw GB/s, 3 reps, exclusive allocation):

| size | w8 default | w8 Ring/pol | Δ | w16 default | w16 best-forced | Δ | w32 default | w32 best-forced | Δ |
|---|---|---|---|---|---|---|---|---|---|
| 4M | 129.1 | 153.1 | **+19%** | 104.6 | 102.0 (Ring) | -2% | 90.9 | 77.8 (Ring) | -14% |
| 8M | 192.0 | 272.5 | **+42%** | 157.0 | 127.7 | -19% | 137.7 | 107.3 | -22% |
| 16M | 275.4 | 370.9 | **+35%** | 240.1 | 230.3 | -4% | 193.6 | 135.4 | -30% |
| 32M | 344.7 | 449.3 | **+30%** | 303.1 | 292.6 | -3% | 273.5 | 255.6 | -7% |
| 64M | 420.1 | 456.3 | **+9%** | 382.6 | 342.6 | -10% | 337.6 | 316.6 | -6% |
| 128M | 595.3 | 628.2 | **+6%** | 440.1 | 430.1 | -2% | 409.4 | 363.7 | -11% |
| >=256M | = | = | 0% | = | = | <=0% | = | = | <=0% |

The w8 window (Ring/LL128 4-32M, Ring/Simple 64-192M) is real and large. By w16 it is gone: the default (which tracks NVLS) is within noise of the best arm at every size, and forcing Ring loses everywhere below 512M. At w32 forcing is strictly harmful. The earlier w16/w32 "candidate windows" (fit on contamination) are refuted by this clean data: forcing 8-32M at w16 costs -3..-10%. So the exploit is a narrow-world phenomenon on this fabric: protocol-only at w4, algorithm+protocol at w8, absent from w16 up — NCCL's default model is good at rack scale, bad at small cliques.

`nvl72_size_aware` behaved exactly per design: it captured the full w8 window (within 0.5 GB/s of the dedicated v2 policy) and tracked baseline within noise wherever it stands down (w16 64M: 383.1 vs 382.6; w32 8G: 680.2 vs 680.1). The transplanted `nvlink_ring_mid_v2` — no scale guard — loses up to -55% at w32.

### AllGather and AlltoAll (MoE patterns)

- AllGather: the default already picks the best algorithm (== env Ring) at every scale and size; env NVLS is substantially worse at large sizes (460 vs 674 GB/s @8G w8; 378 vs 505 @8G w32). No tuning headroom found.
- `nvlink_ring_mid_v2` fires on *all* collectives (no coll_type check) and its Ring/Simple band damages AllGather at scale: -18% @64M w16, **-47% @64M w32**. Policies must be collective-type-guarded, not just size/scale-guarded — folded into `nvl72_size_aware` (AllReduce-only).
- AlltoAll (grouped send/recv, the MoE dispatch path): baseline == noop within noise at w8/w16/w32 — the tuner plugin does not perturb the p2p path. Plateau ~468 GB/s busbw at w16/w32.

### Stability, overhead, divergence

- AllGather 128M @w8, 20 independent runs: default CV **6.4%** vs policy CV **3.7%** (means equal). Direction matches the paper (policy halves variance) but this fabric is ~40x noisier than their single node (0.15%) — cross-tray MNNVL run-to-run variance is its own finding.
- Small-message noop overhead stays zero at scale: mean delta +0.01 us (w8) / +0.09 us (w16) across 8B-256K.
- Divergence hazard, precisified: `adaptive_channels` at w8 did NOT hang — without the profiler attached there is no per-rank telemetry, all ranks compute identical actions (verified: same action word on every rank), and the policy is accidentally rank-uniform. The hazard requires asymmetric telemetry (profiler attached under real load) or rank-asymmetric call counts. The paper's unimplemented taint analysis would need to model exactly this.

Clean w8 result (2 trays, before the collision): the plugin runs multi-tray — the paper's stated missing validation — and **the paper's headline returns over MNNVL, larger than the original**. `nvlink_ring_mid_v2` vs default tuner, AllReduce busbw (3 reps):

| size | default | v2 policy | Δ |
|---|---|---|---|
| 4M | 128.6 | 153.9 | +20% |
| 8M | 191.7 | 272.2 | +42% |
| 16M | 275.6 | 371.0 | +35% |
| 32M | 344.9 | 450.1 | +31% |
| 64M | 420.8 | 458.4 | +9% |
| 128M | 595.2 | 628.0 | +6% |
| >=256M | — | — | 0% (no override) |

At w4 the default tuner already rides Ring mid-range and only the protocol trick pays (+22/+43% at 4M/16M); at w8 the default commits to NVLS too early and the full algo+proto window opens. The paper's 8-GPU number was +27% peak; over MNNVL we see +42%. Noop remains within noise of baseline at both scales (>=4M).

Open questions the rerun answers: noop fixed-overhead vs world size (small-message pairs), whether any Ring/LL128 window survives at w16-w64 (first, contaminated, sweep suggested it narrows to isolated sizes — unverified), variance under a live policy, and the rank-divergence hazard demo (`adaptive_channels` under timeout).

## M3: MNNVL-aware policy

Done: the plugin no longer discards `ncclNvlDomainInfo_v5_t` (`plugin.cpp:1599`); `nccl_policy_ctx` gains `n_nvl_domains` / `min_ranks_per_nvl_domain` / `max_ranks_per_nvl_domain` (appended after `reserved`; old programs keep offsets; harness still 24/24). New policy `nvl72_size_aware`: rank-uniform inputs only, acts only when `n_nvl_domains == 1`, per-scale windows.

Final form after the clean evaluation: overrides only for AllReduce (coll_type guard — the Ring band damages AllGather at scale), only inside one NVL domain, and only at r<=8 where the windows are measured (w4: Ring/LL128 4-32M; w8: + Ring/Simple 64-192M). Everything else: no override, and the evaluation proves that branch right — where it stands down it tracks the default within noise at w16/w32 at every size. Validated at w8: reproduces `nvlink_ring_mid_v2` within 0.5 GB/s (+19/+42/+35/+30% at 4-32M over the default); the `n_nvl_domains == 1` check returned true live, confirming NCCL populates the exposed struct. The punchline inverts the naive reading of the paper: at rack scale the durable value of the mechanism is not a permanent override, it is (a) the measurement loop that finds where windows exist, (b) hot-swappable scale/type-guarded deployment of them, and (c) zero-cost presence everywhere else.

### Reading the results: w4 vs w8, clean data (AllReduce busbw GB/s)

| size | w4 default | w4 policy | Δ | w8 default | w8 policy | Δ | regime |
|---|---|---|---|---|---|---|---|
| 4M | 153.9 | 188.3 | +22% | 128.1 | 153.5 | +20% | proto window opens |
| 8M | 280.8 | 280.2 | 0% | 192.3 | 272.4 | +42% | w4 default guesses LL128 right; w8 does not |
| 16M | 250.3 | 359.0 | +43% | 276.3 | 372.2 | +35% | heart of the window |
| 32M | 407.2 | 439.9 | +8% | 344.7 | 449.5 | +30% | still open |
| 64M | 558.3 | 557.4 | 0% | 420.9 | 458.6 | +9% | closing at w4, open at w8 |
| 128M | 594.6 | 593.4 | 0% | 595.1 | 627.9 | +5.5% | tail |
| >=256M | = | = | 0% | = | = | 0% | NVLS wins; policy stands down |

Read the default columns vertically: at 4M busbw *drops* w4->w8 (154->128, latency-bound — more ranks, more hops), at 8G it *rises* (684->836, bandwidth-bound — more aggregate links). The policy's window sits between the regimes. Interpretation: at w4 the default picks the right algorithm (Ring; baseline == env-Ring at every size) and only mis-picks the protocol at 4-32M (+43% from LL128). At w8, NCCL's cost model swings to NVLS far too early — measured crossover ~256M vs the model's ~4M — so the error becomes algorithm+protocol and the window widens. Expected unfolding at w16-w64 (hypothesis for the slurm evaluation): Ring's per-hop latency grows with ranks while NVLS multicast amortizes, so the window narrows and migrates — meaning no static policy is right twice, and the per-scale, hot-swappable eBPF policy is the mechanism that keeps up.

## Upstream findings (candidate issues/PRs for eunomia-bpf/nccl-eBPF)

1. 8 of 24 policies missing from CMake, including the paper's headline `nvlink_ring_mid_v2`.
2. Root `Makefile` references a nonexistent `src/Makefile`.
3. `vmlinux/vmlinux.h` symlinks to x86 despite an arm64 header sitting next to it.
4. Tuner discards `ncclNvlDomainInfo_v5_t` (`plugin.cpp:1599`) — exactly the MNNVL topology a rackscale policy needs.
5. kernel_observer CPU mask breaks above 32 CPUs (`1u << cpu` guarded to 0): misclassifies on any modern server (their own 240-core EPYC included).
6. `llvm-config-15|llvm-config` lookup fails on stock Ubuntu 24.04 (llvm-config-18).
7. 206 MB of committed x86-64 ELF binaries.
8. No telemetry export path from the profiler maps.
9. arm64: builds and passes the full harness with the four dependency fixes above (worth a CI target).
10. NCCL-side (for NVIDIA): silent NVLS/channel collapse when a clique GPU lacks NVLink (see incident above). Draft WARN patch (fused-clique NVLink audit in `topo.cc` + NVLS-clear message in `init.cc`) in [`.agents/debug/2026-08-14-mnnvl-channel-collapse/report.md`](../../.agents/debug/2026-08-14-mnnvl-channel-collapse/report.md); 2.31.2's remote-device-type discovery is the related upstream improvement.

## NIXL / Mooncake gap note

The paper's title scope ("GPU collective communication") covers NCCL only; the repo has no NIXL hooks. On this rack the KV-transfer plane is Mooncake TE over MNNVL (640 GB/s cross-tray), which is exactly the traffic NCCL plugins never see. The NCCLbpf pattern (verified policy programs at userspace extension points, above the kernel-bypassed transport) would map to Mooncake's transfer scheduling the same way the tuner maps to NCCL's algo choice. Deferred by scope decision (2026-08-14): NCCL first; sketch to be expanded after M2/M3.

## Reproduce

```bash
# one arm, one shape
TRAYS=2 TRAY_LIST="06 07" ARM=policy:nvlink_ring_mid_v2 \
  NCCL_LIB_DIR=$HOME/nccl-eBPF/nccl/build/lib scripts/nccl_bench.sh run   # in this repo

# arm matrix
TRAYS=8 ARMS="baseline noop algo:Ring algo:NVLS" REPS=3 ... nccl_bench.sh sweep

# parse everything to CSV
scripts/parse_nccl_log.py
```

Raw logs + meta sidecars live in the scaling-inference repo (`gb300/scripts/collectives/results/`, quarantine subdir included); the runner/parser are mirrored here under `scripts/`. CPU harness: `~/nccl-eBPF/src/nccl-policy-plugin/build/test_ebpf_plugin`. Plugin branch with all patches: `~/nccl-eBPF` @ `gb300-aarch64`.

## References

[1] Zheng et al. "NCCLbpf: Verified, Composable Policy Execution for GPU Collective Communication." eBPF '26 Workshop, 2026. arXiv:2603.11438.
[2] Zheng et al. "bpftime: Userspace eBPF Runtime for Fast Uprobe and Syscall Hook." 2025.
[3] Gershuni et al. "Simple and Precise Static Analysis of Untrusted Linux Kernel Extensions." PLDI, 2019.
[4] NVIDIA. "NCCL 2.29 Documentation: Tuner and Profiler Plugin Interfaces."
