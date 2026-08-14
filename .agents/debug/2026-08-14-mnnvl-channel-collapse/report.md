> Relocated 2026-08-14 from docs/tmp/ into .agents/debug/2026-08-14-mnnvl-channel-collapse/ (evidence XMLs now in topo/, logs in repro_results/); in-text docs/tmp paths refer to the old layout.

# Root-cause report: GB300 cross-tray MNNVL channel collapse

Date: 2026-08-14  
NCCL under test: 2.29.7+cuda13.1, git `36191590`  
Platform: GB300 NVL72, Grace-Blackwell/aarch64, kernel 6.14 nvidia-64k, driver 595.71.05  
GPU experiments performed only on trays 04 and 05.

## Executive conclusion

The collapse is caused by one malformed per-GPU topology record, not by a generic two-tray MNNVL bandwidth limit, the MNNVL clique merge, IMEX permissions, or multicast allocation.

The topology dumped by the failing tray02/tray03 run proves that tray03 GPU1—MPI rank 5, PCI `0009:06:00.0`—has **no NVLink/NVSwitch edge in NCCL's XML**, while each of the other seven GB300 GPUs has:

```xml
<nvlink target="fffffff:ff:ff.0" count="18" tclass="0x068000"/>
```

The malformed rank is visible at `/mnt/shared/home/dc/topo_2tray.xml:63-68`:

```xml
<pci busid="0009:06:00.0" ...>
  <gpu dev="1" sm="103" rank="5" gdr="1">
    <c2c bw="44712" count="5"/>
  </gpu>
</pci>
```

This is the only substantive GPU-connectivity difference between the failing `/mnt/shared/home/dc/topo_2tray.xml` and the healthy trays04/05 dump `docs/tmp/mnnvl_topo_2tray_0405.xml`. The latter contains `count="18"` for rank 5 at lines 65-67, as it does for the other seven GPUs.

NCCL 2.29.7 silently accepts that missing edge. Its MNNVL XML fusion then creates one shared NVSwitch node, but rank 5 is disconnected from it. The consequences are deterministic:

1. The NVLS search cannot find an NVS-to-every-GPU graph, so it returns zero channels.
2. The normal ring search cannot find a complete high-bandwidth ring and falls back to one 0.1 GB/s synthetic search channel.
3. NCCL's normal post-search duplication turns that one search channel into the reported two collective channels.
4. Because the NVLS graph contains zero channels, NCCL clears the provisional NVLS capability before attempting any multicast/IMEX allocation.

This causal chain was reproduced on healthy trays04/05 by removing only rank 5's NVSwitch edge: the result changed from `32 coll / 24 NVLS` to `2 coll / 0 NVLS`, with `isAllDirectP2p 0` and `isAllCudaP2p 1`, matching the supplied failure.

The immediate workaround is to set `NCCL_TOPO_FILE` to a healthy **single-tray** topology template containing all four `count="18"` NVSwitch edges. This was validated on trays04/05 and restored `32 coll / 24 NVLS` and 507.359 GB/s average bus bandwidth at 512 MiB.

## Confidence and boundary of the diagnosis

The NCCL-level root cause—rank 5's missing NVS edge—is proven directly by the failing topology dump and by controlled reproduction.

The lowest failing driver/NVML call cannot be distinguished from the existing artifacts alone. In NCCL 2.29.7, a link is silently discarded if any of these per-link checks fails:

- `nvmlDeviceGetNvLinkCapability(..., P2P_SUPPORTED)` fails or reports false;
- `NVML_FI_DEV_NVLINK_GET_STATE` fails or reports inactive; or
- `nvmlDeviceGetNvLinkRemotePciInfo` fails.

All 18 links were discarded for tray03 GPU1. The rest of that GPU's record is healthy: it is SM103, has all five C2C links, has `gdr="1"`, and reports the same completed MNNVL fabric UUID/clique/health mask as the other GPUs. This makes a per-device NVML enumeration defect or transient driver state much more likely than a physically absent NVLink fabric.

There is particularly strong circumstantial evidence for the remote-PCI query being the incompatible step: NCCL 2.31.2 changed discovery to query the NVLink remote **device type** first and, for `NVML_NVLINK_DEVICE_TYPE_SWITCH`, directly emits the NVSwitch sentinel without calling `nvmlDeviceGetNvLinkRemotePciInfo`. That exact change removes an unnecessary dependency on remote PCI enumeration for NVSwitch-connected GB300 GPUs. It is a strong upgrade/backport candidate, but it was not tested on the restricted tray03 GPU, so this report does not claim the exact failing NVML subcall as proven.

## Direct evidence

### 1. Failing topology dump isolates rank 5

`/mnt/shared/home/dc/topo_2tray.xml` was produced by the supplied tray02/tray03 `diag_topodump2` run. Its GPU NVSwitch link counts are:

| Host/rank | Device | PCI bus | NVSwitch edge |
|---|---:|---|---:|
| tray02/rank 0 | 0 | `0008:06:00.0` | 18 |
| tray02/rank 1 | 1 | `0009:06:00.0` | 18 |
| tray02/rank 2 | 2 | `0018:06:00.0` | 18 |
| tray02/rank 3 | 3 | `0019:06:00.0` | 18 |
| tray03/rank 4 | 0 | `0008:06:00.0` | 18 |
| **tray03/rank 5** | **1** | **`0009:06:00.0`** | **missing** |
| tray03/rank 6 | 2 | `0018:06:00.0` | 18 |
| tray03/rank 7 | 3 | `0019:06:00.0` | 18 |

The same file records `gdr="1"` and `<c2c bw="44712" count="5"/>` for rank 5. Therefore CUDA's GPUDirect-RDMA capability bit and the Grace-Blackwell C2C links are not the topology fields that gate the failed graph.

The earlier verbose `diag_2tray.log` has `GPU Direct RDMA Enabled ... (rank N)` messages 16 times each for ranks 0, 1, 2, 3, 4, 6, and 7, and zero times for rank 5. This is consistent with the malformed topology reducing rank 5's usable paths. It is not evidence that rank 5's CUDA GDR attribute is zero; the subsequent topology dump explicitly says `gdr="1"`.

### 2. Healthy two-tray fusion is correctly modeled

The fresh trays04/05 dump `docs/tmp/mnnvl_topo_2tray_0405.xml` contains two different host hashes and eight GPUs. All eight GPUs point to the same NVSwitch target with `count="18"`.

The healthy graph dump `docs/tmp/mnnvl_graph_2tray_0405.xml` contains:

- ring graph: 16 searched channels at 45 GB/s per channel, duplicated later to 32 collective channels;
- tree graph: 16 searched channels;
- NVLS graph: 8 heads at `speedintra=80`, `speedinter=90`, resulting in 24 configured NVLS channels on Blackwell.

The fresh no-topology-override trays04/05 control produced:

```text
MNNVL 1 cliqueSize 8
Check P2P Type isAllDirectP2p 1 directMode 0 isAllCudaP2p 1
32 coll channels, 32 collnet channels, 24 nvls channels,
32 p2p channels, 32 p2p channels per peer
# Avg bus bandwidth    : 506.781
```

This uses the same `libnccl.so` as the failing run. It proves that NCCL 2.29.7 can model and allocate cross-tray NVLS on this rack, and that the MNNVL merge itself does not impose a two-channel ceiling.

### 3. Controlled causal reproduction

A healthy single-tray template was supplied to tray04. For the tray05 MPI application context, only GPU1's edge was changed from `count="18"` to `count="0"`; all other topology fields were unchanged. Because NCCL trims each process's input XML to its managed GPU before MNNVL fusion, this made only tray05 GPU1/rank 5 disconnected from the shared NVS.

The result was:

```text
MNNVL 1 cliqueId d26 cliqueSize 8
Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 1
2 coll channels, 2 collnet channels, 0 nvls channels,
2 p2p channels, 2 p2p channels per peer
536870912 ... busbw 33.44 ... in-place busbw 32.98
# Avg bus bandwidth    : 33.2126
```

The synthetic throughput is not expected to equal the failing run's 51.6033 GB/s exactly: the synthetic zero-bandwidth XML can make an intra-host ring segment use SHM, while the real CUDA fabric remains reachable. The significant result is the exact graph/channel/direct-P2P signature.

A complementary experiment changed only GPU1's XML `gdr` attribute from 1 to 0 while retaining its `count="18"` NVS edge. It still built `32 coll / 24 NVLS` and achieved 498.429 GB/s. That separates the graph failure from the GDR attribute.

### 4. IMEX/multicast is downstream, not the failure

The failing logs show every process reporting provisional multicast support and `NVLS_NCHANNELS 24`, but they never show multicast-group creation, handle import, bind, or an IMEX/CUDA failure.

The healthy trays04/05 run does create one 8-rank multicast group and imports its shareable handle across trays. Therefore IMEX permissions and cross-host multicast work with the installed driver and `/dev/nvidia-caps-imex-channels/channel0` configuration.

## NCCL 2.29.7 source trace

Line references below are for the supplied source at git `36191590`.

### Per-rank NVLink discovery silently loses the links

`nccl/src/graph/topo.cc:1479-1486` detects only the GPU managed by the current process. MNNVL later obtains the other GPUs through XML all-gather and fusion.

`nccl/src/graph/xml.cc:766-823` performs NVLink discovery. For SM103 it tries 18 links (`:770`). Each link is silently skipped at:

- `:777-780` if the P2P capability query fails or is false;
- `:782-797` if the NVLink state field fails or is not enabled;
- `:799-801` if remote PCI information cannot be obtained.

Only a link that passes all three checks reaches `:812-821`, where NCCL creates or increments the `<nvlink>` element. No per-link warning or summary is emitted for the skip paths. The complete absence of an `<nvlink>` child for rank 5 means none of its 18 iterations reached `:812`.

### MNNVL fusion is correct and uses a shared NVS

`nccl/src/graph/topo.cc:1543-1579` changes the XML fusion group from same-host ranks to `comm->clique.ranks` when `comm->MNNVL` is true. Thus all eight independently discovered GPU records become one topology.

`nccl/src/graph/topo.cc:588-629` converts `<nvlink>` XML elements into topology links. For a switch-class target, `:617-620` creates one NVS node if needed and otherwise reuses the existing NVS. `:623-627` connects the GPU and NVS in both directions with `count * ncclTopoNVLinkBw(sm)`.

Consequently, the merger intentionally models a single NVS spanning both host hashes. There is no special low cross-host MNNVL bandwidth constant and no default PCI/SM-copy bandwidth substituted for healthy cross-host GPUs. The failing record simply leaves rank 5 without an NVS edge.

### Why NVLS becomes zero without an IMEX warning

`nccl/src/transport/nvls.cc:156-203` checks CUDA multicast support and assigns a provisional count. For a Blackwell communicator wholly contained in one MNNVL domain, `:181-200` chooses 24 channels. No multicast group is allocated at this stage.

`nccl/src/init.cc:1125-1132` invokes the NVLS topology search only after that provisional capability check.

`nccl/src/graph/search.cc:387-417` requires NVS-to-GPU and GPU-to-NVS capacity for **every GPU** before accepting an NVLS head/channel. A disconnected rank 5 makes that test fail.

The generic fallback at `nccl/src/graph/search.cc:1216-1238` explicitly excludes `NCCL_TOPO_PATTERN_NVLS`, so the NVLS graph remains at zero channels. Finally, `nccl/src/init.cc:1255-1273` takes the minimum graph result across ranks and clears `comm->nvlsSupport` and `comm->nvlsChannels` when the NVLS graph has zero channels.

The multicast allocation and IMEX import/bind code is therefore never reached. This is why toggling `NCCL_NVLS_ENABLE`, `NCCL_CUMEM_ENABLE`, or IMEX permissions cannot repair this failure and produces no allocation warning.

### Why the ring ends at two channels

With rank 5 disconnected from the NVS, the high-bandwidth ring search fails. `nccl/src/graph/search.cc:1216-1237` falls back to a simple rank order with one search channel, `bwIntra=0.1`, and `PATH_SYS`.

`nccl/src/graph/connect.cc:436-453` normally duplicates the searched ring/tree channels while connecting them. One fallback search channel therefore becomes the final two collective channels seen in the logs.

`NCCL_MAX_NCHANNELS` cannot fix this. `nccl/src/graph/connect.cc:323-353` defines it as an upper bound, and `:484-492` caps/copies already discovered channels after graph construction. It cannot create a missing NVS path or make the NVLS graph nonzero.

### Why `via P2P/MNNVL` does not contradict the diagnosis

MNNVL clique membership comes from NVML fabric UUID/clique information, independently of the XML NVS edge. `nccl/src/graph/paths.cc:394-412` checks that fabric identity.

For peers in a valid MNNVL clique, `nccl/src/graph/paths.cc:378-388` can report CUDA P2P as available even where NCCL's modeled topology path is not acceptable; the comment at `:386-387` explicitly says NCCL assumes CUDA P2P for MNNVL peers. Thus `isAllCudaP2p 1` and transport messages saying `via P2P/MNNVL` show that CUDA can use the fabric. They do not prove that the graph search received a complete NVS bandwidth model.

## Validated workaround

Use a known-good **single-tray** topology file containing all four GB300 NVSwitch edges:

```bash
export NCCL_TOPO_FILE=/mnt/shared/home/dc/topo_1tray.xml
```

That file was dumped from healthy tray02 and contains `count="18"` for devices 0-3. A single-node file is the correct form even for an MNNVL communicator: NVIDIA's `NCCL_TOPO_FILE` documentation explicitly notes that multi-node NVLink topology dumps contain the full domain, but a topology input should contain only the single-node topology. NCCL refreshes CPU host hashes at `nccl/src/graph/topo.cc:1461-1468`, overwrites the managed GPU's rank at `:1479-1486`, trims other branches, and then fuses one managed-GPU record from each clique rank.

Official documentation: [NCCL `NCCL_TOPO_FILE` and `NCCL_TOPO_DUMP_FILE`](https://github.com/NVIDIA/nccl/blob/master/docs/userguide/source/env.rst#nccl_topo_file).

For the provided runner, add the exported variable; `nccl_bench.sh` propagates all `NCCL_*` environment variables through `mpirun -x`:

```bash
cd /mnt/shared/home/dc/scaling-inference/gb300/scripts/collectives
TRAYS=2 \
TRAY_LIST="02 03" \
NCCL_TOPO_FILE=/mnt/shared/home/dc/topo_1tray.xml \
NCCL_LIB_DIR=/mnt/shared/home/dc/nccl-eBPF/nccl/build/lib \
MSG_MIN=512M MSG_MAX=512M ITERS=5 WARMUP=2 CHECK=0 \
ARM=baseline TAG=diag_2tray_topo_override \
./nccl_bench.sh run
```

That affected-host command was **not run**, because trays02/03 were explicitly excluded from GPU experiments.

### Exact validation on allowed trays04/05

The validation used an equivalent healthy single-tray template dumped from tray04, `docs/tmp/mnnvl_topo_1tray_04.xml`:

```bash
mpirun --bind-to none \
  -H pod4-gb300-3-tray04-f3:4,pod4-gb300-3-tray05-f3:4 \
  --mca oob_tcp_if_include bond0.225 \
  --mca btl_tcp_if_include bond0.225 \
  -x LD_LIBRARY_PATH=/mnt/shared/home/dc/nccl-eBPF/nccl/build/lib \
  -x NCCL_DEBUG=INFO \
  -x NCCL_MNNVL_ENABLE=1 \
  -x NCCL_NVLS_ENABLE=1 \
  -x NCCL_CUMEM_ENABLE=1 \
  -x NCCL_TOPO_FILE=/mnt/shared/home/dc/nccl-eBPF/docs/tmp/mnnvl_topo_1tray_04.xml \
  /mnt/shared/home/dc/nccl-tests/build_mpi/all_reduce_perf \
  -b 512M -e 512M -n 5 -w 2 -c 0 -g 1
```

Validated result:

```text
MNNVL 1 cliqueId d26 cliqueSize 8
Check P2P Type isAllDirectP2p 1 directMode 0 isAllCudaP2p 1
32 coll channels, 32 collnet channels, 24 nvls channels,
32 p2p channels, 32 p2p channels per peer
536870912 ... busbw 506.65 ... in-place busbw 508.07
# Avg bus bandwidth    : 507.359
```

This is essentially identical to the immediately preceding no-override healthy control (506.781 GB/s), so the template restores the graph without introducing a measurable penalty.

Later repeats were excluded from performance validation because an unrelated `openinfer` job started on trays04/05 at 12:09:46 UTC and subsequently held all eight GPUs at 99% utilization. Those processes were observed only; they were not modified or stopped.

## Permanent fix and upstream status

### Recommended operational fix

1. Apply the single-tray `NCCL_TOPO_FILE` override immediately for this uniform rack.
2. During a maintenance window, reset/restart the driver stack for tray03 GPU1/host and repeat `NCCL_TOPO_DUMP_FILE`. Do not declare the device recovered until rank 5 again has `count="18"`.
3. If the bad record recurs, collect the result of the three NVML operations named above for links 0-17 and file it as a driver/NVML issue. The fabric UUID/health state alone is insufficient because those fields were already healthy during the failure.

### NCCL version comparison

- 2.29.7 has the silent remote-PCI-dependent discovery shown at `nccl/src/graph/xml.cc:777-822`.
- 2.30.7 still performs `ncclNvmlDeviceGetNvLinkRemotePciInfo` before it can create the NVLink XML record; it does not contain the relevant repair.
- 2.31.2 changes the logic. In that tag's `src/graph/xml.cc:968-1019`, NCCL queries `nvmlDeviceGetNvLinkRemoteDeviceType`; a switch endpoint is mapped directly to the NVSwitch sentinel and class without a remote-PCI query. It also emits INFO messages for some remote-device failures.

The relevant NVIDIA upstream commits are:

- [`aed770bd`: add the NVLink remote-device-type NVML wrapper](https://github.com/NVIDIA/nccl/commit/aed770bd0764462b8b0b959dbc8935a54465b1bb)
- [`716dce1c`: query NVLink remote device type](https://github.com/NVIDIA/nccl/commit/716dce1c75cddbc4df8c915f6b58ef2f38e8befb)
- [`598c836f`: derive NVLink target class from remote device type](https://github.com/NVIDIA/nccl/commit/598c836fa1bcf2937e87215314b88f8b50c8703c)

NCCL 2.31.2 was officially released on 2026-08-11: [NVIDIA NCCL v2.31.2-1 release](https://github.com/NVIDIA/nccl/releases/tag/v2.31.2-1). The release summary does not explicitly call out this topology change, so the source diff—not a release-note claim—is the basis for its relevance.

### Proposed NCCL 2.29.7 backport

The preferred source fix is to backport the three upstream changes above rather than inventing MNNVL bandwidth or blindly synthesizing 18 links. The core semantic change in `ncclTopoGetXmlFromGpu` is:

```diff
- nvmlPciInfo_t remoteProc;
- if (ncclNvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc)
-     != ncclSuccess) continue;
- /* derive target and class from remoteProc.busId */
+ nvmlIntNvLinkDeviceType_t remoteType;
+ if (ncclNvmlDeviceGetNvLinkRemoteDeviceType(nvmlDev, l, &remoteType)
+     != ncclSuccess) {
+   INFO(NCCL_INIT|NCCL_GRAPH,
+        "Unable to get remote device type for GPU %d NVLink %d", dev, l);
+   continue;
+ }
+ if (remoteType == NVML_NVLINK_DEVICE_TYPE_SWITCH) {
+   strcpy(lowerId, "fffffff:ffff:ff");
+   strcpy(tclass, "0x068000");
+ } else {
+   nvmlPciInfo_t remoteProc = {};
+   if (ncclNvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc)
+       != ncclSuccess) {
+     INFO(NCCL_INIT|NCCL_GRAPH,
+          "Unable to get remote PCI info for GPU %d NVLink %d", dev, l);
+     continue;
+   }
+   /* GPU/IBMNPU target handling */
+ }
```

The backport must also include the NVML enum/function wrapper from `aed770bd`; copying only the `xml.cc` hunk is incomplete.

Irrespective of the functional backport, 2.29 should warn when a Blackwell GPU yields no usable NVLinks. A minimal diagnostic addition after the loop would turn this failure from a silent 12x performance regression into an actionable message:

```diff
@@ ncclTopoGetXmlFromGpu(...)
+ int usableNvLinks = 0;
  for (int l = 0; l < maxNvLinks; ++l) {
    ...
    /* after adding/incrementing the nvlink XML node */
+   usableNvLinks++;
  }
+ if (sm >= 100 && maxNvLinks != 0 && usableNvLinks == 0) {
+   WARN("GPU %d (SM %d) reported no usable NVLinks; topology graph and NVLS will be degraded. "
+        "Set NCCL_TOPO_DUMP_FILE and inspect NVML link capability/state/remote-target queries.",
+        dev, sm);
+ }
```

Blindly synthesizing `count="18"` merely from MNNVL clique membership is not recommended as the general upstream fix: a genuinely degraded GPU should not be modeled at full bandwidth. The explicit topology-file override is acceptable here because the rack is uniform, the fabric is independently known healthy, and the corrected model was validated.

## Environment knobs evaluated

- `NCCL_MNNVL_ENABLE`, `NCCL_NVLS_ENABLE`, and `NCCL_CUMEM_ENABLE`: capability/transport gates; none reconstruct a missing topology link.
- `NCCL_MNNVL_UUID` and `NCCL_MNNVL_CLIQUE_ID`: override fabric identity in `nccl/src/init.cc:660-715`. Clique identity was already correct and these variables cannot add an NVS edge.
- `NCCL_MNNVL_SCATTER_NETS_ENABLE`: affects GPU-first versus channel-first **network-device** selection at `nccl/src/graph/search.cc:508-550`; irrelevant to an intra-domain NVS-to-GPU edge.
- `NCCL_MNNVL_RAIL_PER_HOST`: changes how NET rails are matched at `nccl/src/graph/search.cc:558-571`; irrelevant when the failed collective is modeled as one MNNVL node and the missing edge is GPU-to-NVS.
- `NCCL_MAX_NCHANNELS=32`: only an upper bound after search; it cannot replace the missing link.
- IMEX device permissions: not reached because the NVLS graph is zero before allocation.

## Hypotheses resolved

| Hypothesis | Finding |
|---|---|
| MNNVL XML merge gives remote GPUs low/default bandwidth | Rejected. Healthy 04/05 fusion uses one global NVS with 18 NVLinks per GPU and builds 32/24 channels. |
| Ring search limits cross-host MNNVL to two channels | Rejected. It finds 16 physical search channels on healthy 04/05 and those are duplicated to 32. Two is the one-channel fallback doubled. |
| NVLS multicast/IMEX allocation fails silently | Rejected as the immediate cause. The zero-channel topology graph disables NVLS before allocation begins. |
| Fabric/clique discovery is wrong | Rejected. All ranks have the same completed UUID/clique and `cliqueSize 8`; CUDA MNNVL transfers remain available. |
| One bad GPU topology poisons the communicator | Confirmed. Failing dump identifies tray03 GPU1/rank5; a one-rank synthetic fault reproduces the channel signature. |
| GDR attribute causes the NVLS loss | Rejected. Failing dump has `gdr="1"`; forcing only `gdr="0"` on healthy topology retains 32/24. |
| aarch64/64K pages generically break 2.29.7 | Rejected. Same kernel/architecture/build works on trays04/05 and on healthy GPUs in the failing domain. |
| later NCCL contains relevant work | Confirmed for 2.31.2 source. Its switch discovery bypasses remote PCI info; 2.30.7 does not. Exact affected-host validation remains pending. |

## Artifacts and repository hygiene

Useful generated artifacts retained in the requested report directory:

- `docs/tmp/mnnvl_topo_1tray_04.xml` — healthy single-tray input/workaround template;
- `docs/tmp/mnnvl_topo_2tray_0405.xml` — healthy fused two-tray topology;
- `docs/tmp/mnnvl_graph_1tray_04.xml` — healthy one-tray graph;
- `docs/tmp/mnnvl_graph_2tray_0405.xml` — healthy two-tray graph;
- this report.

No file under `/mnt/shared/home/dc/scaling-inference` was modified. No NCCL source instrumentation was ultimately needed. The NCCL repository remains clean at detached HEAD `36191590`; no commit was created.

---

# Final addendum: hardware confirmation, reproducible fault injection, the exact origin of `2`, and a WARN patch

Date: 2026-08-14  
Applies to NCCL git `36191590` (`2.29.7+cuda13.1`).

This is an append-only correction to the interrupted investigation above. The post-interruption hardware work establishes that the original incident is a **real tray03 GPU1 hardware degradation**, not an NCCL graph-search or NVML-discovery bug. The affected device is tray03 GPU1, PCI bus ID `0009:01:00.0`; it contributes zero `<nvlink>` entries to the fused MNNVL clique topology, and direct `nvidia-smi nvlink` queries hang on tray03. The earlier speculation that the missing entry might be a transient NCCL/NVML remote-PCI enumeration problem is therefore superseded. In particular, using a topology override that lies about this GPU having 18 working links is not an acceptable workaround for the physically degraded tray.

The rack control is decisive. With the same NCCL library, healthy trays06/07 report `32 coll / 24 nvls` and 684.878 GB/s average bus bandwidth at 512 MiB (`scaling-inference/gb300/scripts/collectives/results/diag_2tray_0607.log:256-284,323-328`). The hardware-degraded trays02/03 report `2 coll / 0 nvls` and 51.5903 GB/s (`scaling-inference/gb300/scripts/collectives/results/diag_2tray.log:378-407,585-595`). No new GPU command was issued to trays01/02/03/16 or 06-13 in this investigation; all new experiments below used only trays04/05.

## 1. Fault-injection reproduction

### Result: exact control-plane reproduction; the physical 51 GB/s dataplane is not reproducible with `NCCL_TOPO_FILE`

One synthetic NVLink-less GPU in an otherwise healthy eight-GPU MNNVL clique reproduces the failure's graph signature exactly:

```text
Pattern 4 (ring): nChannels 1, bw 0.1, type SYS/SYS
Pattern 1 (tree): nChannels 1, bw 90, type C2C/PIX
Pattern 5 (NVLS): nChannels 0
2 coll channels, 2 collnet channels, 0 nvls channels,
2 p2p channels, 2 p2p channels per peer
```

The XML injection does **not** reproduce the physical run's 51.5903 GB/s. The repeatable rank-5 XML injection measured 6.66929 GB/s, and an even more faithful temporary discovery hook that omitted rank 5's `<nvlink>` element entirely measured 4.50832 GB/s. This is a mechanism boundary, not a failed causal test: the topology controls graph search, while the real MNNVL CUDA mapping and transfers still execute on a physically healthy tray05 GPU1. Only the broken tray03 hardware supplies the degraded physical datapath that happens to sustain approximately 51 GB/s.

The `33.2126 GB/s` synthetic number in the interrupted section above has no retained raw log and could not be repeated. It is superseded by the audited, retained repetitions in this addendum. The channel signature remains identical in every valid zero-NVLink injection.

### `NCCL_TOPO_FILE` semantics under MNNVL

`NCCL_TOPO_FILE` is local input before clique fusion, not a ready-made global fused topology:

- `nccl/src/graph/topo.cc:1453-1456` reads the named file independently in every rank.
- `nccl/src/graph/topo.cc:1461-1468` rewrites every CPU `host_hash` to the current host.
- `nccl/src/graph/topo.cc:1479-1486` fills and marks only the GPU managed by the current process.
- `nccl/src/graph/topo.cc:1539-1540` trims branches that do not contain the managed GPU.
- `nccl/src/graph/topo.cc:1543-1547` selects the MNNVL clique ranks as the fusion group.
- `nccl/src/graph/topo.cc:1559-1578` all-gathers and fuses those per-rank XML fragments.

Therefore a single full eight-GPU edited file on the shared home filesystem is not a reliable way to assign a fault to only one host: every process treats it as its local template, rewrites its host identity, retains its own managed GPU, and participates in fusion. The working method is to place different single-tray templates at the **same host-local `/tmp` path**: healthy on tray04, GPU1 `count="0"` on tray05. The runner propagates the one common pathname through `mpirun`; see `nccl_bench.sh:131-153`.

The retained inputs are:

```text
docs/tmp/mnnvl_topo_1tray_healthy.xml
  SHA-256 0e7e865d009701825c996cf211d250afae262d6a4cfb0c94f2f38b000e72acec
  GPU0/GPU1/GPU2/GPU3 nvlink counts: 18/18/18/18

docs/tmp/mnnvl_topo_fault_gpu1.xml
  SHA-256 8d758dd8c8e8ce7b8171a37588b4539dd6a90ea4220faee4453be540a3450138
  GPU0/GPU1/GPU2/GPU3 nvlink counts: 18/0/18/18
```

An explicit `count="0"` child is required for topology-file injection. If the `<nvlink>` child is merely deleted, `nccl/src/graph/xml.cc:766-823` sees that no node exists and re-runs NVML discovery, restoring the healthy physical GPU's 18 links.

### Exact reproducible XML-injection command

The following is the exact rank-placement reproduction. Tray05 is second in `TRAY_LIST`, so its local GPU1 becomes global rank 5, matching the incident's position.

```bash
cd /mnt/shared/home/dc/nccl-eBPF

scp docs/tmp/mnnvl_topo_1tray_healthy.xml \
  pod4-gb300-3-tray04-f3:/tmp/nccl_mnnvl_fault_inject_20260814.xml
scp docs/tmp/mnnvl_topo_fault_gpu1.xml \
  pod4-gb300-3-tray05-f3:/tmp/nccl_mnnvl_fault_inject_20260814.xml

env \
  TRAYS=2 TRAY_LIST="04 05" \
  NCCL_LIB_DIR=/mnt/shared/home/dc/nccl-eBPF/nccl/build/lib \
  ARM=baseline TAG=mnnvl_0405_fault_rank5_gpu1_512M_repro \
  RESULTS_DIR=/mnt/shared/home/dc/nccl-eBPF/docs/tmp/mnnvl_repro_results \
  MSG_MIN=512M MSG_MAX=512M ITERS=20 WARMUP=5 CHECK=0 \
  NCCL_TOPO_FILE=/tmp/nccl_mnnvl_fault_inject_20260814.xml \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING \
  NCCL_TOPO_DUMP_FILE=/mnt/shared/home/dc/nccl-eBPF/docs/tmp/mnnvl_topo_0405_fault_rank5_gpu1_fused.xml \
  NCCL_TOPO_DUMP_FILE_RANK=0 \
  /mnt/shared/home/dc/scaling-inference/gb300/scripts/collectives/nccl_bench.sh run
```

The input placement was verified on each host before the run:

```text
tray04: 0e7e865d... ; counts 18/18/18/18
tray05: 8d758dd8... ; counts 18/0/18/18
```

The fused output proves that only global rank 5 is faulted: `docs/tmp/mnnvl_topo_0405_fault_rank5_gpu1_fused.xml:55-92` contains ranks 4-7 from tray05, and rank 5 alone has `count="0"` at `:65-66`. The log records one ring search channel, one tree search channel, and zero NVLS channels at `docs/tmp/mnnvl_repro_results/mnnvl_0405_fault_rank5_gpu1_512M_repro.log:1080-1102`; the final `2 coll / 0 nvls` line is at `:1215` (and once per rank through `:1244`). At 512 MiB the selected collective is `TREE/SIMPLE`, the out-of-place bus bandwidth is 4.10 GB/s, the in-place bus bandwidth is 9.24 GB/s, and the reported average is 6.66929 GB/s (`:1309-1363`).

For comparison, the immediately preceding healthy trays04/05 control used the same library and benchmark settings without `NCCL_TOPO_FILE`:

```bash
env -u NCCL_TOPO_FILE \
  TRAYS=2 TRAY_LIST="04 05" \
  NCCL_LIB_DIR=/mnt/shared/home/dc/nccl-eBPF/nccl/build/lib \
  ARM=baseline TAG=mnnvl_0405_healthy_control_512M \
  RESULTS_DIR=/mnt/shared/home/dc/nccl-eBPF/docs/tmp/mnnvl_repro_results \
  MSG_MIN=512M MSG_MAX=512M ITERS=20 WARMUP=5 CHECK=0 \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV \
  NCCL_TOPO_DUMP_FILE=/mnt/shared/home/dc/nccl-eBPF/docs/tmp/mnnvl_topo_0405_healthy_control_fused.xml \
  NCCL_TOPO_DUMP_FILE_RANK=0 \
  NCCL_GRAPH_DUMP_FILE=/mnt/shared/home/dc/nccl-eBPF/docs/tmp/mnnvl_graph_0405_healthy_control.xml \
  NCCL_GRAPH_DUMP_FILE_RANK=0 \
  /mnt/shared/home/dc/scaling-inference/gb300/scripts/collectives/nccl_bench.sh run
```

It produced `32 coll / 24 nvls` (`docs/tmp/mnnvl_repro_results/mnnvl_0405_healthy_control_512M.log:1794-1822`), 491.14 GB/s out-of-place, 505.29 GB/s in-place, and 498.213 GB/s average (`:1861-1869`). The 06/07 control above is the cleaner rack baseline at 684.878 GB/s; the 04/05 control is reported here because it brackets the allowed-tray injection with identical local conditions.

### Higher-fidelity no-element control

To exclude `count="0"` versus a missing child as the source of the throughput mismatch, a temporary four-line discovery hook set `maxNvLinks=0` only when `dev == 1` and a host-local marker existed. It was compiled into an isolated build, leaving `nccl/build/lib` untouched:

```bash
make -j32 src.build \
  BUILDDIR=/mnt/shared/home/dc/nccl-eBPF/docs/tmp/nccl-fi-build \
  NVCC_GENCODE="-gencode=arch=compute_103,code=sm_103" \
  CUDA_HOME=/usr/local/cuda-13.1
```

Only tray05 had `/tmp/nccl_fault_no_nvlink_gpu1_20260814.marker`; the run used the same runner command with:

```bash
NCCL_LIB_DIR=/mnt/shared/home/dc/nccl-eBPF/docs/tmp/nccl-fi-build/lib
NCCL_FAULT_INJECT_NO_NVLINK_MARKER=/tmp/nccl_fault_no_nvlink_gpu1_20260814.marker
TAG=mnnvl_0405_fault_rank5_discovery_skip_512M
```

Its fused dump has rank 5 with no `<nvlink>` child at all (`docs/tmp/mnnvl_topo_0405_fault_rank5_discovery_skip_fused.xml:65`), yet it still gives one ring search channel, one tree search channel, zero NVLS channels, final `2 coll / 0 nvls`, and only 4.50832 GB/s (`docs/tmp/mnnvl_repro_results/mnnvl_0405_fault_rank5_discovery_skip_512M.log:1038-1060,1185-1202,1321`). The temporary source hook was reverted, the alternate build was moved to trash, and the marker was removed. The NCCL source and production build are pristine.

### Why topology injection cannot synthesize the hardware's 51 GB/s

The XML is consumed by topology/path and graph search. It does not disable a physical NVLink, change the GPU's fabric health, or make CUDA reproduce a hanging NVML/NVLink device:

- `nccl/src/graph/paths.cc:276-391` combines topology distance with CUDA/NVML P2P capability. For remote MNNVL peers it explicitly assumes CUDA P2P at `:386-387`.
- `nccl/src/transport/p2p.cc:128-175` decides whether the P2P transport can connect, including the real `cudaDeviceCanAccessPeer` result for locally visible devices.
- `nccl/src/transport/p2p.cc:409-414` selects the cuMem transport and labels it `P2P/MNNVL` from `comm->MNNVL`; it does not emulate the XML link bandwidth in the CUDA mapping.

Thus the reproducible scientific statement is:

| Case | Physical GPU | Topology presented to search | Final channels | 512 MiB average bus BW |
|---|---|---|---|---:|
| healthy trays06/07 | healthy | 18 links on every GPU | 32 coll / 24 NVLS | 684.878 GB/s |
| original trays02/03 | tray03 GPU1 degraded | zero links on rank 5 | 2 coll / 0 NVLS | 51.5903 GB/s |
| trays04/05 XML fault | healthy | rank 5 `count="0"` | 2 coll / 0 NVLS | 6.66929 GB/s |
| trays04/05 discovery-skip fault | healthy | no rank 5 `<nvlink>` child | 2 coll / 0 NVLS | 4.50832 GB/s |

No attempt was made to disable physical NVLinks on a healthy production tray. Doing so would be a materially riskier hardware mutation and is unnecessary to prove the NCCL control-plane causal chain.

## 2. Why the final collective channel count is exactly `2`

The short answer is that the ring search does **not** derive two channels from the five C2C links. It produces a one-channel emergency fallback; topology postset then duplicates that one search channel into two compute/collective channels.

### The C2C number does not calculate the final `2`

The XML field is:

```xml
<c2c bw="44712" count="5"/>
```

`nccl/src/graph/topo.cc:675-695` converts it to a bidirectional GPU-to-CPU C2C edge with:

```text
c2cBw = (44712 * 5) / 1000 = 223.56 GB/s
```

That edge makes a one-channel balanced tree possible, which is why the measured tree graph says `bw 90, type C2C/PIX`. It does not imply `floor(223.56 / anything) == 2`. In fact, `getTotalBw()` at `nccl/src/graph/search.cc:30-38` considers only direct `LINK_NVL` and `LINK_PCI` bandwidth, and search initialization takes the **maximum** over GPUs at `:39-53`; it is not a bad-GPU minimum that selects two channels. SM100's candidate per-channel speeds are the arrays at `nccl/src/graph/search.cc:987-990`.

### Exact one-to-two sequence

1. `nccl/src/init.cc:1089-1094` initializes the ring graph with `minChannels=1` and `maxChannels=MAXCHANNELS/2`, then calls topology search.

2. A complete high-bandwidth ring cannot traverse every GPU when one GPU has no usable NVSwitch edge. Once ordinary search is exhausted, `nccl/src/graph/search.cc:1216-1237` takes the non-NVLS fallback path. It installs simple rank order, sets `bwIntra=0.1`, sets `typeIntra=PATH_SYS`, and—critically—sets `graph->nChannels=1` at `:1237`. The injected logs show exactly `Pattern 4 ... nChannels 1 ... SYS/SYS`.

3. `nccl/src/init.cc:1097-1103` fixes both the tree minimum and maximum to the ring result. Since the fallback ring has one channel, the balanced-tree search is constrained to one search channel. The surviving C2C/PCIe route satisfies that one channel at the first SM100 intra speed, producing `Pattern 1 ... nChannels 1, bw 90, type C2C/PIX`.

4. The NVLS graph is separate. `nccl/src/graph/search.cc:387-417` must reserve NVS-to-GPU and GPU-to-NVS capacity for every GPU. Rank 5 has no such path, and the generic fallback explicitly excludes `NCCL_TOPO_PATTERN_NVLS` at `nccl/src/graph/search.cc:1216`. Therefore pattern 5 remains at zero channels. `nccl/src/init.cc:1260-1273` takes the minimum graph result across ranks and silently clears provisional NVLS support when that minimum is zero.

5. During topology postset, `nccl/src/graph/connect.cc:436-453` connects the rings and trees and unconditionally performs the normal duplication:

   ```c++
   nChannels = comm->nChannels = std::min(MAXCHANNELS, nChannels*2);
   ```

   Therefore `1 search channel * 2 = 2 final collective channels`. The later performance-oriented doubling at `nccl/src/graph/connect.cc:473-476` does not fire because the fallback ring's `bwIntra` is 0.1, not greater than 45.

There is a separate source of the number two in the same final INFO line: `nccl/src/graph/paths.cc:834-851` assigns two P2P channels per peer to a local non-`PATH_NVL` path. That explains `2 p2p channels per peer`; it is not the cause of `2 coll channels`.

The precise answer for the paper is therefore: **the search settles on one, not two; postset duplicates one to two.** The C2C bandwidth permits the constrained one-channel tree fallback, but no division of `223.56 GB/s` determines the final channel count.

## 3. Minimal silent-degradation WARN patch (draft only; not committed)

The following upstream-facing draft covers both observability gaps:

1. after fused MNNVL XML is materialized, the clique leader names every GPU that has no positive-bandwidth direct NVLink edge; and
2. after graph results are reduced across ranks, rank 0 warns when provisional NVLS capability is being cleared because the NVLS graph has zero channels.

It deliberately does not synthesize links or change algorithm selection. A physically degraded GPU must not be modeled as healthy. The patch only turns the current silent, order-of-magnitude performance collapse into an actionable initialization warning.

```diff
diff --git a/src/graph/topo.cc b/src/graph/topo.cc
--- a/src/graph/topo.cc
+++ b/src/graph/topo.cc
@@ -1584,7 +1584,24 @@ ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** sy
   }
 
   // Only update our topo tracking structure if we aren't dumping (separate steps)
-  if (dumpXmlFile == NULL) NCCLCHECKGOTO(ncclTopoGetSystemFromXml(xml, system, getHostHash()), ret, fail);
+  if (dumpXmlFile == NULL) {
+    NCCLCHECKGOTO(ncclTopoGetSystemFromXml(xml, system, getHostHash()), ret, fail);
+    // An MNNVL clique is expected to be connected through NVLink/NVSwitch.
+    // Emit this once per clique, after all per-rank XML fragments are fused.
+    if (comm->MNNVL && comm->cliqueRank == 0) {
+      for (int g = 0; g < (*system)->nodes[GPU].count; g++) {
+        struct ncclTopoNode* gpu = (*system)->nodes[GPU].nodes + g;
+        bool hasNvlink = false;
+        for (int l = 0; l < gpu->nlinks; l++) {
+          struct ncclTopoLink* link = gpu->links + l;
+          if (link->type == LINK_NVL && link->bw > 0.0f) {
+            hasNvlink = true;
+            break;
+          }
+        }
+        if (!hasNvlink) WARN("MNNVL topology: rank %d busId %lx has no active NVLinks; NVLS will be unavailable and collective performance may be severely degraded",
+                             gpu->gpu.rank, NCCL_TOPO_ID_LOCAL_ID(gpu->id));
+      }
+    }
+  }
 
 exit:
   if (!comm->MNNVL && localRanks) free(localRanks);
diff --git a/src/init.cc b/src/init.cc
--- a/src/init.cc
+++ b/src/init.cc
@@ -1270,7 +1270,12 @@ static ncclResult_t initTransportsRank(struct ncclComm* comm, struct ncclComm* p
     comm->p2pnChannelsPerPeer = std::min(comm->p2pnChannelsPerPeer, allGather3Data[i].p2pnChannelsPerPeer);
   }
   if (graphs[NCCL_ALGO_COLLNET_CHAIN]->nChannels == 0) comm->config.collnetEnable = 0;
-  if (graphs[NCCL_ALGO_NVLS]->nChannels == 0) comm->nvlsSupport = comm->nvlsChannels = 0;
+  if (graphs[NCCL_ALGO_NVLS]->nChannels == 0) {
+    if (comm->nvlsSupport && comm->rank == 0) {
+      WARN("NVLS support was detected, but topology search found zero NVLS channels; disabling NVLS");
+    }
+    comm->nvlsSupport = 0;
+    comm->nvlsChannels = 0;
+  }
 
   comm->nChannels = treeGraph->nChannels = ringGraph->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);
   if (comm->nChannels < nChannelsOrig) {
```

The placement of the first warning is intentional. At `nccl/src/graph/topo.cc:1559-1578`, fusion is complete and the system contains all clique GPUs, so the warning can identify the specific remote rank and bus ID instead of merely reporting that one local NVML loop found nothing. The `link->bw > 0.0f` condition catches both the real missing-child case and the explicit `count="0"` fault-injection case.

The second warning is guarded by the old `comm->nvlsSupport` value. It therefore fires only on the meaningful transition from provisionally supported (`nccl/src/transport/nvls.cc:156-203`) to graph-disabled (`nccl/src/init.cc:1273`), not on systems that never advertised multicast support. Keeping the actual clears after the warning preserves behavior exactly.

This diff is a report-only draft. It was not applied to the retained NCCL tree and was not committed. The temporary fault-injection source change described in item 1 was also reverted. `git status --short` in `nccl/` is empty at the end of the investigation; the shared production build at `nccl/build/lib` was never rebuilt or replaced.

---

# Addendum 2026-08-14 (late): second degraded GPU — tray14 GPU3, quieter variant

The w64 (16-tray) campaign block reproduced the collapse family at rack scale:
AllReduce 208 GB/s @8G (vs 680 at w32), ~5 GB/s at small sizes, `NCCL_ALGO=NVLS`
rejected as invalid usage, "12 coll channels, 12 collnet channels, 0 nvls channels".

A slurm diagnostic (job 47967: pair tests + instrumented w64 + control) resolved it in
one pass:

- Pair tray06+tray14: `12 coll / 0 nvls`, 81.7 GB/s. Pairs with trays 15/17/18: healthy
  (`32 coll / 24 nvls`, ~700 GB/s).
- Fused-topology dump of the 64-rank communicator: `rank=51 dev=3 busid 0019:06:00.0`
  (tray14 GPU3) has **zero `<nvlink>` entries**; the other 63 GPUs have 18.
- 12-tray control without trays 14/15/17/18: healthy, `32/24`, 557 GB/s @512M at 48
  ranks — so no NCCL-at-16-hosts problem; the clique fusion and NVLS are fine at 48
  ranks with all-healthy members.

Differences from the tray03 case:

1. **Quieter failure**: `nvidia-smi nvlink -s` does NOT hang on tray14 (it hangs on
   tray03). Detection required the fused-topo/channel-line recipe — reinforcing that
   the recipe, not ad-hoc smi probing, is the reliable check.
2. **Milder graph damage**: 12 search channels survive instead of the 1x2 emergency
   fallback — consistent with the ring search still closing reduced rings through the
   degraded GPU's C2C path while NVLS (which needs NVS capacity to every GPU) still
   drops to zero. The severity of the collapse varies with what the sick GPU's
   remaining paths permit; the invariant signature is `0 nvls channels` plus a
   channel count far below 32.

Operational state: rack now has three excluded trays (03: GPU1 NVLink dead, smi hangs;
14: GPU3 NVLink dead, smi responsive; 16: node down). Healthy maximum is 15 trays =
w60. Both degraded GPUs need the driver/host reset + re-dump verification described in
the recommendations above.
