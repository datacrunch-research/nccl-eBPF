#!/usr/bin/env bash
# nccl_bench.sh -- NCCL collective benchmarks on the GB300 NVL72 rack, with
# optional NCCLbpf tuner/profiler plugin arms (github.com/eunomia-bpf/nccl-eBPF,
# arxiv 2603.11438). Scripted form of the rackscale recipe in gb300/setup.md
# section 1.2: OpenMPI over ssh, 4 ranks per tray, -g 1, MNNVL scale-up domain.
#
# Subcommands:
#   run       one benchmark invocation (one arm), raw log + json meta sidecar
#   sweep     loop over ARMS x REPS at the current TRAYS count
#   selftest  print the mpirun command line that `run` would execute
#
# Env overrides (defaults in parentheses):
#   TRAYS       trays to use, 4 ranks each (1)
#   TRAY_LIST   explicit tray ids, e.g. "01 03 17" (first TRAYS live trays)
#   SKIP_TRAYS  degraded/dead trays, space-separated ("03 14 16")
#   RACK        rack prefix (pod4-gb300-3)
#   TEST        nccl-tests binary name (all_reduce_perf)
#   BIN_DIR     nccl-tests build dir (~/nccl-tests/build_mpi)
#   MSG_MIN MSG_MAX FACTOR   size sweep, nccl-tests -b/-e/-f (8 / 8G / 2)
#   ITERS WARMUP CHECK       nccl-tests -n/-w/-c (20 / 5 / 0)
#   ARM         baseline | noop | policy:<name> | algo:<Algo>[/<Proto>] (baseline)
#   PROFILER    none | native | ebpf (none) -- attaches profiler plugin arm
#   MAX_NCH     sets NCCL_MAX_NCHANNELS if non-empty
#   NCCL_LIB_DIR  alternate libnccl dir prepended to LD_LIBRARY_PATH
#                 (e.g. ~/nccl-eBPF/nccl/build/lib for the 2.29.7 A/B)
#   PLUGIN_DIR  NCCLbpf build dir (~/nccl-eBPF/src/nccl-policy-plugin/build)
#   ARMS REPS   sweep matrix ("baseline noop" / 3)
#   TAG REP     output naming (auto / 1)
#   RESULTS_DIR (<script dir>/results)
#   MNNVL_ENV   1 -> export NCCL_MNNVL_ENABLE/NVLS/CUMEM/SOCKET_IFNAME (1)
#
# Examples:
#   TRAYS=1 ARM=noop ./nccl_bench.sh run
#   TRAYS=2 ARM=policy:nvlink_ring_mid_v2 REP=2 ./nccl_bench.sh run
#   TRAYS=17 ARMS="baseline noop algo:Ring algo:NVLSTree" REPS=3 ./nccl_bench.sh sweep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAYS="${TRAYS:-1}"
TRAY_LIST="${TRAY_LIST:-}"
SKIP_TRAYS="${SKIP_TRAYS:-03 14 16}"
RACK="${RACK:-pod4-gb300-3}"
TEST="${TEST:-all_reduce_perf}"
BIN_DIR="${BIN_DIR:-$HOME/nccl-tests/build_mpi}"
MSG_MIN="${MSG_MIN:-8}"
MSG_MAX="${MSG_MAX:-8G}"
FACTOR="${FACTOR:-2}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"
CHECK="${CHECK:-0}"
ARM="${ARM:-baseline}"
PROFILER="${PROFILER:-none}"
MAX_NCH="${MAX_NCH:-}"
NCCL_LIB_DIR="${NCCL_LIB_DIR:-}"
PLUGIN_DIR="${PLUGIN_DIR:-$HOME/nccl-eBPF/src/nccl-policy-plugin/build}"
ARMS="${ARMS:-baseline noop}"
REPS="${REPS:-3}"
TAG="${TAG:-}"
REP="${REP:-1}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
MNNVL_ENV="${MNNVL_ENV:-1}"

host_of() { printf '%s-tray%s-f3' "$RACK" "$1"; }

live_trays() {
    local t skip out=()
    for t in $(seq -w 01 18); do
        skip=0
        for s in $SKIP_TRAYS; do [ "$t" = "$(printf '%02d' "${s#0}")" ] && skip=1; done
        [ "$skip" = 0 ] && out+=("$t")
    done
    echo "${out[@]}"
}

build_hostlist() {
    local trays=($TRAY_LIST) list=() t
    if [ ${#trays[@]} -eq 0 ]; then
        trays=($(live_trays))
        trays=("${trays[@]:0:$TRAYS}")
    fi
    if [ ${#trays[@]} -lt "$TRAYS" ]; then
        echo "ERROR: need $TRAYS trays, only ${#trays[@]} live" >&2; exit 1
    fi
    for t in "${trays[@]}"; do list+=("$(host_of "$t"):4"); done
    TRAYS_USED="${trays[*]}"
    HOSTLIST=$(IFS=,; echo "${list[*]}")
}

# Resolve one ARM spec into NCCL env for this run.
apply_arm() {
    ARM_KIND="$ARM" ARM_POLICY="" ARM_ALGO="" ARM_PROTO=""
    case "$ARM" in
        baseline) ;;
        noop|policy:*)
            ARM_KIND="policy"
            ARM_POLICY="${ARM#policy:}"; [ "$ARM" = noop ] && ARM_POLICY=noop
            export NCCL_TUNER_PLUGIN="$PLUGIN_DIR/libnccl-policy.so"
            export NCCL_POLICY_BPF_PATH="$PLUGIN_DIR/ebpf-policies/${ARM_POLICY}.bpf.o"
            export NCCL_POLICY_VERIFY_MODE="${NCCL_POLICY_VERIFY_MODE:-strict}"
            [ -f "$NCCL_POLICY_BPF_PATH" ] || { echo "ERROR: no policy $NCCL_POLICY_BPF_PATH" >&2; exit 1; }
            ;;
        algo:*)
            ARM_KIND="envforce"
            local spec="${ARM#algo:}"
            ARM_ALGO="${spec%%/*}"
            if [ "$spec" != "${spec#*/}" ]; then ARM_PROTO="${spec#*/}"; fi
            export NCCL_ALGO="$ARM_ALGO"
            if [ -n "$ARM_PROTO" ]; then export NCCL_PROTO="$ARM_PROTO"; fi
            ;;
        *) echo "ERROR: unknown ARM=$ARM" >&2; exit 1;;
    esac
    if [ "$PROFILER" != "none" ]; then
        export NCCL_PROFILER_PLUGIN="$PLUGIN_DIR/libnccl-policy.so"
        export NCCL_POLICY_PROFILER_MODE="$PROFILER"
    fi
    if [ -n "$MAX_NCH" ]; then export NCCL_MAX_NCHANNELS="$MAX_NCH"; fi
    if [ "$MNNVL_ENV" = 1 ]; then
        export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"
        export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
        export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
        export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0.225}"
    fi
    if [ -n "$NCCL_LIB_DIR" ]; then
        export LD_LIBRARY_PATH="$NCCL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    return 0
}

# -x flags for every NCCL_*/LD_LIBRARY_PATH var currently exported.
propagate_flags() {
    local v flags=()
    for v in $(compgen -e | grep -E '^NCCL_|^LD_LIBRARY_PATH$'); do flags+=(-x "$v"); done
    echo "${flags[@]}"
}

# Refuse to benchmark trays whose GPUs are already running compute work
# (a colleague's slurm job polluted a whole sweep once). Retries for
# PREFLIGHT_WAIT seconds first: back-to-back runs briefly see their own
# predecessor's ranks draining. FORCE=1 skips entirely.
preflight_busy_check() {
    [ "${FORCE:-0}" = 1 ] && return 0
    local deadline=$(( $(date +%s) + ${PREFLIGHT_WAIT:-60} )) t host busy
    while :; do
        busy=""
        for t in $TRAYS_USED; do
            host=$(host_of "$t")
            local n
            n=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" \
                'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l' 2>/dev/null || echo 0)
            if [ "${n:-0}" -gt 0 ]; then busy="$host:$n"; break; fi
        done
        [ -z "$busy" ] && return 0
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "ERROR: ${busy%%:*} still has ${busy##*:} compute process(es) after ${PREFLIGHT_WAIT:-60}s; aborting (FORCE=1 to override)" >&2
            return 1
        fi
        sleep 5
    done
}

do_run() {
    build_hostlist
    preflight_busy_check || return 1
    apply_arm
    local world=$((TRAYS * 4))
    local tag="${TAG:-${TEST%_perf}_w${world}_${ARM//[:\/]/-}$( [ "$PROFILER" != none ] && echo "_prof-$PROFILER")$( [ -n "$MAX_NCH" ] && echo "_nch$MAX_NCH")_m${MSG_MIN}-${MSG_MAX}_i${ITERS}_r${REP}}"
    mkdir -p "$RESULTS_DIR"
    local log="$RESULTS_DIR/${tag}.log" meta="$RESULTS_DIR/${tag}.meta"

    # Pin OMPI's own TCP (OOB + BTL) to the rack VLAN; with many interfaces
    # (bonds, IB, docker) peer connections otherwise cross VLANs and hang.
    local cmd=(mpirun --bind-to none -H "$HOSTLIST"
               --mca oob_tcp_if_include "${OMPI_IFACE:-bond0.225}"
               --mca btl_tcp_if_include "${OMPI_IFACE:-bond0.225}"
               $(propagate_flags)
               "$BIN_DIR/$TEST" -b "$MSG_MIN" -e "$MSG_MAX" -f "$FACTOR"
               -n "$ITERS" -w "$WARMUP" -c "$CHECK" -g 1)

    if [ "${1:-}" = "print" ]; then echo "${cmd[@]}"; return 0; fi

    echo "[nccl_bench] $tag -> $log"
    "${cmd[@]}" >"$log" 2>&1 || { echo "[nccl_bench] FAILED $tag (see $log)"; return 1; }

    local nccl_ver plugin_sha
    nccl_ver=$(grep -m1 -oE 'NCCL version [0-9.+a-z]+' "$log" || true)
    plugin_sha=$(git -C "$HOME/nccl-eBPF" rev-parse --short HEAD 2>/dev/null || echo none)
    cat >"$meta" <<EOF
{"tag":"$tag","ts":"$(date -Is)","test":"$TEST","world":$world,"trays":"$TRAYS_USED",
 "arm":"$ARM","arm_kind":"$ARM_KIND","policy":"$ARM_POLICY","algo":"${ARM_ALGO}","proto":"${ARM_PROTO}",
 "profiler":"$PROFILER","max_nch":"$MAX_NCH","iters":$ITERS,"warmup":$WARMUP,"check":$CHECK,
 "msg":"$MSG_MIN..$MSG_MAX f$FACTOR","rep":$REP,"nccl":"$nccl_ver","nccl_lib_dir":"$NCCL_LIB_DIR",
 "plugin_sha":"$plugin_sha","bin":"$BIN_DIR/$TEST"}
EOF
    grep -m1 'Avg bus bandwidth' "$log" || true
}

do_sweep() {
    local arm rep
    for arm in $ARMS; do
        for rep in $(seq 1 "$REPS"); do
            ( ARM="$arm" REP="$rep" TAG="" do_run ) || echo "[nccl_bench] sweep continues after failure: $arm r$rep"
        done
    done
    echo "[nccl_bench] sweep done: TRAYS=$TRAYS ARMS='$ARMS' REPS=$REPS"
}

case "${1:-run}" in
    run)      do_run;;
    sweep)    do_sweep;;
    selftest) do_run print;;
    *) echo "usage: $0 [run|sweep|selftest] (config via env, see header)" >&2; exit 1;;
esac
