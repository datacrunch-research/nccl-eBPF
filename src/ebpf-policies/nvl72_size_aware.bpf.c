/* nvl72_size_aware.bpf.c -- GB300 NVL72 rack policy (MNNVL scale-up domain).
 *
 * Derived from measured sweeps on pod4-gb300-3 (NCCL 2.29.7, CUDA 13.1,
 * 4 ranks/tray, one MNNVL fabric domain). See
 * scaling-inference/gb300/ncclbpf_mnnvl.md for the data.
 *
 * Design rules:
 *  - Only rank-uniform inputs (n_bytes, n_ranks, n_nvl_domains): every rank
 *    computes the same action, so ranks cannot diverge and hang.
 *  - Only act inside a single NVL domain; anything else: no override.
 *  - Only force combinations measured faster than NCCL's default; sizes
 *    where the default already wins are left alone.
 *
 * w4 (one tray): Ring/LL128 at 4M-32M beats the default protocol choice
 * (+22% @4M, +43% @16M, +8% @32M).
 * w8..w64 thresholds: filled from the M2 ladder (placeholders return 0 until
 * the measured tables land).
 */
#include "bpf_compat.h"
#include "policy_action.h"
#include "policy_context.h"

static inline uint64_t force(uint32_t algo, uint32_t proto) {
  return nccl_policy_pack_action(
      algo, proto, 0, 0,
      NCCL_POLICY_ACTION_SET_ALGO | NCCL_POLICY_ACTION_SET_PROTO);
}

SEC("uprobe")
uint64_t nvl72_size_aware_policy(struct nccl_policy_ctx *ctx) {
  if (!ctx)
    return 0;

  /* Outside a single NVL domain (or pre-v5 NCCL): never override. */
  if (ctx->n_nvl_domains != 1)
    return 0;

  /* AllReduce only. The measured windows are AllReduce-specific: the same
   * Ring forcing applied to AllGather costs up to -47% at w32 (the default
   * is already optimal there), which is how nvlink_ring_mid_v2 -- which has
   * no coll_type guard -- degrades MoE-pattern collectives at scale. */
  if (ctx->coll_type != NCCL_POLICY_COLL_ALLREDUCE)
    return 0;

  uint64_t b = ctx->n_bytes;
  uint32_t r = ctx->n_ranks;

  if (r <= 4) {
    /* One tray: measured LL128 window (+22% @4M, +43% @16M, +8% @32M). */
    if (b >= (4ULL << 20) && b <= (32ULL << 20))
      return force(NCCL_POLICY_ALGO_RING, NCCL_POLICY_PROTO_LL128);
    return 0;
  }

  if (r <= 8) {
    /* Two trays: paper-shaped window returns over MNNVL
     * (+20/+42/+35/+31% at 4-32M; +9/+6% at 64-128M). */
    if (b >= (4ULL << 20) && b <= (32ULL << 20))
      return force(NCCL_POLICY_ALGO_RING, NCCL_POLICY_PROTO_LL128);
    if (b >= (64ULL << 20) && b <= (192ULL << 20))
      return force(NCCL_POLICY_ALGO_RING, NCCL_POLICY_PROTO_SIMPLE);
    return 0;
  }

  /* r > 8: no override. Measured on an exclusive allocation (slurm 47899,
   * 2026-08-14): at w16/w32 the default tuner is within noise of the best
   * arm at every size, Ring loses everywhere below 512M, and the earlier
   * candidate windows (8-32M @w16, 32M @w32, fit on contaminated data)
   * cost -3..-10% -- refuted. The exploit is a narrow-world phenomenon on
   * this fabric: proto-only at w4, algo+proto at w8, gone by w16. */
  return 0;
}

char LICENSE[] SEC("license") = "GPL";
