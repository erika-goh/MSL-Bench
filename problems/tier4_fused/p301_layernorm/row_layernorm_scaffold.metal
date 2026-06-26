// Scaffold for p301_layernorm. Save as `row_layernorm.metal`.
//
// Buffer bindings:
//   buffer(0) = x     (input, B*K floats, row-major)
//   buffer(1) = gamma (per-column scale, K floats)
//   buffer(2) = beta  (per-column offset, K floats)
//   buffer(3) = out   (output, B*K floats, same shape as x)
//
// Dispatch geometry:
//   grid        = (B*K, 1, 1)
//   threadgroup = (K,   1, 1)
//
// Layernorm formula (per row b):
//   mean[b] = sum(x[b, :]) / K
//   var[b]  = sum((x[b, k] - mean[b])^2) / K
//   out[b, k] = (x[b, k] - mean[b]) / sqrt(var[b] + eps) * gamma[k] + beta[k]
//
// Algebraic identity used here to save a reduction pass:
//   var = E[x^2] - E[x]^2  =  sumsq/K - (sum/K)^2
//
// Lets us compute sum AND sumsq in a single tree-reduction, then
// derive var from them arithmetically. (Numerically risky for large-
// magnitude offset data, but fine for unit-scale randn input.)

#include <metal_stdlib>
using namespace metal;

constant uint K = 256;
constant float EPS = 1e-5f;

kernel void row_layernorm(
    device const float* x     [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta  [[buffer(2)]],
    device float*       out   [[buffer(3)]],
    uint tid_in_tg  [[thread_position_in_threadgroup]],
    uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float sum_scratch[K];
    threadgroup float sqs_scratch[K];

    uint b   = tg_in_grid;
    uint tid = tid_in_tg;
    uint idx = b * K + tid;

    // ============================================================
    // TODO 1: load x[idx] into a local register `v`; initialize
    // sum_scratch[tid] = v and sqs_scratch[tid] = v*v. Keep `v`
    // — you need it in phase 3.
    // ============================================================



    // ============================================================
    // TODO 2: barrier, then combined tree reduction.
    //
    // for stride = K/2; stride > 0; stride >>= 1:
    //     if (tid < stride):
    //         sum_scratch[tid] += sum_scratch[tid + stride];
    //         sqs_scratch[tid] += sqs_scratch[tid + stride];
    //     barrier
    //
    // ONE barrier per stage covers both arrays.
    // ============================================================



    // ============================================================
    // TODO 3: every thread reads scratch[0] of both arrays,
    // computes mean / var / inv_std in registers, and writes its
    // own output element.
    //
    //   mean = sum_scratch[0] / float(K)
    //   var  = sqs_scratch[0] / float(K) - mean * mean
    //   inv_std = rsqrt(var + EPS)
    //   out[idx] = (v - mean) * inv_std * gamma[tid] + beta[tid]
    //
    // `rsqrt` is Metal's reciprocal-square-root intrinsic — faster
    // and slightly more accurate than `1.0 / sqrt(...)`.
    // ============================================================



}
