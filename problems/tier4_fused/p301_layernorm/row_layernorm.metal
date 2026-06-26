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
    // Paired reduction: one scratch for sum, one for sumsq.
    threadgroup float sum_scratch[K];
    threadgroup float sqs_scratch[K];

    uint b   = tg_in_grid;
    uint tid = tid_in_tg;
    uint idx = b * K + tid;

    // Phase 1: load + initialize partial sums.
    float v = x[idx];
    sum_scratch[tid] = v;
    sqs_scratch[tid] = v * v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Combined tree reduction. Both arrays move in lockstep — one
    // barrier per stage covers both reductions.
    for (uint stride = K / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_scratch[tid] += sum_scratch[tid + stride];
            sqs_scratch[tid] += sqs_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Every thread reads the final sum + sumsq from scratch[0].
    // Compute mean, var = sumsq/K - mean^2, and the inverse-stddev
    // scale factor in registers — no further threadgroup traffic.
    float mean = sum_scratch[0] / float(K);
    float var  = sqs_scratch[0] / float(K) - mean * mean;
    float inv_std = rsqrt(var + EPS);

    // Phase 3: per-element affine using the row's mean/var and the
    // per-column gamma/beta.
    out[idx] = (v - mean) * inv_std * gamma[tid] + beta[tid];
}
