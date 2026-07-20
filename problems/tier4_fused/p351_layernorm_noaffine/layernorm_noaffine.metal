// Row-wise layernorm without affine, fused into one dispatch.
// Combined sum + sum-of-squares tree reduction (both scratch arrays move in
// lockstep, one barrier per stage), then var = sumsq/K - mean^2, then each
// thread normalizes its own element. Same skeleton as p301, affine dropped.
#include <metal_stdlib>
using namespace metal;

constant uint  K   = 256;
constant float EPS = 1e-5f;

kernel void layernorm_noaffine(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    uint tid_in_tg  [[thread_position_in_threadgroup]],
    uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float sum_scratch[K];
    threadgroup float sqs_scratch[K];

    uint b   = tg_in_grid;
    uint tid = tid_in_tg;
    uint idx = b * K + tid;

    float v = x[idx];
    sum_scratch[tid] = v;
    sqs_scratch[tid] = v * v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = K / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_scratch[tid] += sum_scratch[tid + stride];
            sqs_scratch[tid] += sqs_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean    = sum_scratch[0] / float(K);
    float var     = sqs_scratch[0] / float(K) - mean * mean;
    float inv_std = rsqrt(var + EPS);

    out[idx] = (v - mean) * inv_std;
}
