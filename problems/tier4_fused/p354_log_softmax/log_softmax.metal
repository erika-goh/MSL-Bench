// Row-wise log-softmax, fused. One threadgroup per row, K threads.
// Pass 1: tree-reduce max (subtracting it keeps exp finite).
// Pass 2: tree-reduce sum of exp(x - max).
// Finish: out = (x - max) - log(sum). Two scratch arrays so the max survives
// while the sum reduction runs.
#include <metal_stdlib>
using namespace metal;

constant uint K = 256;

kernel void log_softmax(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    uint tid_in_tg  [[thread_position_in_threadgroup]],
    uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float max_scratch[K];
    threadgroup float sum_scratch[K];

    uint b   = tg_in_grid;
    uint tid = tid_in_tg;
    uint idx = b * K + tid;

    float v = x[idx];

    // Pass 1: row max.
    max_scratch[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = K / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_scratch[tid] = max(max_scratch[tid], max_scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = max_scratch[0];

    // Pass 2: sum of exp(x - max).
    float e = exp(v - m);
    sum_scratch[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = K / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_scratch[tid] += sum_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float s = sum_scratch[0];

    out[idx] = (v - m) - log(s);
}
