#include <metal_stdlib>
using namespace metal;

constant uint K = 256;  // row width; matches spec's K and threadgroup size

kernel void row_sum(device const float* x   [[buffer(0)]],
                    device float* out       [[buffer(1)]],
                    uint tid_in_tg  [[thread_position_in_threadgroup]],
                    uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[K];

    uint b   = tg_in_grid;
    uint tid = tid_in_tg;

    // TODO 1: each thread loads one element. Row-major: x[b][tid] = x[b*K + tid].
    scratch[tid] = x[b * K + tid];

    // TODO 2: every load must be visible before any reduction step reads it.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // TODO 3: tree reduction. Stages: 128, 64, 32, 16, 8, 4, 2, 1.
    // The barrier is OUTSIDE the `if` — barriers must be hit by all threads
    // in the group, even ones that did no work this stage.
    for (uint stride = K / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // TODO 4: after log2(K)=8 stages, scratch[0] holds the row sum.
    if (tid == 0) {
        out[b] = scratch[0];
    }
}
