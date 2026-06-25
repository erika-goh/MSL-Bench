#include <metal_stdlib>
using namespace metal;

constant uint K = 256;

kernel void row_softmax(device const float* x   [[buffer(0)]],
                        device float* out       [[buffer(1)]],
                        uint tid_in_tg  [[thread_position_in_threadgroup]],
                        uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[K];

    uint b   = tg_in_grid;
    uint tid = tid_in_tg;
    uint idx = b * K + tid;

    // Phase 1: load + max reduction. scratch[0] holds the row max afterward.
    float value = x[idx];
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = K / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Every thread reads the row max from scratch[0]. Need a barrier
    // BEFORE phase 2 overwrites scratch[] so this read finishes first.
    float row_max = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: exp(x - row_max), then sum reduction.
    float e = exp(value - row_max);
    scratch[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = K / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // No further writes to scratch[], so no barrier needed after this read.
    float row_sum = scratch[0];

    out[idx] = e / row_sum;
}
