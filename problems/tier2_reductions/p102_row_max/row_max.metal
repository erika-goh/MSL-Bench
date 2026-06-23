#include <metal_stdlib>
using namespace metal;

constant uint K = 256;

kernel void row_max(device const float* x   [[buffer(0)]],
                    device float* out       [[buffer(1)]],
                    uint tid_in_tg  [[thread_position_in_threadgroup]],
                    uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[K];

    uint b   = tg_in_grid;
    uint tid = tid_in_tg;

    scratch[tid] = x[b * K + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction with max. Barrier outside the `if` — all threads in
    // the group must hit it. Identity element (-INFINITY) is unused here
    // because every thread already loaded a real value above.
    for (uint stride = K / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[b] = scratch[0];
    }
}
