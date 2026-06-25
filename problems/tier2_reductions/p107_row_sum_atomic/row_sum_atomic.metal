#include <metal_stdlib>
using namespace metal;

constant uint K       = 256;
constant uint K_CHUNK = 64;

kernel void row_sum_atomic(
    device const float*   x   [[buffer(0)]],
    device atomic_float*  out [[buffer(1)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[K_CHUNK];

    uint tid = tid_in_tg.x;            // .y is always 0 (TG_Y=1)
    uint row = tg_in_grid.y;
    uint col = tg_in_grid.x * K_CHUNK + tid;

    scratch[tid] = x[row * K + col];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduce 64 → 1 in shared memory (6 stages).
    for (uint stride = K_CHUNK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        atomic_fetch_add_explicit(&out[row], scratch[0], memory_order_relaxed);
    }
}
