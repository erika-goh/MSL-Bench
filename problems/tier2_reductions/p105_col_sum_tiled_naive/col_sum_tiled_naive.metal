#include <metal_stdlib>
using namespace metal;

constant uint K     = 256;
constant uint TG_X  = 32;
constant uint TG_Y  = 8;
constant uint B     = 262144;

kernel void col_sum_tiled_naive(
    device const float* x       [[buffer(0)]],
    device float*       out     [[buffer(1)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[TG_X][TG_Y];

    uint tx  = tid_in_tg.x;
    uint ty  = tid_in_tg.y;
    uint col = tg_in_grid.x * TG_X + tx;

    // 32 threads sharing the same ty (one SIMD-group) read 32 consecutive
    // floats at any single instant → one coalesced device-memory load.
    float partial = 0.0f;
    for (uint r = ty; r < B; r += TG_Y) {
        partial += x[r * K + col];
    }

    scratch[tx][ty] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Combine the TG_Y=8 partials per column. Sequential loop is faster
    // than a tree reduction at this size.
    if (ty == 0) {
        float total = 0.0f;
        for (uint i = 0; i < TG_Y; i++) {
            total += scratch[tx][i];
        }
        out[col] = total;
    }
}
