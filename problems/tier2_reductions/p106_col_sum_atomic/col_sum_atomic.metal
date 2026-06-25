#include <metal_stdlib>
using namespace metal;

constant uint K         = 256;
constant uint TG_X      = 32;
constant uint TG_Y      = 8;
constant uint ROW_CHUNK = 1024;

kernel void col_sum_atomic(
    device const float*   x   [[buffer(0)]],
    device atomic_float*  out [[buffer(1)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[TG_X][TG_Y];

    uint tx = tid_in_tg.x;
    uint ty = tid_in_tg.y;
    uint col = tg_in_grid.x * TG_X + tx;
    uint row_start = tg_in_grid.y * ROW_CHUNK;
    uint row_end   = row_start + ROW_CHUNK;

    // Coalesced loads: at any instant the 32 threads of a SIMD-group
    // share ty and have consecutive tx → 32 consecutive addresses.
    float partial = 0.0f;
    for (uint r = row_start + ty; r < row_end; r += TG_Y) {
        partial += x[r * K + col];
    }

    scratch[tx][ty] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Combine TG_Y partials per column → one value per column per TG.
    if (ty == 0) {
        float total = 0.0f;
        for (uint i = 0; i < TG_Y; i++) {
            total += scratch[tx][i];
        }
        // Combine across TGs. memory_order_relaxed: we need atomicity,
        // not ordering with any other memory op.
        atomic_fetch_add_explicit(&out[col], total, memory_order_relaxed);
    }
}
