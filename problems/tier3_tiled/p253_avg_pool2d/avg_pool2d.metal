// 2x2 average pooling, tiled with threadgroup staging.
// Each 16x16 threadgroup owns a 16x16 output block <-> a 32x32 input tile.
// The 256 threads cooperatively load the 1024-element input tile (4 each,
// via a linearized thread id), barrier, then each thread averages its own
// 2x2 window straight from the staged tile.
#include <metal_stdlib>
using namespace metal;

constant uint W_IN  = 2048;
constant uint W_OUT = 1024;
constant uint TILE  = 16;      // output-tile edge
constant uint ITILE = 32;      // input-tile edge = 2 * TILE

kernel void avg_pool2d(device const float* x   [[buffer(0)]],
                       device float*       out [[buffer(1)]],
                       uint2 tid_in_tg  [[thread_position_in_threadgroup]],
                       uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float tile[ITILE][ITILE];

    uint tx = tid_in_tg.x, ty = tid_in_tg.y;
    uint bx = tg_in_grid.x, by = tg_in_grid.y;

    // Cooperative load: 256 threads * 4 = 1024 = 32*32 elements.
    uint lin = ty * TILE + tx;                 // 0..255
    for (uint n = 0; n < 4; n++) {
        uint idx  = lin + n * 256;             // 0..1023
        uint r    = idx / ITILE;               // 0..31
        uint col  = idx % ITILE;               // 0..31
        uint grow = by * ITILE + r;
        uint gcol = bx * ITILE + col;
        tile[r][col] = x[grow * W_IN + gcol];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread averages its 2x2 window from the staged tile.
    uint r0 = 2 * ty, c0 = 2 * tx;
    float s = tile[r0][c0] + tile[r0][c0 + 1]
            + tile[r0 + 1][c0] + tile[r0 + 1][c0 + 1];

    uint orow = by * TILE + ty;
    uint ocol = bx * TILE + tx;
    out[orow * W_OUT + ocol] = s * 0.25f;
}
