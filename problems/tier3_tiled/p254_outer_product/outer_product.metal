// Outer product C[i,j] = u[i] * v[j], tiled with threadgroup staging.
// Each 16x16 threadgroup stages the TILE u-values (loaded by column tx==0)
// and TILE v-values (loaded by row ty==0) for its block, then every thread
// multiplies one staged u by one staged v. Each loaded value is reused TILE
// times, which is the reason to stage rather than reload from device memory.
#include <metal_stdlib>
using namespace metal;

constant uint N    = 4096;
constant uint TILE = 16;

kernel void outer_product(device const float* u   [[buffer(0)]],
                          device const float* v   [[buffer(1)]],
                          device float*       c   [[buffer(2)]],
                          uint2 tid_in_tg  [[thread_position_in_threadgroup]],
                          uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float u_tile[TILE];
    threadgroup float v_tile[TILE];

    uint tx = tid_in_tg.x, ty = tid_in_tg.y;
    uint bx = tg_in_grid.x, by = tg_in_grid.y;

    uint i = by * TILE + ty;   // row  -> u index
    uint j = bx * TILE + tx;   // col  -> v index

    // One edge of threads loads each staged vector.
    if (tx == 0) u_tile[ty] = u[i];
    if (ty == 0) v_tile[tx] = v[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    c[i * N + j] = u_tile[ty] * v_tile[tx];
}
