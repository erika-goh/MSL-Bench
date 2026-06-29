#include <metal_stdlib>
using namespace metal;

// Tiled transpose. Each threadgroup handles one TILE x TILE block of A,
// staging it through threadgroup memory so that both the device read
// (from A) and the device write (to B) are coalesced across SIMD-group
// lanes. The naïve thread-per-element transpose has uncoalesced writes
// — adjacent threads writing addresses M floats apart.
constant uint M    = 2048;
constant uint N    = 2048;
constant uint TILE = 16;

kernel void transpose_kernel(
    device const float* a    [[buffer(0)]],
    device float*       b    [[buffer(1)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float tile[TILE][TILE];

    uint tx = tid_in_tg.x;
    uint ty = tid_in_tg.y;
    uint bx = tg_in_grid.x;   // column tile index (in A's frame)
    uint by = tg_in_grid.y;   // row tile index (in A's frame)

    // ----- read A's tile into threadgroup memory, coalesced -----
    // Adjacent threads (varying tx, same ty) read addresses one apart
    // in A → one cache-line load per warp row.
    uint a_row = by * TILE + ty;
    uint a_col = bx * TILE + tx;
    tile[ty][tx] = a[a_row * N + a_col];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ----- write B's transposed tile, coalesced -----
    // The tile we just loaded covered A[by*TILE..(by+1)*TILE - 1,
    // bx*TILE..(bx+1)*TILE - 1]. The corresponding region of B is at
    // B[bx*TILE..(bx+1)*TILE - 1, by*TILE..(by+1)*TILE - 1], and the
    // transpose maps tile[ty][tx] → B[bx*TILE + tx][by*TILE + ty].
    //
    // To make the WRITE coalesced (the whole point of staging), we
    // assign each thread an output coordinate (b_row, b_col) where
    // b_col varies with tx and b_row varies with ty — same shape as
    // the read. The TG-memory read becomes tile[tx][ty] (swapped),
    // which has no coalescing penalty because TG memory access is not
    // bandwidth-bound at this tile size.
    uint b_row = bx * TILE + ty;
    uint b_col = by * TILE + tx;
    b[b_row * M + b_col] = tile[tx][ty];
}
