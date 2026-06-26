#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// M is implicit in the grid.y dimension; not referenced explicitly.
constant uint N       = 1024;
constant uint K       = 1024;
constant uint TG_M    = 16;
constant uint TG_N    = 16;
constant uint K_STAGE = 32;
constant uint SG      = 8;

kernel void matmul_simdgroup_staged(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float*       c    [[buffer(2)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    // Staged input slabs. TG_M=16 rows of A by K_STAGE=32 cols, etc.
    threadgroup float A_stage[TG_M * K_STAGE];  // 512 floats
    threadgroup float B_stage[K_STAGE * TG_N];  // 512 floats

    uint tid = tid_in_tg.x;        // 0..31 (TG_Y=1)
    uint by  = tg_in_grid.y;
    uint bx  = tg_in_grid.x;
    uint m0  = by * TG_M;
    uint n0  = bx * TG_N;

    // Four 8x8 accumulators covering the TG's 16x16 output patch.
    simdgroup_matrix<float, 8, 8> C_tl = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> C_tr = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> C_bl = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> C_br = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint ko = 0; ko < K; ko += K_STAGE) {
        // Cooperative load A[m0..m0+15, ko..ko+31] into A_stage.
        // 32 threads x 16 iterations covers 512 elements. At each iter
        // step i, threads 0..31 read consecutive A device addresses
        // (one row's slice) -> coalesced.
        for (uint i = 0; i < (TG_M * K_STAGE) / 32; i++) {
            uint idx = i * 32 + tid;
            uint row = idx / K_STAGE;
            uint col = idx % K_STAGE;
            A_stage[idx] = a[(m0 + row) * K + (ko + col)];
        }
        // Cooperative load B[ko..ko+31, n0..n0+15] into B_stage.
        for (uint i = 0; i < (K_STAGE * TG_N) / 32; i++) {
            uint idx = i * 32 + tid;
            uint row = idx / TG_N;
            uint col = idx % TG_N;
            B_stage[idx] = b[(ko + row) * N + (n0 + col)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner K-loop over the staged slab. Each sub-step loads four
        // 8x8 fragments from threadgroup memory and runs four MACs.
        for (uint ki = 0; ki < K_STAGE; ki += SG) {
            simdgroup_matrix<float, 8, 8> A_top, A_bot, B_left, B_right;
            simdgroup_load(A_top,   A_stage + 0 * K_STAGE + ki, K_STAGE);
            simdgroup_load(A_bot,   A_stage + 8 * K_STAGE + ki, K_STAGE);
            simdgroup_load(B_left,  B_stage + ki * TG_N + 0,    TG_N);
            simdgroup_load(B_right, B_stage + ki * TG_N + 8,    TG_N);

            simdgroup_multiply_accumulate(C_tl, A_top, B_left,  C_tl);
            simdgroup_multiply_accumulate(C_tr, A_top, B_right, C_tr);
            simdgroup_multiply_accumulate(C_bl, A_bot, B_left,  C_bl);
            simdgroup_multiply_accumulate(C_br, A_bot, B_right, C_br);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store the four output tiles to their slots in C.
    simdgroup_store(C_tl, c + (m0    ) * N + (n0    ), N);
    simdgroup_store(C_tr, c + (m0    ) * N + (n0 + 8), N);
    simdgroup_store(C_bl, c + (m0 + 8) * N + (n0    ), N);
    simdgroup_store(C_br, c + (m0 + 8) * N + (n0 + 8), N);
}
