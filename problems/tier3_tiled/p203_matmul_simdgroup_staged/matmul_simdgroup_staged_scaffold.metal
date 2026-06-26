// Scaffold for p203_matmul_simdgroup_staged. Save as
// `matmul_simdgroup_staged.metal`. Don't edit this scaffold file.
//
// Combines two optimization layers on top of p202:
//   1. Threadgroup-memory staging of A and B slabs
//   2. Four 8x8 output tiles accumulated per TG (16x16 output patch)
//
// Buffer bindings:
//   buffer(0) = a (M*K floats, row-major)
//   buffer(1) = b (K*N floats, row-major)
//   buffer(2) = c (M*N floats, row-major, output)
//
// Dispatch geometry:
//   grid        = ((N/TG_N)*TGS, M/TG_M, 1) = (2048, 64, 1)
//   threadgroup = (TGS=32,       1,      1)
//
// One SIMD-group (32 threads) per TG. Each TG produces a 16x16 patch
// of C as four separate 8x8 matrix-unit tiles.
//
// ============================================================
// The two optimizations, and why they need each other
// ============================================================
//
// Staging into TG memory alone doesn't help: device reads are the same,
// just routed through TG memory. The win comes from REUSE — when
// multiple output tiles share staged inputs.
//
// Multi-tile alone (without staging) does give reuse but every K-iter
// still touches device memory four times. Putting them together lets
// each device load fan out across multiple matrix-unit ops AND
// multiple output tiles.
//
// In this kernel:
//   - A staged slab serves both top and bottom rows of C output
//   - B staged slab serves both left and right cols of C output
//   - 2x reuse on each side, 2x*2x = 4x reuse total per stage load.
//
// Device reads per output element drop from p202's ~256 to ~128.
//
// ============================================================
// Two barriers per outer iter
// ============================================================
//
//   1. After staging loads finish, before any matrix-unit op reads
//      the staged tiles.
//   2. After the inner K-loop finishes reading the staged tiles,
//      before the next outer iter overwrites them.
//
// No barrier inside the inner K-loop because matrix-unit ops are
// SIMD-group-synchronous by construction.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

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
    threadgroup float A_stage[TG_M * K_STAGE];  // 16*32 = 512 floats
    threadgroup float B_stage[K_STAGE * TG_N];  // 32*16 = 512 floats

    uint tid = tid_in_tg.x;
    uint by  = tg_in_grid.y;
    uint bx  = tg_in_grid.x;
    uint m0  = by * TG_M;
    uint n0  = bx * TG_N;

    // ============================================================
    // TODO 1: declare the four 8x8 accumulators, all initialized to 0.
    //
    //   simdgroup_matrix<float, 8, 8> C_tl = simdgroup_matrix<float, 8, 8>(0.0f);
    //   simdgroup_matrix<float, 8, 8> C_tr = simdgroup_matrix<float, 8, 8>(0.0f);
    //   simdgroup_matrix<float, 8, 8> C_bl = simdgroup_matrix<float, 8, 8>(0.0f);
    //   simdgroup_matrix<float, 8, 8> C_br = simdgroup_matrix<float, 8, 8>(0.0f);
    // ============================================================



    // ============================================================
    // TODO 2: outer K-loop, stepping K_STAGE columns at a time.
    //
    // for (uint ko = 0; ko < K; ko += K_STAGE) {
    //     // (a) Cooperative loads. 32 threads each handle 16 entries
    //     //     of A_stage and 16 of B_stage. Iteration scheme: at
    //     //     step i, threads 0..31 read consecutive addresses of
    //     //     a row's slice → coalesced.
    //     for (uint i = 0; i < (TG_M * K_STAGE) / 32; i++) {
    //         uint idx = i * 32 + tid;
    //         uint row = idx / K_STAGE;
    //         uint col = idx % K_STAGE;
    //         A_stage[idx] = a[(m0 + row) * K + (ko + col)];
    //     }
    //     for (uint i = 0; i < (K_STAGE * TG_N) / 32; i++) {
    //         uint idx = i * 32 + tid;
    //         uint row = idx / TG_N;
    //         uint col = idx % TG_N;
    //         B_stage[idx] = b[(ko + row) * N + (n0 + col)];
    //     }
    //     threadgroup_barrier(mem_flags::mem_threadgroup);
    //
    //     // (b) Inner K-loop over the staged slab. SG=8 sub-steps.
    //     for (uint ki = 0; ki < K_STAGE; ki += SG) {
    //         simdgroup_matrix<float, 8, 8> A_top, A_bot, B_left, B_right;
    //         simdgroup_load(A_top,   A_stage + 0 * K_STAGE + ki, K_STAGE);
    //         simdgroup_load(A_bot,   A_stage + 8 * K_STAGE + ki, K_STAGE);
    //         simdgroup_load(B_left,  B_stage + ki * TG_N + 0,    TG_N);
    //         simdgroup_load(B_right, B_stage + ki * TG_N + 8,    TG_N);
    //
    //         simdgroup_multiply_accumulate(C_tl, A_top, B_left,  C_tl);
    //         simdgroup_multiply_accumulate(C_tr, A_top, B_right, C_tr);
    //         simdgroup_multiply_accumulate(C_bl, A_bot, B_left,  C_bl);
    //         simdgroup_multiply_accumulate(C_br, A_bot, B_right, C_br);
    //     }
    //     threadgroup_barrier(mem_flags::mem_threadgroup);
    // }
    // ============================================================



    // ============================================================
    // TODO 3: store the four accumulators to their slots in C.
    //
    //   simdgroup_store(C_tl, c + (m0    ) * N + (n0    ), N);
    //   simdgroup_store(C_tr, c + (m0    ) * N + (n0 + 8), N);
    //   simdgroup_store(C_bl, c + (m0 + 8) * N + (n0    ), N);
    //   simdgroup_store(C_br, c + (m0 + 8) * N + (n0 + 8), N);
    // ============================================================



}
