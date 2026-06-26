// Scaffold for p202_matmul_simdgroup. Save as `matmul_simdgroup.metal`.
//
// Buffer bindings (same as p201):
//   buffer(0) = a (M*K floats, row-major)
//   buffer(1) = b (K*N floats, row-major)
//   buffer(2) = c (M*N floats, row-major, output)
//
// Dispatch geometry:
//   grid        = (N/SG * 32, M/SG, 1) = (4096, 128, 1)
//   threadgroup = (32,        1,    1)
//
// 32 threads per TG = one SIMD-group per TG. Each TG computes one 8x8
// tile of C. Number of TGs: (N/SG) * (M/SG) = 128 * 128 = 16,384.
//
// ============================================================
// What's new: the SIMD-group matrix type
// ============================================================
//
// Apple GPUs have a matrix unit accessible at the SIMD-group level.
// `simdgroup_matrix<float, 8, 8>` is a typed handle for an 8x8 float
// matrix whose values are distributed across the 32 threads of one
// SIMD-group. You cannot index it from a single thread — the 32
// threads collectively own it. Three operations are exposed:
//
//   - simdgroup_load(mat, ptr, stride)
//       32 threads cooperatively load an 8x8 tile from device memory.
//       `stride` is the row stride in ELEMENTS (not bytes).
//
//   - simdgroup_multiply_accumulate(D, A, B, C)
//       D = A * B + C, executed in hardware. D and C can be the same
//       matrix object (accumulator pattern).
//
//   - simdgroup_store(mat, ptr, stride)
//       32 threads cooperatively store the 8x8 tile back to memory.
//
// No threadgroup_barrier appears in the hot loop. The matrix ops are
// SIMD-group-synchronous by construction.
//
// ============================================================
// Algorithm
// ============================================================
//
// For TG (by, bx) = (tg_in_grid.y, tg_in_grid.x / 1) — the per-TG
// 8x8 output tile starts at row by*SG, col bx*SG. (We don't index
// individual threads inside the TG; the matrix ops use all 32.)
//
//   accumulator C_frag = 0
//   for k_tile in 0..K/SG:
//       load A_frag = A[by*SG .. by*SG+7, k_tile*SG .. k_tile*SG+7]
//       load B_frag = B[k_tile*SG .. k_tile*SG+7, bx*SG .. bx*SG+7]
//       C_frag = A_frag * B_frag + C_frag
//   store C_frag to C[by*SG .. by*SG+7, bx*SG .. bx*SG+7]

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint M  = 1024;
constant uint N  = 1024;
constant uint K  = 1024;
constant uint SG = 8;

kernel void matmul_simdgroup(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float*       c    [[buffer(2)]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    uint by = tg_in_grid.y;
    uint bx = tg_in_grid.x;
    uint m0 = by * SG;
    uint n0 = bx * SG;

    // ============================================================
    // TODO 1: declare the accumulator and the two operand fragments.
    //
    //   simdgroup_matrix<float, 8, 8> C_frag = simdgroup_matrix<float, 8, 8>(0.0f);
    //   simdgroup_matrix<float, 8, 8> A_frag;
    //   simdgroup_matrix<float, 8, 8> B_frag;
    //
    // C_frag is initialized to zero so the accumulate pattern works
    // on the first iteration.
    // ============================================================



    // ============================================================
    // TODO 2: K-loop walking the inner dimension in SG-sized steps.
    //
    // for (uint kt = 0; kt < K / SG; kt++) {
    //     uint k0 = kt * SG;
    //     simdgroup_load(A_frag, a + m0 * K + k0, K);   // A tile at (m0, k0)
    //     simdgroup_load(B_frag, b + k0 * N + n0, N);   // B tile at (k0, n0)
    //     simdgroup_multiply_accumulate(C_frag, A_frag, B_frag, C_frag);
    // }
    //
    // The matrix-unit op is the expensive line; the loads from device
    // memory are the bottleneck unless we add a threadgroup-staging
    // optimization (future problem).
    // ============================================================



    // ============================================================
    // TODO 3: store C_frag to its 8x8 slot in C.
    //
    //   simdgroup_store(C_frag, c + m0 * N + n0, N);
    // ============================================================



}
