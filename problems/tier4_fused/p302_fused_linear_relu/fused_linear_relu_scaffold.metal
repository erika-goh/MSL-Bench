// Scaffold for p302_fused_linear_relu. Save as `fused_linear_relu.metal`.
//
// Buffer bindings:
//   buffer(0) = x   (input, M*K floats, row-major)
//   buffer(1) = w   (weights, K*N floats, row-major)
//   buffer(2) = b   (per-column bias, N floats)
//   buffer(3) = out (output, M*N floats)
//
// Dispatch geometry (same as p202):
//   grid        = ((N/SG)*32, M/SG, 1) = (4096, 128, 1)
//   threadgroup = (32,         1,    1)
//
// Single SIMD-group per TG produces one 8x8 tile of `out`. Three
// phases:
//
//   Phase 1 (matmul): K-loop over the inner dim, simdgroup matrix
//   unit accumulates into C_frag. Same as p202.
//
//   Phase 2 (epilogue stage): store C_frag to a 64-element
//   threadgroup tile so individual threads can address its values.
//
//   Phase 3 (bias + ReLU + write): each of the 32 threads picks 2
//   of the 64 tile values, adds the per-column bias, applies ReLU,
//   writes to device memory.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint N  = 1024;
constant uint K  = 1024;
constant uint SG = 8;

kernel void fused_linear_relu(
    device const float* x    [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device const float* b    [[buffer(2)]],
    device float*       out  [[buffer(3)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    uint by = tg_in_grid.y;
    uint bx = tg_in_grid.x;
    uint m0 = by * SG;
    uint n0 = bx * SG;

    // ============================================================
    // TODO 1 (matmul phase): same as p202.
    //
    // simdgroup_matrix<float, 8, 8> C_frag = simdgroup_matrix<float, 8, 8>(0.0f);
    // simdgroup_matrix<float, 8, 8> A_frag, B_frag;
    // for (uint kt = 0; kt < K / SG; kt++) {
    //     uint k0 = kt * SG;
    //     simdgroup_load(A_frag, x + m0 * K + k0, K);
    //     simdgroup_load(B_frag, w + k0 * N + n0, N);
    //     simdgroup_multiply_accumulate(C_frag, A_frag, B_frag, C_frag);
    // }
    // ============================================================



    // ============================================================
    // TODO 2: stage C_frag into threadgroup memory so individual
    // threads can address values.
    //
    // threadgroup float tile[SG * SG];   // 64 floats
    // simdgroup_store(C_frag, tile, SG);
    // threadgroup_barrier(mem_flags::mem_threadgroup);
    // ============================================================



    // ============================================================
    // TODO 3 (bias + ReLU + write): each of 32 threads handles 2
    // of the 64 elements.
    //
    // uint tid = tid_in_tg.x;
    // for (uint i = 0; i < 2; i++) {
    //     uint local = tid * 2 + i;          // 0..63
    //     uint row   = local / SG;
    //     uint col   = local % SG;
    //     float v = tile[local] + b[n0 + col];
    //     v = max(v, 0.0f);
    //     out[(m0 + row) * N + (n0 + col)] = v;
    // }
    // ============================================================



}
