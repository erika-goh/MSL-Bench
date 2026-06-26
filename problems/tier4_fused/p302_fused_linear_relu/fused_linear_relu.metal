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

    // Matmul phase: same shape as p202.
    simdgroup_matrix<float, 8, 8> C_frag = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> A_frag;
    simdgroup_matrix<float, 8, 8> B_frag;

    for (uint kt = 0; kt < K / SG; kt++) {
        uint k0 = kt * SG;
        simdgroup_load(A_frag, x + m0 * K + k0, K);
        simdgroup_load(B_frag, w + k0 * N + n0, N);
        simdgroup_multiply_accumulate(C_frag, A_frag, B_frag, C_frag);
    }

    // Epilogue: stage the 8x8 fragment in threadgroup memory, then
    // 32 threads each handle 2 elements — add bias, ReLU, write out.
    threadgroup float tile[SG * SG];  // 64 floats
    simdgroup_store(C_frag, tile, SG);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tid = tid_in_tg.x;
    for (uint i = 0; i < 2; i++) {
        uint local = tid * 2 + i;   // 0..63 covering the 8x8 tile
        uint row   = local / SG;
        uint col   = local % SG;
        float v = tile[local] + b[n0 + col];
        v = max(v, 0.0f);
        out[(m0 + row) * N + (n0 + col)] = v;
    }
}
