#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

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

    simdgroup_matrix<float, 8, 8> C_frag = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> A_frag;
    simdgroup_matrix<float, 8, 8> B_frag;

    for (uint kt = 0; kt < K / SG; kt++) {
        uint k0 = kt * SG;
        simdgroup_load(A_frag, a + m0 * K + k0, K);
        simdgroup_load(B_frag, b + k0 * N + n0, N);
        simdgroup_multiply_accumulate(C_frag, A_frag, B_frag, C_frag);
    }

    simdgroup_store(C_frag, c + m0 * N + n0, N);
}
