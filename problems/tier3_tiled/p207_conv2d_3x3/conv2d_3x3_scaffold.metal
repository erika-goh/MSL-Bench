// Scaffold for p207_conv2d_3x3. Save as `conv2d_3x3.metal`.
//
// y = conv2d(x, w), 3×3 valid (no padding, stride 1).
//   x is (IN_H, IN_W) = (1026, 1026)
//   w is (K, K)       = (3, 3)
//   y is (OUT_H, OUT_W) = (1024, 1024) = (IN_H - K + 1, IN_W - K + 1)
//
// Each thread computes ONE output element y[i_out, j_out] via a
// 9-element 2D dot product:
//
//   y[i, j] = sum_{ki=0..2} sum_{kj=0..2} x[i + ki, j + kj] * w[ki, kj]
//
// This is cross-correlation (no kernel flip), matching PyTorch's
// F.conv2d behavior.
//
// Memory pattern:
//   * At fixed (ki, kj), adjacent threads (varying j_out) read
//     adjacent x addresses → coalesced.
//   * w is 36 bytes total, fits in L1 / constant cache after the
//     first access. No staging needed.
//
// Dispatch:
//   grid        = (OUT_W, OUT_H, 1) = (1024, 1024, 1)   one thread / output
//   threadgroup = (TILE,  TILE,  1) = (16,   16,   1)   256 threads / TG

#include <metal_stdlib>
using namespace metal;

constant uint IN_W  = 1026;
constant uint K     = 3;
constant uint OUT_W = 1024;

kernel void conv2d_3x3_kernel(
    device const float* x   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device float*       y   [[buffer(2)]],
    uint2 idx [[thread_position_in_grid]])
{
    uint j_out = idx.x;
    uint i_out = idx.y;

    // ============================================================
    // TODO: 9-element 2D dot product.
    //
    //   float acc = 0.0f;
    //   for (uint ki = 0; ki < K; ki++) {
    //       uint x_row_off = (i_out + ki) * IN_W;
    //       uint w_row_off = ki * K;
    //       for (uint kj = 0; kj < K; kj++) {
    //           acc += x[x_row_off + j_out + kj] * w[w_row_off + kj];
    //       }
    //   }
    //   y[i_out * OUT_W + j_out] = acc;
    //
    // Note: K (=3) is small enough that the inner loop should
    // unroll automatically. Pre-multiplying ki by IN_W and K once
    // per outer iteration is a micro-optimization the compiler
    // could also do, but writing it explicitly makes the offset
    // arithmetic easier to audit.
    // ============================================================
}
