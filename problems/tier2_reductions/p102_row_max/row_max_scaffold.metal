// Scaffold for p102_row_max. Save your version as `row_max.metal`.
// Don't edit this scaffold file in place; keep it as a reference.
//
// Buffer bindings:
//   buffer(0) = x   (input, B*K floats, row-major)
//   buffer(1) = out (output, B floats)
//
// Dispatch geometry (from the spec):
//   grid        = (B*K, 1, 1)
//   threadgroup = (K,   1, 1)
//
// This is structurally identical to p101 row_sum. The only changes:
//   - replace `+=` with `max(a, b)` in the tree reduction
//   - (if grid > input) the identity element is `-INFINITY`, not 0
//
// Why max needs no tolerance loosening:
//   Float addition is non-associative, so tree-summed vs CPU-summed totals
//   diverge by ~sqrt(K)*eps*|sum| (~1e-5 for K=256). Max is associative AND
//   commutative for non-NaN floats — every reduction order returns the
//   same bit pattern. So `out` should match torch.max exactly.

#include <metal_stdlib>
using namespace metal;

constant uint K = 256;

kernel void row_max(device const float* x   [[buffer(0)]],
                    device float* out       [[buffer(1)]],
                    uint tid_in_tg  [[thread_position_in_threadgroup]],
                    uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[K];

    uint b   = tg_in_grid;
    uint tid = tid_in_tg;

    // ============================================================
    // TODO 1: load x[b][tid] into scratch[tid].
    // Same as p101.
    // ============================================================



    // ============================================================
    // TODO 2: barrier (mem_threadgroup) so every load is visible.
    // ============================================================



    // ============================================================
    // TODO 3: tree reduction with max(). For stride = K/2, K/4, ..., 1:
    //   - if tid < stride: scratch[tid] = max(scratch[tid], scratch[tid+stride]);
    //   - barrier outside the if.
    //
    // metal::max(a, b) is the elementwise float max — IEEE NaN-propagating.
    // Our spec uses randn so NaNs won't appear; for an input that may
    // contain NaN, use `fmax` instead (treats NaN as missing).
    // ============================================================



    // ============================================================
    // TODO 4: thread 0 writes scratch[0] to out[b].
    // ============================================================



}
