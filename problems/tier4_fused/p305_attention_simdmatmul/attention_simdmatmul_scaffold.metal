// Scaffold for p305_attention_simdmatmul. Save as `attention_simdmatmul.metal`.
//
// Same fused attention as p304 (M = D = 512), but the two matmul phases
// run on the Apple GPU matrix unit via simdgroup_matrix<float, 8, 8>.
// The softmax in between stays as a per-row simdgroup lane reduction.
//
// Buffer bindings:
//   buffer(0) = q   (M*D floats, row-major)
//   buffer(1) = k   (M*D floats, row-major)
//   buffer(2) = v   (M*D floats, row-major)
//   buffer(3) = out (M*D floats)
//
// Dispatch geometry:
//   grid        = (M/8 * 256, 1, 1)   = (16384, 1, 1)
//   threadgroup = (256, 1, 1)         = 8 SIMD groups × 32 lanes
//
// Layout:
//   * 64 TGs total. TG t handles query rows [8t, 8t+8).
//   * 8 SIMD groups per TG. SG s handles output column block
//     [s*64, (s+1)*64) in BOTH phase 1 (scores) and phase 3 (out).
//   * In phase 2, SG s does softmax for ROW s (one SG per row).
//
// Threadgroup memory:
//   scores[8 × 512] = 16 KB (well under 32 KB).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint  M     = 512;
constant uint  D     = 512;
constant float SCALE = 0.04419417382415922f;

constant uint  ROWS_PER_TG    = 8;
constant uint  SG             = 8;
constant uint  SGS_PER_TG     = 8;
constant uint  COLS_PER_SG_PHASE1 = M / SGS_PER_TG;  // 64
constant uint  COLS_PER_SG_PHASE3 = D / SGS_PER_TG;  // 64

kernel void attention_simdmatmul(
    device const float* q   [[buffer(0)]],
    device const float* k   [[buffer(1)]],
    device const float* v   [[buffer(2)]],
    device float*       out [[buffer(3)]],
    uint  tg_in_grid                [[threadgroup_position_in_grid]],
    uint  sg_in_tg                  [[simdgroup_index_in_threadgroup]],
    uint  lane                      [[thread_index_in_simdgroup]])
{
    threadgroup float scores[ROWS_PER_TG * M];

    uint m_base = tg_in_grid * ROWS_PER_TG;

    // ====================================================================
    // TODO PHASE 1: tiled QK^T into scores[8 x M] scratch
    // ====================================================================
    // SG s computes 8 output tiles at columns [s*64 .. (s+1)*64). For
    // each tile (top row 0, top col c0):
    //   C_frag = 0
    //   for kt in 0..D/8:
    //       Q_frag = load Q[m_base..m_base+7, kt*8..kt*8+7]          (no transpose)
    //       K_frag = load K[c0..c0+7,        kt*8..kt*8+7]           (TRANSPOSE = true)
    //       C_frag += Q_frag · K_frag
    //   simdgroup_store(C_frag, scores + c0, M);
    //
    // Why transpose-load K: we want C = Q @ K^T. Loading K transposed
    // lets the matrix unit do its standard A @ B operation.



    // After all 8 SGs finish writing their score columns:
    //
    //   threadgroup_barrier(mem_flags::mem_threadgroup);

    // ====================================================================
    // TODO PHASE 2: row-wise softmax with simd_max / simd_sum
    // ====================================================================
    // SG s handles row s. 32 lanes share the M=512 elements (16 per lane).
    //
    //   pass 1 (scale + max):
    //       lane_max = -INFINITY
    //       for i in lane, lane+32, ..., <M:
    //           v = row[i] * SCALE
    //           row[i] = v
    //           lane_max = max(lane_max, v)
    //       row_max = simd_max(lane_max)            // reduces across 32 lanes
    //
    //   pass 2 (exp + sum):
    //       lane_sum = 0
    //       for i in lane, lane+32, ..., <M:
    //           e = exp(row[i] - row_max)
    //           row[i] = e
    //           lane_sum += e
    //       row_sum = simd_sum(lane_sum)
    //
    //   pass 3 (normalize):
    //       for i in lane, lane+32, ..., <M:
    //           row[i] *= 1.0 / row_sum
    //
    // No threadgroup_barriers inside phase 2 — SIMD-group lanes are
    // synchronous, and only the SG that owns a row reads/writes it.



    // After all 8 SGs finish their rows:
    //
    //   threadgroup_barrier(mem_flags::mem_threadgroup);

    // ====================================================================
    // TODO PHASE 3: tiled PV from threadgroup scratch
    // ====================================================================
    // SG s computes 8 output tiles at columns [s*64 .. (s+1)*64). For
    // each tile (top row 0, top col d0):
    //   O_frag = 0
    //   for kt in 0..M/8:
    //       P_frag = simdgroup_load from THREADGROUP: scores + kt*8, stride M
    //       V_frag = simdgroup_load from DEVICE:      v + kt*8 * D + d0, stride D
    //       O_frag += P_frag · V_frag
    //   simdgroup_store(O_frag, out + m_base * D + d0, D);

}
