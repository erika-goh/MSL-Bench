#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Same algorithm as p304 but the two matmul phases run on Apple's
// per-SIMD-group matrix unit via simdgroup_matrix<float, 8, 8>. The
// softmax in the middle stays as a per-row simdgroup reduction.
constant uint  M     = 512;
constant uint  D     = 512;
constant float SCALE = 0.04419417382415922f;   // 1.0 / sqrt(D=512)

constant uint  ROWS_PER_TG    = 8;             // one 8x8 tile high per TG
constant uint  SG             = 8;             // simdgroup_matrix tile dim
constant uint  SGS_PER_TG     = 8;             // 8 SIMD groups per TG
constant uint  COLS_PER_SG_PHASE1 = M / SGS_PER_TG;  // 64 columns each
constant uint  COLS_PER_SG_PHASE3 = D / SGS_PER_TG;  // 64 columns each

kernel void attention_simdmatmul(
    device const float* q   [[buffer(0)]],
    device const float* k   [[buffer(1)]],
    device const float* v   [[buffer(2)]],
    device float*       out [[buffer(3)]],
    uint  tg_in_grid                [[threadgroup_position_in_grid]],
    uint  sg_in_tg                  [[simdgroup_index_in_threadgroup]],
    uint  lane                      [[thread_index_in_simdgroup]])
{
    // scores[row][col] with stride M. Reused across phases:
    //   phase 1: raw scores from QK^T
    //   phase 2: scaled scores → exp(scores - row_max) → probabilities
    //   phase 3: input to PV matmul
    threadgroup float scores[ROWS_PER_TG * M];

    uint m_base = tg_in_grid * ROWS_PER_TG;   // first query row this TG handles

    // ====================================================================
    // PHASE 1: scores = Q @ K^T  (no scale yet — applied in phase 2)
    // ====================================================================
    // Each SG produces 8 tiles in row-block (rows 0..7), columns
    // [sg_in_tg * 64, (sg_in_tg + 1) * 64). All 8 tiles share the same
    // Q rows but cover different K rows, so K must be tile-loaded
    // afresh per output column but Q is re-loaded each iteration for
    // simplicity (relying on L1 to keep it warm).
    {
        uint c_base = sg_in_tg * COLS_PER_SG_PHASE1;   // 0, 64, 128, ..., 448

        for (uint t = 0; t < COLS_PER_SG_PHASE1 / SG; t++) {  // 8 tiles
            uint c0 = c_base + t * SG;  // top-row of K we're computing scores against

            simdgroup_matrix<float, 8, 8> C_frag = simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint kt = 0; kt < D / SG; kt++) {   // 64 K-strips of 8
                uint k_off = kt * SG;

                simdgroup_matrix<float, 8, 8> Q_frag;
                simdgroup_matrix<float, 8, 8> K_frag;

                // Q[m_base..m_base+7, k_off..k_off+7]  (no transpose)
                simdgroup_load(Q_frag, q + m_base * D + k_off, D);

                // K[c0..c0+7, k_off..k_off+7] loaded TRANSPOSED so the
                // matrix unit computes Q @ K^T as a standard A @ B.
                simdgroup_load(K_frag,
                               k + c0 * D + k_off,
                               D,
                               ulong2(0, 0),
                               /*transpose=*/true);

                simdgroup_multiply_accumulate(C_frag, Q_frag, K_frag, C_frag);
            }

            // Store into scratch[0..7][c0..c0+7].
            simdgroup_store(C_frag, scores + c0, M);
        }
    }

    // All 8 SGs must finish writing the score row-block before any SG
    // reads its row for softmax.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ====================================================================
    // PHASE 2: softmax over each of the 8 rows, one SG per row.
    // ====================================================================
    // SG sg_in_tg owns row sg_in_tg. 32 lanes share the M=512 elements
    // (16 elements per lane). simd_max / simd_sum collapse the lane
    // dimension without barriers or scratch.
    {
        uint my_row = sg_in_tg;
        threadgroup float* row = scores + my_row * M;

        // --- pass 1: apply SCALE and find row max ---
        float lane_max = -INFINITY;
        for (uint i = lane; i < M; i += 32) {
            float v_scaled = row[i] * SCALE;
            row[i] = v_scaled;
            lane_max = max(lane_max, v_scaled);
        }
        float row_max = simd_max(lane_max);

        // --- pass 2: exp(s - max), sum ---
        float lane_sum = 0.0f;
        for (uint i = lane; i < M; i += 32) {
            float e = exp(row[i] - row_max);
            row[i] = e;
            lane_sum += e;
        }
        float row_sum = simd_sum(lane_sum);

        // --- pass 3: normalize ---
        float inv = 1.0f / row_sum;
        for (uint i = lane; i < M; i += 32) {
            row[i] *= inv;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ====================================================================
    // PHASE 3: out = probs @ V  (standard tiled matmul)
    // ====================================================================
    // Each SG produces 8 output tiles in row-block (rows 0..7), columns
    // [sg_in_tg * 64, (sg_in_tg + 1) * 64). Reads probs from threadgroup
    // memory; reads V from device.
    {
        uint d_base = sg_in_tg * COLS_PER_SG_PHASE3;

        for (uint t = 0; t < COLS_PER_SG_PHASE3 / SG; t++) {
            uint d0 = d_base + t * SG;

            simdgroup_matrix<float, 8, 8> O_frag = simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint kt = 0; kt < M / SG; kt++) {  // 64 strips
                uint k_off = kt * SG;

                simdgroup_matrix<float, 8, 8> P_frag;
                simdgroup_matrix<float, 8, 8> V_frag;

                // probs[0..7, k_off..k_off+7]  (threadgroup load)
                simdgroup_load(P_frag, scores + k_off, M);

                // V[k_off..k_off+7, d0..d0+7]   (no transpose)
                simdgroup_load(V_frag, v + k_off * D + d0, D);

                simdgroup_multiply_accumulate(O_frag, P_frag, V_frag, O_frag);
            }

            simdgroup_store(O_frag, out + m_base * D + d0, D);
        }
    }
}
