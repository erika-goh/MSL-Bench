#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// p305 + Q staged into threadgroup memory. Single-line algorithmic
// delta vs p305 inside phase 1: Q tiles are read from TG memory
// instead of device. Eliminates the 64× Q redundancy (8 SGs × 8
// score tiles each re-loading the same Q row-block from device).
constant uint  M     = 512;
constant uint  D     = 512;
constant float SCALE = 0.04419417382415922f;

constant uint  ROWS_PER_TG    = 8;
constant uint  SG             = 8;
constant uint  SGS_PER_TG     = 8;
constant uint  TG_THREADS     = 256;
constant uint  COLS_PER_SG_PHASE1 = M / SGS_PER_TG;   // 64
constant uint  COLS_PER_SG_PHASE3 = D / SGS_PER_TG;   // 64

constant uint  Q_STAGE_LEN    = ROWS_PER_TG * D;      // 4096 floats = 16 KB
constant uint  Q_PER_THREAD   = Q_STAGE_LEN / TG_THREADS;  // 16

kernel void attention_qstaged(
    device const float* q   [[buffer(0)]],
    device const float* k   [[buffer(1)]],
    device const float* v   [[buffer(2)]],
    device float*       out [[buffer(3)]],
    uint  tg_in_grid                [[threadgroup_position_in_grid]],
    uint  sg_in_tg                  [[simdgroup_index_in_threadgroup]],
    uint  lane                      [[thread_index_in_simdgroup]],
    uint  tid_in_tg                 [[thread_position_in_threadgroup]])
{
    threadgroup float q_stage[Q_STAGE_LEN];        // 16 KB
    threadgroup float scores[ROWS_PER_TG * M];     // 16 KB

    uint m_base = tg_in_grid * ROWS_PER_TG;

    // ====================================================================
    // PHASE 0: cooperatively stage Q[m_base..m_base+7, :] into TG memory.
    // ====================================================================
    // 256 threads, 4096 elements, 16 per thread. Adjacent threads at
    // each step read adjacent device addresses → coalesced loads.
    {
        uint q_row_base = m_base * D;
        for (uint i = 0; i < Q_PER_THREAD; i++) {
            uint idx = tid_in_tg + i * TG_THREADS;
            q_stage[idx] = q[q_row_base + idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ====================================================================
    // PHASE 1: scores = Q @ K^T  (Q now from TG memory)
    // ====================================================================
    {
        uint c_base = sg_in_tg * COLS_PER_SG_PHASE1;

        for (uint t = 0; t < COLS_PER_SG_PHASE1 / SG; t++) {
            uint c0 = c_base + t * SG;

            simdgroup_matrix<float, 8, 8> C_frag = simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint kt = 0; kt < D / SG; kt++) {
                uint k_off = kt * SG;

                simdgroup_matrix<float, 8, 8> Q_frag;
                simdgroup_matrix<float, 8, 8> K_frag;

                // Q tile now read from TG (stride D, base address
                // q_stage + k_off corresponds to row 0, col k_off).
                simdgroup_load(Q_frag, q_stage + k_off, D);

                simdgroup_load(K_frag,
                               k + c0 * D + k_off,
                               D,
                               ulong2(0, 0),
                               /*transpose=*/true);

                simdgroup_multiply_accumulate(C_frag, Q_frag, K_frag, C_frag);
            }

            simdgroup_store(C_frag, scores + c0, M);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ====================================================================
    // PHASE 2: row-wise softmax (unchanged from p305)
    // ====================================================================
    {
        uint my_row = sg_in_tg;
        threadgroup float* row = scores + my_row * M;

        float lane_max = -INFINITY;
        for (uint i = lane; i < M; i += 32) {
            float v_scaled = row[i] * SCALE;
            row[i] = v_scaled;
            lane_max = max(lane_max, v_scaled);
        }
        float row_max = simd_max(lane_max);

        float lane_sum = 0.0f;
        for (uint i = lane; i < M; i += 32) {
            float e = exp(row[i] - row_max);
            row[i] = e;
            lane_sum += e;
        }
        float row_sum = simd_sum(lane_sum);

        float inv = 1.0f / row_sum;
        for (uint i = lane; i < M; i += 32) {
            row[i] *= inv;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ====================================================================
    // PHASE 3: out = probs @ V (unchanged from p305)
    // ====================================================================
    {
        uint d_base = sg_in_tg * COLS_PER_SG_PHASE3;

        for (uint t = 0; t < COLS_PER_SG_PHASE3 / SG; t++) {
            uint d0 = d_base + t * SG;

            simdgroup_matrix<float, 8, 8> O_frag = simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint kt = 0; kt < M / SG; kt++) {
                uint k_off = kt * SG;

                simdgroup_matrix<float, 8, 8> P_frag;
                simdgroup_matrix<float, 8, 8> V_frag;

                simdgroup_load(P_frag, scores + k_off, M);
                simdgroup_load(V_frag, v + k_off * D + d0, D);

                simdgroup_multiply_accumulate(O_frag, P_frag, V_frag, O_frag);
            }

            simdgroup_store(O_frag, out + m_base * D + d0, D);
        }
    }
}
