// Scaffold for p306_attention_qstaged. Save as `attention_qstaged.metal`.
//
// Single-idea variant of p305: stage Q[m_base..m_base+7, :] into
// threadgroup memory once at the top, then read from there in
// phase 1. Phases 2 and 3 unchanged from p305.
//
// Threadgroup memory budget (32 KB cap):
//   q_stage[8 × 512] = 16 KB
//   scores[8 × 512]  = 16 KB
//   total            = 32 KB (at the limit)
//
// Dispatch same as p305: 64 TGs × 256 threads.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint  M     = 512;
constant uint  D     = 512;
constant float SCALE = 0.04419417382415922f;

constant uint  ROWS_PER_TG    = 8;
constant uint  SG             = 8;
constant uint  SGS_PER_TG     = 8;
constant uint  TG_THREADS     = 256;
constant uint  COLS_PER_SG_PHASE1 = M / SGS_PER_TG;
constant uint  COLS_PER_SG_PHASE3 = D / SGS_PER_TG;

constant uint  Q_STAGE_LEN    = ROWS_PER_TG * D;
constant uint  Q_PER_THREAD   = Q_STAGE_LEN / TG_THREADS;

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
    threadgroup float q_stage[Q_STAGE_LEN];
    threadgroup float scores[ROWS_PER_TG * M];

    uint m_base = tg_in_grid * ROWS_PER_TG;

    // ====================================================================
    // TODO PHASE 0: cooperative Q load
    // ====================================================================
    // 256 threads, 4096 elements → 16 per thread. Striped layout:
    //
    //   for i in 0..Q_PER_THREAD:
    //       idx = tid_in_tg + i * TG_THREADS
    //       q_stage[idx] = q[m_base * D + idx]
    //
    // Adjacent threads at each i read adjacent device addresses
    // → coalesced. Followed by threadgroup_barrier(mem_threadgroup).



    // ====================================================================
    // TODO PHASE 1: scores = Q @ K^T, with Q now from threadgroup memory
    // ====================================================================
    // Identical to p305 phase 1 except:
    //   simdgroup_load(Q_frag, q_stage + k_off, D);
    // (instead of `q + m_base * D + k_off`)
    //
    // K_frag load is unchanged — still from device, still transposed.
    // After all 8 SGs write their score column blocks, barrier.



    // ====================================================================
    // TODO PHASE 2: row-wise softmax (paste p305's phase 2 verbatim)
    // ====================================================================



    // ====================================================================
    // TODO PHASE 3: out = probs @ V (paste p305's phase 3 verbatim)
    // ====================================================================

}
