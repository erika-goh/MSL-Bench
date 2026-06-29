#include <metal_stdlib>
using namespace metal;

// Same algorithm as p303_attention_head; only M, D, and SCALE change.
// The point of the problem is to observe how this kernel's performance
// vs MPS evolves as compute grows from ~0.5M ops (p303) to ~134M (here).
constant uint  M     = 512;
constant uint  D     = 512;
constant float SCALE = 0.04419417382415922f;  // 1.0 / sqrt(D=512)

kernel void attention_large(
    device const float* q   [[buffer(0)]],
    device const float* k   [[buffer(1)]],
    device const float* v   [[buffer(2)]],
    device float*       out [[buffer(3)]],
    uint tid_in_tg  [[thread_position_in_threadgroup]],
    uint tg_in_grid [[threadgroup_position_in_grid]])
{
    // One shared array reused across all phases:
    //   phase 1: scores
    //   phase 2: reduction workspace (max, then sum)
    //   phase 3: probabilities (broadcast for V-weighted output)
    threadgroup float scratch[M];
    // Staged Q row — broadcast value used by all threads in phase 1.
    threadgroup float q_row[D];

    uint m   = tg_in_grid;   // which query row this TG handles
    uint tid = tid_in_tg;    // 0..M-1 == 0..D-1 (since M == D)

    // ---------- phase 0: stage Q[m, :] into threadgroup memory ----------
    q_row[tid] = q[m * D + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------- phase 1: scores ----------
    // Thread tid computes s[tid] = (Q[m] · K[tid]) * SCALE. Dot-product
    // length is now D=512 (vs 64 in p303) — the main per-thread compute
    // bump. K read pattern: sequential WITHIN a thread (good for its
    // own prefetch) but UNCOALESCED across adjacent threads — at fixed
    // d, neighbours are D=512 floats apart, so each SIMD-group load
    // becomes 32 separate cache-line fetches. Real bandwidth cost.
    float s = 0.0f;
    for (uint d = 0; d < D; d++) {
        s += q_row[d] * k[tid * D + d];
    }
    s *= SCALE;

    scratch[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------- phase 2a: row max ----------
    // Tree-reduce-max. Depth grows from 6 (M=64) to 9 (M=512).
    for (uint stride = M / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------- phase 2b: row sum of exp(s - max) ----------
    float e = exp(s - row_max);
    scratch[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = M / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float p = e / row_sum;
    scratch[tid] = p;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------- phase 3: output ----------
    // Thread tid computes out[m, tid] = sum_j scratch[j] * V[j, tid].
    // V reads: stride-1 across threads at fixed j (thread t reads
    // v[j*D + t] — adjacent columns of the same row) → COALESCED.
    // No bandwidth issue here; the uncoalesced reads are in phase 1.
    float o = 0.0f;
    for (uint j = 0; j < M; j++) {
        o += scratch[j] * v[j * D + tid];
    }
    out[m * D + tid] = o;
}
