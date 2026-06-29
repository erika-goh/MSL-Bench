#include <metal_stdlib>
using namespace metal;

constant uint  M     = 64;
constant uint  D     = 64;
constant float SCALE = 0.125f;  // 1.0 / sqrt(D=64)

kernel void attention_head(
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
    uint tid = tid_in_tg;    // 0..M-1 == 0..D-1 (since M == D == 64)

    // ---------- phase 0: stage Q[m, :] into threadgroup memory ----------
    // M == D, so one thread per element. Coalesced (adjacent threads
    // read adjacent device addresses).
    q_row[tid] = q[m * D + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------- phase 1: scores ----------
    // Thread tid computes s[tid] = (Q[m] · K[tid]) * SCALE.
    // Q row read from threadgroup (broadcast across threads).
    // K row read from device: sequential WITHIN a thread (good for that
    // thread's prefetch), but UNCOALESCED across adjacent threads — at
    // fixed d, thread t reads k[t*D + d] so neighbours are D=64 floats
    // apart. The real per-byte bandwidth cost of this kernel lives here.
    float s = 0.0f;
    for (uint d = 0; d < D; d++) {
        s += q_row[d] * k[tid * D + d];
    }
    s *= SCALE;

    // Save my score for the exp step later; also publish to scratch
    // so the upcoming max-reduction can see all scores.
    scratch[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------- phase 2a: row max ----------
    // Standard tree-reduce-max over scratch. Destroys scratch.
    for (uint stride = M / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------- phase 2b: row sum of exp(s - max) ----------
    // Each thread recomputes its exp from its own `s` register.
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

    // Normalized probability for this thread's position.
    float p = e / row_sum;

    // Publish probabilities to scratch so every thread can read them
    // in phase 3 (each output dim needs the full prob vector).
    scratch[tid] = p;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------- phase 3: output ----------
    // Thread tid computes out[m, tid] = sum_j scratch[j] * V[j, tid].
    // V reads are stride-1 across threads at fixed j (thread t reads
    // v[j*D + t], i.e. adjacent columns of the same row) → COALESCED.
    // No bandwidth issue here.
    float o = 0.0f;
    for (uint j = 0; j < M; j++) {
        o += scratch[j] * v[j * D + tid];
    }
    out[m * D + tid] = o;
}
