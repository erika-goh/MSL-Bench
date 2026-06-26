#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint N       = 1024;
constant uint K       = 1024;
constant uint TG_M    = 16;
constant uint TG_N    = 16;
constant uint K_STAGE = 32;
constant uint SG      = 8;

// Helper: 32 threads cooperatively load a TG_M x K_STAGE slab of A and
// a K_STAGE x TG_N slab of B from device into the given threadgroup
// buffers. Caller-side: ensure the buffers are not currently being
// read by the matrix unit (separate set in double-buffer scheme).
static inline void stage_loads(
    threadgroup float* A_buf,
    threadgroup float* B_buf,
    device const float* a,
    device const float* b,
    uint m0, uint n0, uint ko,
    uint tid)
{
    for (uint i = 0; i < (TG_M * K_STAGE) / 32; i++) {
        uint idx = i * 32 + tid;
        uint row = idx / K_STAGE;
        uint col = idx % K_STAGE;
        A_buf[idx] = a[(m0 + row) * K + (ko + col)];
    }
    for (uint i = 0; i < (K_STAGE * TG_N) / 32; i++) {
        uint idx = i * 32 + tid;
        uint row = idx / TG_N;
        uint col = idx % TG_N;
        B_buf[idx] = b[(ko + row) * N + (n0 + col)];
    }
}

// Helper: consume one staged slab — inner K-loop running 4 sub-steps
// of (load 4 fragments, do 4 MACs).
static inline void compute_slab(
    threadgroup const float* A_buf,
    threadgroup const float* B_buf,
    thread simdgroup_matrix<float, 8, 8> & C_tl,
    thread simdgroup_matrix<float, 8, 8> & C_tr,
    thread simdgroup_matrix<float, 8, 8> & C_bl,
    thread simdgroup_matrix<float, 8, 8> & C_br)
{
    for (uint ki = 0; ki < K_STAGE; ki += SG) {
        simdgroup_matrix<float, 8, 8> A_top, A_bot, B_left, B_right;
        simdgroup_load(A_top,   A_buf + 0 * K_STAGE + ki, K_STAGE);
        simdgroup_load(A_bot,   A_buf + 8 * K_STAGE + ki, K_STAGE);
        simdgroup_load(B_left,  B_buf + ki * TG_N + 0,    TG_N);
        simdgroup_load(B_right, B_buf + ki * TG_N + 8,    TG_N);

        simdgroup_multiply_accumulate(C_tl, A_top, B_left,  C_tl);
        simdgroup_multiply_accumulate(C_tr, A_top, B_right, C_tr);
        simdgroup_multiply_accumulate(C_bl, A_bot, B_left,  C_bl);
        simdgroup_multiply_accumulate(C_br, A_bot, B_right, C_br);
    }
}

kernel void matmul_double_buffered_backfires(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float*       c    [[buffer(2)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    // Two sets of staged buffers — ping-pong between them.
    threadgroup float A_stage[2][TG_M * K_STAGE];
    threadgroup float B_stage[2][K_STAGE * TG_N];

    uint tid = tid_in_tg.x;
    uint by  = tg_in_grid.y;
    uint bx  = tg_in_grid.x;
    uint m0  = by * TG_M;
    uint n0  = bx * TG_N;

    simdgroup_matrix<float, 8, 8> C_tl = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> C_tr = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> C_bl = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> C_br = simdgroup_matrix<float, 8, 8>(0.0f);

    // Prologue: load the first slab into buffer 0.
    stage_loads(A_stage[0], B_stage[0], a, b, m0, n0, 0u, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop: each iter loads the NEXT slab into the alternate
    // buffer while consuming the CURRENT slab from its buffer. The
    // compiler is free to schedule the device-memory loads in parallel
    // with the matrix-unit ops because they touch different
    // threadgroup-memory addresses.
    uint num_stages = K / K_STAGE;
    for (uint s = 1; s < num_stages; s++) {
        uint cur  = (s - 1) & 1;
        uint next = s & 1;
        uint ko_next = s * K_STAGE;

        stage_loads(A_stage[next], B_stage[next], a, b, m0, n0, ko_next, tid);
        compute_slab(A_stage[cur], B_stage[cur], C_tl, C_tr, C_bl, C_br);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Epilogue: consume the last slab (no further loads).
    uint last = (num_stages - 1) & 1;
    compute_slab(A_stage[last], B_stage[last], C_tl, C_tr, C_bl, C_br);

    simdgroup_store(C_tl, c + (m0    ) * N + (n0    ), N);
    simdgroup_store(C_tr, c + (m0    ) * N + (n0 + 8), N);
    simdgroup_store(C_bl, c + (m0 + 8) * N + (n0    ), N);
    simdgroup_store(C_br, c + (m0 + 8) * N + (n0 + 8), N);
}
