#include <metal_stdlib>
using namespace metal;

constant uint K       = 256;
constant uint TG      = 256;
constant uint B       = 262144;

kernel void col_sum(device const float* x   [[buffer(0)]],
                    device float* out       [[buffer(1)]],
                    uint tid_in_tg  [[thread_position_in_threadgroup]],
                    uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[TG];

    uint col = tg_in_grid;
    uint tid = tid_in_tg;

    // Interleaved striping: thread t handles rows t, t+TG, t+2*TG, ...
    // Adjacent threads read x[r*K + col] for r=tid..tid+1, stride K=256
    // floats apart at any single instant → uncoalesced.
    float partial = 0.0f;
    for (uint r = tid; r < B; r += TG) {
        partial += x[r * K + col];
    }

    scratch[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction over the TG partials. Same shape as p101.
    for (uint stride = TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[col] = scratch[0];
    }
}
