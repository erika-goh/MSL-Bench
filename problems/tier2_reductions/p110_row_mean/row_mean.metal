// row_mean via SIMD-group shuffle reduction, then divide by K.
//
// Structurally identical to the hand-tuned row_sum kernel from this
// session's investigation -- one threadgroup per row, intra-SIMD reduce
// via simd_shuffle_down, single threadgroup barrier, cross-SIMD reduce,
// then lane 0 writes (sum / K).
#include <metal_stdlib>
using namespace metal;

constant uint K = 256;

kernel void row_mean(constant float *x   [[buffer(0)]],
                     device float   *out [[buffer(1)]],
                     uint lid  [[thread_position_in_threadgroup]],
                     uint gid  [[threadgroup_position_in_grid]],
                     uint sid  [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float simd_sums[8];  // 8 SIMD groups per 256-thread group

    float val = x[gid * K + lid];

    // Intra-SIMD reduce (5 shuffle steps)
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);

    if (lane == 0) {
        simd_sums[sid] = val;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SIMD reduce (first SIMD group only)
    if (sid == 0) {
        float partial = (lane < 8) ? simd_sums[lane] : 0.0f;
        partial += simd_shuffle_down(partial, 4);
        partial += simd_shuffle_down(partial, 2);
        partial += simd_shuffle_down(partial, 1);
        if (lane == 0) {
            out[gid] = partial / float(K);
        }
    }
}
