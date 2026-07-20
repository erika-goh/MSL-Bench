// row_max_abs (L-infinity norm): max over |x| per row, via SIMD-group shuffle.
// Same reduction skeleton as row_sum/row_mean, but the combine op is max(|.|)
// instead of +. 0.0 is a valid padding identity for the cross-SIMD stage
// because absolute values are always >= 0.
#include <metal_stdlib>
using namespace metal;

constant uint K = 256;

kernel void row_max_abs(constant float *x   [[buffer(0)]],
                        device float   *out [[buffer(1)]],
                        uint lid  [[thread_position_in_threadgroup]],
                        uint gid  [[threadgroup_position_in_grid]],
                        uint sid  [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float simd_max[8];  // 8 SIMD groups per 256-thread group

    float val = fabs(x[gid * K + lid]);

    // Intra-SIMD reduce (5 shuffle steps)
    val = max(val, simd_shuffle_down(val, 16));
    val = max(val, simd_shuffle_down(val, 8));
    val = max(val, simd_shuffle_down(val, 4));
    val = max(val, simd_shuffle_down(val, 2));
    val = max(val, simd_shuffle_down(val, 1));

    if (lane == 0) {
        simd_max[sid] = val;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SIMD reduce (first SIMD group only)
    if (sid == 0) {
        float partial = (lane < 8) ? simd_max[lane] : 0.0f;
        partial = max(partial, simd_shuffle_down(partial, 4));
        partial = max(partial, simd_shuffle_down(partial, 2));
        partial = max(partial, simd_shuffle_down(partial, 1));
        if (lane == 0) {
            out[gid] = partial;
        }
    }
}
