// Scaffold for p106_col_sum_atomic. Save as `col_sum_atomic.metal`.
// Don't edit this scaffold file in place.
//
// Buffer bindings:
//   buffer(0) = x   (input, B*K floats, row-major)
//   buffer(1) = out (output, K atomic_floats — host buffer is plain floats,
//                    but kernel views it as atomic_float for atomic ops)
//
// Dispatch geometry (from the spec):
//   grid        = (K,    N_TG_Y * TG_Y, 1) = (256, 2048, 1)
//   threadgroup = (TG_X, TG_Y,          1) = (32,  8,    1)
//
// This is the structural fix that p105 needed. Two things are new
// compared to p105:
//
//   1. Many threadgroups along Y. p105 had 8 TGs total; this has
//      2048 — finally enough to keep Apple Silicon's many cores fed.
//      Each TG handles ROW_CHUNK=1024 rows of a 32-column block.
//
//   2. Atomic accumulation across TGs. Multiple TGs compute partial
//      sums for the same column (one per row block), and there's no
//      way to synchronize across TGs within a single dispatch — so
//      each TG's final 32 partials get combined into out[col] via
//      atomic_fetch_add_explicit. Metal 3 added `atomic_float` and
//      atomic_fetch_add for floats specifically for this pattern.
//
// Atomic contention per output slot: N_TG_Y = 256 atomic ops. They
// serialize on each slot but across K=256 slots they run in parallel,
// so the wall-clock cost is bounded.
//
// CRITICAL: the spec must set `zero_output_each_run: True` because
// the harness re-runs the kernel many times (warmup + 10 timed) and
// each run ADDS to the output. Without per-run zeroing, run N sees
// (N+1)x the true sum and verification fails.

#include <metal_stdlib>
using namespace metal;

constant uint K         = 256;
constant uint TG_X      = 32;
constant uint TG_Y      = 8;
constant uint ROW_CHUNK = 1024;

kernel void col_sum_atomic(
    device const float*   x   [[buffer(0)]],
    device atomic_float*  out [[buffer(1)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[TG_X][TG_Y];

    uint tx = tid_in_tg.x;
    uint ty = tid_in_tg.y;
    uint col = tg_in_grid.x * TG_X + tx;
    uint row_start = tg_in_grid.y * ROW_CHUNK;
    uint row_end   = row_start + ROW_CHUNK;

    // ============================================================
    // TODO 1: each thread sums its slice of the ROW_CHUNK rows.
    //
    // Thread (tx, ty) handles rows row_start+ty, row_start+ty+TG_Y,
    // ..., up to row_end. For each such row r, accumulate
    // x[r * K + col] into a local partial.
    //
    // Coalescing check (same as p105): at fixed ty and r, threads
    // tx=0..31 read 32 consecutive floats → one coalesced load.
    // ============================================================



    // ============================================================
    // TODO 2: scratch[tx][ty] = partial; barrier.
    // ============================================================



    // ============================================================
    // TODO 3: combine the TG_Y=8 partials per column (sequential
    // loop, ty==0 only). Get a single per-column total per TG.
    // ============================================================



    // ============================================================
    // TODO 4: atomic add the per-column total into out[col].
    //
    //   atomic_fetch_add_explicit(&out[col], total,
    //                             memory_order_relaxed);
    //
    // memory_order_relaxed: just atomic, no ordering w.r.t. other
    // memory ops (we have none we need to order against). Cheapest
    // memory order Metal exposes.
    // ============================================================



}
