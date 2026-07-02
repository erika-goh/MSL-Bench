# Metal KernelBench — Leaderboard

| Run | n | fast_0 (correct) | fast_1 (≥MPS) | fast_2 (≥2×MPS) |
|---|---|---|---|---|
| ollama_qwen2.5-coder-14b_one_shot | 6 | 0.0% | 0.0% | 0.0% |

## Per-tier breakdown

### ollama_qwen2.5-coder-14b_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 2 | 0.0% | 0.0% | 0.0% |
| 2 | 2 | 0.0% | 0.0% | 0.0% |
| 3 | 1 | 0.0% | 0.0% | 0.0% |
| 4 | 1 | 0.0% | 0.0% | 0.0% |

## Per-problem failure stage

Legend: `✓` correct · `c` compile · `r` runtime · `v` verify · `n` no code emitted · `·` not run.

| Problem | T | ollama_qwen2.5-coder-14b_one_shot | ✓/n |
|---|---|---|---|
| p001_vector_add | 1 | c | 0/1 |
| p013_gelu | 1 | c | 0/1 |
| p101_row_sum | 2 | c | 0/1 |
| p109_row_prefix_sum | 2 | c | 0/1 |
| p201_matmul_tiled | 3 | v | 0/1 |
| p305_attention_simdmatmul | 4 | c | 0/1 |
