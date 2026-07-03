# Metal KernelBench — Leaderboard

| Run | n | fast_0 (correct) | fast_1 (≥MPS) | fast_2 (≥2×MPS) |
|---|---|---|---|---|
| gemini_gemini-2.5-flash_one_shot | 19 | 63.2% | 63.2% | 5.3% |
| ollama_qwen2.5-coder-14b_one_shot | 6 | 0.0% | 0.0% | 0.0% |

## Per-tier breakdown

### gemini_gemini-2.5-flash_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 12 | 91.7% | 91.7% | 8.3% |
| 2 | 7 | 14.3% | 14.3% | 0.0% |

### ollama_qwen2.5-coder-14b_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 2 | 0.0% | 0.0% | 0.0% |
| 2 | 2 | 0.0% | 0.0% | 0.0% |
| 3 | 1 | 0.0% | 0.0% | 0.0% |
| 4 | 1 | 0.0% | 0.0% | 0.0% |

## Per-problem failure stage

Legend: `✓` correct · `c` compile · `r` runtime · `v` verify · `n` no code emitted · `e` provider/harness error · `·` not run.

| Problem | T | gemini_gemini-2.5-flash_one_shot | ollama_qwen2.5-coder-14b_one_shot | ✓/n |
|---|---|---|---|---|
| p001_vector_add | 1 | ✓ 1.0× | c | 1/2 |
| p002_relu | 1 | ✓ 1.1× | · | 1/1 |
| p003_elementwise_mul | 1 | ✓ 1.1× | · | 1/1 |
| p004_scalar_mul | 1 | ✓ 1.2× | · | 1/1 |
| p005_leaky_relu | 1 | ✓ 1.1× | · | 1/1 |
| p006_axpby | 1 | ✓ 2.5× | · | 1/1 |
| p007_sigmoid | 1 | ✓ 1.1× | · | 1/1 |
| p008_tanh | 1 | ✓ 1.1× | · | 1/1 |
| p010_abs | 1 | c | · | 0/1 |
| p011_exp | 1 | ✓ 1.1× | · | 1/1 |
| p012_clamp | 1 | ✓ 1.1× | · | 1/1 |
| p013_gelu | 1 | ✓ 1.1× | c | 1/2 |
| p101_row_sum | 2 | c | c | 0/2 |
| p102_row_max | 2 | c | · | 0/1 |
| p103_col_sum | 2 | c | · | 0/1 |
| p104_row_softmax | 2 | ✓ 1.1× | · | 1/1 |
| p105_col_sum_tiled_naive | 2 | c | · | 0/1 |
| p106_col_sum_atomic | 2 | n | · | 0/1 |
| p107_row_sum_atomic | 2 | c | · | 0/1 |
| p109_row_prefix_sum | 2 | · | c | 0/1 |
| p201_matmul_tiled | 3 | · | v | 0/1 |
| p305_attention_simdmatmul | 4 | · | c | 0/1 |

### Unbeaten problems (0 correct across all runs)

p010_abs, p101_row_sum, p102_row_max, p103_col_sum, p105_col_sum_tiled_naive, p106_col_sum_atomic, p107_row_sum_atomic, p109_row_prefix_sum, p201_matmul_tiled, p305_attention_simdmatmul
