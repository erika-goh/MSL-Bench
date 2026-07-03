# Metal KernelBench — Leaderboard

| Run | n | fast_0 (correct) | fast_1 (≥MPS) | fast_2 (≥2×MPS) |
|---|---|---|---|---|
| gemini_gemini-2.5-flash_one_shot | 19 | 63.2% | 63.2% | 5.3% |
| gemini_gemini-2.5-flash_repair | 2 | 50.0% | 50.0% | 0.0% |
| groq_llama-3.3-70b-versatile_one_shot | 36 | 38.9% | 33.3% | 2.8% |
| groq_llama-3.3-70b-versatile_repair | 17 | 82.4% | 70.6% | 5.9% |
| ollama_qwen2.5-coder-14b_one_shot | 6 | 0.0% | 0.0% | 0.0% |

## Per-tier breakdown

### gemini_gemini-2.5-flash_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 12 | 91.7% | 91.7% | 8.3% |
| 2 | 7 | 14.3% | 14.3% | 0.0% |

### gemini_gemini-2.5-flash_repair
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 2 | 50.0% | 50.0% | 0.0% |

### groq_llama-3.3-70b-versatile_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 12 | 100.0% | 100.0% | 8.3% |
| 2 | 9 | 22.2% | 0.0% | 0.0% |
| 3 | 8 | 0.0% | 0.0% | 0.0% |
| 4 | 7 | 0.0% | 0.0% | 0.0% |

### groq_llama-3.3-70b-versatile_repair
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 12 | 100.0% | 100.0% | 8.3% |
| 2 | 5 | 40.0% | 0.0% | 0.0% |

### ollama_qwen2.5-coder-14b_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 2 | 0.0% | 0.0% | 0.0% |
| 2 | 2 | 0.0% | 0.0% | 0.0% |
| 3 | 1 | 0.0% | 0.0% | 0.0% |
| 4 | 1 | 0.0% | 0.0% | 0.0% |

## Per-problem failure stage

Legend: `✓` correct · `c` compile · `r` runtime · `v` verify · `n` no code emitted · `e` provider/harness error · `·` not run.

| Problem | T | gemini_gemini-2.5-flash_one_shot | gemini_gemini-2.5-flash_repair | groq_llama-3.3-70b-versatile_one_shot | groq_llama-3.3-70b-versatile_repair | ollama_qwen2.5-coder-14b_one_shot | ✓/n |
|---|---|---|---|---|---|---|---|
| p001_vector_add | 1 | ✓ 1.0× | ✓ 1.1× | ✓ 1.1× | ✓ 1.1× | c | 4/5 |
| p002_relu | 1 | ✓ 1.1× | e | ✓ 1.1× | ✓ 1.1× | · | 3/4 |
| p003_elementwise_mul | 1 | ✓ 1.1× | · | ✓ 1.1× | ✓ 1.1× | · | 3/3 |
| p004_scalar_mul | 1 | ✓ 1.2× | · | ✓ 1.1× | ✓ 1.1× | · | 3/3 |
| p005_leaky_relu | 1 | ✓ 1.1× | · | ✓ 1.1× | ✓ 1.1× | · | 3/3 |
| p006_axpby | 1 | ✓ 2.5× | · | ✓ 2.5× | ✓ 2.5× | · | 3/3 |
| p007_sigmoid | 1 | ✓ 1.1× | · | ✓ 1.2× | ✓ 1.1× | · | 3/3 |
| p008_tanh | 1 | ✓ 1.1× | · | ✓ 1.1× | ✓ 1.2× | · | 3/3 |
| p010_abs | 1 | c | · | ✓ 1.1× | ✓ 1.1× | · | 2/3 |
| p011_exp | 1 | ✓ 1.1× | · | ✓ 1.1× | ✓ 1.1× | · | 3/3 |
| p012_clamp | 1 | ✓ 1.1× | · | ✓ 1.2× | ✓ 1.1× | · | 3/3 |
| p013_gelu | 1 | ✓ 1.1× | · | ✓ 1.1× | ✓ 1.1× | c | 3/4 |
| p101_row_sum | 2 | c | · | ✓ 0.4× | ✓ 0.5× | c | 2/4 |
| p102_row_max | 2 | c | · | c | v | · | 0/3 |
| p103_col_sum | 2 | c | · | ✓ 0.3× | ✓ 0.3× | · | 2/3 |
| p104_row_softmax | 2 | ✓ 1.1× | · | c | v | · | 1/3 |
| p105_col_sum_tiled_naive | 2 | c | · | c | e | · | 0/3 |
| p106_col_sum_atomic | 2 | n | · | c | · | · | 0/2 |
| p107_row_sum_atomic | 2 | c | · | c | · | · | 0/2 |
| p108_row_argmax | 2 | · | · | c | · | · | 0/1 |
| p109_row_prefix_sum | 2 | · | · | c | · | c | 0/2 |
| p201_matmul_tiled | 3 | · | · | c | · | v | 0/2 |
| p202_matmul_simdgroup | 3 | · | · | c | · | · | 0/1 |
| p203_matmul_simdgroup_staged | 3 | · | · | c | · | · | 0/1 |
| p204_matmul_double_buffered_backfires | 3 | · | · | v | · | · | 0/1 |
| p205_transpose | 3 | · | · | c | · | · | 0/1 |
| p206_sgemv | 3 | · | · | c | · | · | 0/1 |
| p207_conv2d_3x3 | 3 | · | · | c | · | · | 0/1 |
| p208_conv2d_5x5_tiled | 3 | · | · | v | · | · | 0/1 |
| p301_layernorm | 4 | · | · | c | · | · | 0/1 |
| p302_fused_linear_relu | 4 | · | · | v | · | · | 0/1 |
| p303_attention_head | 4 | · | · | v | · | · | 0/1 |
| p304_attention_large | 4 | · | · | v | · | · | 0/1 |
| p305_attention_simdmatmul | 4 | · | · | c | · | c | 0/2 |
| p306_attention_qstaged | 4 | · | · | c | · | · | 0/1 |
| p307_rmsnorm | 4 | · | · | c | · | · | 0/1 |

### Unbeaten problems (0 correct across all runs)

p102_row_max, p105_col_sum_tiled_naive, p106_col_sum_atomic, p107_row_sum_atomic, p108_row_argmax, p109_row_prefix_sum, p201_matmul_tiled, p202_matmul_simdgroup, p203_matmul_simdgroup_staged, p204_matmul_double_buffered_backfires, p205_transpose, p206_sgemv, p207_conv2d_3x3, p208_conv2d_5x5_tiled, p301_layernorm, p302_fused_linear_relu, p303_attention_head, p304_attention_large, p305_attention_simdmatmul, p306_attention_qstaged, p307_rmsnorm
