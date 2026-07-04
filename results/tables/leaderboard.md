# Metal KernelBench — Leaderboard

| Run | n | fast_0 (correct) | fast_1 (≥MPS) | fast_2 (≥2×MPS) |
|---|---|---|---|---|
| groq_llama-3.3-70b-versatile_one_shot | 36† | 38.9% | 33.3% | 2.8% |
| groq_qwen-qwen3-32b_one_shot | 36† | 8.3% | 8.3% | 0.0% |
| gemini_gemini-2.5-flash_one_shot | 19† | 63.2% | 63.2% | 5.3% |
| groq_llama-3.3-70b-versatile_repair | 17† | 82.4% | 70.6% | 5.9% |
| ollama_qwen2.5-coder-14b_one_shot | 6† | 0.0% | 0.0% | 0.0% |
| gemini_gemini-2.5-flash_repair | 2† | 50.0% | 50.0% | 0.0% |

† partial sample (n < 60 total problems). Aggregate percentages are not directly comparable across rows — different runs may have tested different subsets of the suite.

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

### groq_qwen-qwen3-32b_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 15 | 20.0% | 20.0% | 0.0% |
| 2 | 15 | 0.0% | 0.0% | 0.0% |
| 3 | 6 | 0.0% | 0.0% | 0.0% |

### ollama_qwen2.5-coder-14b_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 2 | 0.0% | 0.0% | 0.0% |
| 2 | 2 | 0.0% | 0.0% | 0.0% |
| 3 | 1 | 0.0% | 0.0% | 0.0% |
| 4 | 1 | 0.0% | 0.0% | 0.0% |

## Per-problem failure stage

Legend: `✓` correct · `c` compile · `r` runtime · `v` verify · `n` no code emitted · `e` provider/harness error · `·` not run.

| Problem | T | gemini_gemini-2.5-flash_one_shot | gemini_gemini-2.5-flash_repair | groq_llama-3.3-70b-versatile_one_shot | groq_llama-3.3-70b-versatile_repair | groq_qwen-qwen3-32b_one_shot | ollama_qwen2.5-coder-14b_one_shot | ✓/n |
|---|---|---|---|---|---|---|---|---|
| p001_vector_add | 1 | ✓ 1.0× | ✓ 1.1× | ✓ 1.1× | ✓ 1.1× | c | c | 4/6 |
| p002_relu | 1 | ✓ 1.1× | e | ✓ 1.1× | ✓ 1.1× | c | · | 3/5 |
| p003_elementwise_mul | 1 | ✓ 1.1× | · | ✓ 1.1× | ✓ 1.1× | ✓ 1.1× | · | 4/4 |
| p004_scalar_mul | 1 | ✓ 1.2× | · | ✓ 1.1× | ✓ 1.1× | c | · | 3/4 |
| p005_leaky_relu | 1 | ✓ 1.1× | · | ✓ 1.1× | ✓ 1.1× | c | · | 3/4 |
| p006_axpby | 1 | ✓ 2.5× | · | ✓ 2.5× | ✓ 2.5× | c | · | 3/4 |
| p007_sigmoid | 1 | ✓ 1.1× | · | ✓ 1.2× | ✓ 1.1× | c | · | 3/4 |
| p008_tanh | 1 | ✓ 1.1× | · | ✓ 1.1× | ✓ 1.2× | ✓ 1.1× | · | 4/4 |
| p009_neg | 1 | · | · | · | · | c | · | 0/1 |
| p010_abs | 1 | c | · | ✓ 1.1× | ✓ 1.1× | ✓ 1.2× | · | 3/4 |
| p011_exp | 1 | ✓ 1.1× | · | ✓ 1.1× | ✓ 1.1× | c | · | 3/4 |
| p012_clamp | 1 | ✓ 1.1× | · | ✓ 1.2× | ✓ 1.1× | c | · | 3/4 |
| p013_gelu | 1 | ✓ 1.1× | · | ✓ 1.1× | ✓ 1.1× | c | c | 3/5 |
| p014_square | 1 | · | · | · | · | c | · | 0/1 |
| p015_silu | 1 | · | · | · | · | c | · | 0/1 |
| p101_row_sum | 2 | c | · | ✓ 0.4× | ✓ 0.5× | c | c | 2/5 |
| p102_row_max | 2 | c | · | c | v | c | · | 0/4 |
| p103_col_sum | 2 | c | · | ✓ 0.3× | ✓ 0.3× | c | · | 2/4 |
| p104_row_softmax | 2 | ✓ 1.1× | · | c | v | c | · | 1/4 |
| p105_col_sum_tiled_naive | 2 | c | · | c | e | c | · | 0/4 |
| p106_col_sum_atomic | 2 | n | · | c | · | n | · | 0/3 |
| p107_row_sum_atomic | 2 | c | · | c | · | c | · | 0/3 |
| p108_row_argmax | 2 | · | · | c | · | c | · | 0/2 |
| p109_row_prefix_sum | 2 | · | · | c | · | c | c | 0/3 |
| p110_row_mean | 2 | · | · | · | · | c | · | 0/1 |
| p111_row_min | 2 | · | · | · | · | c | · | 0/1 |
| p112_row_var | 2 | · | · | · | · | n | · | 0/1 |
| p113_col_max | 2 | · | · | · | · | n | · | 0/1 |
| p114_row_l2_norm | 2 | · | · | · | · | c | · | 0/1 |
| p115_row_dot | 2 | · | · | · | · | c | · | 0/1 |
| p201_matmul_tiled | 3 | · | · | c | · | n | v | 0/3 |
| p202_matmul_simdgroup | 3 | · | · | c | · | n | · | 0/2 |
| p203_matmul_simdgroup_staged | 3 | · | · | c | · | n | · | 0/2 |
| p204_matmul_double_buffered_backfires | 3 | · | · | v | · | e | · | 0/2 |
| p205_transpose | 3 | · | · | c | · | n | · | 0/2 |
| p206_sgemv | 3 | · | · | c | · | n | · | 0/2 |
| p207_conv2d_3x3 | 3 | · | · | c | · | · | · | 0/1 |
| p208_conv2d_5x5_tiled | 3 | · | · | v | · | · | · | 0/1 |
| p301_layernorm | 4 | · | · | c | · | · | · | 0/1 |
| p302_fused_linear_relu | 4 | · | · | v | · | · | · | 0/1 |
| p303_attention_head | 4 | · | · | v | · | · | · | 0/1 |
| p304_attention_large | 4 | · | · | v | · | · | · | 0/1 |
| p305_attention_simdmatmul | 4 | · | · | c | · | · | c | 0/2 |
| p306_attention_qstaged | 4 | · | · | c | · | · | · | 0/1 |
| p307_rmsnorm | 4 | · | · | c | · | · | · | 0/1 |

### Unbeaten problems (0 correct across all runs)

p009_neg, p014_square, p015_silu, p102_row_max, p105_col_sum_tiled_naive, p106_col_sum_atomic, p107_row_sum_atomic, p108_row_argmax, p109_row_prefix_sum, p110_row_mean, p111_row_min, p112_row_var, p113_col_max, p114_row_l2_norm, p115_row_dot, p201_matmul_tiled, p202_matmul_simdgroup, p203_matmul_simdgroup_staged, p204_matmul_double_buffered_backfires, p205_transpose, p206_sgemv, p207_conv2d_3x3, p208_conv2d_5x5_tiled, p301_layernorm, p302_fused_linear_relu, p303_attention_head, p304_attention_large, p305_attention_simdmatmul, p306_attention_qstaged, p307_rmsnorm
