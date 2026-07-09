# Metal KernelBench — Leaderboard

| Run | n | fast_0 (correct) | fast_1 (≥MPS) | fast_2 (≥2×MPS) |
|---|---|---|---|---|
| groq_llama-3.1-8b-instant_one_shot | 60 | 28.3% | 25.0% | 1.7% |
| groq_openai-gpt-oss-120b_one_shot | 60 | 45.0% | 38.3% | 10.0% |
| groq_qwen-qwen3-32b_one_shot | 60 | 8.3% | 6.7% | 0.0% |
| groq_qwen-qwen3-32b_one_shot_mt5200 | 60 | 0.0% | 0.0% | 0.0% |
| groq_llama-3.3-70b-versatile_one_shot | 43† | 44.2% | 37.2% | 4.7% |
| gemini_gemini-2.5-flash_one_shot | 22† | 63.6% | 63.6% | 4.5% |
| groq_llama-3.3-70b-versatile_repair | 17† | 82.4% | 70.6% | 5.9% |
| ollama_qwen2.5-coder-14b_one_shot | 6† | 0.0% | 0.0% | 0.0% |
| gemini_gemini-2.5-flash_repair | 2† | 50.0% | 50.0% | 0.0% |

† partial sample (n < 60 total problems). Aggregate percentages are not directly comparable across rows — different runs may have tested different subsets of the suite.

## Per-tier breakdown

### gemini_gemini-2.5-flash_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 15 | 86.7% | 86.7% | 6.7% |
| 2 | 7 | 14.3% | 14.3% | 0.0% |

### gemini_gemini-2.5-flash_repair
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 2 | 50.0% | 50.0% | 0.0% |

### groq_llama-3.1-8b-instant_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 15 | 100.0% | 100.0% | 6.7% |
| 2 | 15 | 6.7% | 0.0% | 0.0% |
| 3 | 15 | 0.0% | 0.0% | 0.0% |
| 4 | 15 | 6.7% | 0.0% | 0.0% |

### groq_llama-3.3-70b-versatile_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 15 | 100.0% | 100.0% | 6.7% |
| 2 | 13 | 30.8% | 7.7% | 7.7% |
| 3 | 8 | 0.0% | 0.0% | 0.0% |
| 4 | 7 | 0.0% | 0.0% | 0.0% |

### groq_llama-3.3-70b-versatile_repair
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 12 | 100.0% | 100.0% | 8.3% |
| 2 | 5 | 40.0% | 0.0% | 0.0% |

### groq_openai-gpt-oss-120b_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 15 | 93.3% | 93.3% | 6.7% |
| 2 | 15 | 46.7% | 20.0% | 6.7% |
| 3 | 15 | 20.0% | 20.0% | 20.0% |
| 4 | 15 | 20.0% | 20.0% | 6.7% |

### groq_qwen-qwen3-32b_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 15 | 20.0% | 20.0% | 0.0% |
| 2 | 15 | 0.0% | 0.0% | 0.0% |
| 3 | 15 | 6.7% | 0.0% | 0.0% |
| 4 | 15 | 6.7% | 6.7% | 0.0% |

### groq_qwen-qwen3-32b_one_shot_mt5200
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 15 | 0.0% | 0.0% | 0.0% |
| 2 | 15 | 0.0% | 0.0% | 0.0% |
| 3 | 15 | 0.0% | 0.0% | 0.0% |
| 4 | 15 | 0.0% | 0.0% | 0.0% |

### ollama_qwen2.5-coder-14b_one_shot
| Tier | n | fast_0 | fast_1 | fast_2 |
|---|---|---|---|---|
| 1 | 2 | 0.0% | 0.0% | 0.0% |
| 2 | 2 | 0.0% | 0.0% | 0.0% |
| 3 | 1 | 0.0% | 0.0% | 0.0% |
| 4 | 1 | 0.0% | 0.0% | 0.0% |

## Per-problem failure stage

Legend: `✓` correct · `c` compile · `r` runtime · `v` verify · `n` no code emitted · `e` provider/harness error · `·` not run.

| Problem | T | gemini_gemini-2.5-flash_one_shot | gemini_gemini-2.5-flash_repair | groq_llama-3.1-8b-instant_one_shot | groq_llama-3.3-70b-versatile_one_shot | groq_llama-3.3-70b-versatile_repair | groq_openai-gpt-oss-120b_one_shot | groq_qwen-qwen3-32b_one_shot | groq_qwen-qwen3-32b_one_shot_mt5200 | ollama_qwen2.5-coder-14b_one_shot | ✓/n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p001_vector_add | 1 | ✓ 1.0× | ✓ 1.1× | ✓ 1.1× | ✓ 1.1× | ✓ 1.1× | ✓ 1.1× | c | c | c | 6/9 |
| p002_relu | 1 | ✓ 1.1× | e | ✓ 1.2× | ✓ 1.1× | ✓ 1.1× | ✓ 1.1× | c | c | · | 5/8 |
| p003_elementwise_mul | 1 | ✓ 1.1× | · | ✓ 1.1× | ✓ 1.1× | ✓ 1.1× | ✓ 1.1× | ✓ 1.1× | c | · | 6/7 |
| p004_scalar_mul | 1 | ✓ 1.2× | · | ✓ 1.1× | ✓ 1.1× | ✓ 1.1× | ✓ 1.1× | c | e | · | 5/7 |
| p005_leaky_relu | 1 | ✓ 1.1× | · | ✓ 1.2× | ✓ 1.1× | ✓ 1.1× | ✓ 1.2× | c | c | · | 5/7 |
| p006_axpby | 1 | ✓ 2.5× | · | ✓ 2.4× | ✓ 2.5× | ✓ 2.5× | ✓ 2.5× | c | c | · | 5/7 |
| p007_sigmoid | 1 | ✓ 1.1× | · | ✓ 1.2× | ✓ 1.2× | ✓ 1.1× | ✓ 1.1× | c | c | · | 5/7 |
| p008_tanh | 1 | ✓ 1.1× | · | ✓ 1.2× | ✓ 1.1× | ✓ 1.2× | ✓ 1.1× | ✓ 1.1× | e | · | 6/7 |
| p009_neg | 1 | ✓ 1.1× | · | ✓ 1.2× | ✓ 1.1× | · | c | c | c | · | 3/6 |
| p010_abs | 1 | c | · | ✓ 1.2× | ✓ 1.1× | ✓ 1.1× | ✓ 1.1× | ✓ 1.2× | c | · | 5/7 |
| p011_exp | 1 | ✓ 1.1× | · | ✓ 1.2× | ✓ 1.1× | ✓ 1.1× | ✓ 1.2× | c | c | · | 5/7 |
| p012_clamp | 1 | ✓ 1.1× | · | ✓ 1.1× | ✓ 1.2× | ✓ 1.1× | ✓ 1.1× | c | c | · | 5/7 |
| p013_gelu | 1 | ✓ 1.1× | · | ✓ 1.2× | ✓ 1.1× | ✓ 1.1× | ✓ 1.2× | c | c | c | 5/8 |
| p014_square | 1 | ✓ 1.1× | · | ✓ 1.1× | ✓ 1.1× | · | ✓ 1.1× | c | c | · | 4/6 |
| p015_silu | 1 | e | · | ✓ 1.2× | ✓ 1.1× | · | ✓ 1.1× | c | c | · | 3/6 |
| p101_row_sum | 2 | c | · | c | ✓ 0.4× | ✓ 0.5× | ✓ 0.5× | c | c | c | 3/8 |
| p102_row_max | 2 | c | · | c | c | v | c | c | c | · | 0/7 |
| p103_col_sum | 2 | c | · | c | ✓ 0.3× | ✓ 0.3× | ✓ 0.3× | c | c | · | 3/7 |
| p104_row_softmax | 2 | ✓ 1.1× | · | c | c | v | v | c | c | · | 1/7 |
| p105_col_sum_tiled_naive | 2 | c | · | c | c | e | c | c | c | · | 0/7 |
| p106_col_sum_atomic | 2 | n | · | c | c | · | ✓ 1.2× | n | n | · | 1/6 |
| p107_row_sum_atomic | 2 | c | · | c | c | · | ✓ 0.5× | c | c | · | 1/6 |
| p108_row_argmax | 2 | · | · | c | c | · | ✓ 2.2× | c | c | · | 1/5 |
| p109_row_prefix_sum | 2 | · | · | c | c | · | c | c | n | c | 0/6 |
| p110_row_mean | 2 | · | · | c | ✓ 0.0× | · | ✓ 0.5× | c | c | · | 2/5 |
| p111_row_min | 2 | · | · | v | ✓ 2.6× | · | c | c | c | · | 1/5 |
| p112_row_var | 2 | · | · | ✓ 0.7× | c | · | c | n | c | · | 1/5 |
| p113_col_max | 2 | · | · | c | e | · | c | n | c | · | 0/5 |
| p114_row_l2_norm | 2 | · | · | v | · | · | c | c | c | · | 0/4 |
| p115_row_dot | 2 | · | · | c | · | · | ✓ 1.9× | c | c | · | 1/4 |
| p201_matmul_tiled | 3 | · | · | c | c | · | c | n | c | v | 0/6 |
| p202_matmul_simdgroup | 3 | · | · | c | c | · | c | n | c | · | 0/5 |
| p203_matmul_simdgroup_staged | 3 | · | · | c | c | · | c | n | n | · | 0/5 |
| p204_matmul_double_buffered_backfires | 3 | · | · | c | v | · | c | e | n | · | 0/5 |
| p205_transpose | 3 | · | · | c | c | · | c | n | c | · | 0/5 |
| p206_sgemv | 3 | · | · | c | c | · | c | n | c | · | 0/5 |
| p207_conv2d_3x3 | 3 | · | · | v | c | · | ✓ 6.8× | c | e | · | 1/5 |
| p208_conv2d_5x5_tiled | 3 | · | · | c | v | · | v | e | n | · | 0/5 |
| p209_matmul_naive | 3 | · | · | c | · | · | c | ✓ 0.2× | e | · | 1/4 |
| p210_matmul_bias | 3 | · | · | c | · | · | c | n | c | · | 0/4 |
| p211_batched_matmul | 3 | · | · | c | · | · | c | c | c | · | 0/4 |
| p212_transpose_tiled | 3 | · | · | c | · | · | c | c | n | · | 0/4 |
| p213_conv1d | 3 | · | · | v | · | · | ✓ 27.2× | c | c | · | 1/4 |
| p214_conv2d_stride2 | 3 | · | · | c | · | · | ✓ 40.0× | c | c | · | 1/4 |
| p215_matmul_scaled | 3 | · | · | c | · | · | c | e | n | · | 0/4 |
| p301_layernorm | 4 | · | · | c | c | · | c | c | c | · | 0/5 |
| p302_fused_linear_relu | 4 | · | · | c | v | · | n | c | c | · | 0/5 |
| p303_attention_head | 4 | · | · | e | v | · | c | e | c | · | 0/5 |
| p304_attention_large | 4 | · | · | c | v | · | v | c | n | · | 0/5 |
| p305_attention_simdmatmul | 4 | · | · | c | c | · | n | n | n | c | 0/6 |
| p306_attention_qstaged | 4 | · | · | c | c | · | v | c | c | · | 0/5 |
| p307_rmsnorm | 4 | · | · | c | c | · | c | e | c | · | 0/5 |
| p308_gelu_fused | 4 | · | · | c | · | · | c | n | c | · | 0/4 |
| p309_batchnorm | 4 | · | · | ✓ 1.0× | · | · | c | c | c | · | 1/4 |
| p310_bias_add_relu | 4 | · | · | c | · | · | ✓ 6.9× | c | c | · | 1/4 |
| p311_softmax_stable | 4 | · | · | c | · | · | c | n | c | · | 0/4 |
| p312_attention_causal | 4 | · | · | c | · | · | c | n | n | · | 0/4 |
| p313_attention_masked | 4 | · | · | c | · | · | c | n | n | · | 0/4 |
| p314_swiglu | 4 | · | · | c | · | · | ✓ 1.9× | ✓ 1.9× | c | · | 2/4 |
| p315_glu_fused | 4 | · | · | v | · | · | ✓ 1.8× | c | e | · | 1/4 |

### Unbeaten problems (0 correct across all runs)

p102_row_max, p105_col_sum_tiled_naive, p109_row_prefix_sum, p113_col_max, p114_row_l2_norm, p201_matmul_tiled, p202_matmul_simdgroup, p203_matmul_simdgroup_staged, p204_matmul_double_buffered_backfires, p205_transpose, p206_sgemv, p208_conv2d_5x5_tiled, p210_matmul_bias, p211_batched_matmul, p212_transpose_tiled, p215_matmul_scaled, p301_layernorm, p302_fused_linear_relu, p303_attention_head, p304_attention_large, p305_attention_simdmatmul, p306_attention_qstaged, p307_rmsnorm, p308_gelu_fused, p311_softmax_stable, p312_attention_causal, p313_attention_masked
