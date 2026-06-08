# Muon 系数序列实验结果

## 0. 研究问题

Muon 会把矩阵参数的更新方向做近似正交化。我们研究的问题是：

```text
在模型、数据、训练 token 数、batch 设置和 learning rate schedule 固定时，
不同 Newton-Schulz / Polar Express 系数序列会如何影响：

1. validation loss
2. wall-clock time
3. update matrix 的谱几何
```

这里的核心对象不是“调很多超参数”，而是正交化迭代中的系数序列。

每一步 Newton-Schulz-style 映射可写成：

\[
p(\sigma)=a\sigma+b\sigma^3+c\sigma^5.
\]

其中 \(\sigma\) 是 update matrix 的奇异值。多步 schedule 对应复合映射：

\[
p_T(\sigma)=p_T\circ p_{T-1}\circ\cdots\circ p_1(\sigma).
\]

## 1. 术语和实验代号

| 代号 | 含义 |
|---|---|
| AdamW | 不做矩阵正交化的 optimizer baseline |
| stable5 | 5 步 stable Newton-Schulz 系数 |
| fast5 | 5 步 fast Muon 系数，也是最重要 baseline |
| manual_T5_f3_s2 | 总 5 步，前 3 步 fast，后 2 步 stable |
| manual_T9_f4_s5 | 总 9 步，前 4 步 fast，后 5 步 stable |
| pe_T5_l1e-3 | Polar Express，5 步，lower bound 为 \(10^{-3}\) |
| pe_T9_l3e-5 | Polar Express，9 步，lower bound 为 \(3\times10^{-5}\) |

fast / stable 是两组固定系数：

```text
fast   = (3.4445, -4.7750, 2.0315)
stable = (2.0,    -1.5,    0.5)
```

## 2. 指标

| 指标 | 怎么看 |
|---|---|
| final validation loss | 训练结束时的验证集 loss，越低越好 |
| val-loss AUC | validation loss 曲线面积，越低表示整体学习更快 |
| wall-clock | 真实运行时间，越低越快 |
| g_post semi-orth error | 正交化后 update matrix 的半正交误差，越低表示越接近正交目标 |
| attention / MLP error | 分别统计 attention projection 和 MLP 矩阵的 g_post error |
| gain | \(p_T(\sigma)/\sigma\)，表示某个奇异值经过 schedule 后被放大多少倍 |

`gain` 只用于解释多项式映射，不是训练指标。
例如 \(gain=100\) 表示输入奇异值 \(\sigma\) 被映射到约 \(100\sigma\)。

## 3. 旧实验已经得到的结论

旧实验是固定 T=5 的单 seed 对照，比较：

```text
AdamW / stable5 / manual_T5_f3_s2 / fast5 / Polar Express T=5
```

主要结论：

1. Muon 系列明显优于 AdamW。
2. stable5 明显弱于 fast5。
3. manual_T5_f3_s2 介于 stable5 和 fast5 之间。
4. 同样 T=5 时，Muon 内部 wall-clock 基本一样。
5. PE 的 g_post error 最低，说明几何上更接近半正交。

这些结论主要来自：

```text
results/train/summary.csv
results/benchmark/summary.csv
results/spectral/summary.csv
```

## 4. 新实验 A：固定 T=5，比不同系数序列

目的：控制迭代步数 T=5，只比较系数序列。

| schedule | final val loss | val-loss AUC |
|---|---:|---:|
| AdamW | 5.373726 | 5.967695 |
| stable5 | 4.527259 | 5.316559 |
| manual_T5_f3_s2 | 4.317646 | 5.029547 |
| fast5 | 4.282755 | 4.934028 |
| pe_T5_l3e-3 | 4.287261 | 4.946200 |
| pe_T5_l1e-3 | 4.290226 | 4.933800 |

结论：T=5 固定时，系数序列确实影响训练。`fast5` 明显强于 `stable5`，`manual_T5_f3_s2` 介于两者之间。

结论强度：强。差距远大于 multi-seed 中约 0.003-0.005 的 seed 波动。

相关图：

```text
results/followup_4090_20260608/figures/final_val_loss_by_schedule.png
results/followup_4090_20260608/figures/val_loss_vs_tokens.png
```

图怎么看：

```text
val_loss_vs_tokens:
横轴是训练 token，纵轴是 validation loss。
同一横坐标下，曲线越低表示同样数据量学得越好。

final_val_loss_by_schedule:
每个柱子是一种 schedule 的最终 loss。
柱子越低越好。
```

## 5. 新实验 B：多项式映射和 gain

目的：解释不同系数序列为什么会产生不同几何行为。

证据：

```text
results/followup_4090_20260608/polynomial_maps/map_samples.csv
results/followup_4090_20260608/polynomial_maps/composed_maps.png
results/followup_4090_20260608/polynomial_maps/composed_map_delta.png
```

在 \(\sigma=10^{-5}\) 时：

| schedule | gain |
|---|---:|
| stable5 | 32 |
| fast5 | 485 |
| manual_T9_f4_s5 | 4502 |
| pe_T9_l3e-5 | 71309 |

结论：这些 schedule 对小奇异值的放大强度完全不同。
这说明它们改变的是谱映射 \(p_T(\sigma)\)，不是普通标签。

结论强度：强。多项式映射是确定性计算，不受训练随机性影响。

图怎么看：

```text
composed_maps:
横轴是输入奇异值 sigma，纵轴是经过完整 schedule 后的输出。
越靠近 1，表示越强地把奇异值推向正交目标。

composed_map_delta:
纵轴是 p_T(sigma) - sigma。
正值表示放大，负值表示压缩。
```

## 6. 新实验 C：manual iteration-depth trend

目的：观察 manual fast-to-stable schedule 增加迭代步数后是否更好。

注意：这不是严格的“只改变 T”实验，因为代表点的 fast/stable split 也随 T 变化。更准确地说，它是 manual family 的 depth trend。

| schedule | final val loss | val-loss AUC |
|---|---:|---:|
| manual_T5_f3_s2 | 4.317646 | 5.029547 |
| manual_T7_f4_s3 | 4.287305 | 4.931248 |
| manual_T8_f5_s3 | 4.289020 | 4.925309 |
| manual_T9_f3_s6 | 4.288733 | 4.927095 |
| manual_T9_f4_s5 | 4.292259 | 4.926766 |
| manual_T10_f5_s5 | 4.294906 | 4.928156 |
| fast5 | 4.282755 | 4.934028 |

结论：manual 从 T=5 增加到 T=7/8/9/10 后明显追近 fast5，但没有稳定超过 fast5。

结论强度：中。趋势清楚，但多数点是 seed0。

相关图：

```text
results/followup_4090_20260608/figures/manual_depth_final_loss.png
```

图怎么看：

```text
每个点是一种 manual schedule。
纵轴越低越好。
它用于看 depth trend，不用于宣称“某个算法整体更强”。
```

## 7. 新实验 D：PE lower-bound sensitivity

目的：固定 PE T=5，只改变 lower bound。

| schedule | final val loss | val-loss AUC |
|---|---:|---:|
| pe_T5_l3e-3 | 4.287261 | 4.946200 |
| pe_T5_l1e-3 | 4.290226 | 4.933800 |
| pe_T5_l3e-4 | 4.299781 | 4.934898 |
| pe_T5_l3e-5 | 4.312518 | 4.947870 |

结论：PE lower bound 是实质参数。它改变奇异值区间假设，也改变训练结果。

结论强度：中。足以说明 lower bound 敏感，但不是完整 lower-bound 全局最优搜索。

相关图：

```text
results/followup_4090_20260608/figures/pe_lower_bound_final_loss.png
```

图怎么看：

```text
横轴是 PE lower bound，纵轴是 final validation loss。
固定 T=5 时，柱子变化说明 lower bound 本身会影响结果。
```

## 8. 新实验 E：PE iteration depth

目的：固定 lower bound 为 \(3\times10^{-5}\)，改变 PE iteration count。

| schedule | final val loss | val-loss AUC |
|---|---:|---:|
| pe_T5_l3e-5 | 4.312518 | 4.947870 |
| pe_T9_l3e-5 | 4.296119 | 4.928503 |
| pe_T10_l3e-5 | 4.294432 | 4.927799 |
| fast5 | 4.282755 | 4.934028 |

结论：PE 增加 iteration depth 后 loss 变好，但仍没有超过 fast5。

结论强度：中。只有代表点，不是完整 T=5..10 sweep。

## 9. 新实验 F：spectral breakdown

目的：看不同 schedule 是否真的改变 update geometry。

| schedule | buffer_post error | g_pre error | g_post error |
|---|---:|---:|---:|
| stable5 | 3.0746 | 3.1320 | 0.9006 |
| fast5 | 3.4377 | 3.4914 | 0.6444 |
| manual_T9_f4_s5 | 3.7297 | 3.7885 | 0.4431 |
| pe_T9_l3e-5 | 3.8317 | 3.9036 | 0.3781 |

三个对象含义：

```text
buffer_post: momentum buffer 更新后的矩阵
g_pre: 进入正交化前的 Nesterov mixed matrix
g_post: 正交化后的 update matrix
```

attention / MLP 分解：

| schedule | attention g_post error | MLP g_post error |
|---|---:|---:|
| fast5 | 0.5146 | 0.7743 |
| manual_T9_f4_s5 | 0.1776 | 0.7086 |
| pe_T9_l3e-5 | 0.0481 | 0.7081 |

结论：manual_T9_f4_s5 和 pe_T9_l3e-5 明显降低 g_post error，尤其在 attention matrices 上。但它们的 validation loss 没有超过 fast5。

结论强度：强。几何差异很大；但“几何更好为什么 loss 不一定更好”仍需要理论解释。

相关图：

```text
results/followup_4090_20260608/figures/spectral_object_error_by_schedule.png
results/followup_4090_20260608/figures/attention_vs_mlp_gpost_error.png
```

图怎么看：

```text
spectral_object_error_by_schedule:
比较 buffer_post、g_pre、g_post 三个阶段。
g_post 越低，说明正交化后的 update 越接近目标。

attention_vs_mlp_gpost_error:
分别看 attention 和 MLP 矩阵。
结果显示 attention 上的区分度更大。
```

## 10. 新实验 G：multi-seed confirmation

目的：确认代表配置的差距是否大于 seed noise。

| schedule | seed0 | seed1 | seed2 | mean | std |
|---|---:|---:|---:|---:|---:|
| fast5 | 4.282755 | 4.292107 | 4.292769 | 4.289210 | 0.004573 |
| manual_T9_f4_s5 | 4.292259 | 4.299827 | 4.296642 | 4.296243 | 0.003102 |
| pe_T9_l3e-5 | 4.296119 | 4.304651 | 4.298659 | 4.299810 | 0.003577 |

结论：当前 100M-token clean setup 下，fast5 是最稳的 practical baseline。

结论强度：强，但只限当前模型和 100M token 设置。

## 11. 新实验 H：LR sanity

目的：确认结论不是因为某个 schedule 偶然拿到了更合适的 LR。

| schedule | lr=0.5 | lr=1.0 | lr=2.0 |
|---|---:|---:|---:|
| fast5 | 4.781276 | 4.282755 | 4.858344 |
| manual_T9_f4_s5 | 4.710713 | 4.292259 | 5.228550 |
| pe_T9_l3e-5 | 4.700085 | 4.296119 | 5.302590 |

结论：三个代表 schedule 都在 lr_mul=1.0 下最好。

结论强度：中到强。每个 LR 是 seed0，但差距足够大。

## 12. Benchmark

| schedule | wall-clock |
|---|---:|
| AdamW | 1039.732s |
| stable5 | 1094.865s |
| fast5 | 1095.099s |
| manual_T5_f3_s2 | 1099.423s |
| manual_T9_f4_s5 | 1123.528s |
| pe_T5_l3e-5 | 1098.117s |
| pe_T9_l3e-5 | 1119.423s |

结论：

```text
同 T=5 的 Muon variants wall-clock 很接近。
T=9 manual/PE 会更慢一些。
AdamW 更快，因为它没有矩阵正交化路径。
```

注意：当前代码中 AdamW 和 Muon 都是手写 Torch path。不要把 AdamW 更快解释成 fused kernel 差异。

## 13. 最终结论

1. 固定 T=5 时，系数序列会显著影响训练。
2. fast Muon coefficients 是当前最强 practical baseline。
3. stable Newton-Schulz 更保守，但在当前 LM 预训练中明显落后。
4. manual/PE 可以显著改善正交化几何，尤其 attention matrices。
5. 更低的 semi-orthogonality error 不必然带来更低 validation loss。
6. PE lower bound 和 iteration depth 都是有数学含义的谱映射参数。
7. AdamW 是 non-orthogonalization baseline，不是核心 orthogonalizer 对照。

最适合汇报的主线：

```text
不同 Newton-Schulz / Polar Express 系数序列
通过 composed singular-value map p_T(sigma)
改变 update geometry；
但更强的几何正交化不自动等价于更好的 validation loss。
```
