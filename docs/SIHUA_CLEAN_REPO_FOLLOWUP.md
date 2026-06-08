# Clean Repo 后续实验方向

核心问题：

```text
在模型、数据、token budget、LR schedule、batch 结构和优化器状态固定时，
不同 Newton-Schulz / Polar Express 系数序列如何改变 update geometry
和预训练行为？
```

## 1. Polynomial singular-value map analysis

目的：先从数学上解释不同系数序列诱导的 singular-value dynamics，
避免把实验解释成单纯调参。

对象：

```text
stable5
fast5
manual_f3_s2
manual iteration-depth representative schedules
PE lower-bound schedules
PE T-depth schedules
```

分析：

```text
p(sigma) = a sigma + b sigma^3 + c sigma^5
p_1:T(sigma)
p(sigma) - sigma
p'(sigma)
fixed / near-fixed regions
attracting / repelling intervals
```

用途：

```text
解释小奇异值如何被放大
解释哪些区间被推向 semi-orthogonal target
解释 lower bound 和 iteration depth 为什么会改变 geometry
```

## 2. Manual iteration-depth trend

目的：验证增加 Newton-Schulz 迭代步数后，fast-to-stable 系数序列是否
相对 fast-only 带来真实的 geometry 或 loss 优势。

Primary trend 配置：

```text
p2_T5_f3_s2
p2_T7_f4_s3
p2_T8_f5_s3
p2_T9_f4_s5
p2_T10_f5_s5
```

Local T=9 check：

```text
p2_T9_f9_s0
p2_T9_f5_s4
p2_T9_f4_s5
p2_T9_f3_s6
```

比较：

```text
val loss vs tokens
val loss vs wall-clock
g_pre -> g_post semi_orth_error
attention vs MLP g_post semi_orth_error
orthogonalizer_time_ms
```

## 3. T=5 Polar Express lower-bound sensitivity

目的：验证 PE lower bound 假设如何改变诱导出的 singular value map，
以及最终的 update geometry。

配置：

```text
PE T5 l3e-3
PE T5 l1e-3
PE T5 l3e-4
PE T5 l3e-5
```

比较：

```text
val loss vs tokens
g_pre -> g_post semi_orth_error
singular value spread
stable rank / SVD entropy
attention vs MLP g_post semi_orth_error
```

## 4. Polar Express iteration depth

目的：验证增加 PE 迭代步数是只改善 geometry，还是也能转化为 training
loss 和 wall-clock efficiency 优势。

配置：

```text
PE T5  l3e-5
PE T9  l3e-5
PE T10 l3e-5
```

比较：

```text
same-token val loss
same-wall-clock val loss
g_post semi_orth_error
attention vs MLP g_post semi_orth_error
orthogonalizer_time_ms
```

## 5. Key spectral breakdown

目的：用少量 spectral run 验证不同调度对 attention 和 MLP 矩阵的影响
是否一致。

配置：

```text
stable5
fast5
manual_f3_s2
PE T5 l1e-3
PE T5 l3e-5
PE T9 l3e-5
best manual trend config
```

比较：

```text
buffer_post semi_orth_error
g_pre semi_orth_error
g_post semi_orth_error
attention g_post semi_orth_error
MLP g_post semi_orth_error
singular value spread
stable rank / SVD entropy
```

## 6. Final multi-seed confirmation

目的：确认最强候选配置之间的差异是否大于 seed noise。

配置：

```text
fast5
best manual
best PE
```

Seeds：

```text
0
1
2
```

比较：

```text
mean / std final val loss
tokens_to_target
time_to_target
geometry metrics
```

## 7. LR sanity check

目的：确认系数序列结论不是由某个偶然 learning rate 选择造成的。

配置：

```text
fast5
best manual
best PE
```

LR：

```text
0.5
1.0
2.0
```

比较：

```text
final val loss
loss curve stability
g_post semi_orth_error
```

## 8. AdamW kernel-control benchmark

目的：当 AdamW 被用作 wall-clock 参照时，区分算法成本和
implementation / kernel maturity 的影响。

最小控制组：

```text
AdamW current manual path
AdamW optimized PyTorch fused / foreach path
Muon current path
```

解释口径：

```text
Use AdamW mainly as a non-orthogonalization baseline for loss.
Use AdamW wall-clock only after reporting the optimized-vs-unfused kernel gap.
Do not make AdamW-vs-Muon wall-clock a main mathematical claim.
```

如果时间允许再做：

```text
Muon fused / foreach implementation
```

这属于 systems-extension experiment，不属于核心数学系数序列研究。
