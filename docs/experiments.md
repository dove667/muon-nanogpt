# Muon Newton-Schulz 正交化调度对比实验

## 一、研究问题

Muon 优化器通过 Newton-Schulz（NS）迭代将动量矩阵正交化。每次迭代由五次多项式系数 $(a, b, c)$ 定义：$p(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$。不同系数决定了每次迭代将奇异值推向 1 的速度。

在 **T=5 步固定**的前提下，五种系数序列构成一个从慢到快的谱系：

| 策略 | 每步系数 | 收敛速度 |
|------|---------|----------|
| **AdamW** | 无正交化 | 基线 |
| **Vanilla** | 5× $(2.0, -1.5, 0.5)$ | 最慢，标准 NS 迭代 |
| **Manual** | 3× 快速 + 2× 稳定 | 先快后慢 |
| **Fast** | 5× $(3.4445, -4.7750, 2.0315)$ | 最快，Keller Jordan 调优系数 |
| **Polar Express** | 每步自适应动态计算 | 自适应 |

本实验在完全固定的训练栈上对比这五种策略。

## 二、控制变量

除 NS 系数序列外，**所有变量在各组间保持一致**。每种策略跑 1 次固定 seed 实验（不再考虑种子随机性——收敛稳定性通过验证损失曲线判断）。

### 硬件

| 参数 | 值 |
|------|-----|
| GPU | 1× RTX 4090（24GB VRAM） |
| CUDA | 12.1 |
| PyTorch | 2.5.1+cu121 |

### 训练栈

| 参数 | 值 |
|------|-----|
| 模型 | 11 层标准 prenorm Transformer，hidden_dim=768，6 heads（head_dim=128），MLP ratio=4 |
| 数据 | FineWeb-10B（预分词 token 分片，朴素连续 block 采样） |
| 训练 token 预算 | **100M** |
| 验证 | 每 2M tokens，验证 524,288 tokens |
| 精度 | BF16 |
| 序列长度 | 2048（全程固定） |
| tokens_per_step | 131,072（16 gradient accumulation × 4 seq/microbatch × 2048） |
| 梯度累积 | 16 步 |
| 随机种子 | 固定为代码内常量 0 |
| 位置编码 | 标准 RoPE |
| 注意力 | 标准 causal self-attention |
| LR 调度 | warmup（前 10% 步线性升至峰值）→ cosine decay 至峰值 10% |

### Muon 组（Vanilla / Manual / Fast / Polar Express）

| 参数 | 值 |
|------|-----|
| 矩阵参数学习率 | 0.023（`lr_mul=1.0`） |
| 动量 | 0.95 |
| NS 迭代步数 | **T=5** |
| 权重衰减 | 1.2 |

### 各策略系数序列

- **Vanilla**：5 步全用 $(2.0, -1.5, 0.5)$
- **Manual**：前 3 步 $(3.4445, -4.7750, 2.0315)$ + 后 2 步 $(2.0, -1.5, 0.5)$
- **Fast**：5 步全用 $(3.4445, -4.7750, 2.0315)$
- **Polar Express**：`T=5`，`lower_bound=1e-3`，`cushion=0.02`，`safety_factor=0.02`

### AdamW 组

| 参数 | 值 |
|------|-----|
| 学习率 | 0.008 |
| $\beta_1$, $\beta_2$ | (0.9, 0.95) |
| weight decay | 0.005 |

## 三、运行模式

训练支持三种模式，应分开跑（每种模式有不同程度的 `torch.cuda.synchronize()` 开销）：

| 模式 | 命令 | 用途 |
|------|------|------|
| 默认（纯训练） | `python -m src.training.train --orth <mode> --data-path /data` | 最快，产出 loss 曲线 |
| Benchmark | `... --benchmark` | 测量端到端 wall-clock 时间 |
| Spectral | `... --spectral` | 采集优化器动量矩阵的 SVD 频谱 |


## 四、评估指标

| 维度 | 指标 | 含义 |
|------|------|------|
| 最终效果 | `val/loss` 最终值 | 验证损失 |
| 收敛速度 | `val/loss` 随 token 数变化曲线 | 同等预算下谁收敛更快 |
| 计算开销 | `benchmark/wall_clock_s` | 端到端墙钟时间 |
| 正交性质量 | `spec/update_orth_error` | $\|U^TU - I\|$（仅 spectral 模式输出） |

## 五、预期

- **AdamW** 作为无正交化的下界参照
- **Vanilla**（全慢）收敛最慢、正交质量可能不达标，但计算量最低
- **Fast**（全快）收敛最快，但可能"冲过头"导致正交精度下降
- **Manual** 居于两者之间
- **Polar Express** 期望在收敛速度与正交质量间取得最佳平衡，但有额外的逐步五次多项式计算开销

## 六、实现说明

当前实现已经对齐这份实验设计：

1. 模型为标准 per-layer prenorm Transformer，使用标准 `nn.Linear`
2. tied embedding + lm_head，标准 cross-entropy
3. 数据为朴素连续 block 采样，不做 BOS packing 或变长 attention
4. Adam 参数每步更新，无交错 step trick
5. 默认训练模式零 `torch.cuda.synchronize()`
6. 命名统一：`tokens_per_step` / `tokens_per_microbatch` / `sequences_per_microbatch`
