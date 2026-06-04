# 项目状态

## 当前阶段：fixed T=5 对照实验

- 5 配置 × 1 seed = 5 runs（不再考虑种子随机性）
- 训练入口：`python -m src.training.train --orth <mode> --data-path /data`
- 三种模式分开跑：纯训练（默认）/ Benchmark（`--benchmark`）/ Spectral（`--spectral`）
- 默认模式零 `torch.cuda.synchronize()` 开销
- 所有固定超参数在 `src/config/config.yaml`

## 已完成

- 模型：标准 per-layer prenorm Transformer + RoPE，`nn.Linear`，tied embedding
- 数据：朴素连续 block 采样（`fineweb_train_*.bin` / `fineweb_val_*.bin`）
- 优化器：Muon 走 NS 正交化 → 矩阵参数；Adam → 非矩阵参数和 embedding
- 架构：`build_optimizer(...)` 构造 / `step_optimizer(...)` 驱动 / 训练状态为局部变量
- orth 调度：纯函数 `build_coeff_schedule / orth_record / orth_norm_factor`
- 目录按职责重组：`config/` `data/` `model/` `optim/` `training/` `analysis/`
- 命名统一：`tokens_per_step` / `tokens_per_microbatch` / `sequences_per_microbatch`
- 删除了 `experiment_plan.py`（编排层）、data_generator 动态重配置分支
- 训练循环拆分为三种独立模式，各自无交叉开销
- 去除所有竞速 trick（参数银行、softcap logits、ReLU² MLP、QK norm 等）
- 文档同步更新：README / AGENTS.md / runbook / experiments / status

## 固定训练栈

- train token budget = 100M
- eval every = 2M tokens, eval tokens = 524,288
- tokens_per_step = 131,072
- seq_len = 2048
- grad_accum_steps = 16
- LR = 10% warmup + cosine decay 到峰值 10%
