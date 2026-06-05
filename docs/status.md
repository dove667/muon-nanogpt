# 项目状态

## 当前阶段：fixed T=5 对照实验

- 5 配置 × 固定 seed 1 次 = 5 runs（不再考虑种子随机性）
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
- 训练入口移除 CLI seed，固定 seed 写死在代码内，run 名改为时间戳
- 数据读取改为按 shard 流式加载，避免一次性把全部数据 pin 到内存
- 修复 data_generator 的 microbatch 取样长度错误，按 `(seq_len + 1)` 成对切分 inputs/targets
- 修复 spectral 模式参数名错误、AdamW 模式学习率日志、final val loss 取值错误
- benchmark 只记录端到端总时间，分析图同步改为总 wall-clock 柱状图
- 固定栈 warmup 改为前 2%，Adam 路径改为标准 decoupled AdamW，并对齐常见小 GPT 基线超参数
- run 名时间戳简化为 `MMDD_HHMM`，并将 `min_lr_frac` 提高到 0.2，减轻 70M+ token 后的过早变平
- analysis 脚本移除多 seed / 多 run 平均与宽松兼容；按 `train` / `benchmark` / `spectral` 严格区分，同一 `orth` 同一模式重复即报错
- 训练后汇总输出合并为单个 `results/summary.csv`，每个 `orth` 一行，减少 `run_summary.csv` / `orth_summary.csv` 的重复
- `summarize_runs.py` 汇总流程收成单层，区分 `orthogonalizer_type` 与 `orth_error` 语义，减少 `orth` 歧义命名
- train CLI 将 `--benchmark` 和 `--spectral` 设为互斥参数，禁止单个 run 混合两种诊断模式
- runbook / README / AGENTS / experiments 文档同步为三模式组织：推荐按 `runs/train|benchmark|spectral/` 分目录，分析输出统一为 `results/summary.csv` + `results/figures/`
- 文档进一步明确：`runs/` 是当前分析工作区，历史实验需移到 `runs/` 外归档，避免被递归分析误读
- 文档同步更新：README / AGENTS.md / runbook / experiments / status

## 固定训练栈

- train token budget = 100M
- eval every = 2M tokens, eval tokens = 524,288
- tokens_per_step = 131,072
- seq_len = 2048
- grad_accum_steps = 16
- LR = 2% warmup + cosine decay 到峰值 20%
