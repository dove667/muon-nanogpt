# 运行指南

## 环境

4× RTX 4090 (24GB VRAM)，CUDA 12.1，PyTorch 2.5.1+cu121。使用 `torchrun` 启动分布式训练，无需 `accelerate` 或 `deepspeed`。

数据路径按实际位置设置，以下示例中用 `/data/fineweb10B` 代替。

## 1. 单次训练（`python -m src.run_training`）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
python -m src.run_training \
  --orth vanilla \
  --name smoke_test \
  --lr-mul 1.0 \
  --train-token-budget 2000000 \
  --eval-every-tokens 500000 \
  --eval-tokens 131072 \
  --data-path /data/fineweb10B \
  --wandb off
```

使用 Manual、Fast、AdamW 或 Polar Express 时可这样切换：

```bash
# Manual：5 步中前 3 步快速、后 2 步稳定
python -m src.run_training --orth manual --fast-steps 3 --stable-steps 2 ...

# Fast：5 步全 fast 系数
python -m src.run_training --orth fast ...

# AdamW baseline：矩阵参数不做 Muon 正交化
python -m src.run_training --orth adamw ...

# Polar Express：奇异值下界 1e-3
python -m src.run_training --orth polar_express --pe-lower-bound 1e-3 ...
```

### 参数

#### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--orth` | str | `fast` | 正交化策略：`adamw` / `vanilla` / `manual` / `fast` / `polar_express` |
| `--name` | str | `$WANDB_NAME` / `$RUN_NAME` | 运行名称，同时也是输出目录名 |
| `--lr-mul` | float | `1.0` | 学习率倍率 |
| `--seed` | int | `0` | 随机种子 |
| `--train-token-budget` | int | `100000000` | 训练 token 预算 |
| `--eval-every-tokens` | int | `2000000` | 验证间隔（token 数） |
| `--eval-tokens` | int | `524288` | 每次验证的 token 数 |
| `--data-path` | str | — | 数据集根目录 |

#### Manual 模式专属参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--fast-steps` | int | `5` | 快速系数步数 |
| `--stable-steps` | int | `0` | 稳定系数步数（须满足 fast_steps + stable_steps = 5） |

#### Polar Express 模式专属参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--pe-lower-bound` | str | `1e-3` | 奇异值下界 |
| `--pe-cushion` | float | `2e-2` | cushion 参数 |
| `--pe-safety-factor` | float | `2e-2` | 安全因子 |

#### 训练控制

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--train-grad-accum-steps` | int | `16` | 梯度累积步数 |
| `--eval-batch-size` | int | — | 验证批次大小（None 时自动计算） |
| `--eval-at-start` | flag | False | 训练开始前先做一次验证 |
| `--log-every-steps` | int | `20` | 训练指标日志间隔（步数） |
| `--model-max-seq-len` | int | `0` | 最大序列长度（0 使用默认 2048） |

#### 日志与 W&B

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--wandb` | str | `on` | W&B 开关：`on` / `off` |
| `--wandb-project` | str | `muon-nanogpt` | W&B 项目名 |
| `--wandb-entity` | str | — | W&B entity 名 |
| `--wandb-mode` | str | — | W&B 模式（如 `offline`、`dryrun`） |

#### 谱分析

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--spectral-every-tokens` | int | `10000000` | 谱分析间隔（token 数） |
| `--spectral-max-matrices` | int | `5` | 谱分析最大矩阵数 |
| `--spectral-max-dim` | int | `1024` | 谱分析最大维度 |

#### 分布式

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--nproc-per-node` | int | `1` | 每节点进程数（torchrun 用） |

## 2. 完整实验计划（`python -m src.experiment_plan`）

实验计划详见 [`docs/experiments.md`](experiments.md)。当前入口为固定 5 配置 × 3 seeds = 15 runs：

```bash
# 基础运行
python -m src.experiment_plan --data-path /data/fineweb10B
```

跳过已完成轮次（断点续跑）：

```bash
python -m src.experiment_plan --data-path /data/fineweb10B --skip-completed-runs
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--skip-completed-runs` | flag | False | 跳过已完成的运行 |
| `--train-token-budget` | int | `100000000` | 训练 token 预算 |
| `--eval-every-tokens` | int | `2000000` | 验证间隔（token 数） |
| `--eval-tokens` | int | `524288` | 每次验证的 token 数 |
| `--train-grad-accum-steps` | int | `16` | 梯度累积步数 |
| `--eval-batch-size` | int | — | 验证批次大小 |
| `--eval-at-start` | flag | False | 训练开始前先做一次验证 |
| `--log-every-steps` | int | `20` | 训练指标日志间隔（步数） |
| `--wandb` | str | `on` | W&B 开关：`on` / `off` |
| `--wandb-project` | str | `muon-nanogpt` | W&B 项目名 |
| `--wandb-entity` | str | — | W&B entity 名 |
| `--wandb-mode` | str | — | W&B 模式（如 `offline`、`dryrun`） |
| `--nproc-per-node` | int | `1` | 每节点进程数 |
| `--data-path` | str | — | 数据集根目录 |
| `--model-max-seq-len` | int | `0` | 最大序列长度（0 使用默认 2048） |
| `--spectral-every-tokens` | int | `10000000` | 谱分析间隔（token 数） |
| `--spectral-max-matrices` | int | `5` | 谱分析最大矩阵数 |
| `--spectral-max-dim` | int | `1024` | 谱分析最大维度 |

## 3. 分析报告

### 3.1 汇总 CSV（`python -m src.analysis.summarize_runs`）

```bash
# 汇总为 run-level / config-level CSV
python -m src.analysis.summarize_runs --runs-dir runs --out-dir results --print-top 12
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--runs-dir` | str | `runs` | 读取运行数据的目录 |
| `--out-dir` | str | `results` | 输出 CSV 的目录 |
| `--print-top` | int | `0` | 打印 Top-N 运行终端摘要（0 表示不打印） |
| `--allow-empty` | flag | False | 允许目录为空时继续（不报错退出） |

### 3.2 曲线图（`python -m src.analysis.plot_curves`）

```bash
# 生成曲线图
python -m src.analysis.plot_curves --runs-dir runs --out-dir results/figures
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--runs-dir` | str | `runs` | 读取运行数据的目录 |
| `--out-dir` | str | `results/figures` | 输出图表的目录 |
| `--orths` | str[] | — | 只绘制指定 orth 类型的曲线（不指定则全部绘制） |

### 3.3 仪表板（`python -m src.analysis.build_dashboard`）

```bash
# 构建交互式仪表板
python -m src.analysis.build_dashboard --analysis-dir results --out results/dashboard.html
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--analysis-dir` | str | `results` | 读取分析 CSV 的目录 |
| `--out` | str | `results/dashboard.html` | 输出 HTML 文件路径 |

## 运行产物

验证数据默认来自 `DATA_PATH` 下匹配 `fineweb_val_*.bin` 的 shard；训练数据对应 `fineweb_train_*.bin`。

每轮训练输出到 `runs/<name>/`：

| 文件 | 内容 |
|---|---|
| `config.json` | 完整配置快照 |
| `metrics.jsonl` | 每步一条 JSON（训练指标、验证指标、谱指标） |
| `console.log` | torchrun 标准输出 |

分析脚本读取 `runs/` 并输出到 `results/`：

- `results/run_summary.csv`：逐 run 汇总
- `results/orth_summary.csv`：按配置聚合后的 mean/std
- `results/figures/`：对比图
- `results/dashboard.html`：轻量 HTML 报告

## 注意

- `train.py` 在 `torchrun` 内部运行，使用相对导入（如 `import polar`），**不要改为包导入**。
- 首次运行会编译模型和预热 CUDA 内核（约 7 分钟），后续运行复用 `.torchinductor/` 缓存。
- `5090_results/` 是之前硬件的存档输出，**请勿修改**。
- 固定实验栈默认使用 `seq_len=2048`、`batch=8*2048*8`、`grad_accum=16`、`window=(3,7)` 与 warmup+cosine LR。
