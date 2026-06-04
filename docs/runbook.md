# 运行指南

## 环境

4× RTX 4090 (24GB VRAM)，CUDA 12.1，PyTorch 2.5.1+cu121。使用 `torchrun` 启动分布式训练，无需 `accelerate` 或 `deepspeed`。

数据路径按实际位置设置，以下示例中用 `/data/fineweb10B` 代替。

## 单次训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
python -m src.run_training \
  --orth vanilla \
  --group test \
  --name smoke_test \
  --lr-mul 1.0 \
  --train-token-budget 2000000 \
  --eval-every-tokens 1000000 \
  --eval-tokens 131072 \
  --data-path /data/fineweb10B \
  --wandb off
```

使用 Manual、Fast、AdamW 或 Polar Express 时可这样切换：

```bash
# Manual：5 步中前 3 步快速、后 2 步稳定
python -m src.run_training --orth manual --ns-t 5 --fast-steps 3 --stable-steps 2 ...

# Fast：5 步全 fast 系数
python -m src.run_training --orth fast ...

# AdamW baseline：矩阵参数不做 Muon 正交化
python -m src.run_training --orth adamw ...

# Polar Express：T=5，奇异值下界 1e-3
python -m src.run_training --orth polar_express --pe-t 5 --pe-lower-bound 1e-3 ...
```

## 运行完整实验计划

实验计划详见 [`docs/experiments.md`](experiments.md)。当前项目已经化简为单一固定实验入口：

```bash
# 5 配置 × 3 seeds = 15 runs
python -m src.experiment_plan --data-path /data/fineweb10B
```

跳过已完成轮次（断点续跑）：

```bash
python -m src.experiment_plan --data-path /data/fineweb10B --skip-completed-runs
```

## 生成分析报告

```bash
# 汇总为 CSV
python -m src.analysis.summarize_runs --runs-dir runs --out-dir results --print-top 12

# 生成曲线图
python -m src.analysis.plot_curves --runs-dir runs --out-dir results/figures

# 构建交互式仪表板
python -m src.analysis.build_dashboard --analysis-dir results --out results/dashboard.html
```

## 参数速查

### 通用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--orth` | `fast` | 正交化策略：`adamw` / `vanilla` / `manual` / `fast` / `polar_express` |
| `--lr-mul` | `1.0` | 学习率倍率（Muon 组基础 LR 为 0.023，AdamW 组基础 LR 为 0.008） |
| `--seed` | `0` | 随机种子 |
| `--train-token-budget` | `100000000` | 训练 token 预算 |
| `--eval-every-tokens` | `10000000` | 验证间隔（token 数） |
| `--eval-tokens` | `2097152` | 每次验证的 token 数 |
| `--data-path` | — | 数据集根目录（必填，无默认值） |
| `--wandb` | `on` | W&B 日志开关：`on` / `off` |

### Manual 模式专属参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ns-t` | `5` | Newton-Schulz 总迭代数 |
| `--fast-steps` | `ns_t` | 快速系数步数 |
| `--stable-steps` | `0` | 稳定系数步数（须满足 `fast + stable = ns_t`） |

### Polar Express 模式专属参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--pe-t` | `5` | Polar Express 迭代数 |
| `--pe-lower-bound` | `1e-3` | 奇异值下界 |
| `--pe-cushion` | `2e-2` | cushion 参数 |
| `--pe-safety-factor` | `2e-2` | 安全因子 |

## 运行产物

每轮训练输出到 `runs/<group>/<name>/`：

| 文件 | 内容 |
|---|---|
| `config.json` | 完整配置快照 |
| `metrics.jsonl` | 每步一条 JSON（训练指标、验证指标、谱指标） |
| `console.log` | torchrun 标准输出 |

分析脚本读取 `runs/` 并输出到 `results/`。

## 注意

- `train.py` 在 `torchrun` 内部运行，使用相对导入（如 `import polar`），**不要改为包导入**。
- 首次运行会编译模型和预热 CUDA 内核（约 7 分钟），后续运行复用 `.torchinductor/` 缓存。
- `5090_results/` 是之前硬件的存档输出，**请勿修改**。
- 固定实验栈默认使用 `seq_len=2048`、`batch=16*2048*8`、`window=(3,7)` 与 warmup+cosine LR。
