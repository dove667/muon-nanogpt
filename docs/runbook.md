# 运行指南

## 环境

RTX 4090 (24GB VRAM) 单卡，CUDA 12.1，PyTorch 2.5.1+cu121。当前代码没有分布式路径，训练栈默认就是单卡、标准 Transformer 和固定长度 block 采样。

数据路径按实际位置设置，以下示例中用 `/data/fineweb10B` 代替。

## 配置

所有固定超参数集中在 `src/config/config.yaml`，修改后对全部实验生效。CLI 只暴露必须变化的 3 个参数。

当前固定训练栈：

- 标准 11 层 prenorm Transformer，`model_dim=768`，`num_heads=6`，`head_dim=128`
- `seq_len=2048`
- `tokens_per_step=131072`
- `grad_accum_steps=16`
- `train_token_budget=100M`
- `eval_interval_tokens=2M`
- `eval_tokens=524288`
- LR 为 10% warmup + cosine decay 到峰值 10%

## 1. 单次训练

```bash
python -m src.training.train --orth fast --data-path /data/fineweb10B
```

五种 orth 模式：

```bash
python -m src.training.train --orth adamw         --data-path /data/fineweb10B
python -m src.training.train --orth vanilla       --data-path /data/fineweb10B
python -m src.training.train --orth fast          --data-path /data/fineweb10B
python -m src.training.train --orth manual        --data-path /data/fineweb10B
python -m src.training.train --orth polar_express --data-path /data/fineweb10B
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--orth` | str | `fast` | 正交化策略：`adamw` / `vanilla` / `manual` / `fast` / `polar_express` |
| `--data-path` | str | **必传** | 数据集根目录 |
| `--benchmark` | flag | False | 开启 wall-clock 计时（含 cuda synchronize） |
| `--spectral` | flag | False | 采集优化器状态频谱指标 |

默认模式下训练循环无 `torch.cuda.synchronize()` 开销。`--benchmark` 和 `--spectral` 应分开跑。
随机种子固定写死在代码里，不通过 CLI 暴露；run 名使用时间戳，避免复跑时覆盖或追加到旧日志。

所有其他参数（训练预算、batch、seq_len、LR、正交化细节等）均在 `src/config/config.yaml` 中管理。

## 2. 分析报告

### 汇总 CSV

```bash
python -m src.analysis.summarize_runs
```

输出：`results/run_summary.csv` + `results/orth_summary.csv`

### 曲线图

```bash
python -m src.analysis.plot_curves
```

输出：`results/figures/` 下 `val_loss_vs_tokens` / `benchmark_wall_clock` / `final_val_loss`

## 3. 数据下载

```bash
python scripts/download_fineweb.py [num_chunks]
```

## 运行产物

每轮训练输出到 `runs/<name>/`：

| 文件 | 内容 |
|---|---|
| `config.json` | 实验配置快照（run_name、固定 seed、base_lr、train_token_budget、orth_config） |
| `metrics.jsonl` | 每步一条 JSON（训练指标、验证指标、谱指标） |

分析脚本读取 `runs/` 并输出到 `results/`：

- `results/run_summary.csv`：逐 run 汇总
- `results/orth_summary.csv`：按配置聚合
- `results/figures/`：对比图

## 注意

- 数据格式：FineWeb-10B 预分词 token shard（`fineweb_train_*.bin` / `fineweb_val_*.bin`）
- 训练时使用固定长度 `seq_len=2048` 的朴素连续 block 采样，不再做 BOS packing 或变长 attention
- 固定训练栈详见 `src/config/config.yaml`
