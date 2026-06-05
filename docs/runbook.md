# 运行指南

## 环境

RTX 4090 (24GB VRAM) 单卡，CUDA 12.1，PyTorch 2.5.1+cu121。当前代码没有分布式路径，训练栈默认就是单卡、标准 Transformer 和固定长度 block 采样。

数据路径按实际位置设置，以下示例中用 `/data/fineweb10B` 代替。

## 配置

所有固定超参数集中在 `src/config/config.yaml`，修改后对全部实验生效。CLI 只暴露必须变化的 2 个参数，外加 2 个互斥诊断 flag。

当前固定训练栈：

- 标准 11 层 prenorm Transformer，`model_dim=768`，`num_heads=6`，`head_dim=128`
- `seq_len=2048`
- `tokens_per_step=131072`
- `grad_accum_steps=16`
- `train_token_budget=100M`
- `eval_interval_tokens=2M`
- `eval_tokens=524288`
- LR 为前 2% warmup + cosine decay 到峰值 20%

## 1. 三种运行模式

训练有三种模式，必须分开跑；单个 run 只能属于其中一种。

### 1.1 纯训练模式

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

### 1.2 Benchmark 模式

```bash
python -m src.training.train --orth fast --data-path /data/fineweb10B --benchmark
```

用途：测量端到端 wall-clock 时间。该模式会记录 `benchmark/wall_clock_s`，并引入 `torch.cuda.synchronize()` 开销。

### 1.3 Spectral 模式

```bash
python -m src.training.train --orth fast --data-path /data/fineweb10B --spectral
```

用途：采集优化器状态频谱与正交统计。该模式会记录 `spec/*` 指标。

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--orth` | str | `fast` | 正交化策略：`adamw` / `vanilla` / `manual` / `fast` / `polar_express` |
| `--data-path` | str | **必传** | 数据集根目录 |
| `--benchmark` | flag | False | 开启 wall-clock 计时（含 cuda synchronize） |
| `--spectral` | flag | False | 采集优化器状态频谱指标 |

默认模式下训练循环无 `torch.cuda.synchronize()` 开销。`--benchmark` 和 `--spectral` 是互斥参数，不能同时传。
随机种子固定写死在代码里，不通过 CLI 暴露；run 名使用时间戳，避免复跑时覆盖或追加到旧日志。

所有其他参数（训练预算、batch、seq_len、LR、正交化细节等）均在 `src/config/config.yaml` 中管理。

## 2. 分析报告

### 汇总 CSV

```bash
python -m src.analysis.summarize_runs
```

输出：`results/summary.csv`

该 CSV 以 `orthogonalizer_type` 为行索引口径，同一行内分列放置 `train_*`、`benchmark_*`、`spectral_*` 三种模式的 run 名和关键指标。

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

每轮训练默认输出到 `runs/<name>/`：

| 文件 | 内容 |
|---|---|
| `config.json` | 实验配置快照（run_name、固定 seed、base_lr、train_token_budget、orth_config） |
| `metrics.jsonl` | 该 run 对应模式下的时序指标 |

推荐在训练完成后按模式分目录整理 `runs/`，例如：

```text
runs/
  train/
    0605_1012_fast/
    0605_1013_manual/
  benchmark/
    0605_1110_fast/
  spectral/
    0605_1208_fast/
```

分析脚本会递归扫描 `runs/` 下所有 `metrics.jsonl`，自动根据日志键把 run 判成 `train` / `benchmark` / `spectral`：

- 出现 `benchmark/wall_clock_s`：判为 `benchmark`
- 出现 `spec/sample_count`：判为 `spectral`
- 两者都没有：判为 `train`
- 两者同时出现：直接报错，视为非法混合模式

分析脚本读取 `runs/` 并输出到 `results/`：

- `results/summary.csv`：单表汇总，按 `orthogonalizer_type` 汇总三种模式
- `results/figures/`：对比图

## 注意

- 数据格式：FineWeb-10B 预分词 token shard（`fineweb_train_*.bin` / `fineweb_val_*.bin`）
- 训练时使用固定长度 `seq_len=2048` 的朴素连续 block 采样，不再做 BOS packing 或变长 attention
- 固定训练栈详见 `src/config/config.yaml`
