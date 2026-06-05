# 运行指南

## 环境

RTX 4090 (24GB VRAM) 单卡，CUDA 12.1，PyTorch 2.5.1+cu121。当前代码没有分布式路径，训练栈默认就是单卡、标准 Transformer 和固定长度 block 采样。

数据路径按实际位置设置，以下示例中用 `/data/fineweb10B` 代替。

## 配置

所有固定超参数集中在 `src/config/config.yaml`，修改后对全部实验生效。

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

用途：采集 Muon 真实更新对象的频谱与半正交统计。该模式会记录 `spec/*` 聚合指标，并额外输出逐矩阵的 `spectral_details.jsonl`。

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--orth` | str | `fast` | 正交化策略：`adamw` / `vanilla` / `manual` / `fast` / `polar_express` |
| `--data-path` | str | **必传** | 数据集根目录 |
| `--benchmark` | flag | False | 开启 wall-clock 计时（含 cuda synchronize） |
| `--spectral` | flag | False | 采集 Muon 更新对象的谱指标 |

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

### 谱明细导出

```bash
python -m src.analysis.export_spectral_details
```

输出：`results/spectral_details.csv`

该文件会把所有 spectral run 的 `spectral_details.jsonl` 合并成一个表，便于按层、按对象、按矩阵做离线分析。

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
| `spectral_details.jsonl` | 仅 spectral 模式生成；每次采样的逐矩阵谱统计 |

`runs/` 是当前分析工作区，不是历史实验仓库。运行 `analysis/` 下的脚本时一定要满足以下条件：

- `runs/` 里只保留当前要分析的一组 run，历史 run 需要移到 `runs/` 外的其他目录归档，避免被分析脚本递归误读
- 同一个 `orthogonalizer_type + mode` 在 `runs/` 下只能出现一次

分析脚本会递归扫描 `runs/` 下所有 `metrics.jsonl`，自动根据日志键把 run 判成 `train` / `benchmark` / `spectral`：

- 出现 `benchmark/wall_clock_s`：判为 `benchmark`
- 出现 `spec/sample_count`：判为 `spectral`
- 两者都没有：判为 `train`
- 两者同时出现：直接报错，视为非法混合模式

分析脚本读取 `runs/` 并输出到 `results/`：

- `results/summary.csv`：单表汇总，按 `orthogonalizer_type` 汇总三种模式
- `results/figures/`：对比图
- `results/spectral_details.csv`：逐矩阵谱明细导出

## Spectral 指标口径

当前 spectral 模式不再把 `momentum_buffer` 误当成真实 update 输入，而是在 Muon 真正更新时抓取三个对象：

- `buffer_post`：动量缓存更新后的矩阵
- `g_pre`：进入正交化前的 Nesterov 混合矩阵
- `g_post`：正交化后的矩阵

每个对象都会记录：

- `sv_min` / `sv_max` / `sv_std`
- `stable_rank`
- `svd_entropy`
- `semi_orth_error`

其中 `semi_orth_error` 的定义是：

- tall 矩阵使用 `X^T X`
- wide 矩阵使用 `X X^T`

也就是统一在 short-side Gram 上衡量 semi-orthogonality，而不是用一个会引起 tall / wide 语义混淆的 `orth_error` 名称。

## 注意

- 数据格式：FineWeb-10B 预分词 token shard（`fineweb_train_*.bin` / `fineweb_val_*.bin`）
- 训练时使用固定长度 `seq_len=2048` 的朴素连续 block 采样，不再做 BOS packing 或变长 attention
- 固定训练栈详见 `src/config/config.yaml`
