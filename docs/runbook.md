# 运行指南

## 环境

RTX 4090 (24GB VRAM) 单卡，CUDA 12.1，PyTorch 2.5.1+cu121。162M 模型在单卡上完全够用，无分布式逻辑。

数据路径按实际位置设置，以下示例中用 `/data/fineweb10B` 代替。

## 配置

所有固定超参数集中在项目根 `config.yaml`，修改后对全部实验生效。CLI 只暴露必须变化的 3 个参数。

## 1. 单次训练

```bash
python src/training/train.py --orth fast --seed 0 --data-path /data/fineweb10B
```

五种 orth 模式：

```bash
python src/training/train.py --orth adamw         --seed 0 --data-path /data/fineweb10B
python src/training/train.py --orth vanilla       --seed 0 --data-path /data/fineweb10B
python src/training/train.py --orth fast          --seed 0 --data-path /data/fineweb10B
python src/training/train.py --orth manual        --seed 0 --data-path /data/fineweb10B
python src/training/train.py --orth polar_express --seed 0 --data-path /data/fineweb10B
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--orth` | str | `fast` | 正交化策略：`adamw` / `vanilla` / `manual` / `fast` / `polar_express` |
| `--seed` | int | `0` | 随机种子 |
| `--data-path` | str | **必传** | 数据集根目录 |

所有其他参数（训练预算、batch、seq_len、LR、正交化细节等）均在 `config.yaml` 中管理。

## 2. 完整实验计划

固定 5 配置 × 3 seeds = 15 runs：

```bash
python -m src.experiment_plan --data-path /data/fineweb10B
```

跳过已完成轮次：

```bash
python -m src.experiment_plan --data-path /data/fineweb10B --skip-completed-runs
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--data-path` | str | **必传** | 数据集根目录 |
| `--skip-completed-runs` | flag | False | 跳过已完成的运行 |

## 3. 分析报告

### 汇总 CSV

```bash
python -m src.analysis.summarize_runs
```

输出：`results/run_summary.csv` + `results/orth_summary.csv`

### 曲线图

```bash
python -m src.analysis.plot_curves
```

输出：`results/figures/` 下 7 张 PNG

### 仪表板

```bash
python -m src.analysis.build_dashboard
```

输出：`results/dashboard.html`

## 运行产物

每轮训练输出到 `runs/<name>/`：

| 文件 | 内容 |
|---|---|
| `config.json` | 实验配置快照（run_name、seed、base_lr、train_token_budget、orth_config） |
| `metrics.jsonl` | 每步一条 JSON（训练指标、验证指标、谱指标） |

分析脚本读取 `runs/` 并输出到 `results/`：

- `results/run_summary.csv`：逐 run 汇总
- `results/orth_summary.csv`：按配置聚合后的 mean/std
- `results/figures/`：对比图
- `results/dashboard.html`：轻量 HTML 报告

## 注意

- `5090_results/` 是历史存档，**禁止修改**
- 数据格式：FineWeb-10B 预分词 BOS 对齐 shard（`fineweb_train_*.bin` / `fineweb_val_*.bin`）
- 固定训练栈详见 `config.yaml`
