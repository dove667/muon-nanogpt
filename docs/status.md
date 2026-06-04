# 项目状态

## 当前阶段：fixed T=5 对照实验，代码已彻底化简

- 实验编排：5 配置 × 3 seeds = 15 runs，单一 flat 布局
- CLI 只保留必要参数（`--data-path` `--orth` `--seed` `--name`），其余全部集中到 `config.yaml`
- 单卡 RTX 4090 跑 162M 模型无压力

## 已完成

### 代码化简（本轮最大改动）
- 删除全部分布式训练代码（`DistributedContext`, `torch.distributed`, `sparse_comms.py`, sharding, all-reduce, `broadcast_model`）
- 删除 W&B（`wandb` 初始化/日志/关闭、`--wandb*` CLI 参数、`pyproject.toml` 依赖项）
- 删除 `Hyperparameters` dataclass，所有参数直接传递
- 删除 `LoopConfig`，循环参数直接传给 `run_training_loop`
- 删除 `TrainingStage` / `TrainingSchedule` / `default_training_stages()`
- 删除 `schedule.py`，`compute_lr` 内联到 `manager.py`，`resolve_data_files` 移到 `utils.py`
- `RunLogger` → `Logger`，去掉公共字段注入，config.json 只记实验标识

### 集中化配置
- 新增 `config.yaml`（项目根），统一管理所有固定超参数：
  - training：batch_tokens / seq_len / grad_accum / warmup+cosine LR / eval / spectral 参数
  - model：bigram_vocab_size / block_size / window_sizes / mtp_weights
  - optimizer：完整 param_table + Adam/Muon 默认值 + lr_mul + step_interval
  - orthogonalization：NS 迭代步数 / polar_express 参数
- `src/training/config/loader.py` 加载 YAML，`config/__init__.py` 统一导出 `TRAINING` `MODEL` `OPTIMIZER`

### 目录重组
```
src/training/
├── config/    # yaml 加载 + 导出
├── data/      # Shard, data_generator
├── optim/     # NorMuonAndAdam, TrainingManager
├── orth/      # OrthogonalizerConfig, polar_express
├── model.py
├── metrics.py
├── run_support.py   # Logger, 训练/验证循环, setup_device
└── train.py         # 入口（CLI 仅 4 参数）
```

### 训练栈
- batch = `8 × 2048 × 8` = 131072 tokens/step
- seq_len = 2048, grad_accum = 16, window = (3, 7) blocks
- LR = 10% warmup + cosine decay 到峰值的 10%
- Muon momentum = 0.95, Adam 每 2 步更新一次
- 模型：11L 768D 6 头 head_dim=128，权重在 qk/vo/mlp bank 中
- MTP 权重固定 [1.0, 0.0]，YaRN 逻辑保留但不随阶段切换

### 分析流水线
- `summarize_runs.py` → `run_summary.csv` + `orth_summary.csv`
- `plot_curves.py` → 7 张对比图
- `build_dashboard.py` → 轻量 HTML 报告

## 待确认

- 固定栈窗口配置 `(3, 7)`、MTP 权重 `[1.0, 0.0]`
- YaRN 逻辑未删除，只是不再随阶段切换
- `5090_results/` 是历史存档，禁止修改
