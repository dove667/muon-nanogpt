# AGENTS.md

## 运行环境

- Conda 环境 `muon`（由 `environment.yml` 定义），或回退到 `conda run -n AI`
- Python 依赖用 `uv` 锁定，**禁止执行 `uv sync`**（会删除环境中的 conda 底层包）

## 训练入口

- 单次训练（纯训练模式，无额外同步开销）：`python -m src.training.train --orth <mode> --seed <n> --data-path /data/fineweb10B`
- BenchMark 模式（测量 wall-clock 吞吐）：`python -m src.training.train ... --benchmark`
- 谱分析模式（采集优化器状态频谱）：`python -m src.training.train ... --spectral`
- 162M 模型 4090 单卡完全够用，无分布式逻辑

## 配置管理

- **`src/config/config.yaml`** 是唯一配置来源，包含所有固定训练/模型/优化器/正交化超参数
- `src/config/` 负责 YAML 加载，导出 `TRAINING` `MODEL` `OPTIMIZER` 模块级常量
- train.py CLI 仅暴露必须变化的 3 个参数：`--data-path` `--orth` `--seed`，外加 `--benchmark` / `--spectral` 控制诊断模式

## 五种 orth 模式

| 模式 | 含义 |
|------|------|
| `adamw` | 矩阵参数不走正交化，走 Adam |
| `vanilla` | 5× stable 系数 `(2.0, -1.5, 0.5)` |
| `fast` | 5× fast 系数 `(3.4445, -4.7750, 2.0315)` |
| `manual` | fast_steps 步 fast + stable_steps 步 stable（和为 5） |
| `polar_express` | 每步自适应五次多项式 |

参数值定义在 `src/config/config.yaml` → `orthogonalization` 段。

## 目录结构

```
src/
├── paths.py                     # ROOT, RUNS_ROOT, read_jsonl（项目路径工具）
├── config/                      # YAML 配置加载 → TRAINING, MODEL, OPTIMIZER 常量
│   ├── config.yaml
│   ├── loader.py
│   └── __init__.py
├── data/                        # 数据管线（Shard, data_generator）
│   ├── pipeline.py
│   └── __init__.py
├── model/                       # GPT 模型定义
│   ├── gpt.py                   # GPT, RoPE, TransformerBlock, build_model
│   └── __init__.py
├── optim/                       # 优化器（Muon + Adam + 正交化）
│   ├── normuon.py               # NorMuonAndAdam, ParamConfig
│   ├── manager.py               # build_optimizer, step_optimizer, compute_lr
│   ├── orth.py                  # 系数调度, orth_record, orth_norm_factor
│   ├── polar.py                 # make_polar_express
│   └── __init__.py
├── training/                    # 训练循环编排
│   ├── train.py                 # 单次训练入口
│   ├── logger.py                # 日志写入（Logger）
│   ├── metrics.py               # 频谱指标采集
│   ├── utils.py                 # setup_device, default_run_name, resolve_data_path
│   └── __init__.py
├── analysis/                    # 训练后分析
│   ├── summarize_runs.py        # → results/run_summary.csv + orth_summary.csv
│   ├── plot_curves.py           # → results/figures/（7 张 PNG）
│   ├── build_dashboard.py       # → results/dashboard.html
│   ├── ns_coefficients.py       # Neville-Simpson 系数分析
│   └── __init__.py
```

数据下载脚本独立存放：`scripts/download_fineweb.py`

## 架构要点

- 模型为标准 per-layer prenorm Transformer + RoPE，使用标准 `nn.Linear`
- 每个 `nn.Parameter` 有 `.label` 属性，优化器靠它做 Adam/NorMuon 分发
- `orth_mode != "adamw"` 时：矩阵参数走 Muon，非矩阵参数（embedding、LayerNorm 等）走 Adam
- 无 checkpoint 保存
- LM head 与 embedding 权重绑定（transpose 共享）
- 训练为纯单卡，无 sharding/all-reduce

## 分析流水线

训练完成后依次运行：
1. `python -m src.analysis.summarize_runs` → `results/run_summary.csv` + `results/orth_summary.csv`
2. `python -m src.analysis.plot_curves` → `results/figures/`（7 张 PNG）
3. `python -m src.analysis.build_dashboard` → `results/dashboard.html`

## 项目约定

- 无单元测试、无 linter、无 typechecker 配置——验证靠训练实验
- `5090_results/` 是历史存档，**禁止修改**
- 数据格式：FineWeb-10B 预分词 BOS 对齐 shard（`fineweb_train_*.bin` / `fineweb_val_*.bin`）
- 每轮训练输出到 `runs/<name>/`（`config.json` + `metrics.jsonl`）
- 无 W&B，所有日志为本地 JSONL
- 所有运行命令使用 `python -m <module>` 格式

## 项目状态追踪

每次有意义的改动后，更新 `docs/status.md`，保持总行数不超过 100 行。

## 实验原则

- 控制变量优先于超参数搜索
- 训练栈固定：所有超参数见 `src/config/config.yaml`
- 不做无假设驱动的暴力网格搜索
