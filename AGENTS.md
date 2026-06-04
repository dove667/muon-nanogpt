# AGENTS.md

## 运行环境

- Conda 环境 `muon`（由 `environment.yml` 定义），或回退到 `conda run -n AI`
- Python 依赖用 `uv` 锁定，**禁止执行 `uv sync`**（会删除环境中的 conda 底层包）
- 正确安装流程见 README.md

## 训练入口

- 单次训练：`python src/training/train.py --orth <mode> --name <name> --data-path /data/fineweb10B ...`
- 完整实验（5 配置 × 3 seeds = 15 runs）：`python -m src.experiment_plan --data-path /data/fineweb10B [--skip-completed-runs]`
- 162M 模型 4090 单卡完全够用，无分布式逻辑。如需多卡，手动 `torchrun`
- 并行跑多个独立实验用 `CUDA_VISIBLE_DEVICES=0 python ...` 比 DDP 更推荐

## 五种 orth 模式

| 模式 | 含义 |
|------|------|
| `adamw` | 矩阵参数不走正交化，走 Adam |
| `vanilla` | 5× stable 系数 `(2.0, -1.5, 0.5)` |
| `fast` | 5× fast 系数 `(3.4445, -4.7750, 2.0315)` |
| `manual` | `--fast-steps` 步 fast + `--stable-steps` 步 stable（和为 5） |
| `polar_express` | 每步自适应五次多项式 |

## 架构要点

- 模型权重存储在三个扁平参数 bank（`qk_bank`, `vo_bank`, `mlp_bank`）而非 `nn.Linear`，forward 时切片使用
- 每个 `nn.Parameter` 有 `.label` 属性，优化器靠它做 Adam/NorMuon 分发
- `orth_mode != "adamw"` 时：Adam 参数奇数步更新，NorMuon 参数每步更新
- `train.py` 使用脚本级相对导入（`import polar`），**不要**改为包导入
- 无 checkpoint 保存（`Hyperparameters.save_checkpoint = False` 硬编码）
- LM head 与 embedding 权重绑定（transpose 共享）
- 训练为纯单卡，`NorMuonAndAdam` 中所有 comms 均为 `none`，无 sharding/all-reduce

## 分析流水线

训练完成后依次运行：
1. `python -m src.analysis.summarize_runs` → `results/run_summary.csv` + `results/orth_summary.csv`
2. `python -m src.analysis.plot_curves` → `results/figures/`（7 张 PNG）
3. `python -m src.analysis.build_dashboard` → `results/dashboard.html`

## 项目约定

- 无单元测试、无 linter、无 typechecker 配置——验证靠训练实验
- `5090_results/` 是历史存档，**禁止修改**
- `--data-path` 是必传参数，无默认值
- 数据格式：FineWeb-10B 预分词 BOS 对齐 shard（`fineweb_train_*.bin` / `fineweb_val_*.bin`）
- W&B 日志默认开启，用 `--wandb off` 关闭

## 项目状态追踪

每次有意义的改动后，更新 `docs/status.md`，保持总行数不超过 100 行。

## 实验原则

- 控制变量优先于超参数搜索
- 训练栈固定：batch=`8*2048*8`、seq_len=2048、grad_accum=16、window=(3,7)、warmup+cosine LR
- 不做无假设驱动的暴力网格搜索
