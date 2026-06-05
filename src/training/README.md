# `src/training`

训练主流程、日志与在线诊断。

## 目录职责

- 提供单次训练入口。
- 编排训练、验证、benchmark 和 spectral 三种运行模式。
- 将训练过程中的关键指标写入本地日志。

## 文件说明

- `__init__.py`：训练模块导出占位。
- `train.py`：训练入口与主循环，负责模式切换、验证调度和谱分析触发。
- `logger.py`：把 `config.json`、`metrics.jsonl` 和 `spectral_details.jsonl` 写入 run 目录。
- `metrics.py`：训练期在线指标工具，包含当前梯度范数、SVD 摘要、谱采样与聚合逻辑。
- `utils.py`：设备初始化、固定 seed、默认 run 命名、数据路径校验和主学习率读取。

## 当前谱分析口径

- 只在 `--spectral` 模式下启用。
- 每次触发时，训练循环会在真实 Muon 更新发生的那一步抓取 `buffer_post` / `g_pre` / `g_post`。
- summary 写入 `metrics.jsonl`，detail 写入 `spectral_details.jsonl`。
- 采样策略是“先收集所有候选，再全局均匀选点”，避免只覆盖前几层矩阵。
