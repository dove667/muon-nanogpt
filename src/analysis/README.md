# `src/analysis`

训练后分析与结果导出脚本。

## 目录职责

- 读取 `runs/` 下的训练日志，汇总成便于比较的表格或图像。
- 不参与训练主流程，只处理已经落盘的 `config.json`、`metrics.jsonl`、`spectral_details.jsonl`。

## 文件说明

- `__init__.py`：分析模块导出占位。
- `summarize_runs.py`：递归扫描 `runs/`，输出 `results/summary.csv`，按 `orthogonalizer_type × mode` 汇总 train / benchmark / spectral 关键指标。
- `plot_curves.py`：根据 `metrics.jsonl` 生成训练损失曲线、benchmark wall-clock 柱状图和 final val loss 柱状图；同时读取 `spectral_details.jsonl` 生成谱分析图（`g_post` 半正交误差随 token 的变化、`buffer_post/g_pre/g_post` 对比、attention/MLP 分解）。
- `export_spectral_details.py`：把每次谱分析采样得到的 `spectral_details.jsonl` 合并导出为 `results/spectral_details.csv`，便于离线做分层或逐矩阵分析。

## 当前谱分析口径

- spectral summary 来自 `metrics.jsonl` 中的 `spec/*` 聚合字段。
- spectral detail 来自 `spectral_details.jsonl` 中的逐矩阵字段。
- 当前主要关注的对象是 `buffer_post`、`g_pre`、`g_post` 三类矩阵，以及它们的 `semi_orth_error`、奇异值统计、stable rank 和 entropy。
