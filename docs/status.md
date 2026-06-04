# 项目状态

## 当前阶段：fixed T=5 对照实验已落地到代码

- 实验编排已从 legacy 网格搜索切换为 `5 配置 × 3 seeds = 15 runs`
- 默认计划入口改为 `fixed_t5`，统一使用：
  - train token budget = 100M
  - eval every = 10M tokens
  - eval tokens = 2,097,152

## 已完成

- 新增并打通 5 种 orth 模式：
  - `adamw`
  - `vanilla` = 5× `STABLE_COEFF`
  - `manual` = 3 fast + 2 stable
  - `fast` = 5× `FAST_COEFF`
  - `polar_express` = `T=5`, `lower_bound=1e-3`
- 旧代码里名为 `vanilla` 的 fast 语义已更正为独立 `fast` 模式
- `adamw` 模式下矩阵参数改走 Adam 分支，不再走 Muon 正交化
- 训练调度已改为固定栈：
  - batch = `16 * 2048 * 8`
  - seq_len = `2048`
  - window = `(3, 7)`
  - LR = 10% warmup + cosine decay 到峰值的 10%
  - Muon momentum 固定 `0.95`
- 训练日志已改为记录 fixed stack 的 `base_lr`、`seq_len` 与主矩阵组实际学习率
- 项目结构已化简：
  - 删除 `src/plan/` 多阶段计划层
  - `RunSpec` 与 15-run fixed plan 直接收敛到 `src/experiment_plan.py`
  - `docs/runbook.md` 改为单一实验入口说明

## 为什么这样改

- 与 `docs/experiments.md` 保持一致，避免“实验计划已重设计，但训练代码仍按旧三阶段栈运行”
- 保证唯一自变量是 NS 系数策略，避免 batch/seq/window/LR 联动造成混杂
- fixed 计划已经没有真实的“阶段选择”含义，保留额外目录和别名只会增加理解成本

## 待确认

- 当前固定栈保留了现有第二阶段的窗口配置 `(3, 7)`，MTP 权重固定为 `[1.0, 0.0]`
- YaRN 逻辑未删除，只是不再随阶段切换
- 若后续需要复现实验，可直接从 `python -m src.experiment_plan` 启动
- `src/analysis/` 仍保留部分面向旧多阶段结果的脚本；当前先只化简训练/实验入口
