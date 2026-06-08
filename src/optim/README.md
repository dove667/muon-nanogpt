# `src/optim`

Muon / Adam 优化器实现、正交化调度与相关数学工具。

## 目录职责

- 构建参数到优化器分支的映射。
- 实现 NorMuon + Adam 的联合优化器。
- 提供 Newton-Schulz / Polar Express 正交化系数调度与记录逻辑。

## 文件说明

- `__init__.py`：统一导出优化器构建与正交化工具函数。
- `manager.py`：构建参数表、创建优化器、更新学习率并驱动每步 `step_optimizer`。
- `normuon.py`：`NorMuonAndAdam` 主实现，包含 Adam 更新、Muon 更新、动量状态和方差缩放。
- `orth.py`：系数调度、运行记录和正交化核心逻辑（Newton-Schulz / polar 表达式迭代）。

## 当前谱分析相关约定

- 真实进入正交化的是 `g`，不是单独的 `momentum_buffer`。
- 为了支持 spectral 模式，`normuon.py` 现在可以在真实更新时按需抓取三类对象：
  - `buffer_post`：更新后的动量缓存
  - `g_pre`：进入正交化前的 Nesterov 混合矩阵
  - `g_post`：正交化后的矩阵
