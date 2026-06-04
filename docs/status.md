# 项目状态

## 当前阶段：fixed T=5 对照实验，训练栈已改成干净可解释版本

- 实验编排固定为 5 配置 × 3 seeds = 15 runs
- 训练入口保持两层：
  - `python -m src.training.train`：单次训练
  - `python -m src.experiment_plan`：批量跑 15 个 run
- 所有固定超参数集中在 `src/config/config.yaml`
- 目录已按职责重新组织（model / data / config / optim / training / analysis 各一层）

## 已完成

- 模型已从竞速版参数银行重构为标准 per-layer prenorm Transformer：
  - 标准 `nn.Linear` attention / MLP
  - 标准 RoPE
  - tied embedding + lm_head
  - 标准 cross-entropy
- 删除会干扰研究解释的竞速 trick：
  - 删除 `qk_bank / vo_bank / mlp_bank`
  - 删除 softcapped logits
  - 删除 ReLU-squared MLP
  - 删除 QK norm
  - 删除词表 padding 到 128 倍数
  - 删除交错 Adam 更新，Adam 现在每步更新
- 数据管线已改成朴素固定长度 block 采样：
  - 输入来自 `fineweb_train_*.bin`
  - 验证来自 `fineweb_val_*.bin`
  - 不再做 BOS 对齐 packing 或变长 attention
- 优化器结构已简化：
  - Muon 只作用于标准 Transformer 的矩阵参数
  - token embedding / tied lm_head / LayerNorm 等非矩阵参数走 Adam
  - `adamw` 模式下全部参数都走 Adam
- `TrainingManager` 已移除：
  - 优化器构造改成 `build_optimizer(...)`
  - 每步 LR / momentum 更新改成 `step_optimizer(...)`
  - train tokens、device 等状态回到训练循环中的显式局部变量
- orth 调度层已进一步去中间配置化：
  - `OrthogonalizerConfig` 和后续 `orth_config` 中间对象都已移除
  - `train.py` 直接基于 YAML 里的 `orth_cfg` 生成局部派生变量
  - 保留纯函数负责生成系数、norm factor 和日志记录
- `ForwardScheduleConfig` 也已移除：
  - `model.forward(...)` 回到最直接的 `(inputs, targets)` 签名
- 训练函数传参已统一收口：
  - 运行时对象和实验变量继续显式传递
  - `eval/log/spectral/train budget` 等固定栈参数统一直接从 `config.yaml` 读取
- `manual` 默认已校正为实验设计要求的 `3 fast + 2 stable`
- 批量实验计划已同步修正 run 命名，断点续跑仍然有效
- README / runbook / experiments / dashboard 文案已同步到标准 Transformer + 单卡 + 固定长度 block 采样口径
- 文档中的运行命令统一使用 `python`，不再写 `conda run -n AI`

## 为什么这样改

- 研究目标是比较 Muon 的 NS 系数调度，而不是比较竞速工程技巧
- 标准 Transformer + 固定长度数据 + 每步更新更容易解释收敛差异
- orth 调度层改成纯函数后，训练主路径更短，更容易看清“实验变量到底是什么”
- optimizer 构造和训练状态拆开后，训练循环更像直接可读的实验脚本，而不是小框架
- 固定配置不再层层下传后，函数签名更能反映“哪些东西真的会变”
- 删除分布式和竞速残留后，单卡 4090 上复现实验更直接

## 固定训练栈

- train token budget = 100M
- eval every = 2M tokens
- eval tokens = 524,288
- batch_tokens = 131,072
- seq_len = 2048
- grad_accum_steps = 16
- LR = 10% warmup + cosine decay 到峰值 10%

## 待确认

- 若后续需要进一步减小混杂，可以再评估是否把 `grad_accum_steps=16` 下调
