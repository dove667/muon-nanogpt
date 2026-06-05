# `src/model`

GPT 模型定义与构建入口。

## 目录职责

- 提供标准的单卡 GPT 训练模型。
- 封装 RoPE、Transformer block、最终模型组装。
- 给优化器准备参数标签等结构化信息。

## 文件说明

- `__init__.py`：导出 `GPT` 与 `build_model`。
- `gpt.py`：
  - 定义 RoPE、attention、MLP、TransformerBlock 和 `GPT` 主体。
  - `build_model` 负责按 `src/config/config.yaml` 中的模型配置实例化模型。
  - 为每个 `nn.Parameter` 挂上 `.label`，供优化器按参数类型分发到 Adam 或 NorMuon。

## 架构口径

- 标准 per-layer prenorm Transformer。
- 使用 `nn.Linear`，embedding 与 LM head 权重绑定。
- 不包含额外竞速 trick 或分布式逻辑。
