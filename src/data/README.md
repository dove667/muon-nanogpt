# `src/data`

FineWeb token shard 的读取与训练数据流生成。

## 目录职责

- 读取 `fineweb_train_*.bin` / `fineweb_val_*.bin` 分片。
- 按固定 `seq_len` 和 `tokens_per_step` 生成训练 / 验证所需的 `(inputs, targets)`。
- 保持数据管线简单、稳定，不引入额外实验变量。

## 文件说明

- `__init__.py`：导出数据接口。
- `pipeline.py`：
  - `load_data_shard` 负责读取单个 token shard。
  - `data_generator` 负责跨 shard 流式取样，切成固定长度 block，并直接搬到 CUDA。

## 实现约定

- 使用朴素连续 block 采样。
- 每个样本按 `seq_len + 1` 读取后再拆成 `inputs` / `targets`。
- 不做 BOS packing、变长序列或分布式数据切分。
