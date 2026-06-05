# `src/config`

项目唯一配置源与配置加载逻辑。

## 目录职责

- 维护固定实验栈的全部超参数。
- 将 YAML 配置加载为训练代码可直接访问的模块级常量。
- 保证 CLI 只暴露少量可变项，其他参数统一由配置文件控制。

## 文件说明

- `__init__.py`：对外导出 `TRAINING`、`MODEL`、`OPTIMIZER` 和 `get_orthogonalization`。
- `config.yaml`：唯一配置文件，定义训练预算、模型结构、优化器参数和正交化参数。
- `loader.py`：负责读取 YAML、构造配置对象并导出模块级常量。

## 当前重点

- spectral 相关开关也在这里统一配置，包括 `spectral_interval_tokens`、`spectral_num_matrices` 和 `spectral_dim_cap`。
- 当前默认 `spectral_num_matrices=12`，用于提升不同层之间的谱采样覆盖率。
