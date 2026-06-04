# Muon / Newton-Schulz 正交化调度研究

南方科技大学 (SUSTech) 人工智能数学基础课程项目。

研究 Muon 优化器中 Newton-Schulz 正交化系数调度对 GPT 预训练的影响。核心问题：不同的系数策略（`vanilla`、`manual`、`polar_express`）如何影响验证损失、计算开销和更新矩阵的正交性？

## 系统环境

- 4× RTX 4090 (24GB VRAM)，CUDA 12.1，PyTorch 2.5.1+cu121
- FineWeb-10B 数据集

## 环境搭建

使用 micromamba / conda 管理底层运行时，使用 uv 锁定并安装项目上层 Python 依赖。

- micromamba / conda：负责 Python、PyTorch、CUDA 等二进制 / CUDA 敏感包
- uv：负责锁定并安装项目上层 Python 依赖，但不接管 PyTorch / CUDA 底座

推荐使用 micromamba；如果你习惯 conda，命令可等价替换。

```bash
# 创建环境（推荐 micromamba）
micromamba env create -f environment.yml
micromamba activate muon

# 如果使用 conda，也可以：
# conda env create -f environment.yml
# conda activate muon

# 生成 / 更新 uv.lock
uv lock

# 将 lockfile 导出为 pip 兼容格式，再安装到当前 conda / micromamba 环境
uv export --frozen --no-dev --format requirements.txt --output-file .uv-requirements.txt
uv pip install --python "$CONDA_PREFIX/bin/python" -r .uv-requirements.txt
```

如遇网络问题，可手动指定镜像：

```bash
uv lock --index-url https://pypi.tuna.tsinghua.edu.cn/simple
uv export --frozen --no-dev --format requirements.txt --output-file .uv-requirements.txt
uv pip install --python "$CONDA_PREFIX/bin/python" -r .uv-requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

这样分层后：

- `environment.yml` 负责底座稳定性
- `pyproject.toml` 只描述项目上层依赖
- `uv.lock` 负责锁定上层依赖版本
- `uv pip install` 不会像 `uv sync` 一样删除环境中未声明的底层包

注意：

- 不要对当前 conda / micromamba 环境执行 `uv sync`，因为 `uv sync` 默认是 exact sync，会删除 lockfile 之外的包
- 如果底座包被误删，可直接重新执行 `micromamba env update -f environment.yml`（或对应的 `conda env update -f environment.yml`）恢复

## 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/experiments.md`](docs/experiments.md) | 详细实验计划书 |
| [`docs/runbook.md`](docs/runbook.md) | 运行指南 |
| [`docs/initial_proposal/initial_proposal.pdf`](docs/initial_proposal/initial_proposal.pdf) | 课程项目初期提案 |
