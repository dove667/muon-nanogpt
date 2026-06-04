# 项目状态

## 当前阶段：实验重新设计

已将原 legacy 9 阶段 ~100 轮网格搜索重写为 5 组 × 3 种子 = 15 轮控制变量实验。

## 实验计划（docs/experiments.md）

- **5 个实验对象**：AdamW 基线、Vanilla（全慢）、Manual（快慢混合）、Fast（全快）、Polar Express
- **T=5 统一**，批量/seq_len/LR 调度/window 全部固定
- **LR 调度**：warmup + cosine decay（待实现，当前代码仍为三阶段 + cooldown）
- **AdamW 模式**：待实现（需改 orthogonalization.py / polar.py / manager.py）
- **三阶段批次调度**：待移除，改为固定

## 命名变更

| 旧 | 新 | 含义 |
|----|-----|------|
| vanilla | Fast | 5× 快速系数 |
| — | Vanilla（新增） | 5× 稳定系数 |
| manual（含 Phase‑2） | Manual | 快慢混合 |
| polar_express | Polar Express | 动态系数 |

## 待办

- [ ] 简化训练调度：移除三阶段递增，改为固定 batch/seq_len + warmup+cosine decay
- [ ] 实现 AdamW orth 模式（~30 行改动）
- [ ] 实现 Vanilla orth 模式（5× 稳定系数）
- [ ] 重命名 old "vanilla" → "fast"
- [ ] 决定 YaRN 是否移除（初步结论：暂不动，固定参数即可）
