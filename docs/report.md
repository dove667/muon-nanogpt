# Muon Fixed T=5 实验分析

## 1. 设置与数据

本报告分析当前已经完成的两组实验：

- `train`：本地 `archives/train/`
- `benchmark`：服务器上最新完成的 `archives/benchmark/`

对比对象共 5 个：

- `adamw`
- `vanilla`
- `manual_f3_s2`
- `fast`
- `polar_express`

其中四个 Muon 变体共享同一个基本计算框架：

- 矩阵参数走 Muon，非矩阵参数走 Adam
- Newton-Schulz / quintic 迭代步数固定为 `T=5`
- 差别只在每一步使用的系数表

因此，本轮实验的核心问题可以直接表述为：

1. 不同系数调度是否会影响最终优化结果？
2. 不同系数调度是否会影响端到端 wall-clock？

## 2. Train 结果

### 2.1 最终指标

| orth | train run | final val loss | best val loss | peak mem (MiB) |
|---|---|---:|---:|---:|
| adamw | `adamw_0605_1447` | 5.518270 | 5.518205 | 8380 |
| vanilla | `vanilla_0605_1515` | 4.532286 | 4.532286 | 8074 |
| manual | `manual_f3_s2_0605_1516` | 4.314679 | 4.314679 | 8074 |
| fast | `fast_0605_1448` | 4.280688 | 4.280688 | 8074 |
| polar_express | `polar_express_l1e-3_0605_1516` | 4.289600 | 4.289600 | 8074 |

### 2.2 主要现象

训练结果非常清楚地分成两层。

第一层是 `adamw` 与 Muon 家族的差别。`adamw` 的 final val loss 为 `5.5183`，而四个 Muon 变体全部落在 `4.28 ~ 4.53` 区间，差距非常明显。这说明在当前 100M token 训练预算和固定模型规模下，Muon 路径对优化质量带来了决定性的改善。

第二层是 Muon 内部的排序。四个 Muon 变体中，`fast` 的 final val loss 最低，为 `4.2807`；`polar_express` 以 `4.2896` 紧随其后；`manual_f3_s2` 为 `4.3147`；`vanilla` 最差，为 `4.5323`。因此，在固定 `T=5` 的条件下，系数调度本身确实影响最终优化表现，而且这种影响已经足以在最终验证损失上形成清晰排序。

### 2.3 结果解释

这组结果说明，本项目里真正重要的并不是“是否做 5 次迭代”这么粗粒度的问题，而是“这 5 次迭代用什么系数表”。

一个自然的解释是：不同 schedule 对奇异值的压缩和拉伸方式不同，因此会产生不同的更新几何。`fast` 与 `polar_express` 的表现更好，说明它们在当前训练栈下更接近有利于优化的半正交更新；`manual_f3_s2` 作为 fast-to-stable 的混合调度，也保留了大部分优势；`vanilla` 则明显更保守，因此收敛质量落后。

从数值上看，`fast` 与 `polar_express` 的最终差距只有约 `0.009`，这意味着二者在当前设置下处于同一梯队。相较之下，`vanilla` 与前三者之间的差距则足够大，已经不是“几乎一样”的关系。

## 3. Benchmark 结果

### 3.1 最终指标

| orth | benchmark run | wall clock (s) | final val loss | best val loss | peak mem (MiB) |
|---|---|---:|---:|---:|---:|
| adamw | `adamw_0605_1628` | 1063.990881 | 5.457609 | 5.453564 | 8380 |
| vanilla | `vanilla_0605_1627` | 1133.537138 | 4.530796 | 4.530796 | 8074 |
| manual | `manual_f3_s2_0605_1738` | 1132.338386 | 4.315478 | 4.315478 | 8074 |
| fast | `fast_0605_1626` | 1132.713862 | 4.283905 | 4.283905 | 8074 |
| polar_express | `polar_express_l1e-3_0605_1831` | 1132.577927 | 4.288644 | 4.288644 | 8074 |

### 3.2 主要现象

benchmark 结果同样分成两层。

第一层依然是 `adamw` 与 Muon 家族的差别。`adamw` 的 wall-clock 为 `1063.99s`，而四个 Muon 变体全部落在 `1132.3s ~ 1133.5s` 区间，整体快约 `68.8s`，相对差距约 `6.5%`。这说明 Muon 的额外正交化步骤确实带来了可见的计算成本。

第二层是 Muon 内部不同 schedule 的比较。这里的结果非常整齐：`vanilla`、`manual_f3_s2`、`fast`、`polar_express` 四者几乎完全重合，最大差距只有约 `1.20s`，相对 Muon 均值的偏差不超过 `0.07%`。也就是说，在端到端 wall-clock 这个指标上，四种 Muon 调度没有形成实质性区分。

这一点正是本轮 benchmark 最重要的观察结论。

## 4. 为什么 Muon 四种 schedule 的 wall-clock 几乎完全一样

这个现象并不奇怪，反而与当前实现完全一致。

在这份代码里，四种 Muon schedule 的共同结构是：

- 迭代次数同为 `T=5`
- 每轮执行同类矩阵乘法与加法融合算子
- 张量形状相同
- 内存分配模式相同

换句话说，四种 schedule 之间变化的是标量系数，而不是算子种类、矩阵维度或迭代长度。于是从 GPU 执行角度看，它们的 FLOPs、kernel 数量和访存模式几乎完全一致，最终得到几乎相同的 wall-clock 是预期之内的结果。

这也解释了为什么 benchmark 会出现一个很有代表性的结构：

- `adamw` 和 Muon 家族之间能明显拉开
- Muon 家族内部却几乎无法拉开

原因不是 benchmark “失效”，而是 benchmark 恰好测到了真正会影响成本的那一层差别：是否执行 Muon 正交化；而没有测出几乎不存在的那一层差别：同一计算图下仅仅更换系数表。

## 5. 为什么 AdamW 比 Muon 更快

`adamw` 相比四个 Muon 变体快约 `6.5%`。这一差距至少可以从两层解释。

第一层是算法层。Muon 在每次更新时额外执行了固定 `T=5` 的正交化过程，因此即使完全不考虑实现细节，它也天然比 AdamW 多出一段矩阵迭代计算。这部分额外计算是真实存在的，也是 Muon 相比 AdamW 必然要付出的成本来源。

第二层是系统层，而我认为这很可能是更主要的原因。当前仓库中的 Muon 路径是由 [`src/optim/polar.py`](/Users/dove/Desktop/Math/muon-nanogpt/src/optim/polar.py:20) 和 [`src/optim/normuon.py`](/Users/dove/Desktop/Math/muon-nanogpt/src/optim/normuon.py:197) 里手写的多步张量操作拼接而成，本质上是若干 `matmul`、`addmm`、`baddbmm`、norm、buffer 更新和类型转换的组合。相比之下，AdamW 走的是 PyTorch 标准优化器风格的更新路径，底层实现成熟得多，也更可能直接受益于框架内部已经做好的 kernel 优化、foreach 路径或 fused 更新逻辑。

因此，`adamw` 更快不应仅仅理解为“它少做了一些数学操作”，更合理的理解是：

- Muon 的算法本身更重；
- 而且 Muon 当前的实现方式也更接近“手写组合算子”，系统优化程度大概率不如标准 AdamW。

从这个角度看，本轮 benchmark 更像是在比较“当前仓库里的实际训练系统成本”，而不只是抽象算法复杂度。也正因如此，`adamw` 与 Muon 之间的 `6.5%` 差距既包含算法额外计算，也包含实现成熟度上的差异；其中后者很可能占了相当重要的部分。

## 6. Benchmark 是否因为同步而掩盖了差异

当前证据不支持这个解释。

benchmark 模式的计时方式是：

- 开始前做一次 `torch.cuda.synchronize()`
- 结束后再做一次 `torch.cuda.synchronize()`
- 中间按照正常训练流程执行

因此它测的是完整训练 run 的端到端时间，而不是在每一步 optimizer update 上插入额外同步的微基准。它的作用是确保计时边界准确，而不是强行改变各个 schedule 的相对耗时关系。

如果 benchmark 真把内部差异“同步洗平”，那么一个更强的副作用应该是 `adamw` 与 Muon 之间也不容易被区分；但结果恰恰相反，`adamw` 比 Muon 家族稳定快出约 `6.5%`。这说明 benchmark 对“大类成本差异”是有分辨力的，只是 Muon 内部不同系数调度本来就几乎没有额外成本差别。

## 7. 综合分析

把 train 和 benchmark 合并起来看，本轮 fixed `T=5` 实验给出了一个非常清楚的结论：

- 不同 Muon schedule 会影响优化结果；
- 但在当前实现里，几乎不会影响端到端 wall-clock。

这意味着本轮实验真正需要优化的维度，不是“哪一种 schedule 更省时间”，而是“在相同时间成本下，哪一种 schedule 给出更好的 val loss”。

沿着这个标准看，当前结果最有竞争力的是 `fast` 和 `polar_express`：

- 它们给出了最好的验证损失；
- benchmark 时间与其他 Muon 变体几乎完全一致；
- 因而在“效果/时间”联合视角下占优。

`manual_f3_s2` 表现出明显的折中性质：它比 `vanilla` 好很多，但仍略逊于 `fast` 和 `polar_express`。`vanilla` 则在这轮实验里没有体现出优势，因为它既没有更低的 val loss，也没有更低的 wall-clock。

因此，如果只基于当前已完成结果来总结：

1. `adamw` 可以作为基线，但在当前设置下显著落后于 Muon 家族。
2. Muon 家族内部应主要按 val loss 区分，而不是按 wall-clock 区分。
3. `fast` 与 `polar_express` 是当前最值得重点讨论的两个 schedule。

## 8. 结论

本轮 fixed `T=5` 对照实验表明，Muon 的 schedule 选择主要影响优化质量，而不是计算成本。在训练结果上，`fast` 和 `polar_express` 给出了最优的一组验证损失，`manual_f3_s2` 次之，`vanilla` 最弱；在 benchmark 结果上，四种 Muon schedule 的端到端 wall-clock 基本重合，而 `adamw` 因为不执行 Muon 正交化而整体更快。

因此，这组实验支持如下判断：在固定 `T=5` 的实现下，Muon schedule 的研究重点应放在其诱导的更新几何与最终收敛效果，而不必期待不同 schedule 之间出现显著的 wall-clock 分化。与此同时，AdamW 相对 Muon 的速度优势不应简单归因于“少做了几步运算”，更合理的解释是算法额外计算与系统实现成熟度共同作用，其中系统层面的优化差异很可能是主要因素之一。
