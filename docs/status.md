# 项目状态

## 当前阶段：实验完成，进入报告整理

- 5 配置 × 固定 seed 1 次 = 5 runs（不再考虑种子随机性）
- 训练入口：`python -m src.training.train --orth <mode> --data-path /data`
- 三种模式分开跑：纯训练（默认）/ Benchmark（`--benchmark`）/ Spectral（`--spectral`）
- 默认模式零 `torch.cuda.synchronize()` 开销
- 所有固定超参数在 `src/config/config.yaml`

## 已完成

- 模型：标准 per-layer prenorm Transformer + RoPE，`nn.Linear`，tied embedding
- 数据：朴素连续 block 采样（`fineweb_train_*.bin` / `fineweb_val_*.bin`）
- 优化器：Muon 走 NS 正交化 → 矩阵参数；Adam → 非矩阵参数和 embedding
- 架构：`build_optimizer(...)` 构造 / `step_optimizer(...)` 驱动 / 训练状态为局部变量
- orth 调度：纯函数 `build_coeff_schedule / orth_record / orth_norm_factor`
- 目录按职责重组：`config/` `data/` `model/` `optim/` `training/` `analysis/`
- 命名统一：`tokens_per_step` / `tokens_per_microbatch` / `sequences_per_microbatch`
- 删除了 `experiment_plan.py`（编排层）、data_generator 动态重配置分支
- 训练循环拆分为三种独立模式，各自无交叉开销
- 去除所有竞速 trick（参数银行、softcap logits、ReLU² MLP、QK norm 等）
- 训练入口移除 CLI seed，固定 seed 写死在代码内，run 名改为时间戳
- 数据读取改为按 shard 流式加载，避免一次性把全部数据 pin 到内存
- 修复 data_generator 的 microbatch 取样长度错误，按 `(seq_len + 1)` 成对切分 inputs/targets
- 修复 spectral 模式参数名错误、AdamW 模式学习率日志、final val loss 取值错误
- benchmark 只记录端到端总时间，分析图同步改为总 wall-clock 柱状图
- 固定栈 warmup 改为前 2%，Adam 路径改为标准 decoupled AdamW，并对齐常见小 GPT 基线超参数
- run 名时间戳简化为 `MMDD_HHMM`，并将 `min_lr_frac` 提高到 0.2，减轻 70M+ token 后的过早变平
- analysis 脚本移除多 seed / 多 run 平均与宽松兼容；按 `train` / `benchmark` / `spectral` 严格区分，同一 `orth` 同一模式重复即报错
- 训练后汇总输出合并为单个 `results/summary.csv`，每个 `orth` 一行，减少 `run_summary.csv` / `orth_summary.csv` 的重复
- `summarize_runs.py` 汇总流程收成单层，区分 `orthogonalizer_type` 与 `orth_error` 语义，减少 `orth` 歧义命名
- train CLI 将 `--benchmark` 和 `--spectral` 设为互斥参数，禁止单个 run 混合两种诊断模式
- runbook / README / AGENTS / experiments 文档同步为三模式组织：推荐按 `runs/train|benchmark|spectral/` 分目录，分析输出统一为 `results/summary.csv` + `results/figures/`
- 文档进一步明确：`runs/` 是当前分析工作区，历史实验需移到 `runs/` 外归档，避免被递归分析误读
- 文档同步更新：README / AGENTS.md / runbook / experiments / status
- spectral 采样改为在真实 Muon 更新时抓取 `buffer_post` / `g_pre` / `g_post`，修复过去把 momentum buffer 当作 update 输入的语义错误
- spectral 记录新增 `spectral_details.jsonl` 与 `python -m src.analysis.export_spectral_details`，并将 `orth_error` 重命名为 `semi_orth_error` 以明确其 short-side Gram 含义
- 为 `src/analysis` `src/config` `src/data` `src/model` `src/optim` `src/training` 新增目录 README，并同步更新 runbook / experiments / README 的谱分析文档
- 新增 `docs/report.md`，整理当前 train 与 benchmark 的实验分析：Muon 四种 schedule 的 wall-clock 基本重合，而主要差异体现在 val loss
- `docs/report.md` 补充 AdamW 更快的双重解释：既有 Muon 正交化的额外算法成本，也有系统实现成熟度差异，其中系统层因素被列为更主要的推断
- 修复 spectral 空样本 run 的记录口径：即使某次采样没有 Muon 候选矩阵，也会写入 `spec/sample_count=0`，避免后处理把 `--spectral` run 误判成普通 train
- 谱分析可视化功能并入 `plot_curves.py`，通过 `--spectral` 数据自动触发，不再使用独立的 `plot_spectral.py`
- 全部 train / benchmark / spectral 实验完成，当前文档结论已同步到 `docs/report.md`
- 新增 `docs/final_report/final_report.tex` 与 `docs/final_report/references.bib`，按课程 PDF 的四段结构起草英文 final report，并直接引用 `results/` 中的训练、benchmark 与谱分析图表
- 根据人工反馈继续精修 `docs/final_report/final_report.tex`：补强 Muon 数学动机与 Newton-Schulz 奇异值视角，重写 Technical Approach 为实验设计逻辑，并显著扩写 Main Results 的训练/几何/成本分析
- 重组 final report 的 Main Results：改为结论驱动的段落组织，每段段首明确主结论，合并同类发现并删除冗余解释
- 补回 final report 的图像组织：两张时间序列折线图改回并排双图，并恢复正文对 `curve-panels` 与 `spectral-panels` 的显式引用
- 调整 final report 表格布局：将训练结果的两张表与谱分析的两张表分别合并为并排表格组，以改善单列表格的页面占用
- 继续压缩 Main Results：合并重复性质的训练与谱分析结论，并删除非结果性的图表提示句
- 拆分谱分析中的复合结论段：将“早出现且 attention 更敏感”改为两个单一结论段
- 补强谱分析末尾两段的解释力度，在保持“一段一个发现”的前提下增加结果含义说明
- 重写 final report 的 Conclusion：第一段改为研究问题导向的总结，第二段合并 limitation 与 future directions，取消列表格式
- 重写 Main Results 段首加粗结论句，改为更短、更具体的研究结论表述，去掉抽象的提示性写法

## 固定训练栈

- train token budget = 100M
- eval every = 2M tokens, eval tokens = 524,288
- tokens_per_step = 131,072
- seq_len = 2048
- grad_accum_steps = 16
- LR = 2% warmup + cosine decay 到峰值 20%
