#!/usr/bin/env python
"""Build a Chinese interactive evidence dashboard for the Muon experiments."""


import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

from src.utils import ROOT
WANDB_PROJECT_URL = (
    "https://wandb.ai/yicheng132024-southern-university-of-science-technology/"
    "muon-nanogpt"
)

FINAL_WANDB_RUNS = {
    "final_p2_T9_f3_s6_lr1.0_300M_seed0": "9mr72cth",
    "final_p2_T9_f3_s6_lr1.0_300M_seed1": "rc121dga",
    "final_p2_T9_f3_s6_lr1.0_300M_seed2": "pqs34akn",
    "final_pe_T9_l3e-5_lr1.0_300M_seed0": "wo6nbks3",
    "final_pe_T9_l3e-5_lr1.0_300M_seed1": "194csw0s",
    "final_pe_T9_l3e-5_lr1.0_300M_seed2": "u7clnqnu",
    "final_old_fast5_lr1.0_300M_seed0": "zz5y2o2n",
    "final_old_fast5_lr1.0_300M_seed1": "hgqo98y7",
    "final_old_fast5_lr1.0_300M_seed2": "8gy1q2aw",
}


def esc(text: object) -> str:
    return html.escape("" if text is None else str(text))


def read_rows(summary_csv: Path) -> list[dict[str, str]]:
    if not summary_csv.exists():
        return []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def number(row: dict[str, str], key: str) -> float | None:
    text = (row.get(key) or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: object, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return esc(value)
    return f"{float(value):.{digits}g}"


def fmt_int(value: object) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return ""


def schedule_explain(schedule: str) -> str:
    if not schedule:
        return "未记录 schedule。"
    if schedule == "old_fast5":
        return "vanilla Muon 基线：连续 5 步 fast Newton-Schulz 系数 F。"
    m = re.match(r"p2_T(?P<T>\d+)_f(?P<f>\d+)_s(?P<s>\d+)$", schedule)
    if m:
        return (
            f"Phase2 手工 fast/stable 方案：总 {m['T']} 步，前 {m['f']} 步用 fast 系数 F，"
            f"后 {m['s']} 步用 stable 系数 S。"
        )
    m = re.match(r"pe_T(?P<T>\d+)_l(?P<lb>.+)$", schedule)
    if m:
        return f"Polar Express 方案：做 {m['T']} 步，奇异值 lower bound 设为 {m['lb']}。"
    if schedule.startswith("speedtest"):
        return "测速/校准 run，用来验证吞吐、显存和 pipeline，不作为算法结论。"
    return "实验配置名；具体含义见 run name 和原始 config。"


def run_explain(row: dict[str, str]) -> str:
    group = row.get("group") or ""
    name = row.get("name") or ""
    schedule = row.get("schedule") or ""
    seed = ""
    m_seed = re.search(r"seed(\d+)", name)
    if m_seed:
        seed = f"seed={m_seed.group(1)}；"
    budget = ""
    if "300M" in name:
        budget = "300M token final；"
    elif "100M" in name:
        budget = "100M token main 筛选；"
    elif "30M" in name:
        budget = "30M token grid；"
    role = {
        "calibration": "校准/测速",
        "vanilla_muon": "vanilla Muon 基线",
        "p2_T5_lrgrid": "Phase2 T=5 与 LR 网格",
        "pe_init": "Polar Express 初始 lower-bound 网格",
        "p2_T10": "Phase2 T=10 网格",
        "p2_T78": "Phase2 T=7/8 网格",
        "p2_T69": "Phase2 T=6/9 网格",
        "pe_expand": "Polar Express 扩展搜索",
        "main": "100M top-config 筛选",
        "final": "300M 多 seed 最终对比",
    }.get(group, group or "未知阶段")
    return f"{role}；{budget}{seed}{schedule_explain(schedule)}"


def status_tag(status: str) -> str:
    cls = "pass" if status == "completed" else "warn"
    label = "完成" if status == "completed" else "未完成/运行中"
    return f'<span class="tag {cls}">{label}</span>'


def group_counts(rows: list[dict[str, str]]) -> str:
    counts = Counter((r.get("group") or "unknown", r.get("status") or "unknown") for r in rows)
    order = ["calibration", "vanilla_muon", "p2_T5_lrgrid", "pe_init", "p2_T10", "p2_T78", "p2_T69", "pe_expand", "main", "final"]
    parts = []
    for group in order:
        total = sum(v for (g, _), v in counts.items() if g == group)
        done = counts.get((group, "completed"), 0)
        if total:
            parts.append(f"<li><code>{esc(group)}</code>：{done}/{total} completed</li>")
    return "\n".join(parts)


def final_aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    final = [r for r in rows if r.get("group") == "final" and r.get("status") == "completed"]
    by_schedule: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in final:
        by_schedule[row.get("schedule") or "unknown"].append(row)
    out = []
    for schedule, group_rows in by_schedule.items():
        vals = [v for r in group_rows if (v := number(r, "final_val_loss")) is not None]
        toks = [v for r in group_rows if (v := number(r, "throughput_tokens_per_sec")) is not None]
        err = [v for r in group_rows if (v := number(r, "spec_update_orth_error")) is not None]
        if not vals:
            continue
        out.append({
            "schedule": schedule,
            "n": len(vals),
            "mean_val": mean(vals),
            "std_val": stdev(vals) if len(vals) > 1 else 0.0,
            "min_val": min(vals),
            "max_val": max(vals),
            "tok_s": mean(toks) if toks else None,
            "orth_err": mean(err) if err else None,
        })
    return sorted(out, key=lambda r: float(r["mean_val"]))


def final_aggregate_table(rows: list[dict[str, str]]) -> str:
    agg = final_aggregate(rows)
    if not agg:
        return '<p class="muted">尚未发现 final completed rows。</p>'
    trs = []
    for row in agg:
        trs.append(
            "<tr>"
            f"<td><code>{esc(row['schedule'])}</code><span class=\"hint\">{esc(schedule_explain(str(row['schedule'])))}</span></td>"
            f"<td>{row['n']}</td>"
            f"<td>{fmt(row['mean_val'], 6)}</td>"
            f"<td>{fmt(row['std_val'], 4)}</td>"
            f"<td>{fmt(row['min_val'], 6)} - {fmt(row['max_val'], 6)}</td>"
            f"<td>{fmt(row['tok_s'], 6)}</td>"
            f"<td>{fmt(row['orth_err'], 5)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>配置</th><th>seeds</th><th>平均 final val loss</th><th>std</th>"
        "<th>范围</th><th>throughput tok/s</th><th>update orth error</th></tr></thead><tbody>"
        + "\n".join(trs)
        + "</tbody></table>"
    )


def metric_cards(rows: list[dict[str, str]]) -> str:
    completed = [r for r in rows if r.get("status") == "completed"]
    comparable = [r for r in completed if number(r, "final_val_loss") is not None]
    final = [r for r in comparable if r.get("group") == "final"]
    main = [r for r in comparable if r.get("group") == "main"]
    best_final = min(final, key=lambda r: number(r, "final_val_loss") or 999.0) if final else None
    best_main = min(main, key=lambda r: number(r, "final_val_loss") or 999.0) if main else None
    stage_value = "final 已完成" if final else "未发现 final"
    stage_note = f"{len(completed)}/{len(rows)} 个 run completed，当前没有 active training。"
    cards = [
        ("当前阶段", stage_value, stage_note),
        ("最终候选", "3 configs x 3 seeds", "PE、manual Phase2、vanilla Muon 基线各跑 300M tokens。"),
        ("最低 final loss", fmt(best_final.get("final_val_loss") if best_final else None, 6), best_final.get("name") if best_final else "缺失"),
        ("main 筛选第一", fmt(best_main.get("final_val_loss") if best_main else None, 6), best_main.get("name") if best_main else "缺失"),
    ]
    return "\n".join(
        "<section class=\"metric searchable\" data-kind=\"verdict\">"
        f"<div class=\"metric-label\">{esc(label)}</div>"
        f"<div class=\"metric-value\">{esc(value)}</div>"
        f"<div class=\"metric-note\">{esc(note)}</div>"
        "</section>"
        for label, value, note in cards
    )


def run_table(rows: list[dict[str, str]]) -> str:
    ordered = sorted(
        rows,
        key=lambda r: (
            r.get("group") != "final",
            r.get("group") != "main",
            number(r, "final_val_loss") is None,
            number(r, "final_val_loss") or 999.0,
            r.get("name") or "",
        ),
    )
    trs = []
    for row in ordered:
        final_loss = number(row, "final_val_loss")
        trs.append(
            "<tr class=\"searchable\" data-kind=\"runs\" "
            f"data-group=\"{esc(row.get('group'))}\" data-orth=\"{esc(row.get('orthogonalizer_type'))}\" "
            f"data-status=\"{esc(row.get('status'))}\">"
            f"<td>{esc(row.get('group'))}</td>"
            f"<td><code>{esc(row.get('name'))}</code><span class=\"hint\">{esc(run_explain(row))}</span></td>"
            f"<td><code>{esc(row.get('schedule'))}</code><span class=\"hint\">{esc(schedule_explain(row.get('schedule') or ''))}</span></td>"
            f"<td>{fmt(row.get('lr_mul'), 3)}</td>"
            f"<td>{fmt(final_loss, 6)}</td>"
            f"<td>{fmt(row.get('val_auc_tokens'), 6)}</td>"
            f"<td>{fmt(row.get('throughput_tokens_per_sec'), 6)}</td>"
            f"<td>{status_tag(row.get('status') or '')}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>阶段</th><th>run name 与含义</th><th>schedule 与含义</th>"
        "<th>LR</th><th>final val loss</th><th>AUC/token</th><th>throughput tok/s</th><th>状态</th></tr></thead><tbody>"
        + "\n".join(trs)
        + "</tbody></table>"
    )


def wandb_table(rows: list[dict[str, str]]) -> str:
    final = [r for r in rows if r.get("group") == "final" and r.get("status") == "completed"]
    final.sort(key=lambda r: (r.get("schedule") or "", r.get("name") or ""))
    trs = []
    for row in final:
        name = row.get("name") or ""
        run_id = FINAL_WANDB_RUNS.get(name, "")
        link = f"{WANDB_PROJECT_URL}/runs/{run_id}" if run_id else WANDB_PROJECT_URL
        why = (
            "最终多 seed 证据；用于比较 PE、manual Phase2 和 vanilla Muon 的 final loss、速度与几何指标。"
        )
        trs.append(
            "<tr class=\"searchable\" data-kind=\"wandb\">"
            f"<td><code>{esc(name)}</code><span class=\"hint\">{esc(run_explain(row))}</span></td>"
            f"<td><code>{esc(run_id or 'project')}</code></td>"
            f"<td><a href=\"{esc(link)}\">打开 W&B</a></td>"
            f"<td>{fmt(row.get('final_val_loss'), 6)}</td>"
            f"<td>{esc(why)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>final run</th><th>W&B run id</th><th>链接</th><th>final val loss</th><th>为什么重要</th></tr></thead><tbody>"
        + "\n".join(trs)
        + "</tbody></table>"
    )


def figure_kind(name: str) -> str:
    if "val_loss" in name or "comparison_best" in name:
        return "curves"
    if "heatmap" in name:
        return "metrics"
    if "pareto" in name:
        return "metrics"
    if "spec" in name or "orth" in name:
        return "curves"
    if "throughput_tokens_per_sec" in name:
        return "metrics"
    return "curves"


def figure_title(stem: str) -> str:
    text = stem.replace("_", " ")
    replacements = {
        "val loss vs tokens": "validation loss 随训练 token 变化",
        "val loss vs wall time": "validation loss 随实际时间变化",
        "train throughput tokens per sec": "训练吞吐 tok/s",
        "train tokens per sec": "训练吞吐 tok/s",
        "orthogonalizer time ms": "orthogonalizer 每步耗时",
        "spec update orth error": "update 正交误差",
        "heatmap final val loss": "Phase2 final loss 热力图",
        "heatmap val auc tokens": "Phase2 token AUC 热力图",
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def figure_caption(name: str) -> str:
    if "val_loss_vs_tokens" in name:
        return (
            "这张图回答同 token 预算下谁学得更快。横轴是已训练 token，纵轴是 validation loss，越低越好。"
            "每种颜色代表一个 run 或 schedule；当前结论看谁在相同 token 下更低。"
            "重要性：它直接对应 token efficiency，是判断 abc/schedule 是否更省数据的主证据；它不能单独证明 wall-clock 更划算。"
        )
    if "val_loss_vs_wall_time" in name:
        return (
            "这张图回答同实际时间下谁下降更快。横轴是 wall-clock 时间，纵轴是 validation loss，越低越好。"
            "每种颜色代表一个 run 或 schedule；当前结论看谁在相同时间内更低。"
            "重要性：它直接对应 wall-clock efficiency，能检验更强正交化是否被额外计算成本抵消。"
        )
    if "throughput_tokens_per_sec" in name:
        return (
            "这张图回答训练吞吐是否稳定。横轴是 run 或训练进度，纵轴是 tokens/sec，越高表示单位时间处理 token 更多。"
            "不同颜色代表不同 run 或 schedule；当前结论看速度是否稳定且是否有明显掉速。"
            "重要性：它解释同 wall-clock loss 差异的成本侧来源；它不能单独说明模型质量，需要和 validation loss 一起看。"
        )
    if "spec_update_orth_error" in name:
        return (
            "这张图回答 update 矩阵有多接近正交。横轴是 run 或训练进度，纵轴是正交误差，数值越低表示 update geometry 更接近目标。"
            "不同颜色代表不同 run 或 schedule；当前结论看几何约束是否更干净。"
            "重要性：它对应第二个研究问题，即正交化程度如何改变参数更新；它不能直接等同于 validation loss。"
        )
    if "heatmap" in name:
        return (
            "这张图回答 Phase2 不同 T 和 fast-step 数的趋势。横轴是 fast steps，纵轴是 total steps，颜色表示图题中的指标。"
            "对 loss/AUC 来说颜色对应数值越低越好。重要性：它用于解释为什么选择少数 manual schedule 进入 main/final。"
        )
    if "pareto" in name:
        return (
            "这张图回答质量和成本之间的折中。横轴通常是速度或耗时，纵轴是 final validation loss。"
            "颜色或点代表不同 run/config；靠近低 loss 且低成本/高吞吐的位置更有吸引力。"
            "重要性：它把 loss 收益和计算代价放在同一张图里，避免只按 final loss 排名。"
        )
    return (
        "这张图提供对应阶段的辅助指标。读图时看横轴、纵轴和图例；不同颜色代表不同 run 或 config。"
        "重要性：它保留分析用证据，但单张图不能替代 final 多 seed 对比。"
    )


def phase_blocks(rows: list[dict[str, str]], figures_dir: Path, out_path: Path) -> str:
    phases = [
        ("calibration", "校准配置", "确认训练栈、内存占用、吞吐、eval/W&B 能跑通。", "稳定测速完成，后续 sweep 固定同一训练栈。", True),
        ("vanilla_muon", "vanilla Muon 基线", "建立 old_fast5 在默认 LR 和 sanity LR 下的参考曲线。", "得到 baseline loss、速度和几何指标。", False),
        ("p2_T5_lrgrid", "Phase2 T=5 + LR grid", "测试 5 步 fast/stable split 是否优于 pure fast。", "完成 6 个 split x 3 个 LR。", False),
        ("pe_init", "Polar Express 初始网格", "测试 PE lower bound 1e-2、1e-3、1e-4 与 LR 的耦合。", "完成 3 个 lower bound x 3 个 LR。", False),
        ("p2_T10", "Phase2 T=10", "测试更长 manual schedule 是否带来 token-efficiency 或 wall-clock 收益。", "完成 T=10 的所有 fast/stable split。", False),
        ("p2_T78", "Phase2 T=7/8", "填补 5 到 10 之间的中间迭代预算。", "完成 T=7 与 T=8 全 split。", False),
        ("p2_T69", "Phase2 T=6/9", "补完整个 T=5..10 manual sweep。", "完成 T=6 与 T=9 全 split。", False),
        ("pe_expand", "Polar Express 扩展", "先扩 lower bound，再扩 PE iteration count。", "选出 pe_T9_l3e-5 进入 main/final。", False),
        ("main", "100M main 筛选", "从筛选阶段选 top configs，跑 100M token 比较。", "选出 best PE、best manual 和 vanilla baseline。", True),
        ("final", "300M final 多 seed", "对 3 个最终配置跑 seeds 0/1/2，形成主结论。", "9 个 final runs 全部完成。", True),
    ]
    cards = []
    for group, title, goal, exit_condition, opened in phases:
        group_rows = [r for r in rows if r.get("group") == group]
        best = min(
            [r for r in group_rows if number(r, "final_val_loss") is not None],
            key=lambda r: number(r, "final_val_loss") or 999.0,
            default=None,
        )
        status = "完成" if group_rows and all(r.get("status") == "completed" for r in group_rows) else "缺失或未完成"
        figures = []
        for suffix in ["val_loss_vs_tokens", "val_loss_vs_wall_time", "train_throughput_tokens_per_sec", "spec_update_orth_error"]:
            path = figures_dir / f"{group}_{suffix}.png"
            if path.exists():
                try:
                    rel = path.relative_to(out_path.parent).as_posix()
                except ValueError:
                    rel = path.as_posix()
                figures.append(
                    f"<figure><img src=\"{esc(rel)}\" loading=\"lazy\" alt=\"{esc(path.stem)}\">"
                    f"<figcaption>{esc(figure_caption(path.name))}</figcaption></figure>"
                )
        best_text = (
            f"当前 best row 是 <code>{esc(best.get('name'))}</code>，"
            f"final val loss = {fmt(best.get('final_val_loss'), 6)}。"
            f"<span class=\"hint\">{esc(run_explain(best))}</span>"
            if best else "这个阶段没有可比较的 completed val row。"
        )
        cards.append(
            f"""
            <details class="phase searchable" data-kind="phases" {"open" if opened else ""}>
              <summary><span>{esc(title)}</span><strong>{esc(status)}</strong></summary>
              <div class="phase-body">
                <p><strong>目标：</strong>{esc(goal)}</p>
                <p><strong>退出条件：</strong>{esc(exit_condition)}</p>
                <p><strong>证据解释：</strong>{best_text}</p>
                <div class="phase-figures">{''.join(figures) or '<p class="muted">这个阶段没有单独曲线图。</p>'}</div>
              </div>
            </details>
            """
        )
    return "\n".join(cards)


def terms_lookup() -> str:
    terms = [
        ("Muon", "一种对矩阵 update 做正交化约束的 optimizer 风格；这里关注 orthogonalizer 的系数和迭代次数。"),
        ("Polar Express / PE", "一种根据 lower bound 生成 orthogonalization 系数的方案。"),
        ("Phase2 / p2", "手工 fast/stable schedule：先 fast，后 stable。"),
        ("old_fast5", "vanilla Muon baseline：5 步 fast 系数 F。"),
        ("p2_T9_f3_s6", "总 9 步，3 步 fast + 6 步 stable，是 main 中表现最好的 manual schedule。"),
        ("pe_T9_l3e-5", "Polar Express，9 步，lower bound 为 3e-5，是 final mean loss 最低配置。"),
        ("T / f / s", "T 是总迭代步数，f 是 fast steps，s 是 stable steps。"),
        ("F / S", "F=(3.4445,-4.7750,2.0315)，S=(2,-1.5,0.5)。"),
        ("lr_mul", "相对默认 learning rate 的倍数。"),
        ("seed", "随机种子；final 用 seed 0/1/2 检查稳定性。"),
        ("validation loss / val loss", "验证集 loss，越低表示语言建模质量越好。"),
        ("eval", "evaluation 的缩写；这里指验证集评估，不参与参数更新。"),
        ("run / config / schedule", "run 是一次完整训练记录，config 是配置，schedule 是 orthogonalizer 系数和迭代顺序。"),
        ("AUC/token", "validation loss 对 token 的面积，越低表示同 token 下整体下降更快。"),
        ("wall-clock", "真实运行时间；包含吞吐和计算开销影响。"),
        ("throughput_tokens_per_sec", "训练吞吐，越高表示单位时间处理 token 更多。"),
                ("update orth error", "update 正交误差，越低表示几何性质更接近正交化目标。"),
        ("MathJax / LaTeX", "页面中少量公式使用 LaTeX 写法并交给 MathJax 渲染，方便合作者核对定义。"),
        ("mean / std", "mean 是多个 seed 的平均值，std 是 seed 间波动；final 结论主要看 mean，同时检查 std 是否过大。"),
        ("W&B", "Weights & Biases，用于在线记录曲线和 run 元数据。"),
        ("W&B run links", "点击后可以打开对应 run 的在线曲线、日志和 config。"),
        ("checkpoint", "训练中保存的模型状态；后续复跑或恢复训练时使用。"),
        ("Hardware / Memory", "硬件资源只描述运行环境；显存指标用于解释 batch 与吞吐约束。"),
        ("固定训练栈", "为保证可复现性，所有对比 run 使用同一训练实现、同一注意力路径与同一日志管线。"),
        ("Attention / Normalization / Kernels", "这些名字是训练实现里的工程术语；在论文叙事里应被归入固定训练栈，而不是算法贡献。"),
        ("FineWeb", "训练数据来源；本实验使用本地缓存 shard。"),
        ("run_summary.csv", "本页表格和聚合指标的数据源，由训练日志汇总得到。"),
        ("figures/", "本页曲线图目录；这些 PNG 是从本地 summary 和 W&B/日志曲线生成的静态证据图。"),
        ("main", "100M token 的 top-config 筛选阶段。"),
        ("final", "300M token x 3 seeds 的最终结论阶段。"),
    ]
    return "\n".join(
        f"<dt><code>{esc(term)}</code></dt><dd>{esc(desc)}</dd>"
        for term, desc in terms
    )


def math_panel() -> str:
    return r"""
        <p>这部分把报告里最常见的系数和比较指标写成可渲染公式。符号只用于说明配置，不改变实验数据。</p>
        <ul>
          <li><code>F</code> 是 fast Newton-Schulz 系数：\(F=(3.4445,-4.7750,2.0315)\)。它对应 vanilla Muon 的激进正交化更新。</li>
          <li><code>S</code> 是 stable 系数：\(S=(2,-1.5,0.5)\)。它对应更保守、更稳定的正交化更新。</li>
          <li>Phase2 手工 schedule 定义为
            \[
              \mathrm{P2}(T,f)=\underbrace{F,\ldots,F}_{f\ \text{steps}}+
              \underbrace{S,\ldots,S}_{T-f\ \text{steps}}.
            \]
            这里 \(T\) 是总迭代次数，\(f\) 是 fast steps，\(T-f\) 是 stable steps。</li>
          <li>final 多 seed 排名主要看平均验证损失：
            \[
              \bar L_{\mathrm{val}}=\frac{1}{N}\sum_{i=1}^{N}L_{\mathrm{val}}^{(i)}.
            \]
            同时看 seed 间 std，避免把偶然低点误当作稳定优势。</li>
        </ul>
    """


def decoder_table() -> str:
    rows = [
        ("p2_T9_f3_s6", "p2", "Phase2 manual schedule。"),
        ("p2_T9_f3_s6", "T9", "总共 9 步 Newton-Schulz / orthogonalization iteration。"),
        ("p2_T9_f3_s6", "f3", "前 3 步使用 fast 系数 F。"),
        ("p2_T9_f3_s6", "s6", "后 6 步使用 stable 系数 S。"),
        ("pe_T9_l3e-5", "pe", "Polar Express schedule。"),
        ("pe_T9_l3e-5", "T9", "Polar Express 做 9 步。"),
        ("pe_T9_l3e-5", "l3e-5", "lower bound = 3e-5。"),
        ("final_pe_T9_l3e-5_lr1.0_300M_seed1", "final", "最终 300M token 多 seed 阶段。"),
        ("final_pe_T9_l3e-5_lr1.0_300M_seed1", "lr1.0", "默认 learning-rate multiplier。"),
        ("final_pe_T9_l3e-5_lr1.0_300M_seed1", "seed1", "随机种子为 1。"),
    ]
    trs = [
        f"<tr><td><code>{esc(example)}</code></td><td><code>{esc(fragment)}</code></td><td>{esc(desc)}</td></tr>"
        for example, fragment, desc in rows
    ]
    return "<table><thead><tr><th>例子</th><th>片段</th><th>中文含义</th></tr></thead><tbody>" + "\n".join(trs) + "</tbody></table>"


CSS = """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --ink: #1d2433;
  --muted: #657082;
  --line: #d9dee7;
  --accent: #0b6bcb;
  --accent-2: #0f8a72;
  --warn: #b96b00;
  --bad: #b42318;
  --good-bg: #e9f7f2;
  --warn-bg: #fff4df;
  --shadow: 0 10px 28px rgba(21, 32, 54, 0.10);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--ink);
  line-height: 1.55;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code { font-family: "JetBrains Mono", Consolas, monospace; font-size: .92em; }
header {
  background: #111827;
  color: white;
  padding: 26px clamp(18px, 4vw, 56px);
}
header p { color: #d1d5db; max-width: 1080px; margin: 8px 0 0; }
h1 { margin: 0; font-size: clamp(28px, 4vw, 42px); letter-spacing: 0; }
h2 { margin: 0 0 12px; font-size: 22px; letter-spacing: 0; }
h3 { margin: 18px 0 8px; font-size: 17px; }
.layout {
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  gap: 18px;
  padding: 18px clamp(14px, 3vw, 42px) 42px;
}
nav {
  position: sticky;
  top: 78px;
  align-self: start;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
}
nav a { display: block; padding: 7px 8px; color: var(--ink); border-radius: 6px; }
nav a:hover { background: #eef4ff; text-decoration: none; }
.toolbar {
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(247, 248, 250, .96);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--line);
  padding: 10px clamp(14px, 3vw, 42px);
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}
.search-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.toolbar input, .toolbar select, .search-nav {
  height: 38px;
  border: 1px solid var(--line);
  background: white;
  border-radius: 7px;
  padding: 0 10px;
}
.toolbar input {
  min-width: 220px;
}
.search-nav {
  min-width: 38px;
  cursor: pointer;
  font-weight: 800;
  color: var(--ink);
}
.search-status {
  color: var(--muted);
  min-width: 54px;
  font-size: 13px;
}
.tabs { display: flex; gap: 6px; flex-wrap: wrap; }
.tab {
  border: 1px solid var(--line);
  background: white;
  border-radius: 999px;
  padding: 8px 11px;
  cursor: pointer;
}
.tab.active { background: var(--accent); color: white; border-color: var(--accent); }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
}
.metric-grid, .phase-figures {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
}
.metric {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: 14px;
}
.metric-label { color: var(--muted); font-size: 13px; }
.metric-value { font-size: 24px; font-weight: 750; margin: 5px 0; overflow-wrap: anywhere; }
.metric-note, .muted, .hint { color: var(--muted); font-size: 13px; }
.hint { display: block; margin-top: 4px; }
.callout {
  border-left: 4px solid var(--accent-2);
  background: var(--good-bg);
  padding: 12px;
  border-radius: 8px;
  margin-top: 12px;
}
.tag {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 3px 9px;
  font-size: 12px;
  border: 1px solid var(--line);
}
.tag.pass { background: var(--good-bg); color: var(--accent-2); }
.tag.warn { background: var(--warn-bg); color: var(--warn); }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; }
th { background: #f2f5f9; color: #344054; position: sticky; top: 59px; z-index: 5; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
dl { display: grid; grid-template-columns: 210px minmax(0, 1fr); gap: 8px 14px; }
dt { font-weight: 700; }
dd { margin: 0; color: #3f4a5f; }
.phase {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
  margin: 10px 0;
}
.phase summary {
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 13px 14px;
  font-weight: 700;
}
.phase-body { padding: 0 14px 14px; }
figure {
  margin: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
  overflow: hidden;
}
figure img { display: block; width: 100%; height: auto; background: white; }
figcaption { padding: 10px 12px; color: #3f4a5f; font-size: 13px; }
.hidden { display: none !important; }
mark.search-hit {
  background: #fff176;
  color: inherit;
  padding: 0 2px;
  border-radius: 3px;
}
mark.search-hit.current-hit {
  background: #ffb300;
  outline: 2px solid #9a5b00;
}
::highlight(search-hit) {
  background: #fff176;
  color: inherit;
}
::highlight(current-hit) {
  background: #ffb300;
  color: inherit;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  nav { position: static; }
  dl { grid-template-columns: 1fr; }
  th { position: static; }
}
"""


JS = """
const searchBox = document.getElementById("searchInput");
const prevHit = document.getElementById("prevHit");
const nextHit = document.getElementById("nextHit");
const searchStatus = document.getElementById("searchStatus");
const tabs = [...document.querySelectorAll(".tab")];
const searchRoot = document.querySelector("main") || document.body;
const supportsCssHighlights = Boolean(CSS.highlights && window.Highlight);
let activeKind = "all";
let searchRanges = [];
let fallbackMarks = [];
let currentHit = -1;
let searchTimer = 0;
let textIndex = [];

function applyKindFilter() {
  document.querySelectorAll(".searchable").forEach(el => {
    const kind = el.dataset.kind || "";
    const kindOk = activeKind === "all" || kind === activeKind;
    el.classList.toggle("hidden", !kindOk);
  });
}

function clearHighlights() {
  if (supportsCssHighlights) {
    CSS.highlights.delete("search-hit");
    CSS.highlights.delete("current-hit");
  }
  fallbackMarks.forEach(mark => {
    mark.replaceWith(document.createTextNode(mark.textContent));
  });
  if (fallbackMarks.length) searchRoot.normalize();
  fallbackMarks = [];
  searchRanges = [];
  currentHit = -1;
}

function collectTextNodes(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      if (parent.closest("script, style, textarea, input, mark")) return NodeFilter.FILTER_REJECT;
      if (!node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    }
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  return nodes;
}

function rebuildSearchIndex() {
  textIndex = collectTextNodes(searchRoot).map(node => ({
    node,
    lower: node.nodeValue.toLowerCase()
  }));
}

function isVisibleForSearch(node) {
  const parent = node.parentElement;
  return parent && !parent.closest(".hidden");
}

function rangesForNode(node, lower, queryLower) {
  const ranges = [];
  let index = lower.indexOf(queryLower, 0);
  while (index !== -1) {
    const range = document.createRange();
    range.setStart(node, index);
    range.setEnd(node, index + queryLower.length);
    ranges.push(range);
    index = lower.indexOf(queryLower, index + queryLower.length);
  }
  return ranges;
}

function highlightFallback(queryLower) {
  const nodes = collectTextNodes(searchRoot);
  nodes.forEach(node => {
    if (!isVisibleForSearch(node)) return;
    const text = node.nodeValue;
    const lower = text.toLowerCase();
    let cursor = 0;
    let index = lower.indexOf(queryLower, cursor);
    if (index === -1) return;
    const fragment = document.createDocumentFragment();
    while (index !== -1) {
      if (index > cursor) fragment.appendChild(document.createTextNode(text.slice(cursor, index)));
      const mark = document.createElement("mark");
      mark.className = "search-hit";
      mark.textContent = text.slice(index, index + queryLower.length);
      fragment.appendChild(mark);
      const range = document.createRange();
      range.selectNodeContents(mark);
      searchRanges.push(range);
      fallbackMarks.push(mark);
      cursor = index + queryLower.length;
      index = lower.indexOf(queryLower, cursor);
    }
    if (cursor < text.length) fragment.appendChild(document.createTextNode(text.slice(cursor)));
    node.parentNode.replaceChild(fragment, node);
  });
}

function highlightWithCssRanges(queryLower) {
  const ranges = [];
  textIndex.forEach(item => {
    if (!isVisibleForSearch(item.node)) return;
    if (!item.lower.includes(queryLower)) return;
    ranges.push(...rangesForNode(item.node, item.lower, queryLower));
  });
  searchRanges = ranges;
  if (searchRanges.length) {
    CSS.highlights.set("search-hit", new Highlight(...searchRanges));
  }
}

function updateCurrentHighlight() {
  if (supportsCssHighlights) {
    CSS.highlights.delete("current-hit");
    if (searchRanges[currentHit]) {
      CSS.highlights.set("current-hit", new Highlight(searchRanges[currentHit]));
    }
    return;
  }
  fallbackMarks.forEach(mark => mark.classList.remove("current-hit"));
  const mark = fallbackMarks[currentHit];
  if (mark) mark.classList.add("current-hit");
}

function rangeElement(range) {
  const node = range.startContainer;
  return node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
}

function scrollRangeIntoView(range) {
  const el = rangeElement(range);
  if (!el) return;
  const details = el.closest("details");
  if (details) details.open = true;
  const rect = range.getBoundingClientRect();
  if (rect && rect.height >= 0) {
    const target = Math.max(0, rect.top + window.scrollY - window.innerHeight * 0.35);
    window.scrollTo({ top: target, behavior: "smooth" });
  } else {
    el.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function updateSearchStatus() {
  if (!searchRanges.length) {
    searchStatus.textContent = searchBox.value.trim() ? "0/0" : "";
    return;
  }
  searchStatus.textContent = `${currentHit + 1}/${searchRanges.length}`;
}

function focusHit(index) {
  if (!searchRanges.length) {
    updateSearchStatus();
    return;
  }
  currentHit = (index + searchRanges.length) % searchRanges.length;
  updateCurrentHighlight();
  scrollRangeIntoView(searchRanges[currentHit]);
  updateSearchStatus();
}

function runSearchNow() {
  clearHighlights();
  const query = searchBox.value.trim().toLowerCase();
  if (!query) {
    updateSearchStatus();
    return;
  }
  if (supportsCssHighlights) {
    highlightWithCssRanges(query);
  } else {
    highlightFallback(query);
  }
  if (searchRanges.length) focusHit(0);
  updateSearchStatus();
}

function applySearch() {
  window.clearTimeout(searchTimer);
  searchTimer = window.setTimeout(runSearchNow, 90);
}

searchBox.addEventListener("input", applySearch);
searchBox.addEventListener("keydown", event => {
  if (event.key === "Enter") {
    event.preventDefault();
    focusHit(event.shiftKey ? currentHit - 1 : currentHit + 1);
  }
});
prevHit.addEventListener("click", () => focusHit(currentHit - 1));
nextHit.addEventListener("click", () => focusHit(currentHit + 1));
tabs.forEach(btn => {
  btn.addEventListener("click", () => {
    tabs.forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    activeKind = btn.dataset.kind || btn.dataset.filter || "all";
    applyKindFilter();
    runSearchNow();
  });
});

rebuildSearchIndex();
window.addEventListener("load", rebuildSearchIndex);
"""


def build_html(rows: list[dict[str, str]], figures_dir: Path, out_path: Path) -> str:
    counts = Counter(r.get("group") or "unknown" for r in rows)
    payload = {"row_count": len(rows), "groups": dict(counts)}
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Muon / Polar Express 实验报告</title>
  <style>{CSS}</style>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
  <header>
    <h1>Muon / Polar Express 实验报告</h1>
    <p>读法：先看当前结论，再看名词速查和 run-name 解码，最后按阶段查看 W&B、曲线和表格证据。</p>
  </header>
  <section class="toolbar">
    <div class="search-controls" role="search">
      <label class="sr-only" for="searchInput">关键词搜索</label>
      <input id="searchInput" type="search" placeholder="搜索 phase、run name、loss、W&B、p2、PE">
      <button id="prevHit" class="search-nav" type="button" aria-label="上一个搜索结果">&lt;</button>
      <button id="nextHit" class="search-nav" type="button" aria-label="下一个搜索结果">&gt;</button>
      <span id="searchStatus" class="search-status" aria-live="polite"></span>
    </div>
    <div class="tabs" aria-label="view filters">
      <button class="tab active" type="button" data-kind="all" data-filter="all">全部</button>
      <button class="tab" type="button" data-kind="verdict" data-filter="verdict">结论</button>
      <button class="tab" type="button" data-kind="phases" data-filter="phases">阶段</button>
      <button class="tab" type="button" data-kind="runs" data-filter="runs">运行</button>
      <button class="tab" type="button" data-kind="curves" data-filter="curves">曲线</button>
      <button class="tab" type="button" data-kind="metrics" data-filter="metrics">指标</button>
      <button class="tab" type="button" data-kind="wandb" data-filter="wandb">W&B</button>
    </div>
  </section>
  <div class="layout">
    <aside class="sidebar" aria-label="section navigation">
      <nav>
        <a href="#verdict">当前结论</a>
        <a href="#terms">名词速查</a>
        <a href="#decoder">run-name 解码</a>
        <a href="#math">核心公式</a>
        <a href="#sources">数据来源</a>
        <a href="#timeline">阶段时间线</a>
        <a href="#wandb">W&B 证据</a>
        <a href="#runs">完整 run 表</a>
        <a href="#rules">Go / No-Go</a>
      </nav>
    </aside>
    <main>
      <section id="verdict" class="panel searchable" data-kind="verdict">
        <h2>当前结论</h2>
        <div class="metric-grid">{metric_cards(rows)}</div>
        <div class="callout">
          <strong>当前状态：</strong>实验已完成。最稳妥的结论是：<code>pe_T9_l3e-5</code> 的 final mean loss 略低，
          <code>old_fast5</code> 的计算成本最低，<code>p2_T9_f3_s6</code> 的 wall-clock 曲线和几何诊断有竞争力。
          loss 差距很小，因此报告中应强调多维比较，而不是只按 final loss 排名。
        </div>
        <h3>Final 多 seed 聚合</h3>
        <div class="table-wrap">{final_aggregate_table(rows)}</div>
        <h3>完成情况</h3>
        <ul>{group_counts(rows)}</ul>
      </section>

      <section id="terms" class="panel searchable" data-kind="verdict">
        <h2>名词速查</h2>
        <dl>{terms_lookup()}</dl>
      </section>

      <section id="decoder" class="panel searchable" data-kind="verdict">
        <h2>run-name 解码</h2>
        <p>所有 run name 旁边都附了中文含义。下面用最重要的配置拆开说明。</p>
        <div class="table-wrap">{decoder_table()}</div>
      </section>

      <section id="math" class="panel searchable" data-kind="metrics">
        <h2>核心公式</h2>
        {math_panel()}
      </section>

      <section id="sources" class="panel searchable" data-kind="metrics">
        <h2>数据来源</h2>
        <p>表格和聚合指标来自 <code>results/run_summary.csv</code>；曲线图来自 <code>results/figures/</code>；final run 的在线证据来自下方 W&B run links。</p>
        <p>本页是静态 HTML 快照。当前快照日期按本地生成时间处理；如果后续继续训练，新的 W&B 曲线、日志、checkpoint 或 eval 结论需要同步写回对应 phase。</p>
      </section>

      <section id="timeline" class="panel">
        <h2>阶段时间线</h2>
        {phase_blocks(rows, figures_dir, out_path)}
      </section>

      <section id="wandb" class="panel searchable" data-kind="wandb">
        <h2>W&B 证据</h2>
        <p>全局项目：<a href="{esc(WANDB_PROJECT_URL)}">muon-nanogpt</a>。下表列出 final 9 个核心 run 的 W&B 链接。</p>
        <div class="table-wrap">{wandb_table(rows)}</div>
      </section>

      <section id="runs" class="panel">
        <h2>完整 run 表</h2>
        <p>这张表来自 <code>results/run_summary.csv</code>。每个 run name 都带中文配置说明。</p>
        <div class="table-wrap">{run_table(rows)}</div>
      </section>

      <section id="rules" class="panel searchable" data-kind="verdict">
        <h2>Go / No-Go 规则</h2>
        <ul>
          <li><strong>继续：</strong>如果要扩展实验，优先围绕 <code>pe_T9_l3e-5</code>、<code>old_fast5</code>、<code>p2_T9_f3_s6</code> 做更长 budget 或小模型复核。</li>
          <li><strong>暂停：</strong>如果 W&B、eval、数据 shard 或 seed 固定失败，先修证据链，再继续训练。</li>
          <li><strong>停止：</strong>NaN/Inf、OOM、连续 eval 明显劣化、tokens/sec 比 baseline 慢 40% 以上，进入失败记录而不是继续烧 GPU。</li>
          <li><strong>监控：</strong>远程 GPU 实验运行时最长 10 分钟检查一次，检查进程、GPU、日志、W&B、eval 和 checkpoint。</li>
        </ul>
      </section>
    </main>
  </div>
  <script id="dashboardData" type="application/json">{esc(json.dumps(payload, ensure_ascii=True))}</script>
  <script>{JS}</script>
</body>
</html>
"""
    return "\n".join(line.rstrip() for line in html_text.splitlines()) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="results")
    parser.add_argument("--out", default="results/dashboard.html")
    args = parser.parse_args()

    analysis_dir = (ROOT / args.analysis_dir).resolve()
    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(analysis_dir / "run_summary.csv")
    html_text = build_html(rows, analysis_dir / "figures", out_path)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
