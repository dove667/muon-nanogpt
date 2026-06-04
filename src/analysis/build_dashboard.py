#!/usr/bin/env python
"""Build a lightweight HTML report for the fixed 5x3 experiment."""

import argparse
import csv
import html
from pathlib import Path

from src.utils import ROOT


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def figure_tag(path: Path, out_path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    try:
        rel = path.relative_to(out_path.parent).as_posix()
    except ValueError:
        rel = path.as_posix()
    return (
        "<figure>"
        f"<img src=\"{esc(rel)}\" alt=\"{esc(path.stem)}\">"
        f"<figcaption>{esc(caption)}</figcaption>"
        "</figure>"
    )


def summary_cards(orth_rows: list[dict[str, str]], run_rows: list[dict[str, str]]) -> str:
    best = next((row for row in orth_rows if row.get("final_val_loss_mean")), None)
    completed = sum(int(row.get("completed_count") or 0) for row in orth_rows)
    total = sum(int(row.get("run_count") or 0) for row in orth_rows)
    best_name = best.get("orth_label") if best else ""
    best_loss = best.get("final_val_loss_mean") if best else ""
    cards = [
        ("实验规模", f"{total} runs", "固定 5 配置 × 3 seeds"),
        ("完成情况", f"{completed}/{total}", "completed / total"),
        ("最佳平均 final loss", best_loss, best_name),
        ("比较对象", f"{len(orth_rows)} configs", "AdamW / Vanilla / Manual / Fast / Polar Express"),
    ]
    return "\n".join(
        "<section class=\"card\">"
        f"<div class=\"label\">{esc(label)}</div>"
        f"<div class=\"value\">{esc(value)}</div>"
        f"<div class=\"note\">{esc(note)}</div>"
        "</section>"
        for label, value, note in cards
    )


def orth_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td><code>{esc(row.get('orth_label'))}</code></td>"
            f"<td>{esc(row.get('completed_count'))}/{esc(row.get('run_count'))}</td>"
            f"<td>{esc(row.get('final_val_loss_mean'))}</td>"
            f"<td>{esc(row.get('final_val_loss_std'))}</td>"
            f"<td>{esc(row.get('val_auc_tokens_mean'))}</td>"
            f"<td>{esc(row.get('throughput_tokens_per_sec_mean'))}</td>"
            f"<td>{esc(row.get('step_time_ms_mean'))}</td>"
            f"<td>{esc(row.get('spec_update_orth_error_mean'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>配置</th><th>完成</th><th>mean final val/loss</th><th>std</th>"
        "<th>mean val AUC</th><th>mean throughput</th><th>mean step time (ms)</th><th>mean orth error</th>"
        "</tr></thead><tbody>"
        + "\n".join(body)
        + "</tbody></table>"
    )


def run_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{esc(row.get('orth_label'))}</td>"
            f"<td><code>{esc(row.get('name'))}</code></td>"
            f"<td>{esc(row.get('seed'))}</td>"
            f"<td><code>{esc(row.get('schedule'))}</code></td>"
            f"<td>{esc(row.get('final_val_loss'))}</td>"
            f"<td>{esc(row.get('throughput_tokens_per_sec'))}</td>"
            f"<td>{esc(row.get('step_time_ms'))}</td>"
            f"<td>{esc(row.get('spec_update_orth_error'))}</td>"
            f"<td>{esc(row.get('status'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>配置</th><th>run</th><th>seed</th><th>schedule</th><th>final val/loss</th>"
        "<th>throughput</th><th>step time (ms)</th><th>orth error</th><th>status</th>"
        "</tr></thead><tbody>"
        + "\n".join(body)
        + "</tbody></table>"
    )


CSS = """
:root {
  --bg: #f5f7fb;
  --panel: #ffffff;
  --ink: #1c2431;
  --muted: #667085;
  --line: #d7dee8;
  --accent: #0b6bcb;
  --shadow: 0 12px 28px rgba(17, 24, 39, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: linear-gradient(180deg, #eef4fb 0%, var(--bg) 180px);
  color: var(--ink);
}
header {
  padding: 40px 24px 24px;
  max-width: 1180px;
  margin: 0 auto;
}
h1 { margin: 0 0 10px; font-size: 36px; }
h2 { margin: 0 0 14px; font-size: 24px; }
p { color: var(--muted); }
main {
  max-width: 1180px;
  margin: 0 auto;
  padding: 0 24px 40px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 18px;
  margin-bottom: 18px;
}
.cards, .figures {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
}
.card {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 16px;
  background: #fbfdff;
}
.label { color: var(--muted); font-size: 13px; }
.value { font-size: 28px; font-weight: 700; margin: 6px 0; }
.note { color: var(--muted); font-size: 13px; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); }
th { background: #f8fafc; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
figure {
  margin: 0;
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
  background: white;
}
figure img { width: 100%; display: block; }
figcaption { padding: 10px 12px; color: var(--muted); font-size: 13px; }
ul { color: var(--muted); }
@media (max-width: 720px) {
  header, main { padding-left: 16px; padding-right: 16px; }
  h1 { font-size: 30px; }
}
"""


def build_html(run_rows: list[dict[str, str]], orth_rows: list[dict[str, str]], figures_dir: Path, out_path: Path) -> str:
    figure_specs = [
        ("val_loss_mean_vs_tokens.png", "5 个配置的 mean±std val/loss-token 曲线。"),
        ("val_loss_mean_vs_wall_time.png", "5 个配置的 mean±std val/loss-wall-time 曲线。"),
        ("val_loss_by_seed_tokens.png", "每个配置内部 3 个 seeds 的单独曲线。"),
        ("throughput_vs_final_val_loss.png", "吞吐与最终验证损失的散点对比。"),
        ("final_val_loss_mean_std.png", "最终验证损失均值与标准差柱状图。"),
        ("orth_error_mean_std.png", "update orthogonality error 的均值与标准差柱状图。"),
        ("step_time_ms_mean_std.png", "平均 step time 对比。"),
    ]
    figures = "\n".join(
        figure_tag(figures_dir / filename, out_path, caption)
        for filename, caption in figure_specs
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Muon Schedule Study Report</title>
  <style>{CSS}</style>
</head>
<body>
  <header>
    <h1>Muon Schedule Study 实验报告</h1>
    <p>当前报告只针对新的固定实验设置：AdamW、Vanilla、Manual、Fast、Polar Express，各 3 个 seeds，总计 15 runs。</p>
  </header>
  <main>
    <section class="panel">
      <h2>实验设置</h2>
      <ul>
        <li>固定训练栈：100M train tokens，2M eval interval，524,288 eval tokens。</li>
        <li>固定 batch / seq：131,072 tokens/step，seq_len=2048，grad_accum=16。</li>
        <li>模型是标准 prenorm Transformer + RoPE；不使用 BOS packing、softcap logits 或交错 Adam 更新。</li>
        <li>Muon 组固定 T=5；AdamW 作为无正交化基线。</li>
      </ul>
    </section>

    <section class="panel">
      <h2>概览</h2>
      <div class="cards">{summary_cards(orth_rows, run_rows)}</div>
    </section>

    <section class="panel">
      <h2>配置聚合</h2>
      <div class="table-wrap">{orth_table(orth_rows)}</div>
    </section>

    <section class="panel">
      <h2>关键图表</h2>
      <div class="figures">{figures}</div>
    </section>

    <section class="panel">
      <h2>逐 run 明细</h2>
      <div class="table-wrap">{run_table(run_rows)}</div>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="results")
    parser.add_argument("--out", default="results/dashboard.html")
    args = parser.parse_args()

    analysis_dir = (ROOT / args.analysis_dir).resolve()
    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_rows = read_csv(analysis_dir / "run_summary.csv")
    orth_rows = read_csv(analysis_dir / "orth_summary.csv")
    html_text = build_html(run_rows, orth_rows, analysis_dir / "figures", out_path)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
