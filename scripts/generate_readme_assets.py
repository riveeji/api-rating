from __future__ import annotations

import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "agent_eval.db"
OUTPUT_DIR = ROOT / "docs" / "images"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = load_leaderboard()
    write_architecture_svg(OUTPUT_DIR / "architecture.svg")
    write_results_svg(OUTPUT_DIR / "benchmark-results.svg", leaderboard)


def load_leaderboard() -> list[dict[str, float | str]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                a.config_id,
                a.display_name,
                AVG(e.success) AS success_rate,
                AVG(r.total_latency_ms) AS avg_latency_ms,
                AVG(r.total_tokens) AS avg_tokens
            FROM runs r
            JOIN evaluations e ON e.run_id = r.run_id
            JOIN agent_configs a ON a.config_id = r.config_id
            WHERE a.config_id IN ('baseline_heuristic', 'planner_heuristic', 'verifier_heuristic')
            GROUP BY a.config_id, a.display_name
            ORDER BY success_rate DESC, avg_latency_ms ASC
            """
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def write_architecture_svg(path: Path) -> None:
    boxes = [
        ("用户 / 产品入口", 40, 50, 220, 72, "#f4d8ca"),
        ("在线知识库助手", 310, 50, 240, 72, "#f8ecdd"),
        ("工具层", 610, 40, 220, 180, "#fff8ef"),
        ("文档检索 / 结构化数据", 880, 50, 250, 72, "#f8ecdd"),
        ("离线任务集", 40, 250, 220, 72, "#f4d8ca"),
        ("实验编排与 Runner", 310, 250, 240, 72, "#f8ecdd"),
        ("评测器 / 失败分类", 610, 250, 220, 72, "#fff8ef"),
        ("排行榜 / 实验详情", 880, 250, 250, 72, "#f8ecdd"),
        ("SQLite 持久化层", 400, 430, 360, 78, "#f4d8ca"),
    ]
    arrows = [
        ((260, 86), (310, 86), "真实提问"),
        ((550, 86), (610, 86), "tool calling"),
        ((830, 86), (880, 86), "检索 / SQL / API"),
        ((260, 286), (310, 286), "benchmark 任务"),
        ((550, 286), (610, 286), "run 结果"),
        ((830, 286), (880, 286), "聚合分析"),
        ((430, 122), (530, 430), ""),
        ((740, 220), (650, 430), ""),
        ((430, 322), (520, 430), ""),
        ((1010, 322), (690, 430), ""),
    ]

    def rect(label: str, x: int, y: int, w: int, h: int, fill: str) -> str:
        return (
            f'<rect x="{x}" y="{y}" rx="18" ry="18" width="{w}" height="{h}" '
            f'fill="{fill}" stroke="#b55d31" stroke-width="2"/>'
            f'<text x="{x + w / 2}" y="{y + h / 2 + 6}" text-anchor="middle" '
            f'font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="24" fill="#1f1f1f">{label}</text>'
        )

    def arrow(start: tuple[int, int], end: tuple[int, int], label: str) -> str:
        sx, sy = start
        ex, ey = end
        label_svg = ""
        if label:
            lx = (sx + ex) / 2
            ly = (sy + ey) / 2 - 10
            label_svg = (
                f'<text x="{lx}" y="{ly}" text-anchor="middle" '
                f'font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="18" fill="#6b665d">{label}</text>'
            )
        return (
            f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" stroke="#704931" stroke-width="3" marker-end="url(#arrow)"/>'
            f"{label_svg}"
        )

    content = "\n".join(rect(*box) for box in boxes) + "\n" + "\n".join(
        arrow(start, end, label) for start, end, label in arrows
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="560" viewBox="0 0 1180 560">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#fffaf3"/>
      <stop offset="100%" stop-color="#f5f1e8"/>
    </linearGradient>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <path d="M0,0 L10,3 L0,6 z" fill="#704931"/>
    </marker>
  </defs>
  <rect width="1180" height="560" fill="url(#bg)"/>
  <text x="40" y="28" font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="28" font-weight="700" fill="#1f1f1f">System Architecture</text>
  {content}
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_results_svg(path: Path, leaderboard: list[dict[str, float | str]]) -> None:
    width = 1180
    height = 760
    margin_left = 110
    chart_width = 940
    bar_height = 48
    gap = 32
    start_y = 120

    success_max = 1.0
    latency_max = max(float(item["avg_latency_ms"]) for item in leaderboard) if leaderboard else 1.0

    success_bars: list[str] = []
    latency_points: list[str] = []
    labels: list[str] = []

    for idx, item in enumerate(leaderboard):
        y = start_y + idx * (bar_height + gap)
        name = str(item["display_name"])
        success = float(item["success_rate"])
        latency = float(item["avg_latency_ms"])
        bar_w = chart_width * (success / success_max)
        success_bars.append(
            f'<rect x="{margin_left}" y="{y}" width="{bar_w}" height="{bar_height}" rx="12" fill="#b55d31" opacity="0.88"/>'
            f'<text x="{margin_left + bar_w + 12}" y="{y + 31}" font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="20" fill="#1f1f1f">{success * 100:.1f}%</text>'
        )
        point_x = margin_left + chart_width * (latency / latency_max)
        point_y = 520 + idx * 42
        latency_points.append(
            f'<circle cx="{point_x}" cy="{point_y}" r="8" fill="#704931"/>'
            f'<text x="{point_x + 14}" y="{point_y + 6}" font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="18" fill="#1f1f1f">{latency:.1f} ms</text>'
        )
        labels.append(
            f'<text x="24" y="{y + 31}" font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="20" fill="#1f1f1f">{name}</text>'
            f'<text x="24" y="{point_y + 6}" font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="20" fill="#1f1f1f">{name}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#fffaf3"/>
      <stop offset="100%" stop-color="#f5f1e8"/>
    </linearGradient>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#bg)"/>
  <text x="40" y="40" font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="32" font-weight="700" fill="#1f1f1f">Heuristic Benchmark Snapshot</text>
  <text x="40" y="82" font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="22" fill="#6b665d">Success rate comparison and average latency by strategy</text>

  <text x="40" y="120" font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="24" font-weight="700" fill="#1f1f1f">Success Rate</text>
  <line x1="{margin_left}" y1="410" x2="{margin_left + chart_width}" y2="410" stroke="#d8d0c3" stroke-width="2"/>
  <text x="{margin_left}" y="438" font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="18" fill="#6b665d">0%</text>
  <text x="{margin_left + chart_width - 26}" y="438" font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="18" fill="#6b665d">100%</text>
  {''.join(labels[:len(leaderboard)])}
  {''.join(success_bars)}

  <text x="40" y="500" font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="24" font-weight="700" fill="#1f1f1f">Average Latency</text>
  <line x1="{margin_left}" y1="540" x2="{margin_left + chart_width}" y2="540" stroke="#d8d0c3" stroke-width="2"/>
  <text x="{margin_left}" y="568" font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="18" fill="#6b665d">0 ms</text>
  <text x="{margin_left + chart_width - 70}" y="568" font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" font-size="18" fill="#6b665d">{latency_max:.1f} ms</text>
  {''.join(latency_points)}
</svg>"""
    path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
