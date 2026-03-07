"""
Report exporter for SentinelAI.

- Markdown report (default)
- Optional PDF export with basic Markdown rendering (headings/lists/tables)
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Optional, Tuple

import pandas as pd


def _top_value_counts(df: pd.DataFrame, col: str, topn: int = 5) -> list[tuple[str, int]]:
    if df is None or df.empty or col not in df.columns:
        return []
    vc = df[col].astype(str).value_counts().head(topn)
    return [(str(k), int(v)) for k, v in vc.items()]


def build_markdown_report(
    analysis_df: Optional[pd.DataFrame],
    rule_alerts_df: Optional[pd.DataFrame],
    pcap_df: Optional[pd.DataFrame] = None,
    incident_df: Optional[pd.DataFrame] = None,
    action_df: Optional[pd.DataFrame] = None,
    llm_assessment: str = "",
) -> str:
    """Build a Markdown security report from current detection results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_flows = int(len(analysis_df)) if analysis_df is not None else 0
    total_packets = int(len(pcap_df)) if pcap_df is not None else 0
    total_alerts = int(len(rule_alerts_df)) if rule_alerts_df is not None else 0
    total_incidents = int(len(incident_df)) if incident_df is not None else 0
    total_actions = int(len(action_df)) if action_df is not None else 0

    lines: list[str] = [
        "# SentinelAI 安全检测报告",
        "",
        f"- 生成时间：{now}",
        f"- 流量记录数：{total_flows}",
        f"- 报文记录数（PCAP解析）：{total_packets}",
        f"- 规则告警总数：{total_alerts}",
        f"- 聚合事件总数：{total_incidents}",
        f"- 响应动作记录数：{total_actions}",
        "",
        "## 一、规则告警概览",
        "",
    ]

    if rule_alerts_df is None or rule_alerts_df.empty:
        lines.extend(["- 当前无规则告警。", ""])
    else:
        for rule, count in _top_value_counts(rule_alerts_df, "rule_name", topn=10):
            lines.append(f"- {rule}: {count}")
        lines.append("")
        lines.append("### 告警明细（前10条）")
        lines.append("")
        lines.append("| 时间 | 规则 | 严重级别 | 源IP | 目标IP | 证据摘要 |")
        lines.append("|---|---|---|---|---|---|")
        top10 = rule_alerts_df.head(10)
        for _, row in top10.iterrows():
            lines.append(
                f"| {row.get('timestamp', '')} | {row.get('rule_name', '')} | "
                f"{row.get('severity', '')} | {row.get('src_ip', '')} | "
                f"{row.get('dst_ip', '')} | {str(row.get('evidence', ''))[:80]} |"
            )
        lines.append("")

    lines.extend(["## 二、事件链概览", ""])
    if incident_df is None or incident_df.empty:
        lines.extend(["- 当前无聚合事件。", ""])
    else:
        lines.append("| 事件ID | 源IP | 风险分 | 最高级别 | 告警数 | 规则链 |")
        lines.append("|---|---|---:|---|---:|---|")
        for _, row in incident_df.head(10).iterrows():
            lines.append(
                f"| {row.get('incident_id', '')} | {row.get('src_ip', '')} | "
                f"{row.get('risk_score', 0)} | {row.get('max_severity', '')} | "
                f"{row.get('alert_count', 0)} | {row.get('event_chain', '')} |"
            )
        lines.append("")

    lines.extend(["## 三、模型与流量统计", ""])
    if analysis_df is None or analysis_df.empty:
        lines.extend(["- 无可用流量数据。", ""])
    else:
        lines.append(f"- 流量总数：{len(analysis_df)}")
        for p, c in _top_value_counts(analysis_df, "protocol", topn=5):
            lines.append(f"- 协议 {p}: {c}")
        if "score_iso_norm" in analysis_df.columns:
            mean_score = float(analysis_df["score_iso_norm"].fillna(0).mean())
            lines.append(f"- IsolationForest 平均异常分数：{mean_score:.4f}")
        lines.append("")

    lines.extend(["## 四、响应动作审计", ""])
    if action_df is None or action_df.empty:
        lines.extend(["- 当前无响应动作记录。", ""])
    else:
        lines.append("| 时间 | 动作类型 | 状态 | 目标 | 详情 |")
        lines.append("|---|---|---|---|---|")
        for _, row in action_df.head(10).iterrows():
            lines.append(
                f"| {row.get('timestamp', '')} | {row.get('action_type', '')} | "
                f"{row.get('status', '')} | {row.get('target', '')} | {row.get('detail', '')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 五、处置建议",
            "- 优先处理高风险事件对应的源IP，并结合终端日志进行溯源。",
            "- 对 DNS 异常与 ARP 异常同时出现的主机执行网络隔离与资产核验。",
            "- 对周期性心跳/低慢扫描行为启用更长窗口监控与白名单校验。",
            "",
        ]
    )

    if llm_assessment:
        lines.extend(["## 六、LLM 威胁研判", "", llm_assessment.strip(), ""])

    return "\n".join(lines)


def build_report_filename(prefix: str = "sentinel_report", ext: str = "md") -> str:
    """Build timestamped report filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = ext.strip(".")
    return f"{prefix}_{ts}.{ext}"


def _is_table_separator(line: str) -> bool:
    s = line.strip().strip("|").replace(":", "").replace("-", "").replace(" ", "")
    return s == ""


def _parse_md_table(lines: list[str], start_idx: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    idx = start_idx
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not _is_table_separator(line):
            rows.append(cells)
        idx += 1
    return rows, idx


def markdown_to_pdf_bytes(markdown_text: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Convert markdown text to PDF bytes with basic Markdown rendering.
    Supports headings, bullet lists, and pipe tables.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception:
        return None, "未安装 reportlab，无法导出 PDF。请执行: pip install reportlab"

    import io

    font_name = "Helvetica"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        font_name = "STSong-Light"
    except Exception:
        font_name = "Helvetica"

    styles = getSampleStyleSheet()
    base = ParagraphStyle("cn_base", parent=styles["BodyText"], fontName=font_name, fontSize=10, leading=14)
    h1 = ParagraphStyle("cn_h1", parent=base, fontSize=18, leading=24, spaceBefore=8, spaceAfter=6)
    h2 = ParagraphStyle("cn_h2", parent=base, fontSize=14, leading=20, spaceBefore=8, spaceAfter=4)
    h3 = ParagraphStyle("cn_h3", parent=base, fontSize=12, leading=18, spaceBefore=6, spaceAfter=4)
    bullet = ParagraphStyle("cn_bullet", parent=base, leftIndent=14, bulletIndent=2)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    story = []

    lines = markdown_text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if line == "":
            story.append(Spacer(1, 6))
            i += 1
            continue

        if line.startswith("|"):
            table_rows, next_i = _parse_md_table(lines, i)
            if table_rows:
                # normalize row width
                width = max(len(r) for r in table_rows)
                table_rows = [r + [""] * (width - len(r)) for r in table_rows]
                t = Table(table_rows, repeatRows=1)
                t.setStyle(
                    TableStyle(
                        [
                            ("FONTNAME", (0, 0), (-1, -1), font_name),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ]
                    )
                )
                story.append(t)
                story.append(Spacer(1, 8))
                i = next_i
                continue

        if line.startswith("### "):
            story.append(Paragraph(escape(line[4:]), h3))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(escape(line[3:]), h2))
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(escape(line[2:]), h1))
            i += 1
            continue

        if line.startswith("- "):
            story.append(Paragraph(escape(line[2:]), bullet, bulletText="•"))
            i += 1
            continue

        story.append(Paragraph(escape(raw), base))
        i += 1

    try:
        doc.build(story)
    except Exception as e:
        return None, f"PDF 渲染失败: {e}"

    return buf.getvalue(), None
