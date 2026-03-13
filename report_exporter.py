"""
Report exporter for SentinelAI.

- Markdown report (default)
- Optional PDF export with basic Markdown rendering (headings/lists/tables)
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Optional, Tuple
import json

import pandas as pd


def _top_value_counts(df: Optional[pd.DataFrame], col: str, topn: int = 5) -> list[tuple[str, int]]:
    if df is None or df.empty or col not in df.columns:
        return []
    vc = df[col].astype(str).value_counts().head(topn)
    return [(str(k), int(v)) for k, v in vc.items()]


def _infer_incident_stage(rules_text: object) -> str:
    text = str(rules_text).upper()
    if "PORT_SCAN" in text or "ARP_SCAN" in text:
        return "侦察"
    if "BRUTE_FORCE" in text:
        return "尝试入侵"
    if "ANOMALOUS_DNS" in text or "PERIODIC_BEACON" in text:
        return "控制通信"
    if "TRAFFIC_BURST" in text:
        return "影响/外传"
    if "ARP_SPOOF" in text or "ARP_MITM_SUSPECT" in text:
        return "中间人风险"
    return "综合异常"


def _evidence_summary(evidence_text: object) -> str:
    text = str(evidence_text or "").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except Exception:
        return text[:120]
    keys = [
        "window_start",
        "window_end",
        "flow_count",
        "attempt_count",
        "unique_queries",
        "nxdomain_count",
        "arp_packet_count",
    ]
    parts = [f"{k}={data[k]}" for k in keys if k in data]
    return "; ".join(parts) if parts else text[:120]


def _is_local_llm_fallback(llm_assessment: str) -> bool:
    t = (llm_assessment or "").strip()
    return t.startswith("【本地研判")


def _has_meaningful_agent_appendix(agent_report_markdown: str) -> bool:
    t = (agent_report_markdown or "").strip()
    if not t:
        return False
    if ("规则告警数：0" in t or "规则告警数: 0" in t) and ("事件数：0" in t or "事件数: 0" in t):
        return False
    return True


def _build_status_section(
    analysis_df: Optional[pd.DataFrame],
    rule_alerts_df: Optional[pd.DataFrame],
    incident_df: Optional[pd.DataFrame],
    action_df: Optional[pd.DataFrame],
) -> list[str]:
    lines = ["## 零、数据与执行状态", ""]
    lines.append(f"- 流量特征状态：{'可用' if analysis_df is not None and not analysis_df.empty else '为空'}")
    if rule_alerts_df is None:
        lines.append("- 规则检测状态：未执行（建议点击“执行规则检测”后再导出）")
    elif rule_alerts_df.empty:
        lines.append("- 规则检测状态：已执行，当前未命中规则告警")
    else:
        lines.append(f"- 规则检测状态：已执行，命中 {len(rule_alerts_df)} 条告警")

    if incident_df is None:
        lines.append("- 事件聚合状态：未执行或无输入")
    elif incident_df.empty:
        lines.append("- 事件聚合状态：已执行，当前无聚合事件")
    else:
        lines.append(f"- 事件聚合状态：已执行，聚合 {len(incident_df)} 起事件")

    lines.append(
        f"- 响应动作状态：{'无记录' if action_df is None or action_df.empty else f'已有 {len(action_df)} 条记录'}"
    )
    lines.append("")
    return lines


def _build_recommendation_section(
    rule_alerts_df: Optional[pd.DataFrame],
    incident_df: Optional[pd.DataFrame],
    analysis_df: Optional[pd.DataFrame],
) -> list[str]:
    lines = ["## 五、处置建议", ""]
    total_alerts = int(len(rule_alerts_df)) if isinstance(rule_alerts_df, pd.DataFrame) else 0
    total_incidents = int(len(incident_df)) if isinstance(incident_df, pd.DataFrame) else 0

    if total_alerts == 0 and total_incidents == 0:
        lines.extend(
            [
                "- 当前未见明确规则攻击信号，建议先保留本次快照作为基线样本。",
                "- 建议在受控实验中注入已知攻击流（端口扫描/DNS异常/ARP样本）验证检测灵敏度。",
                "- 若要触发更稳定的 AI 研判，建议延长抓包时长并提高流量多样性。",
                "",
            ]
        )
        return lines

    lines.extend(
        [
            "- 优先处置高风险事件对应的源IP，并结合终端日志进行溯源。",
            "- 对 DNS 异常与 ARP 异常同时出现的主机执行网络隔离与资产核验。",
            "- 对周期性心跳/低慢扫描行为启用更长窗口监控与白名单核验。",
        ]
    )

    if isinstance(analysis_df, pd.DataFrame) and "score_ensemble_norm" in analysis_df.columns:
        high_anom = int((analysis_df["score_ensemble_norm"].fillna(0) >= 0.8).sum())
        lines.append(f"- 高异常分（>=0.8）流量数量：{high_anom}，建议人工复核 Top 异常流。")
    lines.append("")
    return lines


def build_markdown_report(
    analysis_df: Optional[pd.DataFrame],
    rule_alerts_df: Optional[pd.DataFrame],
    pcap_df: Optional[pd.DataFrame] = None,
    incident_df: Optional[pd.DataFrame] = None,
    action_df: Optional[pd.DataFrame] = None,
    llm_assessment: str = "",
    agent_report_markdown: str = "",
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
    ]

    lines.extend(_build_status_section(analysis_df, rule_alerts_df, incident_df, action_df))

    lines.extend(["## 一、规则告警概览", ""])
    if rule_alerts_df is None:
        lines.extend(["- 规则检测尚未执行。", ""])
    elif rule_alerts_df.empty:
        lines.extend(["- 当前无规则告警。", ""])
    else:
        for rule, count in _top_value_counts(rule_alerts_df, "rule_name", topn=10):
            lines.append(f"- {rule}: {count}")
        lines.extend(["", "### 告警明细（前10条）", ""])
        lines.append("| 时间 | 规则 | 严重级别 | 源IP | 目标IP | 证据摘要 |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in rule_alerts_df.head(10).iterrows():
            lines.append(
                f"| {row.get('timestamp', '')} | {row.get('rule_name', '')} | "
                f"{row.get('severity', '')} | {row.get('src_ip', '')} | "
                f"{row.get('dst_ip', '')} | {_evidence_summary(row.get('evidence', ''))} |"
            )
        lines.append("")

    lines.extend(["## 二、事件链概览", ""])
    if incident_df is None:
        lines.extend(["- 事件聚合尚未执行。", ""])
    elif incident_df.empty:
        lines.extend(["- 当前无聚合事件。", ""])
    else:
        lines.append("| 事件ID | 阶段 | 源IP | 风险分 | 最高级别 | 告警数 | 规则链 |")
        lines.append("|---|---|---|---:|---|---:|---|")
        for _, row in incident_df.head(10).iterrows():
            rules_text = row.get("rules", row.get("event_chain", ""))
            lines.append(
                f"| {row.get('incident_id', '')} | {_infer_incident_stage(rules_text)} | {row.get('src_ip', '')} | "
                f"{row.get('risk_score', 0)} | {row.get('max_severity', '')} | "
                f"{row.get('alert_count', 0)} | {row.get('event_chain', '')} |"
            )
        lines.append("")

    lines.extend(["## 三、模型与流量统计", ""])
    if analysis_df is None or analysis_df.empty:
        lines.extend(["- 无可用流量数据。", ""])
    else:
        lines.append(f"- 流量总数：{len(analysis_df)}")
        for p, c in _top_value_counts(analysis_df, "protocol", topn=8):
            lines.append(f"- 协议 {p}: {c}")
        if "score_iso_norm" in analysis_df.columns:
            mean_score = float(analysis_df["score_iso_norm"].fillna(0).mean())
            lines.append(f"- IsolationForest 平均异常分数：{mean_score:.4f}")
        if "score_ensemble_norm" in analysis_df.columns:
            mean_ensemble = float(analysis_df["score_ensemble_norm"].fillna(0).mean())
            lines.append(f"- 集成模型平均异常分数：{mean_ensemble:.4f}")

        score_col = "score_ensemble_norm" if "score_ensemble_norm" in analysis_df.columns else (
            "score_iso_norm" if "score_iso_norm" in analysis_df.columns else None
        )
        if score_col:
            top_anom = analysis_df.sort_values(score_col, ascending=False).head(5)
            lines.extend(["", f"### 高异常流量 Top5（按 {score_col}）", ""])
            lines.append("| 时间 | 源IP | 目标IP | 协议 | 目标端口 | 异常分数 |")
            lines.append("|---|---|---|---|---:|---:|")
            for _, row in top_anom.iterrows():
                lines.append(
                    f"| {row.get('timestamp', '')} | {row.get('src_ip', '')} | {row.get('dst_ip', '')} | "
                    f"{row.get('protocol', '')} | {row.get('dst_port', '')} | {float(row.get(score_col, 0)):.4f} |"
                )
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

    lines.extend(_build_recommendation_section(rule_alerts_df, incident_df, analysis_df))

    llm_text = (llm_assessment or "").strip()
    agent_text = (agent_report_markdown or "").strip()

    append_llm = bool(llm_text)
    if append_llm and agent_text and llm_text in agent_text:
        append_llm = False

    if append_llm:
        lines.extend(["## 六、LLM 威胁研判", "", llm_text, ""])
        if _is_local_llm_fallback(llm_text):
            lines.extend(
                [
                    "> 说明：本次为本地兜底研判（未调用外部LLM）。",
                    "> 如需真实大模型研判，请配置 DEEPSEEK_API_KEY 或在页面输入 API Key。",
                    "",
                ]
            )

    if agent_text:
        if _has_meaningful_agent_appendix(agent_text):
            lines.extend(["## 七、AI Agent 综合研判附录", "", agent_text, ""])
        else:
            lines.extend(
                [
                    "## 七、AI Agent 综合研判附录",
                    "",
                    "- 本次 Agent 附录未提供额外高价值信息（规则/事件为空），已省略详细内容。",
                    "",
                ]
            )

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

