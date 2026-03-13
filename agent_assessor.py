"""
AI Agent assessor for SentinelAI.

Builds a compact multi-source context, calls LLM threat assessment,
and generates a consolidated markdown report for analyst use.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from llm_threat_assessor import generate_threat_assessment


def _safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    return pd.DataFrame()


def _top_value_counts(df: pd.DataFrame, col: str, topn: int = 10) -> list[dict[str, Any]]:
    if df.empty or col not in df.columns:
        return []
    vc = df[col].astype(str).value_counts().head(topn)
    return [{"value": str(k), "count": int(v)} for k, v in vc.items()]


def _infer_stage_from_rules(rules_text: str) -> str:
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


def build_agent_context(
    analysis_df: Optional[pd.DataFrame],
    pcap_df: Optional[pd.DataFrame],
    rule_alerts_df: Optional[pd.DataFrame],
    incident_df: Optional[pd.DataFrame],
    action_df: Optional[pd.DataFrame],
    anomalies_df: Optional[pd.DataFrame] = None,
    topn: int = 20,
) -> Dict[str, Any]:
    """Build structured context for the LLM agent."""
    flows = _safe_df(analysis_df)
    packets = _safe_df(pcap_df)
    alerts = _safe_df(rule_alerts_df)
    incidents = _safe_df(incident_df)
    actions = _safe_df(action_df)
    anomalies = _safe_df(anomalies_df)

    protocol_counts = []
    if not packets.empty and "protocol" in packets.columns:
        protocol_counts = _top_value_counts(packets, "protocol", topn=10)
    elif not flows.empty and "protocol" in flows.columns:
        protocol_counts = _top_value_counts(flows, "protocol", topn=10)

    top_rules = _top_value_counts(alerts, "rule_name", topn=10)
    top_alert_src = _top_value_counts(alerts, "src_ip", topn=10)

    top_incidents: list[dict[str, Any]] = []
    if not incidents.empty:
        inc = incidents.copy()
        if "risk_score" in inc.columns:
            inc = inc.sort_values("risk_score", ascending=False)
        inc = inc.head(topn)
        for _, r in inc.iterrows():
            rules_text = r.get("rules", r.get("event_chain", ""))
            top_incidents.append(
                {
                    "incident_id": str(r.get("incident_id", "")),
                    "src_ip": str(r.get("src_ip", "")),
                    "risk_score": float(r.get("risk_score", 0)),
                    "max_severity": str(r.get("max_severity", "")),
                    "stage": str(r.get("stage", _infer_stage_from_rules(rules_text))),
                    "alert_count": int(r.get("alert_count", 0)),
                    "rules": str(rules_text),
                }
            )

    top_anomalies: list[dict[str, Any]] = []
    if not anomalies.empty:
        score_col = None
        for c in ["anomaly_score", "score_ensemble_norm", "score_iso_norm", "score_ocsvm_norm", "score_dbscan_norm"]:
            if c in anomalies.columns:
                score_col = c
                break
        an = anomalies.copy()
        if score_col:
            an = an.sort_values(score_col, ascending=False)
        an = an.head(topn)
        for _, r in an.iterrows():
            top_anomalies.append(
                {
                    "flow_id": r.get("flow_id"),
                    "timestamp": str(r.get("timestamp", "")),
                    "src_ip": str(r.get("src_ip", "")),
                    "dst_ip": str(r.get("dst_ip", "")),
                    "protocol": str(r.get("protocol", "")),
                    "dst_port": r.get("dst_port"),
                    "score": float(r.get(score_col, 0)) if score_col else 0.0,
                }
            )

    return {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "flow_count": int(len(flows)),
            "packet_count": int(len(packets)),
            "alert_count": int(len(alerts)),
            "incident_count": int(len(incidents)),
            "anomaly_count": int(len(anomalies)),
            "action_count": int(len(actions)),
        },
        "protocol_distribution": protocol_counts,
        "top_rules": top_rules,
        "top_alert_sources": top_alert_src,
        "top_incidents": top_incidents,
        "top_anomalies": top_anomalies,
    }


def build_agent_markdown_report(context: Dict[str, Any], llm_assessment: str) -> str:
    """Build final markdown report shown in Agent tab."""
    s = context.get("summary", {})
    lines = [
        "# AI 智能研判 Agent 报告",
        "",
        f"- 生成时间：{context.get('generated_at', '')}",
        f"- 报文数：{s.get('packet_count', 0)}",
        f"- 流量数：{s.get('flow_count', 0)}",
        f"- 规则告警数：{s.get('alert_count', 0)}",
        f"- 事件数：{s.get('incident_count', 0)}",
        f"- 异常流量数：{s.get('anomaly_count', 0)}",
        "",
        "## 一、机器摘要",
        "",
        "### 协议分布 Top",
    ]
    for item in context.get("protocol_distribution", [])[:8]:
        lines.append(f"- {item['value']}: {item['count']}")

    lines.extend(["", "### 规则命中 Top"])
    for item in context.get("top_rules", [])[:8]:
        lines.append(f"- {item['value']}: {item['count']}")

    lines.extend(["", "### 事件摘要 Top"])
    for item in context.get("top_incidents", [])[:8]:
        lines.append(
            f"- {item.get('incident_id')} | src={item.get('src_ip')} | risk={item.get('risk_score')} | "
            f"stage={item.get('stage')} | rules={item.get('rules')}"
        )

    lines.extend(["", "## 二、LLM 综合研判", "", llm_assessment.strip() if llm_assessment else "无可用LLM研判输出。", ""])
    return "\n".join(lines)


def run_agent_assessment(
    analysis_df: Optional[pd.DataFrame],
    pcap_df: Optional[pd.DataFrame],
    rule_alerts_df: Optional[pd.DataFrame],
    incident_df: Optional[pd.DataFrame],
    action_df: Optional[pd.DataFrame],
    anomalies_df: Optional[pd.DataFrame] = None,
    model: str = "auto",
    provider: str = "auto",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: int = 60,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """End-to-end agent run: context -> LLM text -> markdown report."""
    context = build_agent_context(
        analysis_df=analysis_df,
        pcap_df=pcap_df,
        rule_alerts_df=rule_alerts_df,
        incident_df=incident_df,
        action_df=action_df,
        anomalies_df=anomalies_df,
    )

    alerts_df = _safe_df(rule_alerts_df)
    anomalies = _safe_df(anomalies_df)
    llm_assessment = generate_threat_assessment(
        alerts_df=alerts_df,
        anomalies_df=anomalies,
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
    report_markdown = build_agent_markdown_report(context, llm_assessment)
    return {
        "context": context,
        "llm_assessment": llm_assessment,
        "report_markdown": report_markdown,
    }
