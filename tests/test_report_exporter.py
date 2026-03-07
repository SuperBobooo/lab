from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report_exporter import build_markdown_report


def test_build_markdown_report_contains_incident_and_action_sections():
    analysis_df = pd.DataFrame(
        [{"protocol": "TCP", "score_iso_norm": 0.4}, {"protocol": "UDP", "score_iso_norm": 0.6}]
    )
    alerts_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-03T12:00:00",
                "rule_name": "PORT_SCAN",
                "severity": "medium",
                "src_ip": "10.0.0.1",
                "dst_ip": "MULTI",
                "evidence": '{"x":1}',
            }
        ]
    )
    incident_df = pd.DataFrame(
        [{"incident_id": "INC-00001", "src_ip": "10.0.0.1", "risk_score": 72, "max_severity": "high", "alert_count": 3, "event_chain": "PORT_SCAN -> BRUTE_FORCE"}]
    )
    action_df = pd.DataFrame(
        [{"timestamp": "2026-03-03T12:01:00", "action_type": "ALERT_LOG", "status": "success", "target": "data/logs", "detail": "ok"}]
    )
    pcap_df = pd.DataFrame([{"protocol": "ARP"}, {"protocol": "TCP"}])

    content = build_markdown_report(
        analysis_df=analysis_df,
        rule_alerts_df=alerts_df,
        pcap_df=pcap_df,
        incident_df=incident_df,
        action_df=action_df,
        llm_assessment="测试研判文本",
    )

    assert "SentinelAI 安全检测报告" in content
    assert "规则告警总数" in content
    assert "事件链概览" in content
    assert "响应动作审计" in content
    assert "LLM 威胁研判" in content
    assert "PORT_SCAN" in content
