from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_assessor import build_agent_context, run_agent_assessment


def test_agent_context_and_report_generation_local_fallback():
    flows_df = pd.DataFrame(
        [
            {"flow_id": 1, "timestamp": "2026-03-08T10:00:00", "src_ip": "10.0.0.1", "dst_ip": "8.8.8.8", "protocol": "UDP", "score_iso_norm": 0.91},
            {"flow_id": 2, "timestamp": "2026-03-08T10:00:01", "src_ip": "10.0.0.2", "dst_ip": "1.1.1.1", "protocol": "TCP", "score_iso_norm": 0.12},
        ]
    )
    packets_df = pd.DataFrame([{"protocol": "TCP"}, {"protocol": "UDP"}, {"protocol": "ARP"}])
    alerts_df = pd.DataFrame(
        [
            {"timestamp": "2026-03-08T10:00:10", "rule_name": "PORT_SCAN", "severity": "high", "src_ip": "10.0.0.1", "dst_ip": "MULTI", "evidence": "{}"}
        ]
    )
    incident_df = pd.DataFrame(
        [
            {
                "incident_id": "INC-00001",
                "src_ip": "10.0.0.1",
                "risk_score": 88,
                "max_severity": "high",
                "alert_count": 1,
                "rules": "PORT_SCAN",
                "event_chain": "PORT_SCAN",
            }
        ]
    )

    ctx = build_agent_context(flows_df, packets_df, alerts_df, incident_df, action_df=None, anomalies_df=flows_df.head(1))
    assert ctx["summary"]["flow_count"] == 2
    assert ctx["summary"]["packet_count"] == 3
    assert len(ctx["top_rules"]) == 1

    result = run_agent_assessment(
        analysis_df=flows_df,
        pcap_df=packets_df,
        rule_alerts_df=alerts_df,
        incident_df=incident_df,
        action_df=None,
        anomalies_df=flows_df.head(1),
        provider="auto",
        api_key=None,
    )
    assert "AI 智能研判 Agent 报告" in result["report_markdown"]
    assert "本地研判" in result["llm_assessment"]

