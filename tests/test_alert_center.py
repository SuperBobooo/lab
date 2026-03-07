import pandas as pd

from alert_center import build_incidents


def test_build_incidents_groups_and_splits_by_time_window():
    alerts_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-03T10:00:00",
                "rule_name": "PORT_SCAN",
                "severity": "medium",
                "src_ip": "10.0.0.5",
                "dst_ip": "192.168.1.1",
            },
            {
                "timestamp": "2026-03-03T10:00:20",
                "rule_name": "ANOMALOUS_DNS",
                "severity": "high",
                "src_ip": "10.0.0.5",
                "dst_ip": "8.8.8.8",
            },
            {
                "timestamp": "2026-03-03T10:10:00",
                "rule_name": "BRUTE_FORCE",
                "severity": "medium",
                "src_ip": "10.0.0.5",
                "dst_ip": "192.168.1.10",
            },
        ]
    )

    incidents = build_incidents(alerts_df, correlation_window_s=60)

    assert len(incidents) == 2
    assert set(incidents.columns) >= {
        "incident_id",
        "start_time",
        "end_time",
        "duration_s",
        "src_ip",
        "alert_count",
        "risk_score",
        "event_chain",
    }
    assert incidents["alert_count"].max() == 2
