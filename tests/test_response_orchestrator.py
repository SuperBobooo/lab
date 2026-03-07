import pandas as pd

from response_orchestrator import execute_responses


def test_execute_responses_logs_audit_file(tmp_path):
    alerts_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-03T10:00:00",
                "rule_name": "PORT_SCAN",
                "severity": "high",
                "src_ip": "10.0.0.7",
                "dst_ip": "192.168.1.10",
            }
        ]
    )

    action_df = execute_responses(
        alerts_df=alerts_df,
        log_all_alerts=True,
        enable_webhook=False,
        enable_email=False,
        output_dir=str(tmp_path),
    )

    assert not action_df.empty
    assert (action_df["action_type"] == "ALERT_LOG").any()

    files = list(tmp_path.glob("response_audit_*.jsonl"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "PORT_SCAN" in content
