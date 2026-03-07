import json
from pathlib import Path

import pandas as pd

from log_center import persist_snapshot


def test_persist_snapshot_writes_jsonl_and_manifest(tmp_path: Path):
    packet_df = pd.DataFrame([{"timestamp": "2026-03-03T10:00:00", "protocol": "TCP"}])
    flow_df = pd.DataFrame([{"flow_id": 1, "src_ip": "10.0.0.1"}])
    alert_df = pd.DataFrame([{"rule_name": "PORT_SCAN", "severity": "high"}])
    incident_df = pd.DataFrame([{"incident_id": "INC-00001", "risk_score": 78.0}])
    action_df = pd.DataFrame([{"action_type": "ALERT_LOG", "status": "success"}])

    paths = persist_snapshot(
        output_root=str(tmp_path),
        source="test",
        packet_df=packet_df,
        flow_df=flow_df,
        alert_df=alert_df,
        incident_df=incident_df,
        action_df=action_df,
    )

    assert "manifest" in paths
    for key in ["packet", "flow", "alert", "incident", "action"]:
        assert key in paths
        assert Path(paths[key]).exists()

    manifest_lines = Path(paths["manifest"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest_lines) >= 1
    row = json.loads(manifest_lines[-1])
    assert row["source"] == "test"
