from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rule_engine import detect_statistical_anomalies


def test_detect_statistical_anomalies_hits_traffic_burst():
    base = pd.Timestamp("2026-03-03T12:00:00")
    rows = []
    # Two baseline windows (low bytes)
    rows.append({"timestamp": base, "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "protocol": "TCP", "total_bytes": 1000})
    rows.append({"timestamp": base + pd.Timedelta(seconds=65), "src_ip": "10.0.0.1", "dst_ip": "10.0.0.3", "protocol": "TCP", "total_bytes": 1200})
    # Burst window
    rows.append({"timestamp": base + pd.Timedelta(seconds=130), "src_ip": "10.0.0.1", "dst_ip": "10.0.0.4", "protocol": "TCP", "total_bytes": 15000})

    df = pd.DataFrame(rows)
    alerts = detect_statistical_anomalies(
        df,
        window_s=60,
        burst_multiplier=3.0,
        burst_min_bytes=5000,
        beacon_min_events=10,
        beacon_max_iat_cv=0.1,
    )

    assert not alerts.empty
    assert "TRAFFIC_BURST" in alerts["rule_name"].tolist()
