from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rule_engine import detect_brute_force


def test_detect_brute_force_hits_on_repeated_auth_port_attempts():
    base = pd.Timestamp("2026-03-03T13:00:00")
    rows = []
    for i in range(12):
        rows.append(
            {
                "timestamp": base + pd.Timedelta(seconds=i * 3),
                "src_ip": "10.0.0.50",
                "dst_ip": "10.0.0.10",
                "src_port": 50000 + i,
                "dst_port": 22,
                "protocol": "TCP",
                "total_bytes": 120,
            }
        )
    df = pd.DataFrame(rows)
    alerts = detect_brute_force(df, window_s=60, attempt_threshold=10)

    assert not alerts.empty
    assert "BRUTE_FORCE" in alerts["rule_name"].tolist()
