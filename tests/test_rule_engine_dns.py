from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rule_engine import detect_anomalous_dns


def test_detect_anomalous_dns_hits_on_high_unique_queries():
    base_ts = pd.Timestamp("2026-03-02T12:00:00")
    rows = []
    for i in range(6):
        rows.append(
            {
                "timestamp": base_ts + pd.Timedelta(seconds=i * 5),
                "src_ip": "192.168.1.10",
                "dst_ip": "8.8.8.8",
                "dst_port": 53,
                "l7_protocol": "DNS",
                "dns_query": f"sub{i}.example.com",
                "dns_rcode": 0,
            }
        )

    df = pd.DataFrame(rows)
    alerts_df = detect_anomalous_dns(
        df,
        window_s=60,
        unique_query_threshold=5,
        nxdomain_threshold=10
    )

    assert len(alerts_df) == 1
    assert alerts_df.iloc[0]["rule_name"] == "ANOMALOUS_DNS"
    assert alerts_df.iloc[0]["src_ip"] == "192.168.1.10"
