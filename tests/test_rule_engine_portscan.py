from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rule_engine import detect_port_scan


def test_detect_port_scan_hits_when_unique_ports_exceed_threshold():
    base_ts = pd.Timestamp("2026-03-02T10:00:00")
    rows = []
    for i in range(6):
        rows.append(
            {
                "timestamp": base_ts + pd.Timedelta(seconds=i * 5),
                "src_ip": "10.0.0.10",
                "dst_ip": "10.0.0.20",
                "src_port": 40000 + i,
                "dst_port": 1000 + i,
                "protocol": "TCP",
            }
        )

    flows_df = pd.DataFrame(rows)
    alerts_df = detect_port_scan(flows_df, window_s=60, unique_port_threshold=5)

    assert len(alerts_df) == 1
    assert alerts_df.iloc[0]["rule_name"] == "PORT_SCAN"
    assert alerts_df.iloc[0]["src_ip"] == "10.0.0.10"
    assert int(alerts_df.iloc[0]["unique_ports"]) == 6


def test_detect_port_scan_accepts_start_time_when_timestamp_missing():
    base_ts = pd.Timestamp("2026-03-02T11:00:00")
    rows = []
    for i in range(7):
        rows.append(
            {
                "start_time": base_ts + pd.Timedelta(seconds=i * 3),
                "src_ip": "10.0.0.99",
                "dst_ip": "10.0.0.77",
                "src_port": 50000 + i,
                "dst_port": 2000 + i,
                "protocol": "TCP",
            }
        )

    flows_df = pd.DataFrame(rows)
    alerts_df = detect_port_scan(flows_df, window_s=60, unique_port_threshold=5)

    assert len(alerts_df) == 1
    assert alerts_df.iloc[0]["rule_name"] == "PORT_SCAN"
    assert alerts_df.iloc[0]["src_ip"] == "10.0.0.99"
    assert int(alerts_df.iloc[0]["unique_ports"]) == 7
