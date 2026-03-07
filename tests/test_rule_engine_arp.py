from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rule_engine import detect_arp_anomalies


def test_detect_arp_spoof_hits_when_same_ip_has_multiple_macs():
    base = pd.Timestamp("2026-03-03T10:00:00")
    df = pd.DataFrame(
        [
            {
                "timestamp": base,
                "protocol": "ARP",
                "src_ip": "192.168.0.10",
                "src_mac": "00:11:22:33:44:55",
                "dst_ip": "192.168.0.1",
            },
            {
                "timestamp": base + pd.Timedelta(seconds=10),
                "protocol": "ARP",
                "src_ip": "192.168.0.10",
                "src_mac": "00:11:22:33:44:66",
                "dst_ip": "192.168.0.1",
            },
        ]
    )

    alerts = detect_arp_anomalies(
        df,
        window_s=60,
        arp_flood_threshold=100,
        arp_scan_target_threshold=100,
        arp_spoof_mac_threshold=2,
    )

    assert not alerts.empty
    assert "ARP_SPOOF" in alerts["rule_name"].tolist()
