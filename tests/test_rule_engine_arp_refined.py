from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rule_engine import detect_arp_anomalies


def test_detect_arp_mitm_suspect_hits_for_multi_ip_claim_by_same_mac():
    base = pd.Timestamp("2026-03-03T14:00:00")
    src_mac = "00:11:22:33:44:55"
    rows = [
        {"timestamp": base, "protocol": "ARP", "src_ip": "192.168.0.1", "src_mac": src_mac, "dst_ip": "192.168.0.10"},
        {"timestamp": base + pd.Timedelta(seconds=5), "protocol": "ARP", "src_ip": "192.168.0.2", "src_mac": src_mac, "dst_ip": "192.168.0.11"},
        {"timestamp": base + pd.Timedelta(seconds=10), "protocol": "ARP", "src_ip": "192.168.0.3", "src_mac": src_mac, "dst_ip": "192.168.0.12"},
    ]
    df = pd.DataFrame(rows)
    alerts = detect_arp_anomalies(
        df,
        window_s=60,
        arp_flood_threshold=100,
        arp_scan_target_threshold=100,
        arp_spoof_mac_threshold=10,
        arp_mitm_ip_claim_threshold=2,
        arp_abuse_gratuitous_threshold=100,
    )

    assert not alerts.empty
    assert "ARP_MITM_SUSPECT" in alerts["rule_name"].tolist()
