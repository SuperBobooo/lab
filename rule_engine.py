"""
Rule engine for SentinelAI.

Current rules:
- PORT_SCAN: unique destination ports per source IP in a time window.
- ANOMALOUS_DNS: suspicious DNS burst/high query diversity/high NXDOMAIN rate.
- BRUTE_FORCE: repeated auth-service attempts from source to target.
- ARP_MITM_SUSPECT / ARP_ABUSE: refined ARP behavior detections.
"""

from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd

ALERT_COLUMNS = [
    "timestamp",
    "rule_name",
    "severity",
    "src_ip",
    "dst_ip",
    "unique_ports",
    "ports_sample",
    "evidence",
]


def _is_ephemeral_port(port: int) -> bool:
    """Return True if port is likely an ephemeral/client-side port."""
    return int(port) >= 49152


def _severity_from_unique_ports(unique_ports: int, threshold: int) -> str:
    """Map unique port count to alert severity."""
    if unique_ports >= threshold * 3:
        return "high"
    if unique_ports >= threshold * 2:
        return "medium"
    return "low"


def _resolve_time_column(df: pd.DataFrame) -> str:
    """Resolve usable time column from dataframe."""
    if "timestamp" in df.columns:
        return "timestamp"
    if "start_time" in df.columns:
        return "start_time"
    raise ValueError("flows_df missing required time column: need 'timestamp' or 'start_time'")


def detect_port_scan(
    flows_df: pd.DataFrame,
    window_s: int = 60,
    unique_port_threshold: int = 20
) -> pd.DataFrame:
    """
    Detect potential port scans from flow data.

    Rule:
    - For each src_ip, within each time bucket of `window_s`, if number of
      distinct dst_port values is greater than `unique_port_threshold`,
      emit one PORT_SCAN alert.
    """
    if flows_df is None or flows_df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    time_col = _resolve_time_column(flows_df)

    required = {"src_ip", "dst_port"}
    missing = required - set(flows_df.columns)
    if missing:
        raise ValueError(f"flows_df missing required columns for port-scan rule: {sorted(missing)}")

    df = flows_df.copy()
    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "src_ip", "dst_port"]).copy()
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    df["src_ip"] = df["src_ip"].astype(str)
    df["dst_port"] = pd.to_numeric(df["dst_port"], errors="coerce")
    df = df.dropna(subset=["dst_port"]).copy()
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    df["dst_port"] = df["dst_port"].astype(int)
    # Port-scan rule focuses on TCP/UDP probe-like flows.
    if "protocol" in df.columns:
        df = df[df["protocol"].astype(str).str.upper().isin(["TCP", "UDP"])].copy()
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    # Drop invalid destination ports.
    df = df[(df["dst_port"] > 0) & (df["dst_port"] <= 65535)].copy()
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    # Reduce common reverse-direction false positives:
    # server-response style flow often appears as src_port well-known, dst_port ephemeral.
    if "src_port" in df.columns:
        df["src_port_num"] = pd.to_numeric(df["src_port"], errors="coerce")
        server_response_like = (
            df["src_port_num"].notna()
            & (df["src_port_num"] <= 1024)
            & df["dst_port"].map(_is_ephemeral_port)
        )
        df = df[~server_response_like].copy()
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    df["dst_ip"] = df["dst_ip"].astype(str) if "dst_ip" in df.columns else "MULTI"

    window = f"{int(window_s)}s"
    df["window_start"] = df["timestamp"].dt.floor(window)
    df["window_end"] = df["window_start"] + pd.to_timedelta(window_s, unit="s")

    alerts: List[dict] = []
    grouped = df.groupby(["src_ip", "window_start", "window_end"], dropna=False, sort=True)
    for (src_ip, w_start, w_end), group in grouped:
        ports = sorted(group["dst_port"].dropna().unique().tolist())
        unique_ports = len(ports)
        if unique_ports <= int(unique_port_threshold):
            continue

        dst_ips = sorted(group["dst_ip"].dropna().unique().tolist())
        dst_ip_val = dst_ips[0] if len(dst_ips) == 1 else "MULTI"
        ports_sample = ports[:20]

        evidence = {
            "window_start": w_start.isoformat(),
            "window_end": w_end.isoformat(),
            "flow_count": int(len(group)),
            "unique_ports": int(unique_ports),
            "unique_dst_ips": int(len(dst_ips)),
        }

        alerts.append(
            {
                "timestamp": w_end.isoformat(),
                "rule_name": "PORT_SCAN",
                "severity": _severity_from_unique_ports(unique_ports, int(unique_port_threshold)),
                "src_ip": src_ip,
                "dst_ip": dst_ip_val,
                "unique_ports": int(unique_ports),
                "ports_sample": ",".join(str(p) for p in ports_sample),
                "evidence": json.dumps(evidence, ensure_ascii=False),
            }
        )

    if not alerts:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    alerts_df = pd.DataFrame(alerts).sort_values(
        by=["unique_ports", "timestamp"], ascending=[False, False]
    ).reset_index(drop=True)
    return alerts_df[ALERT_COLUMNS]


def detect_anomalous_dns(
    flows_df: pd.DataFrame,
    window_s: int = 60,
    unique_query_threshold: int = 20,
    nxdomain_threshold: int = 10
) -> pd.DataFrame:
    """
    Detect anomalous DNS behaviors in time windows per source IP.

    Trigger when at least one condition is met within (src_ip, window):
    - distinct dns_query count > unique_query_threshold
    - NXDOMAIN count > nxdomain_threshold (dns_rcode == 3)
    - dns flow/request count > 2 * unique_query_threshold (burst fallback)
    """
    if flows_df is None or flows_df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    time_col = _resolve_time_column(flows_df)

    required = {"src_ip"}
    missing = required - set(flows_df.columns)
    if missing:
        raise ValueError(f"flows_df missing required columns for dns rule: {sorted(missing)}")

    df = flows_df.copy()
    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "src_ip"]).copy()
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    df["src_ip"] = df["src_ip"].astype(str)
    df["dst_ip"] = df["dst_ip"].astype(str) if "dst_ip" in df.columns else "MULTI"

    # DNS filtering:
    # 1) Prefer explicit l7_protocol == DNS
    # 2) Fallback dst_port == 53
    is_dns = pd.Series(False, index=df.index)
    if "l7_protocol" in df.columns:
        is_dns = is_dns | (df["l7_protocol"].astype(str).str.upper() == "DNS")
    if "dst_port" in df.columns:
        dst_port_num = pd.to_numeric(df["dst_port"], errors="coerce")
        is_dns = is_dns | (dst_port_num == 53)
    df = df[is_dns].copy()
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    if "dns_query" not in df.columns:
        df["dns_query"] = None
    if "dns_rcode" not in df.columns:
        df["dns_rcode"] = None

    df["dns_query_norm"] = (
        df["dns_query"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df["dns_rcode_num"] = pd.to_numeric(df["dns_rcode"], errors="coerce")

    window = f"{int(window_s)}s"
    df["window_start"] = df["timestamp"].dt.floor(window)
    df["window_end"] = df["window_start"] + pd.to_timedelta(window_s, unit="s")

    alerts: List[dict] = []
    grouped = df.groupby(["src_ip", "window_start", "window_end"], dropna=False, sort=True)
    for (src_ip, w_start, w_end), group in grouped:
        flow_count = int(len(group))
        unique_queries = int((group["dns_query_norm"] != "").sum())
        if unique_queries > 0:
            unique_queries = int(group.loc[group["dns_query_norm"] != "", "dns_query_norm"].nunique())

        nxdomain_count = int((group["dns_rcode_num"] == 3).sum())
        burst_threshold = int(unique_query_threshold) * 2

        hit = (
            unique_queries > int(unique_query_threshold)
            or nxdomain_count > int(nxdomain_threshold)
            or flow_count > burst_threshold
        )
        if not hit:
            continue

        severity = "low"
        if unique_queries >= int(unique_query_threshold) * 2 or nxdomain_count >= int(nxdomain_threshold) * 2:
            severity = "high"
        elif unique_queries >= int(unique_query_threshold) or nxdomain_count >= int(nxdomain_threshold):
            severity = "medium"

        q_sample = (
            group.loc[group["dns_query_norm"] != "", "dns_query_norm"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        evidence = {
            "window_start": w_start.isoformat(),
            "window_end": w_end.isoformat(),
            "dns_flow_count": flow_count,
            "unique_queries": unique_queries,
            "nxdomain_count": nxdomain_count,
        }
        alerts.append(
            {
                "timestamp": w_end.isoformat(),
                "rule_name": "ANOMALOUS_DNS",
                "severity": severity,
                "src_ip": src_ip,
                "dst_ip": "MULTI",
                "unique_ports": unique_queries,
                "ports_sample": ",".join(q_sample),
                "evidence": json.dumps(evidence, ensure_ascii=False),
            }
        )

    if not alerts:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    alerts_df = pd.DataFrame(alerts).sort_values(
        by=["timestamp", "severity"], ascending=[False, True]
    ).reset_index(drop=True)
    return alerts_df[ALERT_COLUMNS]


def detect_arp_anomalies(
    packets_df: pd.DataFrame,
    window_s: int = 60,
    arp_flood_threshold: int = 50,
    arp_scan_target_threshold: int = 20,
    arp_spoof_mac_threshold: int = 2,
    arp_mitm_ip_claim_threshold: int = 2,
    arp_abuse_gratuitous_threshold: int = 20,
) -> pd.DataFrame:
    """
    Detect ARP-specific anomalies from packet-level data.

    Rules (per src_ip/src_mac within each window):
    - ARP_FLOOD: packet count > arp_flood_threshold
    - ARP_SCAN: unique destination IP count > arp_scan_target_threshold
    - ARP_SPOOF: same src_ip observed with >= arp_spoof_mac_threshold MACs
    - ARP_MITM_SUSPECT: same MAC claims multiple source IPs and targets multiple peers
    - ARP_ABUSE: excessive gratuitous ARP pattern (src_ip == dst_ip)
    """
    if packets_df is None or packets_df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    time_col = _resolve_time_column(packets_df)
    required = {"src_ip", "src_mac", "dst_ip", "protocol"}
    missing = required - set(packets_df.columns)
    if missing:
        raise ValueError(f"packets_df missing required columns for arp rules: {sorted(missing)}")

    df = packets_df.copy()
    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "src_ip", "src_mac", "dst_ip"]).copy()
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    df = df[df["protocol"].astype(str).str.upper() == "ARP"].copy()
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    df["src_ip"] = df["src_ip"].astype(str)
    df["src_mac"] = df["src_mac"].astype(str).str.lower()
    df["dst_ip"] = df["dst_ip"].astype(str)

    window = f"{int(window_s)}s"
    df["window_start"] = df["timestamp"].dt.floor(window)
    df["window_end"] = df["window_start"] + pd.to_timedelta(window_s, unit="s")

    alerts: List[dict] = []

    # ARP_FLOOD + ARP_SCAN
    grouped_src = df.groupby(["src_ip", "src_mac", "window_start", "window_end"], dropna=False, sort=True)
    for (src_ip, src_mac, w_start, w_end), group in grouped_src:
        pkt_count = int(len(group))
        unique_targets = int(group["dst_ip"].nunique())
        target_sample = group["dst_ip"].drop_duplicates().head(20).tolist()

        if pkt_count > int(arp_flood_threshold):
            evidence = {
                "window_start": w_start.isoformat(),
                "window_end": w_end.isoformat(),
                "src_mac": src_mac,
                "arp_packet_count": pkt_count,
                "unique_target_ips": unique_targets,
            }
            alerts.append(
                {
                    "timestamp": w_end.isoformat(),
                    "rule_name": "ARP_FLOOD",
                    "severity": "high" if pkt_count >= int(arp_flood_threshold) * 2 else "medium",
                    "src_ip": src_ip,
                    "dst_ip": "MULTI",
                    "unique_ports": pkt_count,
                    "ports_sample": ",".join(target_sample),
                    "evidence": json.dumps(evidence, ensure_ascii=False),
                }
            )

        if unique_targets > int(arp_scan_target_threshold):
            evidence = {
                "window_start": w_start.isoformat(),
                "window_end": w_end.isoformat(),
                "src_mac": src_mac,
                "arp_packet_count": pkt_count,
                "unique_target_ips": unique_targets,
            }
            alerts.append(
                {
                    "timestamp": w_end.isoformat(),
                    "rule_name": "ARP_SCAN",
                    "severity": "high" if unique_targets >= int(arp_scan_target_threshold) * 2 else "medium",
                    "src_ip": src_ip,
                    "dst_ip": "MULTI",
                    "unique_ports": unique_targets,
                    "ports_sample": ",".join(target_sample),
                    "evidence": json.dumps(evidence, ensure_ascii=False),
                }
            )

        gratuitous_count = int((group["src_ip"] == group["dst_ip"]).sum())
        if gratuitous_count >= int(arp_abuse_gratuitous_threshold):
            evidence = {
                "window_start": w_start.isoformat(),
                "window_end": w_end.isoformat(),
                "src_mac": src_mac,
                "gratuitous_arp_count": gratuitous_count,
                "arp_packet_count": pkt_count,
            }
            alerts.append(
                {
                    "timestamp": w_end.isoformat(),
                    "rule_name": "ARP_ABUSE",
                    "severity": "high" if gratuitous_count >= int(arp_abuse_gratuitous_threshold) * 2 else "medium",
                    "src_ip": src_ip,
                    "dst_ip": "MULTI",
                    "unique_ports": gratuitous_count,
                    "ports_sample": ",".join(target_sample),
                    "evidence": json.dumps(evidence, ensure_ascii=False),
                }
            )

    # ARP_SPOOF: same src_ip with many distinct src_mac in same window
    grouped_spoof = df.groupby(["src_ip", "window_start", "window_end"], dropna=False, sort=True)
    for (src_ip, w_start, w_end), group in grouped_spoof:
        macs = sorted(group["src_mac"].drop_duplicates().tolist())
        if len(macs) < int(arp_spoof_mac_threshold):
            continue

        evidence = {
            "window_start": w_start.isoformat(),
            "window_end": w_end.isoformat(),
            "observed_macs": macs,
            "mac_count": len(macs),
            "arp_packet_count": int(len(group)),
        }
        alerts.append(
            {
                "timestamp": w_end.isoformat(),
                "rule_name": "ARP_SPOOF",
                "severity": "high" if len(macs) > int(arp_spoof_mac_threshold) else "medium",
                "src_ip": src_ip,
                "dst_ip": "MULTI",
                "unique_ports": len(macs),
                "ports_sample": ",".join(macs[:20]),
                "evidence": json.dumps(evidence, ensure_ascii=False),
            }
        )

    # ARP_MITM_SUSPECT: same src_mac claims multiple src_ip and targets multiple hosts in same window.
    grouped_mitm = df.groupby(["src_mac", "window_start", "window_end"], dropna=False, sort=True)
    for (src_mac, w_start, w_end), group in grouped_mitm:
        claimed_ips = sorted(group["src_ip"].drop_duplicates().tolist())
        target_ips = sorted(group["dst_ip"].drop_duplicates().tolist())
        if len(claimed_ips) < int(arp_mitm_ip_claim_threshold):
            continue
        if len(target_ips) < 2:
            continue

        evidence = {
            "window_start": w_start.isoformat(),
            "window_end": w_end.isoformat(),
            "src_mac": src_mac,
            "claimed_src_ips": claimed_ips,
            "target_ips_count": len(target_ips),
            "arp_packet_count": int(len(group)),
        }
        alerts.append(
            {
                "timestamp": w_end.isoformat(),
                "rule_name": "ARP_MITM_SUSPECT",
                "severity": "high" if len(claimed_ips) >= int(arp_mitm_ip_claim_threshold) + 1 else "medium",
                "src_ip": claimed_ips[0] if claimed_ips else "UNKNOWN",
                "dst_ip": "MULTI",
                "unique_ports": len(claimed_ips),
                "ports_sample": ",".join(claimed_ips[:20]),
                "evidence": json.dumps(evidence, ensure_ascii=False),
            }
        )

    if not alerts:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    alerts_df = pd.DataFrame(alerts).sort_values(
        by=["timestamp", "rule_name", "severity"], ascending=[False, True, True]
    ).reset_index(drop=True)
    return alerts_df[ALERT_COLUMNS]


def detect_brute_force(
    flows_df: pd.DataFrame,
    window_s: int = 60,
    attempt_threshold: int = 15,
    auth_ports: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Detect brute-force style repeated attempts on authentication service ports.
    """
    if flows_df is None or flows_df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    time_col = _resolve_time_column(flows_df)
    required = {"src_ip", "dst_ip", "dst_port"}
    missing = required - set(flows_df.columns)
    if missing:
        raise ValueError(f"flows_df missing required columns for brute-force rule: {sorted(missing)}")

    if auth_ports is None:
        auth_ports = [21, 22, 23, 25, 110, 143, 445, 3389, 5900]

    df = flows_df.copy()
    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "src_ip", "dst_ip", "dst_port"]).copy()
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    df["src_ip"] = df["src_ip"].astype(str)
    df["dst_ip"] = df["dst_ip"].astype(str)
    df["dst_port"] = pd.to_numeric(df["dst_port"], errors="coerce")
    df = df.dropna(subset=["dst_port"]).copy()
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    df["dst_port"] = df["dst_port"].astype(int)

    df = df[df["dst_port"].isin(auth_ports)].copy()
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    window = f"{int(window_s)}s"
    df["window_start"] = df["timestamp"].dt.floor(window)
    df["window_end"] = df["window_start"] + pd.to_timedelta(window_s, unit="s")

    alerts: List[dict] = []
    grouped = df.groupby(["src_ip", "dst_ip", "dst_port", "window_start", "window_end"], dropna=False, sort=True)
    for (src_ip, dst_ip, dst_port, w_start, w_end), group in grouped:
        attempts = int(len(group))
        if attempts <= int(attempt_threshold):
            continue

        unique_src_ports = int(group["src_port"].nunique()) if "src_port" in group.columns else 0
        evidence = {
            "window_start": w_start.isoformat(),
            "window_end": w_end.isoformat(),
            "dst_port": int(dst_port),
            "attempt_count": attempts,
            "unique_src_ports": unique_src_ports,
        }
        severity = "medium"
        if attempts >= int(attempt_threshold) * 2:
            severity = "high"

        alerts.append(
            {
                "timestamp": w_end.isoformat(),
                "rule_name": "BRUTE_FORCE",
                "severity": severity,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "unique_ports": attempts,
                "ports_sample": str(dst_port),
                "evidence": json.dumps(evidence, ensure_ascii=False),
            }
        )

    if not alerts:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    alerts_df = pd.DataFrame(alerts).sort_values(
        by=["timestamp", "unique_ports"], ascending=[False, False]
    ).reset_index(drop=True)
    return alerts_df[ALERT_COLUMNS]


def detect_statistical_anomalies(
    flows_df: pd.DataFrame,
    window_s: int = 60,
    burst_multiplier: float = 3.0,
    burst_min_bytes: int = 50000,
    beacon_min_events: int = 5,
    beacon_max_iat_cv: float = 0.2
) -> pd.DataFrame:
    """
    Detect statistical anomalies from flow-level data.

    Rules:
    - TRAFFIC_BURST: per src_ip window bytes significantly exceed historical baseline.
    - PERIODIC_BEACON: low-variance inter-arrival (IAT) pattern on src->dst->protocol.
    """
    if flows_df is None or flows_df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    time_col = _resolve_time_column(flows_df)
    required = {"src_ip", "dst_ip", "protocol", "total_bytes"}
    missing = required - set(flows_df.columns)
    if missing:
        raise ValueError(f"flows_df missing required columns for statistical rules: {sorted(missing)}")

    df = flows_df.copy()
    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "src_ip", "dst_ip", "protocol"]).copy()
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    df["src_ip"] = df["src_ip"].astype(str)
    df["dst_ip"] = df["dst_ip"].astype(str)
    df["protocol"] = df["protocol"].astype(str)
    df["total_bytes"] = pd.to_numeric(df["total_bytes"], errors="coerce").fillna(0.0)

    window = f"{int(window_s)}s"
    df["window_start"] = df["timestamp"].dt.floor(window)
    df["window_end"] = df["window_start"] + pd.to_timedelta(window_s, unit="s")

    alerts: List[dict] = []

    # Rule 1: TRAFFIC_BURST
    src_window = (
        df.groupby(["src_ip", "window_start", "window_end"], as_index=False)
        .agg(window_total_bytes=("total_bytes", "sum"), flow_count=("src_ip", "count"))
        .sort_values(["src_ip", "window_start"])
    )
    for src_ip, group in src_window.groupby("src_ip", sort=False):
        historical: List[float] = []
        for _, row in group.iterrows():
            cur_bytes = float(row["window_total_bytes"])
            baseline = float(pd.Series(historical).median()) if historical else 0.0
            if (
                baseline > 0
                and cur_bytes >= float(burst_min_bytes)
                and cur_bytes > baseline * float(burst_multiplier)
            ):
                severity = "medium"
                if cur_bytes > baseline * (float(burst_multiplier) * 2):
                    severity = "high"
                evidence = {
                    "window_start": row["window_start"].isoformat(),
                    "window_end": row["window_end"].isoformat(),
                    "window_total_bytes": int(cur_bytes),
                    "baseline_bytes_median": int(baseline),
                    "burst_multiplier": float(burst_multiplier),
                }
                alerts.append(
                    {
                        "timestamp": row["window_end"].isoformat(),
                        "rule_name": "TRAFFIC_BURST",
                        "severity": severity,
                        "src_ip": src_ip,
                        "dst_ip": "MULTI",
                        "unique_ports": int(row["flow_count"]),
                        "ports_sample": "",
                        "evidence": json.dumps(evidence, ensure_ascii=False),
                    }
                )
            historical.append(cur_bytes)

    # Rule 2: PERIODIC_BEACON
    # Detect stable inter-arrival for repeated src->dst->protocol communications.
    comm = df.sort_values("timestamp").groupby(["src_ip", "dst_ip", "protocol"], sort=False)
    for (src_ip, dst_ip, proto), group in comm:
        if len(group) < int(beacon_min_events):
            continue
        iats = group["timestamp"].diff().dt.total_seconds().dropna()
        if iats.empty:
            continue
        mean_iat = float(iats.mean())
        std_iat = float(iats.std(ddof=0)) if len(iats) > 1 else 0.0
        if mean_iat <= 0:
            continue
        cv = std_iat / mean_iat
        if cv <= float(beacon_max_iat_cv):
            severity = "medium" if cv <= beacon_max_iat_cv else "low"
            if cv <= beacon_max_iat_cv / 2:
                severity = "high"
            evidence = {
                "event_count": int(len(group)),
                "mean_iat_s": round(mean_iat, 6),
                "std_iat_s": round(std_iat, 6),
                "iat_cv": round(cv, 6),
                "time_start": group["timestamp"].iloc[0].isoformat(),
                "time_end": group["timestamp"].iloc[-1].isoformat(),
            }
            alerts.append(
                {
                    "timestamp": group["timestamp"].iloc[-1].isoformat(),
                    "rule_name": "PERIODIC_BEACON",
                    "severity": severity,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "unique_ports": int(len(group)),
                    "ports_sample": proto,
                    "evidence": json.dumps(evidence, ensure_ascii=False),
                }
            )

    if not alerts:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    alerts_df = pd.DataFrame(alerts).sort_values(
        by=["timestamp", "rule_name", "severity"], ascending=[False, True, True]
    ).reset_index(drop=True)
    return alerts_df[ALERT_COLUMNS]
