"""
Flow builder for SentinelAI: packet records -> flow records.

Expected packet input columns (from pcap_ingest.parse_pcap):
- timestamp
- src_ip, dst_ip, src_port, dst_port, protocol
- packet_length, payload_size

Notes for model compatibility (future integration):
Current model pipeline expects columns like:
- duration_ms, total_bytes, packet_count, byte_rate, packets_per_second, bytes_per_packet

Recommended mapping from flows_df:
- duration_ms <- duration * 1000
- total_bytes <- total_bytes
- packet_count <- packet_count
- byte_rate <- total_bytes / (duration_ms + 1e-3)
- packets_per_second <- packet_count / (duration + 1e-6)
- bytes_per_packet <- total_bytes / (packet_count + 1e-3)
- is_tcp/is_udp <- protocol == "TCP"/"UDP"
- port category features <- based on dst_port
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_PACKET_COLUMNS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "packet_length",
    "payload_size",
]

FLOW_OUTPUT_COLUMNS = [
    "flow_id",
    "start_time",
    "end_time",
    "duration",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "packet_count",
    "total_bytes",
    "total_payload_bytes",
    "bytes_per_second",
    "avg_packet_size",
    "std_packet_size",
    "mean_iat",
    "std_iat",
]


def _validate_input_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_PACKET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"packets_df missing required columns: {missing}")


def _build_single_flow(flow_id: int, group_df: pd.DataFrame) -> Dict:
    ts = group_df["timestamp"]
    pkt_sizes = group_df["packet_length"].astype(float).fillna(0.0)
    payload_sizes = group_df["payload_size"].astype(float).fillna(0.0)

    start_ts = ts.iloc[0]
    end_ts = ts.iloc[-1]
    duration = float((end_ts - start_ts).total_seconds())

    packet_count = int(len(group_df))
    total_bytes = float(pkt_sizes.sum())
    total_payload_bytes = float(payload_sizes.sum())
    if duration <= 0:
        bytes_per_second = 0.0
    else:
        bytes_per_second = float(total_bytes / duration)

    avg_packet_size = float(pkt_sizes.mean()) if packet_count > 0 else 0.0
    std_packet_size = float(pkt_sizes.std(ddof=0)) if packet_count > 1 else 0.0

    iats = ts.diff().dt.total_seconds().dropna()
    mean_iat = float(iats.mean()) if not iats.empty else 0.0
    std_iat = float(iats.std(ddof=0)) if len(iats) > 1 else 0.0

    first = group_df.iloc[0]
    return {
        "flow_id": flow_id,
        "start_time": start_ts.isoformat(),
        "end_time": end_ts.isoformat(),
        "duration": duration,
        "src_ip": first["src_ip"],
        "dst_ip": first["dst_ip"],
        "src_port": first["src_port"],
        "dst_port": first["dst_port"],
        "protocol": first["protocol"],
        "packet_count": packet_count,
        "total_bytes": total_bytes,
        "total_payload_bytes": total_payload_bytes,
        "bytes_per_second": bytes_per_second,
        "avg_packet_size": avg_packet_size,
        "std_packet_size": std_packet_size,
        "mean_iat": mean_iat,
        "std_iat": std_iat,
    }


def build_flows(packets_df: pd.DataFrame, idle_timeout_s: int = 60) -> pd.DataFrame:
    """
    Aggregate packet records into flow records.

    Rules:
    - 5-tuple aggregation: (src_ip, dst_ip, src_port, dst_port, protocol)
    - Session split by idle timeout: if adjacent packet gap > idle_timeout_s, start a new flow.
    - Ignore records missing src_ip/dst_ip.
    - ARP packets are excluded from flow aggregation.
    """
    _validate_input_columns(packets_df)

    if packets_df.empty:
        return pd.DataFrame(columns=FLOW_OUTPUT_COLUMNS)

    df = packets_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Ignore invalid L3 tuples and exclude ARP from flow aggregation.
    df = df.dropna(subset=["src_ip", "dst_ip"])
    df = df[df["protocol"].astype(str).str.upper() != "ARP"]

    if df.empty:
        return pd.DataFrame(columns=FLOW_OUTPUT_COLUMNS)

    tuple_cols = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
    df = df.sort_values(tuple_cols + ["timestamp"]).reset_index(drop=True)

    flows: List[Dict] = []
    flow_id = 1

    for _, group in df.groupby(tuple_cols, dropna=False, sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if group.empty:
            continue

        split_points = [0]
        time_gaps = group["timestamp"].diff().dt.total_seconds().fillna(0.0)
        for idx, gap in enumerate(time_gaps):
            if idx == 0:
                continue
            if float(gap) > float(idle_timeout_s):
                split_points.append(idx)
        split_points.append(len(group))

        for start_idx, end_idx in zip(split_points[:-1], split_points[1:]):
            sub = group.iloc[start_idx:end_idx]
            if sub.empty:
                continue
            flows.append(_build_single_flow(flow_id, sub))
            flow_id += 1

    if not flows:
        return pd.DataFrame(columns=FLOW_OUTPUT_COLUMNS)

    flows_df = pd.DataFrame(flows)
    return flows_df[FLOW_OUTPUT_COLUMNS]
