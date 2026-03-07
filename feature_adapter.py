"""
Feature adapter for SentinelAI.

Convert flow-level records (from flow_builder.build_flows) into the feature
schema expected by the existing ML pipeline in data_generator.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def adapt_flows_for_model(flows_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map/augment flow records to model-compatible dataframe columns.

    Input expected (from flow_builder):
    - flow_id, start_time, end_time, duration
    - src_ip, dst_ip, src_port, dst_port, protocol
    - packet_count, total_bytes, total_payload_bytes, ...

    Output includes (at minimum) all current model feature columns:
    - duration_ms, total_bytes, packet_count
    - byte_rate, packets_per_second, bytes_per_packet
    - hour_sin, hour_cos
    - is_tcp, is_udp, is_icmp
    - is_syn, is_ack, is_rst, is_fin, is_psh
    - is_web_port, is_db_port, is_mail_port, is_file_port, is_dns_port
    """
    if flows_df is None or flows_df.empty:
        return pd.DataFrame()

    df = flows_df.copy()

    # Basic timestamp normalization used by existing app pages.
    df["timestamp"] = pd.to_datetime(df.get("start_time"), errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    # Ensure core numeric fields.
    df["duration"] = pd.to_numeric(df.get("duration", 0), errors="coerce").fillna(0.0)
    df["total_bytes"] = pd.to_numeric(df.get("total_bytes", 0), errors="coerce").fillna(0.0)
    df["packet_count"] = pd.to_numeric(df.get("packet_count", 0), errors="coerce").fillna(0.0)
    df["src_port"] = pd.to_numeric(df.get("src_port"), errors="coerce").fillna(0).astype(int)
    df["dst_port"] = pd.to_numeric(df.get("dst_port"), errors="coerce").fillna(0).astype(int)
    df["protocol"] = df.get("protocol", "UNKNOWN").fillna("UNKNOWN").astype(str).str.upper()

    # Model-compatible core features.
    df["duration_ms"] = df["duration"] * 1000.0
    df["byte_rate"] = np.where(df["duration_ms"] > 0, df["total_bytes"] / df["duration_ms"], 0.0)
    df["packets_per_second"] = np.where(df["duration"] > 0, df["packet_count"] / df["duration"], 0.0)
    df["bytes_per_packet"] = np.where(df["packet_count"] > 0, df["total_bytes"] / df["packet_count"], 0.0)

    # Time features.
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # Protocol features.
    df["is_tcp"] = (df["protocol"] == "TCP").astype(int)
    df["is_udp"] = (df["protocol"] == "UDP").astype(int)
    df["is_icmp"] = (df["protocol"] == "ICMP").astype(int)

    # Packet-level TCP flag features are unavailable at flow-only granularity.
    # Keep them as 0 for compatibility with current model input schema.
    df["is_syn"] = 0
    df["is_ack"] = 0
    df["is_rst"] = 0
    df["is_fin"] = 0
    df["is_psh"] = 0

    # Port category features.
    df["is_web_port"] = df["dst_port"].isin([80, 443, 8080]).astype(int)
    df["is_db_port"] = df["dst_port"].isin([3306, 5432, 1433, 1521]).astype(int)
    df["is_mail_port"] = df["dst_port"].isin([25, 143, 465, 587, 993]).astype(int)
    df["is_file_port"] = df["dst_port"].isin([20, 21, 22, 139, 445]).astype(int)
    df["is_dns_port"] = (df["dst_port"] == 53).astype(int)

    # Compatibility columns expected by app views/training code.
    if "flow_id" not in df.columns:
        df["flow_id"] = [f"flow_{i+1}" for i in range(len(df))]
    else:
        df["flow_id"] = df["flow_id"].astype(str)

    if "src_ip" not in df.columns:
        df["src_ip"] = "0.0.0.0"
    if "dst_ip" not in df.columns:
        df["dst_ip"] = "0.0.0.0"

    # For unsupervised baseline training in current pipeline, default to normal.
    # (Real labels can be replaced later when rule/event engine is added.)
    df["label"] = "normal"
    if "attack_type" not in df.columns:
        df["attack_type"] = None

    return df

