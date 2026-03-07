"""
Alert center: aggregate rule alerts into incident chains.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


SEVERITY_WEIGHT: Dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


def _normalize_alerts(alerts_df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "rule_name", "severity", "src_ip", "dst_ip"}
    missing = required - set(alerts_df.columns)
    if missing:
        raise ValueError(f"alerts_df missing required columns: {sorted(missing)}")

    df = alerts_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "src_ip", "rule_name"]).copy()
    if df.empty:
        return pd.DataFrame(columns=list(alerts_df.columns))

    df["src_ip"] = df["src_ip"].astype(str)
    df["dst_ip"] = df["dst_ip"].fillna("MULTI").astype(str)
    df["rule_name"] = df["rule_name"].astype(str)
    df["severity"] = df["severity"].fillna("low").astype(str).str.lower()
    df.loc[~df["severity"].isin(SEVERITY_WEIGHT.keys()), "severity"] = "low"
    return df.sort_values(["src_ip", "timestamp"]).reset_index(drop=True)


def build_incidents(
    alerts_df: pd.DataFrame,
    correlation_window_s: int = 300,
) -> pd.DataFrame:
    """
    Aggregate alert rows into incident-level chains.

    Grouping strategy:
    - First by src_ip.
    - Within each src_ip, if gap between adjacent alerts <= correlation_window_s,
      they belong to the same incident, otherwise start a new incident.
    """
    if alerts_df is None or alerts_df.empty:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "start_time",
                "end_time",
                "duration_s",
                "src_ip",
                "dst_scope",
                "alert_count",
                "rule_count",
                "max_severity",
                "risk_score",
                "rules",
                "event_chain",
            ]
        )

    df = _normalize_alerts(alerts_df)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "start_time",
                "end_time",
                "duration_s",
                "src_ip",
                "dst_scope",
                "alert_count",
                "rule_count",
                "max_severity",
                "risk_score",
                "rules",
                "event_chain",
            ]
        )

    incidents: List[dict] = []
    incident_seq = 1
    gap = pd.Timedelta(seconds=max(1, int(correlation_window_s)))

    for src_ip, group in df.groupby("src_ip", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        chain_start = 0

        for idx in range(1, len(group) + 1):
            is_cut = False
            if idx == len(group):
                is_cut = True
            else:
                cur_ts = group.loc[idx, "timestamp"]
                prev_ts = group.loc[idx - 1, "timestamp"]
                if (cur_ts - prev_ts) > gap:
                    is_cut = True

            if not is_cut:
                continue

            segment = group.iloc[chain_start:idx].copy()
            start_time = segment["timestamp"].min()
            end_time = segment["timestamp"].max()
            duration_s = float(max(0.0, (end_time - start_time).total_seconds()))
            rules = sorted(segment["rule_name"].unique().tolist())
            dst_hosts = sorted([v for v in segment["dst_ip"].unique().tolist() if v and v != "None"])
            max_sev_weight = int(segment["severity"].map(SEVERITY_WEIGHT).max())
            max_severity = {v: k for k, v in SEVERITY_WEIGHT.items()}.get(max_sev_weight, "low")

            # Risk score: weighted by severity and rule diversity.
            rule_count = len(rules)
            alert_count = int(len(segment))
            risk_score = float(
                min(
                    100.0,
                    round(
                        max_sev_weight * 20
                        + rule_count * 10
                        + min(alert_count, 20) * 2,
                        2,
                    ),
                )
            )
            event_chain = " -> ".join(segment.sort_values("timestamp")["rule_name"].astype(str).tolist())

            incidents.append(
                {
                    "incident_id": f"INC-{incident_seq:05d}",
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_s": duration_s,
                    "src_ip": src_ip,
                    "dst_scope": "MULTI" if len(dst_hosts) > 1 else (dst_hosts[0] if dst_hosts else "MULTI"),
                    "alert_count": alert_count,
                    "rule_count": rule_count,
                    "max_severity": max_severity,
                    "risk_score": risk_score,
                    "rules": ", ".join(rules),
                    "event_chain": event_chain,
                }
            )
            incident_seq += 1
            chain_start = idx

    result = pd.DataFrame(incidents)
    if result.empty:
        return result
    return result.sort_values(["risk_score", "end_time"], ascending=[False, False]).reset_index(drop=True)
