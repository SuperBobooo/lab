"""
Structured log center for SentinelAI.

Writes packet/flow/alert/incident/action records into JSONL files and
maintains a manifest for retrieval.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Dict, Optional

import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_jsonable_records(df: Optional[pd.DataFrame], max_rows: int) -> list:
    if df is None or df.empty:
        return []
    records = df.head(max_rows).copy()
    # Normalize datetime-like columns for JSONL
    for col in records.columns:
        if pd.api.types.is_datetime64_any_dtype(records[col]):
            records[col] = records[col].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    return records.to_dict(orient="records")


def _write_jsonl(path: str, records: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def persist_snapshot(
    output_root: str = "data/logs",
    source: str = "unknown",
    packet_df: Optional[pd.DataFrame] = None,
    flow_df: Optional[pd.DataFrame] = None,
    alert_df: Optional[pd.DataFrame] = None,
    incident_df: Optional[pd.DataFrame] = None,
    action_df: Optional[pd.DataFrame] = None,
    max_rows_per_type: int = 5000,
) -> Dict[str, str]:
    """
    Persist a pipeline snapshot to structured JSONL logs.
    Returns mapping from data type to file path.
    """
    now = dt.datetime.now()
    day_dir = os.path.join(output_root, now.strftime("%Y%m%d"))
    _ensure_dir(day_dir)

    snap_id = now.strftime("%H%M%S_%f")
    paths: Dict[str, str] = {}

    type_to_df = {
        "packet": packet_df,
        "flow": flow_df,
        "alert": alert_df,
        "incident": incident_df,
        "action": action_df,
    }

    for name, df in type_to_df.items():
        rows = _to_jsonable_records(df, max_rows=max_rows_per_type)
        if not rows:
            continue
        path = os.path.join(day_dir, f"{name}_{snap_id}.jsonl")
        _write_jsonl(path, rows)
        paths[name] = path

    manifest_path = os.path.join(output_root, "manifest.jsonl")
    manifest_row = {
        "timestamp": now.isoformat(),
        "snapshot_id": snap_id,
        "source": source,
        "counts": {
            "packet": int(len(packet_df)) if packet_df is not None else 0,
            "flow": int(len(flow_df)) if flow_df is not None else 0,
            "alert": int(len(alert_df)) if alert_df is not None else 0,
            "incident": int(len(incident_df)) if incident_df is not None else 0,
            "action": int(len(action_df)) if action_df is not None else 0,
        },
        "paths": paths,
    }
    _ensure_dir(output_root)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

    paths["manifest"] = manifest_path
    return paths
