"""
Minimal response orchestration:
- Alert audit logging (JSONL)
- Optional webhook notification
- Optional email notification (SMTP via env vars)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import smtplib
import ssl
import urllib.request
from email.message import EmailMessage
from typing import Dict, List, Optional

import pandas as pd


def _now_iso() -> str:
    return dt.datetime.now().isoformat()


def _append_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _severity_rank(sev: str) -> int:
    s = str(sev).lower()
    if s == "high":
        return 3
    if s == "medium":
        return 2
    return 1


def _build_summary(alerts_df: pd.DataFrame) -> Dict:
    by_rule = {}
    if "rule_name" in alerts_df.columns and not alerts_df.empty:
        by_rule = {str(k): int(v) for k, v in alerts_df["rule_name"].value_counts().to_dict().items()}

    by_sev = {}
    if "severity" in alerts_df.columns and not alerts_df.empty:
        by_sev = {str(k): int(v) for k, v in alerts_df["severity"].value_counts().to_dict().items()}

    return {
        "generated_at": _now_iso(),
        "alert_count": int(len(alerts_df)),
        "src_ip_unique": int(alerts_df["src_ip"].nunique()) if "src_ip" in alerts_df.columns else 0,
        "by_rule": by_rule,
        "by_severity": by_sev,
    }


def _send_webhook(webhook_url: str, payload: Dict, timeout_s: int = 8) -> str:
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return f"HTTP {resp.status}"


def _send_email(summary: Dict, smtp_to: str) -> str:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    from_addr = os.getenv("SMTP_FROM", user or "sentinelai@localhost")

    if not host or not smtp_to:
        raise ValueError("missing SMTP_HOST or SMTP_TO")

    msg = EmailMessage()
    msg["Subject"] = f"[SentinelAI] 告警摘要 {summary.get('alert_count', 0)} 条"
    msg["From"] = from_addr
    msg["To"] = smtp_to
    msg.set_content(
        "SentinelAI 告警摘要\n\n"
        f"生成时间: {summary.get('generated_at')}\n"
        f"告警总数: {summary.get('alert_count')}\n"
        f"源IP去重: {summary.get('src_ip_unique')}\n"
        f"规则分布: {summary.get('by_rule')}\n"
        f"严重度分布: {summary.get('by_severity')}\n"
    )

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=10) as server:
        server.starttls(context=context)
        if user and password:
            server.login(user, password)
        server.send_message(msg)
    return f"SMTP {host}:{port}"


def execute_responses(
    alerts_df: Optional[pd.DataFrame],
    log_all_alerts: bool = True,
    enable_webhook: bool = False,
    webhook_url: str = "",
    enable_email: bool = False,
    email_min_severity: str = "high",
    smtp_to: str = "",
    output_dir: str = "data",
) -> pd.DataFrame:
    """
    Execute response actions and return action audit records.
    """
    action_rows: List[Dict] = []
    if alerts_df is None:
        alerts_df = pd.DataFrame(columns=["timestamp", "rule_name", "severity", "src_ip", "dst_ip"])

    summary = _build_summary(alerts_df)
    ts = dt.datetime.now().strftime("%Y%m%d")
    audit_path = os.path.join(output_dir, f"response_audit_{ts}.jsonl")

    if log_all_alerts:
        payload = {
            "action": "ALERT_LOG",
            "timestamp": _now_iso(),
            "summary": summary,
            "alerts_sample": alerts_df.head(100).to_dict(orient="records"),
        }
        _append_jsonl(audit_path, [payload])
        action_rows.append(
            {
                "timestamp": payload["timestamp"],
                "action_type": "ALERT_LOG",
                "status": "success",
                "target": audit_path,
                "detail": f"logged {summary['alert_count']} alerts",
            }
        )

    if enable_webhook:
        if not webhook_url.strip():
            action_rows.append(
                {
                    "timestamp": _now_iso(),
                    "action_type": "WEBHOOK",
                    "status": "skipped",
                    "target": "",
                    "detail": "webhook_url empty",
                }
            )
        else:
            webhook_payload = {
                "source": "SentinelAI",
                "type": "RULE_ALERT_SUMMARY",
                "summary": summary,
            }
            try:
                resp = _send_webhook(webhook_url.strip(), webhook_payload)
                action_rows.append(
                    {
                        "timestamp": _now_iso(),
                        "action_type": "WEBHOOK",
                        "status": "success",
                        "target": webhook_url.strip(),
                        "detail": resp,
                    }
                )
            except Exception as e:
                action_rows.append(
                    {
                        "timestamp": _now_iso(),
                        "action_type": "WEBHOOK",
                        "status": "failed",
                        "target": webhook_url.strip(),
                        "detail": str(e),
                    }
                )

    if enable_email:
        min_rank = _severity_rank(email_min_severity)
        should_send = False
        if not alerts_df.empty and "severity" in alerts_df.columns:
            sev_rank = alerts_df["severity"].astype(str).map(_severity_rank)
            should_send = bool((sev_rank >= min_rank).any())

        if not should_send:
            action_rows.append(
                {
                    "timestamp": _now_iso(),
                    "action_type": "EMAIL",
                    "status": "skipped",
                    "target": smtp_to or os.getenv("SMTP_TO", ""),
                    "detail": "no alert reaches severity threshold",
                }
            )
        else:
            target = smtp_to or os.getenv("SMTP_TO", "")
            try:
                resp = _send_email(summary=summary, smtp_to=target)
                action_rows.append(
                    {
                        "timestamp": _now_iso(),
                        "action_type": "EMAIL",
                        "status": "success",
                        "target": target,
                        "detail": resp,
                    }
                )
            except Exception as e:
                action_rows.append(
                    {
                        "timestamp": _now_iso(),
                        "action_type": "EMAIL",
                        "status": "failed",
                        "target": target,
                        "detail": str(e),
                    }
                )

    if not action_rows:
        action_rows.append(
            {
                "timestamp": _now_iso(),
                "action_type": "NONE",
                "status": "skipped",
                "target": "",
                "detail": "no action enabled",
            }
        )

    # Persist action audit itself
    _append_jsonl(audit_path, [{"action": "ACTION_EXECUTION", "rows": action_rows, "timestamp": _now_iso()}])

    return pd.DataFrame(action_rows)
