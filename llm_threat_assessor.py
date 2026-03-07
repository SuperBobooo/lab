"""
LLM threat assessment helper.

If API key is available, call Chat Completions API.
Otherwise fallback to a local summary.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

import pandas as pd


def _local_fallback_summary(
    alerts_df: Optional[pd.DataFrame],
    anomalies_df: Optional[pd.DataFrame],
    reason: Optional[str] = None,
) -> str:
    alert_count = int(len(alerts_df)) if alerts_df is not None else 0
    anom_count = int(len(anomalies_df)) if anomalies_df is not None else 0
    rule_stats = ""
    if alerts_df is not None and not alerts_df.empty and "rule_name" in alerts_df.columns:
        vc = alerts_df["rule_name"].value_counts().head(5)
        rule_stats = "; ".join([f"{k}:{int(v)}" for k, v in vc.items()])

    lines = [
        "【本地研判（未启用外部LLM）】",
        f"- 规则告警总数：{alert_count}",
        f"- 异常流量数量：{anom_count}",
        f"- 规则Top：{rule_stats if rule_stats else '无'}",
        "- 建议优先处置高危源IP，并结合ARP/DNS异常进行主机侧排查。",
    ]
    if reason:
        lines.append(f"- 外部LLM失败原因：{reason}")
    return "\n".join(lines)


def _resolve_provider_and_key(provider: str, api_key: Optional[str]) -> tuple[str, Optional[str]]:
    deepseek_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    openai_key = api_key or os.getenv("OPENAI_API_KEY")

    if provider == "deepseek":
        return "deepseek", deepseek_key
    if provider == "openai":
        return "openai", openai_key

    # auto
    if deepseek_key:
        return "deepseek", deepseek_key
    if openai_key:
        return "openai", openai_key
    return "none", None


def _resolve_model(provider: str, model: str) -> str:
    m = (model or "auto").strip().lower()
    if provider == "deepseek":
        if m in {"", "auto", "chat", "deepseek"}:
            return "deepseek-chat"
        return model
    if provider == "openai":
        if m in {"", "auto", "chat"}:
            return "gpt-4o-mini"
        return model
    return model


def generate_threat_assessment(
    alerts_df: Optional[pd.DataFrame],
    anomalies_df: Optional[pd.DataFrame],
    model: str = "auto",
    provider: str = "auto",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: int = 30,
) -> str:
    """
    Generate threat assessment text.
    Returns fallback summary when API key is missing or request fails.
    """
    provider = (provider or "auto").strip().lower()
    chosen_provider, chosen_key = _resolve_provider_and_key(provider, api_key)

    if not chosen_key:
        return _local_fallback_summary(alerts_df, anomalies_df, reason="未配置可用API Key")

    alert_preview = ""
    if alerts_df is not None and not alerts_df.empty:
        cols = [c for c in ["timestamp", "rule_name", "severity", "src_ip", "dst_ip", "evidence"] if c in alerts_df.columns]
        alert_preview = alerts_df[cols].head(20).to_json(orient="records", force_ascii=False)

    anom_preview = ""
    if anomalies_df is not None and not anomalies_df.empty:
        cols = [c for c in ["timestamp", "src_ip", "dst_ip", "protocol", "score_iso_norm", "score_ensemble_norm"] if c in anomalies_df.columns]
        anom_preview = anomalies_df[cols].head(20).to_json(orient="records", force_ascii=False)

    prompt = (
        "你是网络安全分析师。请根据以下告警与异常样本，输出简明威胁研判：\n"
        "1) 关键风险摘要\n2) 可能攻击链\n3) 处置优先级与建议\n"
        f"告警样本: {alert_preview}\n"
        f"异常样本: {anom_preview}\n"
    )

    resolved_model = _resolve_model(chosen_provider, model)

    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": "你是严谨的SOC分析师，请输出中文。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")

    if base_url:
        endpoint = base_url.rstrip("/")
    else:
        endpoint = (
            "https://api.deepseek.com/chat/completions"
            if chosen_provider == "deepseek"
            else "https://api.openai.com/v1/chat/completions"
        )

    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {chosen_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            obj = json.loads(body)
            return obj["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            err_body = ""
        reason = f"HTTP {e.code}"
        if err_body:
            reason = f"{reason}: {err_body[:200]}"
        return _local_fallback_summary(alerts_df, anomalies_df, reason=reason)
    except Exception as e:
        return _local_fallback_summary(alerts_df, anomalies_df, reason=str(e))
