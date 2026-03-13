"""
Context-aware chat helper for SentinelAI Agent tab.

Uses DeepSeek Chat Completions API by default and falls back to a local
response when API key/network is unavailable.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional


def _to_json_safe(obj):
    """Recursively convert objects into JSON-serializable values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(v) for v in obj]

    # numpy scalar / pandas scalar often implement item()
    if hasattr(obj, "item"):
        try:
            return _to_json_safe(obj.item())
        except Exception:
            pass

    # pandas.Timestamp and many datetime-like types implement isoformat()
    if hasattr(obj, "isoformat"):
        try:
            return str(obj.isoformat())
        except Exception:
            pass

    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("utf-8", errors="ignore")

    return str(obj)


def _local_fallback_chat(question: str, context: Dict, reason: str) -> str:
    summary = context.get("summary", {}) if isinstance(context, dict) else {}
    return (
        "当前无法调用外部大模型，已切换为本地兜底答复。\n"
        f"- 原因: {reason}\n"
        f"- 本次上下文: 报文={summary.get('packet_count', 0)}，流量={summary.get('flow_count', 0)}，"
        f"告警={summary.get('alert_count', 0)}，事件={summary.get('incident_count', 0)}\n"
        f"- 你的问题: {question}\n"
        "建议检查 DEEPSEEK_API_KEY 环境变量或网络连通性后重试。"
    )


def _request_chat_completion(
    endpoint: str,
    api_key: str,
    payload: dict,
    timeout_s: int,
) -> str:
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        obj = json.loads(body)
        return obj["choices"][0]["message"]["content"].strip()


def generate_contextual_chat_reply(
    question: str,
    context: Dict,
    history: Optional[List[Dict[str, str]]] = None,
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: int = 60,
    max_retries: int = 2,
) -> str:
    """
    Answer a user question using prior SentinelAI analysis context.

    history item format:
    - {"role": "user"|"assistant", "content": "..."}
    """
    q = (question or "").strip()
    if not q:
        return "请输入问题。"

    key = (api_key or os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not key:
        return _local_fallback_chat(q, context, reason="未配置 DEEPSEEK_API_KEY")

    endpoint = (base_url or "https://api.deepseek.com/chat/completions").rstrip("/")
    model_name = (model or "deepseek-chat").strip()
    if model_name.lower() in {"", "auto"}:
        model_name = "deepseek-chat"

    safe_context = context if isinstance(context, dict) else {}
    safe_context_for_json = _to_json_safe(safe_context)
    context_json = json.dumps(safe_context_for_json, ensure_ascii=False)
    if len(context_json) > 12000:
        context_json = context_json[:12000] + "...(truncated)"

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "你是 SentinelAI 的网络安全研判助手。"
                "你必须结合提供的结构化上下文回答问题，并优先给出可执行结论。"
                "若上下文不足，请明确指出缺失信息，不要编造。"
            ),
        },
        {
            "role": "user",
            "content": (
                "以下是本次检测会话上下文（JSON）：\n"
                f"{context_json}\n\n"
                "请记住这些信息，并在后续问答中优先依据这些信息进行回答。"
            ),
        },
    ]

    # Keep recent short history for continuity without exploding token usage.
    history = history or []
    recent = history[-8:]
    for item in recent:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:2000]})

    messages.append({"role": "user", "content": q})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.2,
    }

    attempt = 0
    last_err = ""
    while attempt <= max(0, int(max_retries)):
        try:
            return _request_chat_completion(
                endpoint=endpoint,
                api_key=key,
                payload=payload,
                timeout_s=max(5, int(timeout_s)),
            )
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                err_body = ""
            reason = f"HTTP {e.code}"
            if err_body:
                reason = f"{reason}: {err_body[:240]}"
            return _local_fallback_chat(q, safe_context, reason=reason)
        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            last_err = str(e)
            if attempt >= max_retries:
                break
            time.sleep(min(3.0, 0.8 * (2 ** attempt)))
        except Exception as e:
            return _local_fallback_chat(q, safe_context, reason=str(e))
        attempt += 1

    return _local_fallback_chat(
        q,
        safe_context,
        reason=f"请求超时/网络异常（已重试{max_retries + 1}次）: {last_err}",
    )
