"""
Live capture module for SentinelAI.

Primary runtime path uses scapy.sniff (stable in Streamlit threads).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime
from typing import Callable, Dict, List, Optional

from scapy.all import ARP, DNS, DNSQR, Ether, IP, IPv6, Raw, TCP, UDP, get_if_list, sniff  # type: ignore

_stop_event = threading.Event()


def _find_tshark() -> Optional[str]:
    env_path = os.environ.get("TSHARK_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    found = shutil.which("tshark")
    if found:
        return found

    if os.name == "nt":
        candidates = [
            r"C:\Program Files\Wireshark\tshark.exe",
            r"C:\Program Files (x86)\Wireshark\tshark.exe",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
    return None


def list_interfaces() -> List[str]:
    """List capture interfaces (tshark first, scapy fallback)."""
    # Prefer scapy-native interface names to avoid mismatch when sniff() starts.
    try:
        scapy_ifaces = list(get_if_list())
        if scapy_ifaces:
            return scapy_ifaces
    except Exception:
        pass

    tshark_path = _find_tshark()
    if tshark_path:
        proc = subprocess.run(
            [tshark_path, "-D"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode == 0:
            interfaces: List[str] = []
            output = proc.stdout or ""
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                if ". " in line:
                    interfaces.append(line.split(". ", 1)[1].strip())
                else:
                    interfaces.append(line)
            if interfaces:
                return interfaces

    try:
        return list(get_if_list())
    except Exception as exc:
        raise RuntimeError(f"无法列出网卡接口（tshark/scapy均失败）: {exc}")


def _resolve_interface_for_scapy(interface: str) -> str:
    """
    Resolve user-selected interface label to a scapy-accepted interface name.

    Handles Windows strings like:
    \\Device\\NPF_{GUID} (WLAN)  ->  \\Device\\NPF_{GUID}
    """
    try:
        ifaces = list(get_if_list())
    except Exception:
        ifaces = []

    if interface in ifaces:
        return interface

    # Strip trailing descriptive suffix: " (...)".
    if " (" in interface:
        candidate = interface.split(" (", 1)[0].strip()
        if candidate in ifaces:
            return candidate
        interface = candidate

    # Match by GUID fragment if present.
    if "{" in interface and "}" in interface:
        guid = interface[interface.find("{"): interface.find("}") + 1].lower()
        for name in ifaces:
            if guid in name.lower():
                return name

    # Fallback to original; caller will receive underlying error if invalid.
    return interface


def _decode_http_payload(payload: bytes) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if not payload:
        return None, None, None
    try:
        text = payload.decode("utf-8", errors="ignore")
    except Exception:
        return None, None, None
    lines = text.split("\r\n")
    if not lines:
        return None, None, None

    first = lines[0].strip().split()
    method = None
    path = None
    if len(first) >= 2 and first[0] in {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}:
        method = first[0]
        path = first[1]

    host = None
    for line in lines[1:]:
        if line.lower().startswith("host:"):
            host = line.split(":", 1)[1].strip() if ":" in line else None
            break

    return method, host, path


def _scapy_packet_to_record(pkt) -> Dict:
    record: Dict = {
        "timestamp": datetime.fromtimestamp(float(getattr(pkt, "time", time.time()))).isoformat(),
        "src_mac": None,
        "dst_mac": None,
        "src_ip": None,
        "dst_ip": None,
        "src_port": None,
        "dst_port": None,
        "protocol": "UNKNOWN",
        "packet_length": int(len(pkt)),
        "payload_size": 0,
        "payload_entropy": 0.0,
        "l7_protocol": "UNKNOWN",
        "tcp_flags": "",
        "dns_query": None,
        "dns_qtype": None,
        "dns_rcode": None,
        "http_method": None,
        "http_host": None,
        "http_path": None,
    }

    if Ether in pkt:
        record["src_mac"] = pkt[Ether].src
        record["dst_mac"] = pkt[Ether].dst

    if ARP in pkt:
        record["protocol"] = "ARP"
        record["l7_protocol"] = "ARP"
        record["src_ip"] = pkt[ARP].psrc
        record["dst_ip"] = pkt[ARP].pdst
    elif IP in pkt:
        record["src_ip"] = pkt[IP].src
        record["dst_ip"] = pkt[IP].dst
        if TCP in pkt:
            record["protocol"] = "TCP"
            record["src_port"] = int(pkt[TCP].sport)
            record["dst_port"] = int(pkt[TCP].dport)
            record["tcp_flags"] = str(pkt[TCP].flags)
        elif UDP in pkt:
            record["protocol"] = "UDP"
            record["src_port"] = int(pkt[UDP].sport)
            record["dst_port"] = int(pkt[UDP].dport)
        else:
            record["protocol"] = f"IP_PROTO_{int(pkt[IP].proto)}"
    elif IPv6 in pkt:
        record["protocol"] = "IPV6"
        record["src_ip"] = pkt[IPv6].src
        record["dst_ip"] = pkt[IPv6].dst

    payload = bytes(pkt[Raw].load) if Raw in pkt else b""
    record["payload_size"] = len(payload)
    if payload:
        freq = Counter(payload)
        total = len(payload)
        ent = 0.0
        for c in freq.values():
            p = c / total
            ent -= p * math.log2(p)
        record["payload_entropy"] = float(ent)

    if DNS in pkt:
        record["l7_protocol"] = "DNS"
        try:
            if DNSQR in pkt and pkt[DNS].qd is not None and hasattr(pkt[DNS].qd, "qname"):
                qn = pkt[DNS].qd.qname
                record["dns_query"] = qn.decode("utf-8", errors="ignore").rstrip(".") if isinstance(qn, bytes) else str(qn)
                record["dns_qtype"] = int(pkt[DNS].qd.qtype)
            record["dns_rcode"] = int(pkt[DNS].rcode)
        except Exception:
            pass
    else:
        method, host, path = _decode_http_payload(payload)
        if method is not None:
            record["l7_protocol"] = "HTTP"
            record["http_method"] = method
            record["http_host"] = host
            record["http_path"] = path

    return record


def stop_capture() -> None:
    """Request graceful stop for ongoing capture."""
    _stop_event.set()


def _capture_live_records(
    interface: str,
    bpf_filter: Optional[str] = None,
    duration_s: Optional[int] = None,
    packet_limit: Optional[int] = None,
    on_record: Optional[Callable[[Dict, List[Dict]], None]] = None,
    jsonl_path: Optional[str] = None,
) -> List[Dict]:
    _stop_event.clear()
    records: List[Dict] = []
    writer = None
    iface_name = _resolve_interface_for_scapy(interface)

    try:
        if jsonl_path:
            os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
            writer = open(jsonl_path, "w", encoding="utf-8")

        def _on_pkt(pkt):
            record = _scapy_packet_to_record(pkt)
            records.append(record)

            if writer:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()

            if on_record is not None:
                on_record(record, records)

        sniff(
            iface=iface_name,
            filter=bpf_filter if bpf_filter else None,
            prn=_on_pkt,
            store=False,
            timeout=int(duration_s) if duration_s is not None else None,
            count=int(packet_limit) if packet_limit is not None else 0,
            stop_filter=lambda _: _stop_event.is_set(),
        )
    finally:
        if writer:
            writer.close()

    return records


def start_capture(
    interface: str,
    bpf_filter: Optional[str] = None,
    duration_s: Optional[int] = None,
    packet_limit: Optional[int] = None,
) -> List[Dict]:
    """Start live capture and return normalized packet records."""
    return _capture_live_records(
        interface=interface,
        bpf_filter=bpf_filter,
        duration_s=duration_s,
        packet_limit=packet_limit,
    )


def capture_with_callback(
    interface: str,
    bpf_filter: Optional[str] = None,
    duration_s: Optional[int] = None,
    packet_limit: Optional[int] = None,
    on_record: Optional[Callable[[Dict, List[Dict]], None]] = None,
    jsonl_path: Optional[str] = None,
) -> List[Dict]:
    """Capture helper with callback/jsonl sink for UI streaming."""
    return _capture_live_records(
        interface=interface,
        bpf_filter=bpf_filter,
        duration_s=duration_s,
        packet_limit=packet_limit,
        on_record=on_record,
        jsonl_path=jsonl_path,
    )
