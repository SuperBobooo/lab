"""
PCAP ingest module for SentinelAI.

Provides offline PCAP parsing using scapy and returns normalized packet records.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

from scapy.all import ARP, DNS, DNSQR, Ether, IP, Raw, TCP, UDP, rdpcap  # type: ignore
from scapy.error import Scapy_Exception  # type: ignore


def _safe_iso_timestamp(ts: float) -> str:
    """Convert packet timestamp to ISO-8601 string."""
    return datetime.fromtimestamp(float(ts)).isoformat()


def _calc_entropy(data: bytes) -> float:
    """Calculate Shannon entropy for payload bytes."""
    if not data:
        return 0.0
    total = len(data)
    freq = Counter(data)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)


def _base_record() -> Dict:
    """Create a base record with stable keys."""
    return {
        "timestamp": None,
        "src_mac": None,
        "dst_mac": None,
        "src_ip": None,
        "dst_ip": None,
        "src_port": None,
        "dst_port": None,
        "protocol": "UNKNOWN",
        "packet_length": 0,
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


def _extract_l4_info(pkt) -> tuple[Optional[int], Optional[int], str]:
    """Extract L4 ports/protocol label from an IP packet."""
    if TCP in pkt:
        return int(pkt[TCP].sport), int(pkt[TCP].dport), "TCP"
    if UDP in pkt:
        return int(pkt[UDP].sport), int(pkt[UDP].dport), "UDP"
    # Fallback to IP protocol number when not TCP/UDP.
    if IP in pkt:
        return None, None, f"IP_PROTO_{int(pkt[IP].proto)}"
    return None, None, "IP"


def _decode_http_payload(payload: bytes) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse minimal HTTP request fields from raw payload."""
    if not payload:
        return None, None, None
    try:
        text = payload.decode("utf-8", errors="ignore")
    except Exception:
        return None, None, None

    lines = text.split("\r\n")
    if not lines:
        return None, None, None

    first = lines[0].strip()
    parts = first.split()
    method = None
    path = None
    if len(parts) >= 2 and parts[0] in {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}:
        method = parts[0]
        path = parts[1]

    host = None
    for line in lines[1:]:
        if line.lower().startswith("host:"):
            host = line.split(":", 1)[1].strip() if ":" in line else None
            break

    return method, host, path


def parse_pcap(file_path: str) -> List[Dict]:
    """
    Parse a PCAP/PCAPNG file and return normalized packet records.

    Only IP packets and ARP packets are parsed. Other packet types are skipped.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PCAP file not found: {file_path}")

    try:
        packets = rdpcap(file_path)
    except Scapy_Exception as exc:
        raise ValueError(f"Invalid or unsupported capture format: {file_path}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read PCAP file: {file_path}") from exc

    records: List[Dict] = []

    for pkt in packets:
        # Parse ARP packets separately.
        if ARP in pkt:
            record = _base_record()
            record["timestamp"] = _safe_iso_timestamp(pkt.time)
            record["packet_length"] = int(len(pkt))
            record["protocol"] = "ARP"
            record["l7_protocol"] = "ARP"

            if Ether in pkt:
                record["src_mac"] = pkt[Ether].src
                record["dst_mac"] = pkt[Ether].dst

            arp = pkt[ARP]
            record["src_ip"] = arp.psrc
            record["dst_ip"] = arp.pdst

            payload = bytes(pkt[Raw].load) if Raw in pkt else b""
            record["payload_size"] = len(payload)
            record["payload_entropy"] = _calc_entropy(payload)

            records.append(record)
            continue

        # Only parse IP packets for non-ARP traffic.
        if IP not in pkt:
            continue

        record = _base_record()
        record["timestamp"] = _safe_iso_timestamp(pkt.time)
        record["packet_length"] = int(len(pkt))

        if Ether in pkt:
            record["src_mac"] = pkt[Ether].src
            record["dst_mac"] = pkt[Ether].dst

        ip = pkt[IP]
        record["src_ip"] = ip.src
        record["dst_ip"] = ip.dst

        src_port, dst_port, protocol = _extract_l4_info(pkt)
        record["src_port"] = src_port
        record["dst_port"] = dst_port
        record["protocol"] = protocol
        if TCP in pkt:
            record["tcp_flags"] = str(pkt[TCP].flags)

        payload = bytes(pkt[Raw].load) if Raw in pkt else b""
        record["payload_size"] = len(payload)
        record["payload_entropy"] = _calc_entropy(payload)

        # DNS deep fields (UDP/TCP DNS packets).
        if DNS in pkt:
            record["l7_protocol"] = "DNS"
            dns_layer = pkt[DNS]
            if DNSQR in dns_layer and dns_layer.qd is not None and hasattr(dns_layer.qd, "qname"):
                try:
                    qname = dns_layer.qd.qname
                    record["dns_query"] = qname.decode("utf-8", errors="ignore").rstrip(".") if isinstance(qname, bytes) else str(qname)
                except Exception:
                    record["dns_query"] = None
                try:
                    record["dns_qtype"] = int(dns_layer.qd.qtype)
                except Exception:
                    record["dns_qtype"] = None
            try:
                record["dns_rcode"] = int(dns_layer.rcode)
            except Exception:
                record["dns_rcode"] = None
        else:
            # HTTP minimal parsing from TCP payload.
            method, host, path = _decode_http_payload(payload)
            if method is not None:
                record["l7_protocol"] = "HTTP"
                record["http_method"] = method
                record["http_host"] = host
                record["http_path"] = path

        records.append(record)

    return records


if __name__ == "__main__":
    try:
        data = parse_pcap("example.pcap")
        print(data[:5])
    except Exception as exc:
        print(f"Error: {exc}")
