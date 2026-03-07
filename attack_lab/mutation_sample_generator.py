"""
Offline mutation/non-traditional attack flow generator.

Purpose:
- Build lab replay samples that are harder for classic single-rule detection.
- Output PCAP for end-to-end validation in SentinelAI pipeline.
"""

from __future__ import annotations

import os
import random
import time
from typing import Dict, List

from scapy.all import DNS, DNSQR, Ether, IP, Raw, TCP, UDP, wrpcap  # type: ignore


def _rand_mac(rng: random.Random) -> str:
    return "02:%02x:%02x:%02x:%02x:%02x" % tuple(rng.randint(0, 255) for _ in range(5))


def _baseline_packets(rng: random.Random, count: int, src_mac: str, dst_mac: str) -> List:
    packets = []
    src_ip = "192.168.56.10"
    dst_candidates = ["8.8.8.8", "1.1.1.1", "114.114.114.114"]
    for i in range(max(10, count // 6)):
        dst_ip = dst_candidates[i % len(dst_candidates)]
        if i % 2 == 0:
            pkt = Ether(src=src_mac, dst=dst_mac) / IP(src=src_ip, dst=dst_ip) / TCP(
                sport=41000 + i, dport=443, flags="PA"
            ) / Raw(load=b"GET /health HTTP/1.1\r\nHost: test.local\r\n\r\n")
        else:
            q = f"cdn-{rng.randint(100,999)}.example.com"
            pkt = Ether(src=src_mac, dst=dst_mac) / IP(src=src_ip, dst="8.8.8.8") / UDP(
                sport=52000 + i, dport=53
            ) / DNS(rd=1, qd=DNSQR(qname=q))
        packets.append(pkt)
    return packets


def generate_mutation_attack_pcap(
    output_path: str,
    scenario: str = "stealth_scan",
    packet_count: int = 120,
    seed: int = 1337,
) -> Dict:
    """
    Generate mutation attack samples into a PCAP.

    scenarios:
    - stealth_scan: low-rate port scan with jitter (harder for short-window threshold rules).
    - dns_tunnel_lowrate: low-rate long-subdomain DNS queries (simulated covert channel).
    - jitter_beacon: near-periodic C2-like traffic with jitter.
    """
    rng = random.Random(seed)
    scenario = scenario.strip().lower()
    valid = {"stealth_scan", "dns_tunnel_lowrate", "jitter_beacon"}
    if scenario not in valid:
        raise ValueError(f"Unsupported scenario: {scenario}. Valid: {sorted(valid)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    attacker_mac = _rand_mac(rng)
    gateway_mac = _rand_mac(rng)
    attacker_ip = "192.168.56.66"
    target_ip = "192.168.56.20"

    packets: List = []
    packets.extend(_baseline_packets(rng, packet_count, attacker_mac, gateway_mac))

    attack_packets = []
    if scenario == "stealth_scan":
        ports = list(range(20, 20 + max(20, packet_count)))
        rng.shuffle(ports)
        for i in range(packet_count):
            dport = ports[i % len(ports)]
            pkt = Ether(src=attacker_mac, dst=gateway_mac) / IP(src=attacker_ip, dst=target_ip) / TCP(
                sport=35000 + i, dport=dport, flags="S"
            )
            attack_packets.append(pkt)

    elif scenario == "dns_tunnel_lowrate":
        exfil_domain = "exfil.lab.local"
        for i in range(packet_count):
            label = f"{rng.randint(10**10, 10**11-1):x}"[:20]
            qname = f"{label}.{i % 7}.{exfil_domain}"
            pkt = (
                Ether(src=attacker_mac, dst=gateway_mac)
                / IP(src=attacker_ip, dst="8.8.8.8")
                / UDP(sport=46000 + (i % 1000), dport=53)
                / DNS(rd=1, qd=DNSQR(qname=qname))
            )
            attack_packets.append(pkt)

    elif scenario == "jitter_beacon":
        c2_ip = "45.77.12.34"
        for i in range(packet_count):
            pkt = Ether(src=attacker_mac, dst=gateway_mac) / IP(src=attacker_ip, dst=c2_ip) / TCP(
                sport=37000 + (i % 500), dport=443, flags="PA"
            ) / Raw(load=b"beacon")
            attack_packets.append(pkt)

    packets.extend(attack_packets)

    # Attach synthetic timestamps to emulate low-rate/jitter behavior.
    base_ts = time.time()
    idx = 0
    for pkt in packets:
        if idx < len(packets) - len(attack_packets):
            delta = 0.02 * idx
        else:
            attack_idx = idx - (len(packets) - len(attack_packets))
            if scenario == "stealth_scan":
                delta = 2.5 * attack_idx + rng.uniform(0.0, 1.2)
            elif scenario == "dns_tunnel_lowrate":
                delta = 1.5 * attack_idx + rng.uniform(0.0, 0.8)
            else:  # jitter_beacon
                delta = 6.0 * attack_idx + rng.uniform(-0.6, 0.6)
        pkt.time = base_ts + delta
        idx += 1

    wrpcap(output_path, packets)

    return {
        "output_path": output_path,
        "scenario": scenario,
        "packet_count_total": len(packets),
        "packet_count_attack": len(attack_packets),
        "attacker_ip": attacker_ip,
        "target_ip": target_ip,
    }
