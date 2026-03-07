"""
Offline ARP attack sample generator for SentinelAI attack lab.

Generates real L2 packets (PCAP) for safe lab validation in isolated envs.
"""

from __future__ import annotations

import ipaddress
import os
import random
from typing import Dict, List

from scapy.all import ARP, Ether, IP, TCP, UDP, Raw, wrpcap  # type: ignore


def _random_mac(rng: random.Random) -> str:
    return "02:%02x:%02x:%02x:%02x:%02x" % tuple(rng.randint(0, 255) for _ in range(5))


def _baseline_ip_packets(
    normal_src_ip: str,
    normal_dst_ip: str,
    normal_src_mac: str,
    normal_dst_mac: str,
    n: int,
) -> List:
    packets = []
    for i in range(max(1, n)):
        if i % 2 == 0:
            pkt = (
                Ether(src=normal_src_mac, dst=normal_dst_mac)
                / IP(src=normal_src_ip, dst=normal_dst_ip)
                / TCP(sport=40000 + i, dport=443, flags="S")
                / Raw(load=b"hello")
            )
        else:
            pkt = (
                Ether(src=normal_src_mac, dst=normal_dst_mac)
                / IP(src=normal_src_ip, dst=normal_dst_ip)
                / UDP(sport=50000 + i, dport=53)
                / Raw(load=b"\x12\x34")
            )
        packets.append(pkt)
    return packets


def generate_arp_attack_pcap(
    output_path: str,
    scenario: str = "spoof",
    packet_count: int = 100,
    seed: int = 42,
    victim_ip: str = "192.168.56.10",
    gateway_ip: str = "192.168.56.1",
    attacker_ip: str = "192.168.56.66",
    network_cidr: str = "192.168.56.0/24",
) -> Dict:
    """
    Generate offline ARP attack sample packets to a PCAP file.

    Supported scenarios:
    - spoof: ARP spoofing toward victim
    - flood: ARP reply flood
    - scan: ARP who-has scan over network range
    - mitm: bidirectional spoofing (victim <-> gateway)
    - abuse: gratuitous ARP abuse/broadcast replies
    """
    rng = random.Random(seed)
    scenario = scenario.lower().strip()
    valid = {"spoof", "flood", "scan", "mitm", "abuse"}
    if scenario not in valid:
        raise ValueError(f"Unsupported scenario: {scenario}. Valid: {sorted(valid)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    victim_mac = _random_mac(rng)
    gateway_mac = _random_mac(rng)
    attacker_mac = _random_mac(rng)
    broadcast = "ff:ff:ff:ff:ff:ff"

    packets: List = []

    # Baseline benign traffic so downstream flow/AI pages still have data.
    packets.extend(
        _baseline_ip_packets(
            normal_src_ip=victim_ip,
            normal_dst_ip="8.8.8.8",
            normal_src_mac=victim_mac,
            normal_dst_mac=gateway_mac,
            n=max(10, packet_count // 8),
        )
    )

    if scenario == "spoof":
        for _ in range(packet_count):
            packets.append(
                Ether(src=attacker_mac, dst=victim_mac)
                / ARP(op=2, hwsrc=attacker_mac, psrc=gateway_ip, hwdst=victim_mac, pdst=victim_ip)
            )

    elif scenario == "flood":
        for _ in range(packet_count):
            fake_ip = f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
            fake_mac = _random_mac(rng)
            packets.append(
                Ether(src=fake_mac, dst=broadcast)
                / ARP(op=2, hwsrc=fake_mac, psrc=fake_ip, hwdst=broadcast, pdst=victim_ip)
            )

    elif scenario == "scan":
        net = ipaddress.ip_network(network_cidr, strict=False)
        targets = [str(ip) for ip in net.hosts()]
        if not targets:
            targets = [victim_ip]
        for i in range(packet_count):
            target = targets[i % len(targets)]
            packets.append(
                Ether(src=attacker_mac, dst=broadcast)
                / ARP(op=1, hwsrc=attacker_mac, psrc=attacker_ip, hwdst="00:00:00:00:00:00", pdst=target)
            )

    elif scenario == "mitm":
        for _ in range(packet_count):
            # Tell victim that gateway is attacker
            packets.append(
                Ether(src=attacker_mac, dst=victim_mac)
                / ARP(op=2, hwsrc=attacker_mac, psrc=gateway_ip, hwdst=victim_mac, pdst=victim_ip)
            )
            # Tell gateway that victim is attacker
            packets.append(
                Ether(src=attacker_mac, dst=gateway_mac)
                / ARP(op=2, hwsrc=attacker_mac, psrc=victim_ip, hwdst=gateway_mac, pdst=gateway_ip)
            )

    elif scenario == "abuse":
        for _ in range(packet_count):
            abuse_ip = f"192.168.56.{rng.randint(2, 254)}"
            packets.append(
                Ether(src=attacker_mac, dst=broadcast)
                / ARP(op=2, hwsrc=attacker_mac, psrc=abuse_ip, hwdst=broadcast, pdst=abuse_ip)
            )

    wrpcap(output_path, packets)

    return {
        "output_path": output_path,
        "scenario": scenario,
        "packet_count_total": len(packets),
        "packet_count_attack": len(packets) - max(10, packet_count // 8),
        "victim_ip": victim_ip,
        "gateway_ip": gateway_ip,
        "attacker_ip": attacker_ip,
        "attacker_mac": attacker_mac,
    }

