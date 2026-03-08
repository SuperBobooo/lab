"""
G4 mixed traffic sender (final demo edition) for SentinelAI.

Purpose:
- Generate comprehensive, bounded lab traffic for end-to-end validation:
  baseline + classic + variant patterns.
- Designed for authorized cyber-security training labs only.

Coverage:
1) Baseline normal traffic: TCP(80/443), DNS, UDP, low-rate steady flows.
2) Port scan: TCP + UDP fast scan, slow scan, segmented scan.
3) Brute-force pattern: repeated short TCP connects to single target port.
4) DoS-like pattern: bounded burst to fixed service ports.
5) Exfil-like pattern: burst large UDP + low-slow chunk transfer.
6) DNS anomaly: high-frequency random subdomain queries.
7) ARP optional (same L2): scan/flood/conflict-like samples.
8) Variant additions: jittered beacon, protocol-mixed low-rate probes.

Notes:
- This script does not exploit vulnerabilities.
- Keep traffic bounded with --rate-factor and internal hard caps.
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    from scapy.all import ARP, Ether, sendp  # type: ignore

    SCAPY_AVAILABLE = True
except Exception:
    SCAPY_AVAILABLE = False


TCP_SCAN_PORTS = [
    21, 22, 23, 25, 53, 80, 110, 123, 135, 139, 143, 443, 445, 993, 995,
    1433, 1521, 3306, 3389, 5432, 5900, 6379, 8080, 8443, 9001,
]
UDP_SCAN_PORTS = [53, 67, 68, 69, 123, 137, 161, 500, 514, 1900, 5353, 9999]


@dataclass
class TrafficContext:
    target_ip: str
    src_ips: list[str]
    http_port: int
    tls_port: int
    dns_port: int
    brute_force_port: int
    exfil_port: int
    dos_ports: list[int]
    phase_tick_s: float
    jitter_s: float
    dns_domain: str
    rate_factor: float
    log_path: Path
    arp_enable: bool
    arp_iface: Optional[str]
    arp_gateway_ip: Optional[str]
    arp_victim_ip: Optional[str]
    arp_scan_cidr_prefix: Optional[str]


@dataclass
class Phase:
    name: str
    duration_s: int


def _now() -> float:
    return time.time()


def _sleep_jitter(base_s: float, jitter_s: float) -> None:
    wait_s = base_s + random.uniform(-jitter_s, jitter_s) if jitter_s > 0 else base_s
    time.sleep(max(0.0, wait_s))


def _log(path: Path, event: str, detail: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"timestamp": _now(), "event": event, "detail": detail}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _pick_src_ip(ctx: TrafficContext) -> Optional[str]:
    if not ctx.src_ips:
        return None
    return random.choice(ctx.src_ips)


def _tcp_connect(target_ip: str, target_port: int, timeout_s: float = 0.25, src_ip: Optional[str] = None) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        if src_ip:
            sock.bind((src_ip, 0))
        sock.connect((target_ip, int(target_port)))
        sock.sendall(b"LAB_TRAFFIC\r\n")
        return True
    except Exception:
        return False
    finally:
        sock.close()


def _udp_send(target_ip: str, target_port: int, size: int, src_ip: Optional[str] = None) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if src_ip:
            sock.bind((src_ip, 0))
        payload = b"U" * max(1, int(size))
        sock.sendto(payload, (target_ip, int(target_port)))
        return True
    except Exception:
        return False
    finally:
        sock.close()


def _build_dns_query(domain: str, txid: Optional[int] = None) -> bytes:
    if txid is None:
        txid = random.randint(0, 65535)
    header = struct.pack("!HHHHHH", txid, 0x0100, 1, 0, 0, 0)
    labels = domain.split(".")
    qname = b"".join(len(x).to_bytes(1, "big") + x.encode("ascii", errors="ignore") for x in labels) + b"\x00"
    return header + qname + struct.pack("!HH", 1, 1)


def _dns_query(target_ip: str, dns_port: int, domain: str, src_ip: Optional[str] = None) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.3)
    try:
        if src_ip:
            sock.bind((src_ip, 0))
        sock.sendto(_build_dns_query(domain), (target_ip, int(dns_port)))
        return True
    except Exception:
        return False
    finally:
        sock.close()


def _http_get(target_ip: str, port: int, path: str = "/", src_ip: Optional[str] = None) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.4)
    try:
        if src_ip:
            sock.bind((src_ip, 0))
        sock.connect((target_ip, int(port)))
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {target_ip}\r\n"
            "User-Agent: SentinelLab-G4/2.0\r\n"
            "Connection: close\r\n\r\n"
        ).encode()
        sock.sendall(req)
        return True
    except Exception:
        return False
    finally:
        sock.close()


def _random_subdomain(base_domain: str, min_len: int = 8, max_len: int = 20) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    n = random.randint(min_len, max_len)
    sub = "".join(random.choice(chars) for _ in range(n))
    return f"{sub}.{base_domain}"


def _bounded(v: int, low: int, high: int) -> int:
    return max(low, min(high, v))


def _phase_baseline(ctx: TrafficContext, elapsed: int) -> None:
    src = _pick_src_ip(ctx)
    _http_get(ctx.target_ip, ctx.http_port, "/", src_ip=src)
    _tcp_connect(ctx.target_ip, ctx.tls_port, src_ip=src)
    _dns_query(ctx.target_ip, ctx.dns_port, ctx.dns_domain, src_ip=src)
    _udp_send(ctx.target_ip, 12345, size=64, src_ip=src)
    _log(ctx.log_path, "baseline", {"elapsed_s": elapsed})


def _phase_classic(ctx: TrafficContext, elapsed: int) -> None:
    # A) Fast TCP scan
    tcp_n = _bounded(int(10 * ctx.rate_factor), 4, 18)
    for p in random.sample(TCP_SCAN_PORTS, k=tcp_n):
        _tcp_connect(ctx.target_ip, p, timeout_s=0.12, src_ip=_pick_src_ip(ctx))

    # B) Fast UDP scan
    udp_n = _bounded(int(6 * ctx.rate_factor), 3, 10)
    for p in random.sample(UDP_SCAN_PORTS, k=udp_n):
        _udp_send(ctx.target_ip, p, size=56, src_ip=_pick_src_ip(ctx))

    # C) Brute-force style
    bf_n = _bounded(int(14 * ctx.rate_factor), 6, 30)
    for _ in range(bf_n):
        _tcp_connect(ctx.target_ip, ctx.brute_force_port, timeout_s=0.10, src_ip=_pick_src_ip(ctx))

    # D) DoS-like bounded burst
    dos_n = _bounded(int(22 * ctx.rate_factor), 8, 45)
    for _ in range(dos_n):
        port = random.choice(ctx.dos_ports)
        _udp_send(ctx.target_ip, port, size=160, src_ip=_pick_src_ip(ctx))

    # E) Exfil burst
    exfil_burst_n = _bounded(int(8 * ctx.rate_factor), 3, 14)
    for _ in range(exfil_burst_n):
        _udp_send(ctx.target_ip, ctx.exfil_port, size=1300, src_ip=_pick_src_ip(ctx))

    # F) DNS anomaly high rate
    dns_n = _bounded(int(7 * ctx.rate_factor), 3, 14)
    for _ in range(dns_n):
        _dns_query(ctx.target_ip, ctx.dns_port, _random_subdomain(ctx.dns_domain, 10, 18), src_ip=_pick_src_ip(ctx))

    _log(
        ctx.log_path,
        "classic",
        {
            "elapsed_s": elapsed,
            "tcp_scan_n": tcp_n,
            "udp_scan_n": udp_n,
            "bf_n": bf_n,
            "dos_n": dos_n,
            "exfil_burst_n": exfil_burst_n,
            "dns_n": dns_n,
        },
    )


def _phase_variant(ctx: TrafficContext, elapsed: int) -> None:
    # A) Slow scan (1-2s style when combined with phase tick+jitter)
    p = random.choice(TCP_SCAN_PORTS)
    _tcp_connect(ctx.target_ip, p, timeout_s=0.2, src_ip=_pick_src_ip(ctx))

    # B) Segmented scan (small batch now, next batch in later ticks)
    seg_n = _bounded(int(3 * ctx.rate_factor), 1, 5)
    for p in random.sample(TCP_SCAN_PORTS, k=seg_n):
        _tcp_connect(ctx.target_ip, p, timeout_s=0.16, src_ip=_pick_src_ip(ctx))

    # C) Low-slow exfil
    chunk = random.choice([180, 220, 260, 320, 420])
    _udp_send(ctx.target_ip, ctx.exfil_port, size=chunk, src_ip=_pick_src_ip(ctx))

    # D) DNS variant jitter + random labels
    dns_n = _bounded(int(2 * ctx.rate_factor), 1, 4)
    for _ in range(dns_n):
        _dns_query(ctx.target_ip, ctx.dns_port, _random_subdomain(ctx.dns_domain, 12, 24), src_ip=_pick_src_ip(ctx))

    # E) Protocol mixed probing (avoid single-rule dominance)
    _udp_send(ctx.target_ip, random.choice([53, 80, 443, 8080, 18002]), size=72, src_ip=_pick_src_ip(ctx))
    _tcp_connect(ctx.target_ip, random.choice([80, 443, 8080, 9001]), timeout_s=0.18, src_ip=_pick_src_ip(ctx))

    # F) Jitter beacon
    _udp_send(ctx.target_ip, 18002, size=random.choice([48, 64, 80]), src_ip=_pick_src_ip(ctx))

    _log(
        ctx.log_path,
        "variant",
        {"elapsed_s": elapsed, "slow_port": p, "seg_n": seg_n, "exfil_chunk": chunk, "dns_n": dns_n},
    )


def _arp_optional(ctx: TrafficContext, elapsed: int) -> None:
    if not ctx.arp_enable:
        return
    if not SCAPY_AVAILABLE:
        _log(ctx.log_path, "arp_skip", {"elapsed_s": elapsed, "reason": "scapy_missing"})
        return
    if not ctx.arp_iface:
        _log(ctx.log_path, "arp_skip", {"elapsed_s": elapsed, "reason": "arp_iface_missing"})
        return

    try:
        # 1) ARP scan-like requests (multi target in /24 prefix if provided)
        targets = []
        if ctx.arp_scan_cidr_prefix:
            for i in random.sample(range(2, 254), k=6):
                targets.append(f"{ctx.arp_scan_cidr_prefix}.{i}")
        elif ctx.arp_victim_ip:
            targets = [ctx.arp_victim_ip]
        else:
            targets = []

        for ip in targets:
            req = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(op=1, pdst=ip)
            sendp(req, iface=ctx.arp_iface, verbose=False, count=1)

        # 2) ARP flood-like small burst (bounded)
        flood_n = _bounded(int(4 * ctx.rate_factor), 2, 8)
        if targets:
            for _ in range(flood_n):
                ip = random.choice(targets)
                req = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(op=1, pdst=ip)
                sendp(req, iface=ctx.arp_iface, verbose=False, count=1)

        # 3) ARP conflict/spoof-like reply sample (single bounded message)
        if ctx.arp_gateway_ip and ctx.arp_victim_ip:
            spoof = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(
                op=2,
                psrc=ctx.arp_gateway_ip,
                pdst=ctx.arp_victim_ip,
                hwdst="ff:ff:ff:ff:ff:ff",
            )
            sendp(spoof, iface=ctx.arp_iface, verbose=False, count=1)

        _log(ctx.log_path, "arp", {"elapsed_s": elapsed, "target_count": len(targets)})
    except Exception as exc:
        _log(ctx.log_path, "arp_error", {"elapsed_s": elapsed, "error": str(exc)})


def _run_phase(ctx: TrafficContext, phase: Phase) -> None:
    print(f"[+] Phase start: {phase.name} ({phase.duration_s}s)")
    t0 = _now()
    ticks = 0
    while True:
        elapsed = int(_now() - t0)
        if elapsed >= phase.duration_s:
            break

        if phase.name == "baseline":
            _phase_baseline(ctx, elapsed)
        elif phase.name == "classic":
            _phase_classic(ctx, elapsed)
            _arp_optional(ctx, elapsed)
        elif phase.name == "variant":
            _phase_variant(ctx, elapsed)
            _arp_optional(ctx, elapsed)

        ticks += 1
        _sleep_jitter(ctx.phase_tick_s / max(0.1, ctx.rate_factor), ctx.jitter_s)

    print(f"[+] Phase done: {phase.name}, ticks={ticks}")


def build_g4_phases(total_s: int) -> list[Phase]:
    # 25% baseline + 45% classic + 30% variant
    baseline = max(20, int(total_s * 0.25))
    classic = max(30, int(total_s * 0.45))
    variant = max(20, total_s - baseline - classic)
    return [
        Phase("baseline", baseline),
        Phase("classic", classic),
        Phase("variant", variant),
    ]


def _parse_src_ips(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comprehensive G4 sender for SentinelAI final demo.")
    p.add_argument("--target-ip", required=True, help="SentinelAI host IP.")
    p.add_argument("--duration-s", type=int, default=300, help="Total scenario duration.")
    p.add_argument("--rate-factor", type=float, default=1.0, help="Traffic intensity factor (0.5~2.0 recommended).")
    p.add_argument("--seed", type=int, default=20260308, help="Random seed.")
    p.add_argument("--phase-tick-s", type=float, default=0.8, help="Base tick interval.")
    p.add_argument("--jitter-s", type=float, default=0.25, help="Tick jitter.")
    p.add_argument("--http-port", type=int, default=80)
    p.add_argument("--tls-port", type=int, default=443)
    p.add_argument("--dns-port", type=int, default=53)
    p.add_argument("--bruteforce-port", type=int, default=22)
    p.add_argument("--exfil-port", type=int, default=9001)
    p.add_argument("--dos-ports", default="80,443,8080", help="Comma-separated ports for dos-like bursts.")
    p.add_argument("--dns-domain", default="example.com")
    p.add_argument("--src-ips", default="", help="Optional comma-separated local source IPs for pseudo-distributed mode.")
    p.add_argument("--log-path", default="sender_g4_traffic.jsonl")

    # Optional ARP
    p.add_argument("--enable-arp", action="store_true")
    p.add_argument("--arp-iface", default=None)
    p.add_argument("--arp-gateway-ip", default=None)
    p.add_argument("--arp-victim-ip", default=None)
    p.add_argument("--arp-scan-cidr-prefix", default=None, help="e.g. 192.168.253 (for /24 style ARP scan targets).")

    p.add_argument(
        "--confirm-lab",
        action="store_true",
        help="Required safety switch: confirm authorized lab use.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_lab:
        print("[!] Refuse to run without --confirm-lab")
        return 2

    random.seed(args.seed)

    try:
        dos_ports = [int(x.strip()) for x in str(args.dos_ports).split(",") if x.strip()]
    except Exception:
        print("[!] --dos-ports parse error")
        return 2
    if not dos_ports:
        dos_ports = [80, 443, 8080]

    ctx = TrafficContext(
        target_ip=args.target_ip,
        src_ips=_parse_src_ips(args.src_ips),
        http_port=int(args.http_port),
        tls_port=int(args.tls_port),
        dns_port=int(args.dns_port),
        brute_force_port=int(args.bruteforce_port),
        exfil_port=int(args.exfil_port),
        dos_ports=dos_ports,
        phase_tick_s=max(0.2, float(args.phase_tick_s)),
        jitter_s=max(0.0, float(args.jitter_s)),
        dns_domain=args.dns_domain,
        rate_factor=max(0.2, min(3.0, float(args.rate_factor))),
        log_path=Path(args.log_path),
        arp_enable=bool(args.enable_arp),
        arp_iface=args.arp_iface,
        arp_gateway_ip=args.arp_gateway_ip,
        arp_victim_ip=args.arp_victim_ip,
        arp_scan_cidr_prefix=args.arp_scan_cidr_prefix,
    )

    phases = build_g4_phases(int(args.duration_s))
    _log(
        ctx.log_path,
        "run_start",
        {
            "target_ip": ctx.target_ip,
            "duration_s": int(args.duration_s),
            "rate_factor": ctx.rate_factor,
            "phases": [f"{x.name}:{x.duration_s}" for x in phases],
            "arp_enable": ctx.arp_enable,
        },
    )

    print("[+] G4 final demo sender started")
    print(f"    target={ctx.target_ip}, duration={int(args.duration_s)}s, rate_factor={ctx.rate_factor}")
    print(f"    phases={[(x.name, x.duration_s) for x in phases]}")
    print(f"    log={ctx.log_path}")

    for phase in phases:
        _run_phase(ctx, phase)

    _log(ctx.log_path, "run_end", {"target_ip": ctx.target_ip})
    print("[+] Scenario done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

