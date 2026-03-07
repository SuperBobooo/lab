"""
G4 mixed traffic sender for SentinelAI validation.

Run this script on another host in the same test network, while SentinelAI host
is doing live capture. This script sends:
1) Baseline normal traffic
2) Classic attack-like traffic patterns
3) Variant / stealth traffic patterns

Important:
- Use ONLY in authorized lab environments.
- This script generates traffic patterns for detection validation, not exploitation.
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
from typing import Callable, Iterable

try:
    # Optional: only for ARP lab scenarios in same L2.
    from scapy.all import ARP, Ether, sendp  # type: ignore

    SCAPY_AVAILABLE = True
except Exception:
    SCAPY_AVAILABLE = False


@dataclass
class TrafficContext:
    target_ip: str
    http_port: int
    dns_port: int
    brute_force_port: int
    exfil_port: int
    phase_sleep_s: float
    jitter_s: float
    log_path: Path
    dns_domain: str
    rate_factor: float
    arp_enable: bool
    arp_iface: str | None
    arp_gateway_ip: str | None
    arp_victim_ip: str | None


def _now_ts() -> float:
    return time.time()


def _jitter(base: float, jitter: float) -> float:
    if jitter <= 0:
        return max(0.0, base)
    return max(0.0, base + random.uniform(-jitter, jitter))


def _log(log_path: Path, event: str, detail: dict) -> None:
    record = {
        "timestamp": _now_ts(),
        "event": event,
        "detail": detail,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_connect_tcp(target_ip: str, target_port: int, timeout_s: float = 0.3) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect((target_ip, target_port))
        payload = b"HELLO_SENTINELAI\r\n"
        sock.sendall(payload)
        return True
    except Exception:
        return False
    finally:
        sock.close()


def _safe_send_udp(target_ip: str, target_port: int, payload_size: int = 64) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        payload = b"A" * max(1, payload_size)
        sock.sendto(payload, (target_ip, target_port))
        return True
    except Exception:
        return False
    finally:
        sock.close()


def _build_dns_query(domain: str, txid: int | None = None) -> bytes:
    if txid is None:
        txid = random.randint(0, 65535)
    # Standard query, recursion desired.
    header = struct.pack("!HHHHHH", txid, 0x0100, 1, 0, 0, 0)
    qname = b"".join(len(x).to_bytes(1, "big") + x.encode("ascii", errors="ignore") for x in domain.split(".")) + b"\x00"
    qtype_qclass = struct.pack("!HH", 1, 1)  # A / IN
    return header + qname + qtype_qclass


def _send_dns_query(dns_server_ip: str, dns_port: int, domain: str) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.3)
    try:
        pkt = _build_dns_query(domain)
        sock.sendto(pkt, (dns_server_ip, dns_port))
        return True
    except Exception:
        return False
    finally:
        sock.close()


def _http_get(target_ip: str, target_port: int, path: str = "/") -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        sock.connect((target_ip, target_port))
        req = f"GET {path} HTTP/1.1\r\nHost: {target_ip}\r\nUser-Agent: SentinelLab/1.0\r\nConnection: close\r\n\r\n".encode()
        sock.sendall(req)
        return True
    except Exception:
        return False
    finally:
        sock.close()


def _random_subdomain(base_domain: str, prefix_len: int = 10) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    prefix = "".join(random.choice(alphabet) for _ in range(prefix_len))
    return f"{prefix}.{base_domain}"


def _phase_baseline(ctx: TrafficContext, elapsed_s: int) -> None:
    # Normal business-like traffic.
    _http_get(ctx.target_ip, ctx.http_port, "/")
    _safe_connect_tcp(ctx.target_ip, 443, timeout_s=0.2)
    _send_dns_query(ctx.target_ip, ctx.dns_port, ctx.dns_domain)
    _safe_send_udp(ctx.target_ip, 12345, payload_size=64)
    _log(ctx.log_path, "baseline_tick", {"elapsed_s": elapsed_s})


def _phase_classic(ctx: TrafficContext, elapsed_s: int) -> None:
    # 1) Port scan pattern.
    scan_ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 3306, 3389, 5432, 6379, 8080]
    random.shuffle(scan_ports)
    # Rate is controlled by selecting subset each tick.
    pick_n = max(3, min(len(scan_ports), int(6 * ctx.rate_factor)))
    for p in scan_ports[:pick_n]:
        _safe_connect_tcp(ctx.target_ip, p, timeout_s=0.15)

    # 2) Brute-force-like repeated attempts.
    bf_attempts = max(4, int(8 * ctx.rate_factor))
    for _ in range(bf_attempts):
        _safe_connect_tcp(ctx.target_ip, ctx.brute_force_port, timeout_s=0.15)

    # 3) DoS-like burst (bounded).
    burst_n = max(6, int(15 * ctx.rate_factor))
    for _ in range(burst_n):
        _safe_send_udp(ctx.target_ip, ctx.http_port, payload_size=128)

    # 4) Data exfiltration-like large payload bursts.
    exfil_n = max(2, int(4 * ctx.rate_factor))
    for _ in range(exfil_n):
        _safe_send_udp(ctx.target_ip, ctx.exfil_port, payload_size=1200)

    _log(
        ctx.log_path,
        "classic_tick",
        {"elapsed_s": elapsed_s, "scan_ports": pick_n, "bf_attempts": bf_attempts, "dos_burst": burst_n, "exfil_burst": exfil_n},
    )


def _phase_variant(ctx: TrafficContext, elapsed_s: int) -> None:
    # Variant 1: slow scan with jitter (harder for simple threshold).
    slow_scan_ports = [22, 25, 53, 80, 110, 143, 443, 3389, 5900, 8080]
    p = random.choice(slow_scan_ports)
    _safe_connect_tcp(ctx.target_ip, p, timeout_s=0.2)

    # Variant 2: low-and-slow exfil chunks.
    chunk_size = random.choice([200, 280, 360, 420])
    _safe_send_udp(ctx.target_ip, ctx.exfil_port, payload_size=chunk_size)

    # Variant 3: DNS tunnel-like random subdomains.
    for _ in range(max(1, int(2 * ctx.rate_factor))):
        domain = _random_subdomain(ctx.dns_domain, prefix_len=random.choice([8, 12, 16]))
        _send_dns_query(ctx.target_ip, ctx.dns_port, domain)

    # Variant 4: jittered beacon pattern.
    _safe_send_udp(ctx.target_ip, 18002, payload_size=random.choice([48, 64, 80]))

    _log(ctx.log_path, "variant_tick", {"elapsed_s": elapsed_s, "slow_scan_port": p, "exfil_chunk": chunk_size})


def _phase_arp_optional(ctx: TrafficContext, elapsed_s: int) -> None:
    if not ctx.arp_enable:
        return
    if not SCAPY_AVAILABLE:
        _log(ctx.log_path, "arp_skip", {"reason": "scapy_not_available"})
        return
    if not ctx.arp_iface or not ctx.arp_gateway_ip or not ctx.arp_victim_ip:
        _log(ctx.log_path, "arp_skip", {"reason": "missing_iface_or_ips"})
        return
    try:
        # Controlled ARP samples: scan-style requests + spoof-like gratuitous replies.
        req = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(op=1, pdst=ctx.arp_victim_ip)
        sendp(req, iface=ctx.arp_iface, verbose=False, count=max(1, int(2 * ctx.rate_factor)))

        spoof = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(
            op=2,
            psrc=ctx.arp_gateway_ip,
            pdst=ctx.arp_victim_ip,
            hwdst="ff:ff:ff:ff:ff:ff",
        )
        sendp(spoof, iface=ctx.arp_iface, verbose=False, count=max(1, int(2 * ctx.rate_factor)))
        _log(ctx.log_path, "arp_tick", {"elapsed_s": elapsed_s, "iface": ctx.arp_iface})
    except Exception as exc:
        _log(ctx.log_path, "arp_error", {"error": str(exc)})


@dataclass
class Phase:
    name: str
    duration_s: int
    action: Callable[[TrafficContext, int], None]


def _run_phase(ctx: TrafficContext, phase: Phase) -> None:
    print(f"[+] Phase start: {phase.name} ({phase.duration_s}s)")
    phase_start = time.time()
    tick = 0
    while True:
        elapsed = time.time() - phase_start
        if elapsed >= phase.duration_s:
            break
        phase.action(ctx, int(elapsed))
        # Optional ARP samples are attached in classic/variant windows.
        if phase.name in {"classic", "variant"}:
            _phase_arp_optional(ctx, int(elapsed))
        tick += 1
        sleep_s = _jitter(ctx.phase_sleep_s / max(ctx.rate_factor, 0.1), ctx.jitter_s)
        time.sleep(sleep_s)
    print(f"[+] Phase done: {phase.name}, ticks={tick}")


def build_g4_phases(total_s: int) -> Iterable[Phase]:
    # Keep 30/40/30 split: baseline / classic / variant.
    baseline_s = max(10, int(total_s * 0.30))
    classic_s = max(10, int(total_s * 0.40))
    variant_s = max(10, total_s - baseline_s - classic_s)

    return [
        Phase(name="baseline", duration_s=baseline_s, action=_phase_baseline),
        Phase(name="classic", duration_s=classic_s, action=_phase_classic),
        Phase(name="variant", duration_s=variant_s, action=_phase_variant),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G4 mixed traffic sender for SentinelAI lab.")
    parser.add_argument("--target-ip", required=True, help="SentinelAI host IP (capture target).")
    parser.add_argument("--duration-s", type=int, default=300, help="Total scenario duration in seconds (default: 300).")
    parser.add_argument("--http-port", type=int, default=80, help="HTTP target port for normal traffic.")
    parser.add_argument("--dns-port", type=int, default=53, help="DNS target port.")
    parser.add_argument("--bruteforce-port", type=int, default=22, help="Port used for brute-force-like attempts.")
    parser.add_argument("--exfil-port", type=int, default=9001, help="UDP port used for exfil-like traffic.")
    parser.add_argument("--phase-sleep-s", type=float, default=0.5, help="Base sleep interval per tick.")
    parser.add_argument("--jitter-s", type=float, default=0.15, help="Random jitter on sleep interval.")
    parser.add_argument("--dns-domain", default="example.com", help="Base domain used for DNS queries.")
    parser.add_argument("--rate-factor", type=float, default=1.0, help="Traffic intensity factor (0.5~3.0 recommended).")
    parser.add_argument("--seed", type=int, default=20260305, help="Random seed.")
    parser.add_argument("--log-path", default="sender_g4_traffic.jsonl", help="Sender local audit log path.")

    # Optional ARP lab options.
    parser.add_argument("--enable-arp", action="store_true", help="Enable optional ARP sample generation.")
    parser.add_argument("--arp-iface", default=None, help="Interface name for ARP packets (scapy sendp).")
    parser.add_argument("--arp-gateway-ip", default=None, help="Gateway IP used in ARP spoof-like sample.")
    parser.add_argument("--arp-victim-ip", default=None, help="Victim IP used in ARP sample.")

    parser.add_argument(
        "--confirm-lab",
        action="store_true",
        help="Required safety flag. Must be set to run traffic generation in authorized lab.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.confirm_lab:
        print("[!] Refuse to run without --confirm-lab.")
        print("    Use only in authorized lab environments.")
        return 2

    random.seed(args.seed)

    ctx = TrafficContext(
        target_ip=args.target_ip,
        http_port=args.http_port,
        dns_port=args.dns_port,
        brute_force_port=args.bruteforce_port,
        exfil_port=args.exfil_port,
        phase_sleep_s=args.phase_sleep_s,
        jitter_s=args.jitter_s,
        log_path=Path(args.log_path),
        dns_domain=args.dns_domain,
        rate_factor=max(0.1, float(args.rate_factor)),
        arp_enable=bool(args.enable_arp),
        arp_iface=args.arp_iface,
        arp_gateway_ip=args.arp_gateway_ip,
        arp_victim_ip=args.arp_victim_ip,
    )

    _log(
        ctx.log_path,
        "run_start",
        {
            "target_ip": ctx.target_ip,
            "duration_s": args.duration_s,
            "rate_factor": ctx.rate_factor,
            "arp_enable": ctx.arp_enable,
        },
    )

    print("[+] Start G4 mixed scenario")
    print(f"    target={ctx.target_ip}, duration={args.duration_s}s, rate_factor={ctx.rate_factor}")
    print(f"    sender_log={ctx.log_path}")

    phases = list(build_g4_phases(args.duration_s))
    for phase in phases:
        _run_phase(ctx, phase)

    _log(ctx.log_path, "run_end", {"target_ip": ctx.target_ip})
    print("[+] Scenario done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

