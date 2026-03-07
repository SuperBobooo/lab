from pathlib import Path
import sys

from scapy.all import DNS, DNSQR, Ether, IP, Raw, TCP, UDP, wrpcap  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcap_ingest import parse_pcap


def test_parse_pcap_extracts_dns_and_http_fields(tmp_path):
    pcap_path = tmp_path / "proto_test.pcap"

    dns_pkt = (
        Ether(src="00:aa:bb:cc:dd:01", dst="00:aa:bb:cc:dd:02")
        / IP(src="10.1.1.1", dst="8.8.8.8")
        / UDP(sport=53000, dport=53)
        / DNS(rd=1, qd=DNSQR(qname="example.com"))
    )

    http_payload = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
    http_pkt = (
        Ether(src="00:aa:bb:cc:dd:03", dst="00:aa:bb:cc:dd:04")
        / IP(src="10.1.1.2", dst="10.1.1.3")
        / TCP(sport=12345, dport=80, flags="PA")
        / Raw(load=http_payload)
    )

    wrpcap(str(pcap_path), [dns_pkt, http_pkt])
    records = parse_pcap(str(pcap_path))

    assert len(records) == 2

    dns_records = [r for r in records if r.get("l7_protocol") == "DNS"]
    assert dns_records, "DNS record was not identified"
    assert dns_records[0].get("dns_query") == "example.com"
    assert dns_records[0].get("dns_qtype") == 1

    http_records = [r for r in records if r.get("l7_protocol") == "HTTP"]
    assert http_records, "HTTP record was not identified"
    assert http_records[0].get("http_method") == "GET"
    assert http_records[0].get("http_host") == "example.com"
    assert http_records[0].get("http_path") == "/index.html"
