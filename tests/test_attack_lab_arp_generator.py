from pathlib import Path
import sys

from scapy.all import ARP, rdpcap  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack_lab.arp_sample_generator import generate_arp_attack_pcap


def test_generate_arp_attack_pcap_creates_valid_pcap_with_arp(tmp_path):
    output = tmp_path / "arp_spoof_sample.pcap"
    meta = generate_arp_attack_pcap(
        output_path=str(output),
        scenario="spoof",
        packet_count=40,
        seed=7,
    )

    assert output.exists()
    assert meta["packet_count_total"] > 0

    packets = rdpcap(str(output))
    assert len(packets) > 0
    assert any(ARP in p for p in packets)
