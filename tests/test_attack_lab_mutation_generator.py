from pathlib import Path

from attack_lab.mutation_sample_generator import generate_mutation_attack_pcap
from pcap_ingest import parse_pcap


def test_generate_mutation_attack_pcap_and_parse(tmp_path: Path):
    out = tmp_path / "mutation_stealth_scan.pcap"
    meta = generate_mutation_attack_pcap(
        output_path=str(out),
        scenario="stealth_scan",
        packet_count=40,
    )
    assert out.exists()
    assert meta["scenario"] == "stealth_scan"
    assert meta["packet_count_total"] > 0

    records = parse_pcap(str(out))
    assert len(records) > 0
