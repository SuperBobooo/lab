from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcap_ingest import parse_pcap


REQUIRED_FIELDS = {
    "timestamp",
    "src_mac",
    "dst_mac",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "packet_length",
    "payload_size",
    "payload_entropy",
}


def test_parse_pcap_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        parse_pcap("data/does_not_exist.pcap")


def test_parse_pcap_non_pcap_raises_value_error():
    with pytest.raises(ValueError):
        parse_pcap("data/not_pcap.txt")


def test_parse_pcap_valid_returns_list_of_dict_with_required_fields():
    pcap_path = Path("data/test_minimal.pcap")
    assert pcap_path.exists(), "Missing test capture: data/test_minimal.pcap"

    records = parse_pcap(str(pcap_path))

    assert isinstance(records, list)
    assert len(records) > 0
    assert all(isinstance(item, dict) for item in records)
    assert all(REQUIRED_FIELDS.issubset(item.keys()) for item in records)
