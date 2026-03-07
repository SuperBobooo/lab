from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flow_builder import FLOW_OUTPUT_COLUMNS, build_flows
from pcap_ingest import parse_pcap


def test_build_flows_from_minimal_pcap_has_required_columns():
    pcap_path = Path("data/test_minimal.pcap")
    assert pcap_path.exists(), "Missing test capture: data/test_minimal.pcap"

    packets = parse_pcap(str(pcap_path))
    packets_df = pd.DataFrame(packets)

    flows_df = build_flows(packets_df, idle_timeout_s=60)

    assert isinstance(flows_df, pd.DataFrame)
    assert len(flows_df) >= 1
    assert set(FLOW_OUTPUT_COLUMNS).issubset(set(flows_df.columns))
    zero_duration = flows_df[flows_df["duration"] == 0]
    if not zero_duration.empty:
        assert (zero_duration["bytes_per_second"] == 0).all()
