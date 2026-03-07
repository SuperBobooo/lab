from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_generator import DataGenerator
from feature_adapter import adapt_flows_for_model


def test_adapt_flows_for_model_contains_required_feature_columns():
    flows_df = pd.DataFrame(
        [
            {
                "flow_id": 1,
                "start_time": "2026-03-02T12:00:00",
                "end_time": "2026-03-02T12:00:10",
                "duration": 10.0,
                "src_ip": "10.0.0.1",
                "dst_ip": "10.0.0.2",
                "src_port": 12345,
                "dst_port": 80,
                "protocol": "TCP",
                "packet_count": 4,
                "total_bytes": 400.0,
                "total_payload_bytes": 300.0,
            }
        ]
    )

    adapted = adapt_flows_for_model(flows_df)
    required_features = set(DataGenerator().get_feature_columns())

    assert not adapted.empty
    assert required_features.issubset(set(adapted.columns))
    assert {"timestamp", "label", "src_ip", "dst_ip", "src_port", "dst_port", "protocol"}.issubset(
        set(adapted.columns)
    )
