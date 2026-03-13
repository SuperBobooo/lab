"""
SentinelAI - Network Intrusion Detection System
Data Generation and Model Training Module

This module provides functions for:
1. Generating synthetic network flow data
2. Feature engineering for network flows
3. Training anomaly detection models
4. Generating mock datasets for testing
"""

import pandas as pd
import numpy as np
import datetime
import uuid
import json
import os
from faker import Faker
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib

class DataGenerator:
    """Class for generating synthetic network flow data and training models"""

    def __init__(self, seed=42):
        """Initialize the data generator with a random seed for reproducibility"""
        self.seed = seed
        np.random.seed(seed)
        self.faker = Faker()
        Faker.seed(seed)

    def get_attack_catalog(self):
        """Return supported synthetic attack types and UI description."""
        return {
            'port_scan': '端口扫描（高频低载荷）',
            'brute_force': '暴力破解（重复短连接）',
            'data_exfiltration': '数据外传（大字节/长时）',
            'dos': 'DoS突发（短时高压）',
            'dns_tunnel': 'DNS隧道样式（长域名高频）',
            'slow_scan': '低慢扫描（规避型）',
            'periodic_beacon': '周期心跳（带抖动）',
            'lateral_movement': '横向移动（内网多目标）',
            'protocol_mismatch': '协议伪装（端口-协议不匹配）',
            'low_slow_exfiltration': '低慢外传（小包长时）',
        }

    def generate_synthetic_flows(self, n_normal=1000, n_attack=200, attack_types=None):
        """
        Generate synthetic network flows with normal and attack patterns

        Parameters:
        -----------
        n_normal : int
            Number of normal flows to generate
        n_attack : int
            Number of attack flows to generate
        attack_types : list
            List of attack types to generate (default: all types)

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing synthetic network flows
        """
        # Define attack types if not provided
        if not attack_types:
            attack_types = ['port_scan', 'brute_force', 'data_exfiltration', 'dos']

        now = datetime.datetime.now()
        records = []

        # Normal traffic patterns
        for _ in range(n_normal):
            ts = now + datetime.timedelta(seconds=np.random.randint(0, 3600))
            flags = np.random.choice(['SYN', 'ACK', 'PSH-ACK', 'SYN-ACK', 'RST', 'FIN', 'FIN-ACK', ''])
            proto = np.random.choice(['TCP', 'UDP', 'ICMP'], p=[0.7, 0.25, 0.05])

            # Select realistic destination ports based on protocol
            if proto == 'TCP':
                dst_port = np.random.choice([80, 443, 8080, 22, 25, 143, 3306, 5432],
                                            p=[0.3, 0.3, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
            elif proto == 'UDP':
                dst_port = np.random.choice([53, 123, 161, 1900, 5353],
                                            p=[0.5, 0.2, 0.1, 0.1, 0.1])
            else:  # ICMP
                dst_port = 0

            # Adjust parameters for different connection types
            if dst_port in [80, 443, 8080]:  # Web traffic
                duration = np.random.exponential(scale=200)
                bytes_count = np.random.randint(100, 15000)
                packet_count = max(1, int(bytes_count / np.random.randint(40, 1500)))
            elif dst_port == 53:  # DNS
                duration = np.random.exponential(scale=20)
                bytes_count = np.random.randint(50, 300)
                packet_count = np.random.randint(1, 3)
            elif dst_port in [22, 3306, 5432]:  # SSH, DB
                duration = np.random.exponential(scale=500)
                bytes_count = np.random.randint(100, 5000)
                packet_count = max(1, int(bytes_count / np.random.randint(50, 150)))
            else:  # Other protocols
                duration = np.random.exponential(scale=100)
                bytes_count = np.random.randint(40, 1500)
                packet_count = max(1, int(bytes_count / np.random.randint(40, 100)))

            records.append({
                'timestamp': ts,
                'src_ip': self.faker.ipv4_private(),
                'dst_ip': self.faker.ipv4_private() if np.random.random() < 0.7 else self.faker.ipv4_public(),
                'src_port': np.random.randint(1024, 65535),
                'dst_port': dst_port,
                'protocol': proto,
                'duration_ms': duration,
                'total_bytes': bytes_count,
                'packet_count': packet_count,
                'tcp_flags': flags if proto == 'TCP' else '',
                'label': 'normal',
                'attack_type': None
            })

        # Attack traffic (rule-aligned generation)
        hotspot_src = [self.faker.ipv4_private() for _ in range(5)]
        hotspot_dst = [self.faker.ipv4_private() for _ in range(10)]
        public_targets = [self.faker.ipv4_public() for _ in range(12)]
        common_scan_ports = [21, 22, 23, 25, 53, 80, 110, 123, 135, 139, 143, 443, 445, 993, 995, 1433, 1521, 3306, 3389, 5432, 5900, 6379, 8080, 8443, 9001]

        def _make_record(ts, src_ip, dst_ip, src_port, dst_port, protocol, duration_ms, total_bytes, packet_count, tcp_flags, attack_type, l7_protocol=None, dns_query=None, dns_rcode=None):
            return {
                'timestamp': ts,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'duration_ms': duration_ms,
                'total_bytes': total_bytes,
                'packet_count': packet_count,
                'tcp_flags': tcp_flags,
                'label': 'attack',
                'attack_type': attack_type,
                'l7_protocol': l7_protocol,
                'dns_query': dns_query,
                'dns_rcode': dns_rcode,
            }

        def _random_dns_query():
            sub = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=np.random.randint(12, 28)))
            return f"{sub}.{self.faker.domain_name()}"

        attack_counts = {}
        if n_attack > 0 and attack_types:
            base = n_attack // len(attack_types)
            rem = n_attack % len(attack_types)
            for idx, attack_type in enumerate(attack_types):
                attack_counts[attack_type] = base + (1 if idx < rem else 0)

        for attack_type in attack_types:
            remaining = int(attack_counts.get(attack_type, 0))
            if remaining <= 0:
                continue

            # Spread scenarios across the hour so incidents can form and be separated.
            scenario_anchor = now + datetime.timedelta(seconds=int(np.random.randint(0, 3300)))
            anchor_src = str(np.random.choice(hotspot_src))
            anchor_dst = str(np.random.choice(hotspot_dst))
            anchor_public = str(np.random.choice(public_targets))

            if attack_type == 'port_scan':
                while remaining > 0:
                    cluster_size = int(min(remaining, np.random.randint(24, 72)))
                    window_start = scenario_anchor + datetime.timedelta(seconds=int(np.random.randint(0, 180)))
                    ports = np.random.choice(range(1, 4096), size=cluster_size, replace=False)
                    for port in ports:
                        ts = window_start + datetime.timedelta(seconds=float(np.random.uniform(0, 45)))
                        records.append(_make_record(
                            ts, anchor_src, anchor_dst,
                            int(np.random.randint(1024, 65535)), int(port),
                            'TCP', float(np.random.exponential(scale=6)),
                            int(np.random.randint(40, 80)), 1, 'SYN', 'port_scan'
                        ))
                    remaining -= cluster_size

            elif attack_type == 'slow_scan':
                while remaining > 0:
                    cluster_size = int(min(remaining, np.random.randint(12, 36)))
                    window_start = scenario_anchor + datetime.timedelta(seconds=int(np.random.randint(0, 240)))
                    ports = np.random.choice(common_scan_ports, size=cluster_size, replace=True)
                    for port in ports:
                        ts = window_start + datetime.timedelta(seconds=float(np.random.uniform(0, 110)))
                        records.append(_make_record(
                            ts, anchor_src, anchor_dst,
                            int(np.random.randint(1024, 65535)), int(port),
                            np.random.choice(['TCP', 'UDP'], p=[0.75, 0.25]),
                            float(np.random.exponential(scale=16)),
                            int(np.random.randint(48, 140)), 1, 'SYN', 'slow_scan'
                        ))
                    remaining -= cluster_size

            elif attack_type == 'brute_force':
                while remaining > 0:
                    cluster_size = int(min(remaining, np.random.randint(18, 64)))
                    target_port = int(np.random.choice([22, 23, 3389, 5900]))
                    window_start = scenario_anchor + datetime.timedelta(seconds=int(np.random.randint(0, 240)))
                    for _ in range(cluster_size):
                        ts = window_start + datetime.timedelta(seconds=float(np.random.uniform(0, 50)))
                        records.append(_make_record(
                            ts, anchor_src, anchor_dst,
                            int(np.random.randint(1024, 65535)), target_port,
                            'TCP', float(np.random.exponential(scale=35)),
                            int(np.random.randint(60, 220)),
                            int(np.random.randint(1, 3)),
                            str(np.random.choice(['SYN', 'SYN-ACK', 'ACK'])),
                            'brute_force'
                        ))
                    remaining -= cluster_size

            elif attack_type == 'dns_tunnel':
                while remaining > 0:
                    cluster_size = int(min(remaining, np.random.randint(24, 80)))
                    window_start = scenario_anchor + datetime.timedelta(seconds=int(np.random.randint(0, 240)))
                    for _ in range(cluster_size):
                        ts = window_start + datetime.timedelta(seconds=float(np.random.uniform(0, 55)))
                        rcode = 3 if np.random.random() < 0.35 else 0
                        records.append(_make_record(
                            ts, anchor_src, anchor_public,
                            int(np.random.randint(1024, 65535)), 53,
                            'UDP', float(np.random.exponential(scale=90)),
                            int(np.random.randint(220, 2000)),
                            int(np.random.randint(2, 20)),
                            '', 'dns_tunnel',
                            l7_protocol='DNS',
                            dns_query=_random_dns_query(),
                            dns_rcode=rcode
                        ))
                    remaining -= cluster_size

            elif attack_type == 'periodic_beacon':
                while remaining > 0:
                    cluster_size = int(min(remaining, np.random.randint(8, 24)))
                    interval_s = float(np.random.choice([8, 10, 12, 15]))
                    start_time = scenario_anchor + datetime.timedelta(seconds=int(np.random.randint(0, 300)))
                    for idx in range(cluster_size):
                        jitter = float(np.random.uniform(-1.0, 1.0))
                        ts = start_time + datetime.timedelta(seconds=(idx * interval_s + jitter))
                        records.append(_make_record(
                            ts, anchor_src, anchor_public,
                            int(np.random.randint(1024, 65535)),
                            int(np.random.choice([443, 8080, 9001])),
                            str(np.random.choice(['TCP', 'UDP'], p=[0.65, 0.35])),
                            float(np.random.exponential(scale=45)),
                            int(np.random.randint(80, 420)),
                            int(np.random.randint(1, 4)),
                            str(np.random.choice(['ACK', 'PSH-ACK', ''])),
                            'periodic_beacon'
                        ))
                    remaining -= cluster_size

            elif attack_type == 'dos':
                while remaining > 0:
                    cluster_size = int(min(remaining, np.random.randint(24, 90)))
                    base_window = scenario_anchor + datetime.timedelta(seconds=int(np.random.randint(0, 240)))

                    # Create two low-load windows, then one burst window for TRAFFIC_BURST rule.
                    baseline_points = min(cluster_size // 4, 12)
                    for _ in range(baseline_points):
                        ts = base_window + datetime.timedelta(seconds=float(np.random.uniform(0, 110)))
                        records.append(_make_record(
                            ts, anchor_src, anchor_dst,
                            int(np.random.randint(1024, 65535)),
                            int(np.random.choice([80, 443, 8080, 53])),
                            str(np.random.choice(['TCP', 'UDP', 'ICMP'], p=[0.6, 0.25, 0.15])),
                            float(np.random.exponential(scale=20)),
                            int(np.random.randint(800, 3500)),
                            int(np.random.randint(1, 4)),
                            str(np.random.choice(['SYN', 'ACK', ''])),
                            'dos'
                        ))

                    burst_points = cluster_size - baseline_points
                    burst_window = base_window + datetime.timedelta(seconds=120)
                    for _ in range(burst_points):
                        ts = burst_window + datetime.timedelta(seconds=float(np.random.uniform(0, 40)))
                        records.append(_make_record(
                            ts, anchor_src, anchor_dst,
                            int(np.random.randint(1024, 65535)),
                            int(np.random.choice([80, 443, 8080, 53])),
                            str(np.random.choice(['TCP', 'UDP', 'ICMP'], p=[0.7, 0.2, 0.1])),
                            float(np.random.exponential(scale=10)),
                            int(np.random.randint(3000, 14000)),
                            int(np.random.randint(2, 8)),
                            'SYN',
                            'dos'
                        ))
                    remaining -= cluster_size

            elif attack_type == 'data_exfiltration':
                while remaining > 0:
                    cluster_size = int(min(remaining, np.random.randint(10, 36)))
                    start_time = scenario_anchor + datetime.timedelta(seconds=int(np.random.randint(0, 280)))
                    for _ in range(cluster_size):
                        ts = start_time + datetime.timedelta(seconds=float(np.random.uniform(0, 100)))
                        records.append(_make_record(
                            ts, anchor_src, anchor_public,
                            int(np.random.randint(1024, 65535)),
                            int(np.random.choice([443, 80, 53])),
                            str(np.random.choice(['TCP', 'UDP'], p=[0.7, 0.3])),
                            float(np.random.exponential(scale=420)),
                            int(np.random.randint(12000, 120000)),
                            int(np.random.randint(12, 120)),
                            str(np.random.choice(['PSH-ACK', 'ACK'])),
                            'data_exfiltration'
                        ))
                    remaining -= cluster_size

            elif attack_type == 'lateral_movement':
                while remaining > 0:
                    cluster_size = int(min(remaining, np.random.randint(10, 40)))
                    start_time = scenario_anchor + datetime.timedelta(seconds=int(np.random.randint(0, 300)))
                    for _ in range(cluster_size):
                        ts = start_time + datetime.timedelta(seconds=float(np.random.uniform(0, 70)))
                        records.append(_make_record(
                            ts, anchor_src, str(np.random.choice(hotspot_dst)),
                            int(np.random.randint(1024, 65535)),
                            int(np.random.choice([135, 139, 445, 3389, 5985, 22])),
                            'TCP',
                            float(np.random.exponential(scale=90)),
                            int(np.random.randint(120, 2600)),
                            int(np.random.randint(2, 20)),
                            str(np.random.choice(['SYN', 'SYN-ACK', 'ACK'])),
                            'lateral_movement'
                        ))
                    remaining -= cluster_size

            elif attack_type == 'protocol_mismatch':
                while remaining > 0:
                    cluster_size = int(min(remaining, np.random.randint(8, 30)))
                    start_time = scenario_anchor + datetime.timedelta(seconds=int(np.random.randint(0, 300)))
                    for _ in range(cluster_size):
                        proto = str(np.random.choice(['ICMP', 'UDP']))
                        dst_port = int(np.random.choice([53, 80, 443, 22, 3389, 3306])) if proto != 'ICMP' else 0
                        src_port = 0 if proto == 'ICMP' else int(np.random.randint(1024, 65535))
                        ts = start_time + datetime.timedelta(seconds=float(np.random.uniform(0, 70)))
                        records.append(_make_record(
                            ts, anchor_src, str(np.random.choice(hotspot_dst)),
                            src_port, dst_port, proto,
                            float(np.random.exponential(scale=25)),
                            int(np.random.randint(60, 700)),
                            int(np.random.randint(1, 6)),
                            '', 'protocol_mismatch'
                        ))
                    remaining -= cluster_size

            elif attack_type == 'low_slow_exfiltration':
                while remaining > 0:
                    cluster_size = int(min(remaining, np.random.randint(12, 42)))
                    start_time = scenario_anchor + datetime.timedelta(seconds=int(np.random.randint(0, 300)))
                    for _ in range(cluster_size):
                        ts = start_time + datetime.timedelta(seconds=float(np.random.uniform(0, 300)))
                        records.append(_make_record(
                            ts, anchor_src, anchor_public,
                            int(np.random.randint(1024, 65535)),
                            int(np.random.choice([443, 53, 8080])),
                            str(np.random.choice(['TCP', 'UDP'])),
                            float(np.random.exponential(scale=2400)),
                            int(np.random.randint(700, 10000)),
                            int(np.random.randint(8, 80)),
                            str(np.random.choice(['PSH-ACK', 'ACK', ''])),
                            'low_slow_exfiltration'
                        ))
                    remaining -= cluster_size

        df = pd.DataFrame(records)

        # Add unique flow ID
        df['flow_id'] = [str(uuid.uuid4())[:8] for _ in range(len(df))]

        # Apply feature engineering
        df = self._engineer_features(df)

        # Return sorted by timestamp
        return df.sort_values('timestamp').reset_index(drop=True)

    def _engineer_features(self, df):
        """
        Engineer features from raw network flow data

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing raw network flow data

        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional engineered features
        """
        # Rate-based features
        df['byte_rate'] = df['total_bytes'] / (df['duration_ms'] + 1e-3)  # Avoid division by zero
        df['packets_per_second'] = df['packet_count'] / (df['duration_ms'] + 1e-3) * 1000
        df['bytes_per_packet'] = df['total_bytes'] / (df['packet_count'] + 1e-3)

        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Protocol and flag features
        df['is_tcp'] = (df['protocol'] == 'TCP').astype(int)
        df['is_udp'] = (df['protocol'] == 'UDP').astype(int)
        df['is_icmp'] = (df['protocol'] == 'ICMP').astype(int)
        df['is_syn'] = df['tcp_flags'].str.contains('SYN').fillna(False).astype(int)
        df['is_ack'] = df['tcp_flags'].str.contains('ACK').fillna(False).astype(int)
        df['is_rst'] = df['tcp_flags'].str.contains('RST').fillna(False).astype(int)
        df['is_fin'] = df['tcp_flags'].str.contains('FIN').fillna(False).astype(int)
        df['is_psh'] = df['tcp_flags'].str.contains('PSH').fillna(False).astype(int)

        # Port categories
        df['is_web_port'] = df['dst_port'].isin([80, 443, 8080]).astype(int)
        df['is_db_port'] = df['dst_port'].isin([3306, 5432, 1433, 1521]).astype(int)
        df['is_mail_port'] = df['dst_port'].isin([25, 143, 465, 587, 993]).astype(int)
        df['is_file_port'] = df['dst_port'].isin([20, 21, 22, 139, 445]).astype(int)
        df['is_dns_port'] = (df['dst_port'] == 53).astype(int)

        return df

    def get_feature_columns(self):
        """
        Get the list of feature columns used for modeling

        Returns:
        --------
        list
            List of feature column names
        """
        return [
            'duration_ms', 'total_bytes', 'packet_count',
            'byte_rate', 'packets_per_second', 'bytes_per_packet',
            'hour_sin', 'hour_cos',
            'is_tcp', 'is_udp', 'is_icmp',
            'is_syn', 'is_ack', 'is_rst', 'is_fin', 'is_psh',
            'is_web_port', 'is_db_port', 'is_mail_port', 'is_file_port', 'is_dns_port'
        ]

    def train_isolation_forest(self, df, contamination=0.1):
        """
        Train an Isolation Forest model

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing network flow data
        contamination : float
            Proportion of outliers in the data

        Returns:
        --------
        sklearn.ensemble.IsolationForest
            Trained Isolation Forest model
        """
        # Get feature columns
        features = self.get_feature_columns()

        # Extract normal flows for training
        X = df[df['label'] == 'normal'][features]

        # Train model
        model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=self.seed,
            n_jobs=-1
        )
        model.fit(X)

        return model

    def train_ocsvm(self, df, nu=0.1):
        """
        Train a One-Class SVM model

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing network flow data
        nu : float
            An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors

        Returns:
        --------
        tuple
            (OneClassSVM model, StandardScaler)
        """
        # Get feature columns
        features = self.get_feature_columns()

        # Extract normal flows for training
        X = df[df['label'] == 'normal'][features]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        model.fit(X_scaled)

        return model, scaler

    def train_dbscan(self, df, eps=0.5, min_samples=5):
        """
        Train a DBSCAN model

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing network flow data
        eps : float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_samples : int
            The number of samples in a neighborhood for a point to be considered as a core point

        Returns:
        --------
        tuple
            (DBSCAN model, StandardScaler)
        """
        # Get feature columns
        features = self.get_feature_columns()

        # Extract normal flows for training
        X = df[df['label'] == 'normal'][features]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        model.fit(X_scaled)

        return model, scaler

    def score_flows(self, df, iso_model, ocsvm_model, ocsvm_scaler, dbscan_model, dbscan_scaler):
        """
        Score flows using trained models

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing network flow data
        iso_model : sklearn.ensemble.IsolationForest
            Trained Isolation Forest model
        ocsvm_model : sklearn.svm.OneClassSVM
            Trained One-Class SVM model
        ocsvm_scaler : sklearn.preprocessing.StandardScaler
            Scaler used for OCSVM
        dbscan_model : sklearn.cluster.DBSCAN
            Trained DBSCAN model
        dbscan_scaler : sklearn.preprocessing.StandardScaler
            Scaler used for DBSCAN

        Returns:
        --------
        pandas.DataFrame
            DataFrame with added anomaly scores
        """
        # Create a copy of the dataframe
        scored_df = df.copy()

        # Get feature columns
        features = self.get_feature_columns()
        X = scored_df[features]

        # Score with Isolation Forest
        scored_df['score_iso'] = -iso_model.decision_function(X)  # Higher = more anomalous
        scored_df['pred_iso'] = (iso_model.predict(X) == -1).astype(int)

        # Score with One-Class SVM
        X_scaled_ocsvm = ocsvm_scaler.transform(X)
        scored_df['score_ocsvm'] = -ocsvm_model.decision_function(X_scaled_ocsvm)
        scored_df['pred_ocsvm'] = (ocsvm_model.predict(X_scaled_ocsvm) == -1).astype(int)

        # Score with DBSCAN
        X_scaled_dbscan = dbscan_scaler.transform(X)
        dbscan_labels = dbscan_model.fit_predict(X_scaled_dbscan)
        # Convert DBSCAN labels to anomaly scores: -1 = outlier, other values = inliers
        scored_df['pred_dbscan'] = (dbscan_labels == -1).astype(int)
        # For DBSCAN we don't have a direct anomaly score, so use binary prediction
        scored_df['score_dbscan'] = scored_df['pred_dbscan'].astype(float)

        # Normalize scores to [0, 1] range for easier comparison
        for score_col in ['score_iso', 'score_ocsvm', 'score_dbscan']:
            min_val = scored_df[score_col].min()
            max_val = scored_df[score_col].max()
            if max_val > min_val:
                scored_df[score_col + '_norm'] = (scored_df[score_col] - min_val) / (max_val - min_val)
            else:
                scored_df[score_col + '_norm'] = 0  # Avoid division by zero

        # Create ensemble score (average of normalized scores)
        scored_df['score_ensemble_norm'] = (
                                                   scored_df['score_iso_norm'] + scored_df['score_ocsvm_norm'] + scored_df['score_dbscan_norm']
                                           ) / 3

        # Create ensemble prediction (majority vote)
        scored_df['pred_ensemble'] = (
            (scored_df['pred_iso'] + scored_df['pred_ocsvm'] + scored_df['pred_dbscan'] >= 2)
        ).astype(int)

        return scored_df

    def generate_and_save_mock_data(self, n_normal=2000, n_attack=500, filename='network_flows.csv'):
        """
        Generate mock data and save to CSV

        Parameters:
        -----------
        n_normal : int
            Number of normal flows to generate
        n_attack : int
            Number of attack flows to generate
        filename : str
            Filename to save the data

        Returns:
        --------
        pandas.DataFrame
            Generated mock data
        """
        print(f"Generating {n_normal} normal flows and {n_attack} attack flows...")
        df = self.generate_synthetic_flows(n_normal, n_attack)

        print("Training models...")
        iso_model = self.train_isolation_forest(df)
        ocsvm_model, ocsvm_scaler = self.train_ocsvm(df)
        dbscan_model, dbscan_scaler = self.train_dbscan(df)

        print("Scoring flows...")
        scored_df = self.score_flows(df, iso_model, ocsvm_model, ocsvm_scaler, dbscan_model, dbscan_scaler)

        print(f"Saving to {filename}...")
        scored_df.to_csv(filename, index=False)

        # Save models
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)

        joblib.dump(iso_model, os.path.join(models_dir, 'isolation_forest.joblib'))
        joblib.dump(ocsvm_model, os.path.join(models_dir, 'ocsvm.joblib'))
        joblib.dump(ocsvm_scaler, os.path.join(models_dir, 'ocsvm_scaler.joblib'))
        joblib.dump(dbscan_model, os.path.join(models_dir, 'dbscan.joblib'))
        joblib.dump(dbscan_scaler, os.path.join(models_dir, 'dbscan_scaler.joblib'))

        # Save model performance metrics
        y_true = (scored_df['label'] == 'attack').astype(int)
        metrics_dict = {}

        for name, pred_col, score_col in [
            ('isolation_forest', 'pred_iso', 'score_iso_norm'),
            ('ocsvm', 'pred_ocsvm', 'score_ocsvm_norm'),
            ('dbscan', 'pred_dbscan', 'score_dbscan_norm'),
            ('ensemble', 'pred_ensemble', 'score_ensemble_norm')
        ]:
            try:
                metrics_dict[name] = {
                    'roc_auc': metrics.roc_auc_score(y_true, scored_df[score_col]),
                    'precision': metrics.precision_score(y_true, scored_df[pred_col]),
                    'recall': metrics.recall_score(y_true, scored_df[pred_col]),
                    'f1': metrics.f1_score(y_true, scored_df[pred_col]),
                    'accuracy': metrics.accuracy_score(y_true, scored_df[pred_col])
                }
            except Exception as e:
                print(f"Error calculating metrics for {name}: {e}")
                metrics_dict[name] = {
                    'error': str(e)
                }

        with open(os.path.join(models_dir, 'model_metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        print("Done!")
        return scored_df

# Helper function for explainability
def explain_anomaly(model, row, features, top_n=5):
    """
    Simple explanation function that returns the top features contributing to anomaly

    Parameters:
    -----------
    model : object
        Trained anomaly detection model
    row : pandas.Series
        Row of data to explain
    features : list
        List of feature names
    top_n : int
        Number of top features to return

    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature contributions
    """
    # Get the feature values for the current row
    feature_values = row[features].values

    # Get the feature importance
    if isinstance(model, IsolationForest):
        # Check if the model has the feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Fallback if feature_importances_ is not available
            importances = np.ones(len(features)) / len(features)
    else:
        # For other models, use a simpler approach
        importances = np.ones(len(features)) / len(features)

    # Create a DataFrame with feature names, values, and importances
    explanation_df = pd.DataFrame({
        'feature': features,
        'value': feature_values,
        'importance': importances
    })

    # Normalize the feature values to help interpret them
    scaler = StandardScaler()
    explanation_df['normalized_value'] = scaler.fit_transform(explanation_df[['value']])

    # Calculate a crude "contribution score" - this is a simplification
    explanation_df['contribution'] = explanation_df['importance'] * np.abs(explanation_df['normalized_value'])

    # Sort by contribution and return top contributors
    return explanation_df.sort_values('contribution', ascending=False).head(top_n)

def extract_key_insights(explanation_df, top_n=3):
    """
    Extract key insights from the explanation

    Parameters:
    -----------
    explanation_df : pandas.DataFrame
        DataFrame with feature contributions
    top_n : int
        Number of top insights to return

    Returns:
    --------
    list
        List of insight strings
    """
    top_features = explanation_df.head(top_n)
    insights = []

    for _, row in top_features.iterrows():
        feature = row['feature']
        value = row['value']
        normalized = row['normalized_value']

        # Custom insights based on feature types
        if feature == 'byte_rate' and normalized > 1.5:
            insights.append(f"Unusually high byte rate ({value:.2f} bytes/ms)")
        elif feature == 'byte_rate' and normalized < -1.5:
            insights.append(f"Unusually low byte rate ({value:.2f} bytes/ms)")
        elif feature == 'packets_per_second' and normalized > 1.5:
            insights.append(f"High packet rate ({value:.2f} packets/sec)")
        elif feature == 'duration_ms' and normalized < -1.5:
            insights.append(f"Very short connection ({value:.2f} ms)")
        elif feature == 'duration_ms' and normalized > 1.5:
            insights.append(f"Unusually long connection ({value:.2f} ms)")
        elif feature == 'bytes_per_packet' and normalized > 1.5:
            insights.append(f"Large packets ({value:.2f} bytes/packet)")
        elif feature == 'bytes_per_packet' and normalized < -1.5:
            insights.append(f"Very small packets ({value:.2f} bytes/packet)")
        elif feature.startswith('is_') and value == 1:
            # Convert feature name to readable format
            readable = feature.replace('is_', '').replace('_', ' ')
            insights.append(f"Connection uses {readable}")
        else:
            # Generic insight
            if normalized > 1.5:
                insights.append(f"High {feature.replace('_', ' ')} ({value:.2f})")
            elif normalized < -1.5:
                insights.append(f"Low {feature.replace('_', ' ')} ({value:.2f})")

    return insights

# If this file is run directly, generate mock data
if __name__ == "__main__":
    generator = DataGenerator()
    generator.generate_and_save_mock_data()
