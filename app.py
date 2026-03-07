"""
SentinelAI - Network Intrusion Detection System
Streamlit Application

This module provides the Streamlit-based web UI for:
1. Loading network flow data
2. Visualizing network traffic
3. Detecting and visualizing anomalies
4. Explaining anomalies to security analysts
5. Real-time simulation of network traffic

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import json
import io
import os
import base64
import glob
import matplotlib.pyplot as plt
import warnings
import altair as alt
import joblib
import threading
import queue
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Import the data generator
from data_generator import DataGenerator, explain_anomaly, extract_key_insights
from pcap_ingest import parse_pcap
from flow_builder import build_flows
from feature_adapter import adapt_flows_for_model
from live_capture import capture_with_callback, list_interfaces, stop_capture
from attack_lab.arp_sample_generator import generate_arp_attack_pcap
from attack_lab.mutation_sample_generator import generate_mutation_attack_pcap
from alert_center import build_incidents
from llm_threat_assessor import generate_threat_assessment
from rule_engine import (
    detect_anomalous_dns,
    detect_arp_anomalies,
    detect_brute_force,
    detect_port_scan,
    detect_statistical_anomalies,
)
from response_orchestrator import execute_responses
from log_center import persist_snapshot
from report_exporter import build_markdown_report, build_report_filename, markdown_to_pdf_bytes

# Set up page
st.set_page_config(page_title="智能网络威胁检测平台", layout="wide", initial_sidebar_state="expanded")

# Matplotlib Chinese font fallback to avoid glyph warnings in Streamlit pyplot.
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph .* missing from font", category=UserWarning)

# Create a global queue for real-time flow updates
flow_queue = queue.Queue(maxsize=100)

# Early session defaults (needed by sidebar controls)
if 'live_capture_running' not in st.session_state:
    st.session_state['live_capture_running'] = False
if 'capture_source' not in st.session_state:
    st.session_state['capture_source'] = None

# Custom CSS (Hermes-like cool gray + teal style)
st.markdown("""
<style>
    :root {
        --bg-main: #edf2f7;
        --bg-panel: #ffffff;
        --bg-panel-soft: #f4f8fb;
        --text-main: #10243c;
        --text-muted: #6e8096;
        --accent: #169e8f;
        --accent-deep: #0f7f75;
        --accent-soft: #e6f6f3;
        --border: #d2ddea;
        --shadow: 0 10px 24px rgba(35, 60, 90, 0.07);
    }
    .stApp {
        background: radial-gradient(circle at 80% 0%, #f2f7fb 0%, #edf2f7 45%, #eaf0f6 100%);
        color: var(--text-main);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e7edf4 0%, #e2e9f2 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        color: var(--text-main);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-muted);
    }
    [data-testid="stSidebar"] .stExpander {
        border: 1px solid var(--border);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 4px 10px rgba(38, 70, 112, 0.04);
        margin-bottom: 8px;
    }
    [data-testid="stSidebar"] .stExpander > details > summary {
        font-weight: 600;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 5rem;
        max-width: 1380px;
    }
    h1, h2, h3, h4 {
        color: var(--text-main);
        letter-spacing: 0.1px;
    }
    hr {
        border-color: var(--border);
    }
    div[data-testid="stMetric"] {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.8rem 1rem;
        box-shadow: var(--shadow);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding: 8px 0;
        background: transparent;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding-left: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 8px 16px !important;
        color: #51657e;
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--border);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: #ffffff !important;
        font-weight: 600;
        border: 1px solid var(--accent-deep);
        box-shadow: 0 6px 14px rgba(22, 158, 143, 0.25);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
        background: var(--bg-panel);
        box-shadow: 0 4px 14px rgba(46, 79, 119, 0.04);
    }
    .stAlert {
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        background: #f9fcff !important;
        color: #21425f !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #b9cfdf !important;
        background: #eef5fb !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(180deg, #1aa99a 0%, #159687 100%) !important;
        color: white !important;
        border-color: #128173 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(180deg, #149387 0%, #117f74 100%) !important;
    }
    [data-testid="stSidebar"] .stButton > button:disabled {
        background: #d9e2eb !important;
        color: #8b98a8 !important;
        border-color: #c8d3df !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }
    [data-baseweb="select"] > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea textarea {
        border-radius: 10px !important;
        border-color: var(--border) !important;
        background: #fcfeff !important;
    }
    [data-baseweb="tag"] {
        background: #e7f6f3 !important;
        border: 1px solid #bde8e1 !important;
        color: #16685f !important;
    }
    .anomaly-high {
        color: #8b1f2b;
        font-weight: bold;
        background-color: rgba(216, 76, 90, 0.14);
        padding: 2px 6px;
        border-radius: 6px;
    }
    .anomaly-medium {
        color: #805720;
        font-weight: bold;
        background-color: rgba(229, 187, 92, 0.18);
        padding: 2px 6px;
        border-radius: 6px;
    }
    .alert-box {
        background-color: #eef5ff;
        border-left: 4px solid #6d99c5;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .metrics-box {
        background-color: var(--bg-panel-soft);
        border-left: 4px solid var(--accent);
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .hero-panel {
        background: linear-gradient(135deg, #f5f9fd 0%, #ecf2f8 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: var(--shadow);
    }
    .hero-title {
        color: #1a3558;
        font-weight: 700;
        margin: 0 0 6px 0;
        font-size: 1rem;
    }
    .hero-sub {
        color: var(--text-muted);
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .status-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 14px 0;
    }
    .status-card {
        background: rgba(255,255,255,0.9);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: var(--shadow);
    }
    .status-label {
        color: var(--text-muted);
        font-size: 0.86rem;
        margin-bottom: 4px;
        display: block;
    }
    .status-value {
        color: var(--text-main);
        font-weight: 700;
        font-size: 1.28rem;
    }
    .status-online {
        color: #1b8d5f;
    }
    .app-footer {
        position: static;
        width: 100%;
        background: rgba(246, 249, 253, 0.75);
        border-top: 1px solid var(--border);
        text-align: center;
        padding: 10px 14px;
        color: #4f627a;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 14px;
    }
    @media (max-width: 980px) {
        .status-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .app-footer {
            font-size: 0.82rem;
            padding: 8px 10px;
        }
    }
    @media (max-width: 640px) {
        .status-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

def create_flow_simulator(df, speed_factor, flow_queue, stop_event=None):
    """Create a flow generator that simulates real-time flow data"""
    def simulate_flows():
        # Get flow data from dataframe
        flows = df.copy()
        if flows.empty or 'timestamp' not in flows.columns:
            return
        flows = flows.sort_values('timestamp')

        # Keep replay smooth: use inter-arrival time and cap max sleep to avoid long freeze.
        prev_ts = flows.iloc[0]['timestamp']
        speed = max(float(speed_factor), 0.1)
        max_sleep_s = 0.5

        # Process each flow
        for _, flow in flows.iterrows():
            if stop_event is not None and stop_event.is_set():
                break
            current_ts = flow.get('timestamp', prev_ts)
            iat_s = 0.0
            if pd.notna(current_ts) and pd.notna(prev_ts):
                try:
                    iat_s = max(0.0, float((current_ts - prev_ts).total_seconds()))
                except Exception:
                    iat_s = 0.0

            sleep_s = min(iat_s / speed, max_sleep_s)
            if sleep_s > 0:
                if stop_event is not None and stop_event.is_set():
                    break
                time.sleep(sleep_s)

            # Add to queue
            try:
                flow_queue.put(flow.to_dict(), block=False)
            except queue.Full:
                # If queue is full, remove oldest item
                try:
                    flow_queue.get(block=False)
                    flow_queue.put(flow.to_dict(), block=False)
                except queue.Empty:
                    pass
            prev_ts = current_ts

    return threading.Thread(target=simulate_flows)

def get_download_link(df, filename="sentinel_ai_data.csv", text="下载 CSV"):
    """Generate a link to download the dataframe as a CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def plot_roc_curve(y_true, y_scores_dict):
    """Plot ROC curve for multiple models"""
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, y_score in y_scores_dict.items():
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        roc_auc = metrics.auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率')
    ax.set_ylabel('真阳性率')
    ax.set_title('ROC 曲线')
    ax.legend(loc="lower right")

    return fig

def plot_precision_recall_curve(y_true, y_scores_dict):
    """Plot 精确率-召回率 curve for multiple models"""
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, y_score in y_scores_dict.items():
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        pr_auc = metrics.auc(recall, precision)
        ax.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('召回率')
    ax.set_ylabel('精确率')
    ax.set_title('PR 曲线')
    ax.legend(loc="lower left")

    return fig

def load_models():
    """Load trained models from disk if available"""
    models = {}
    models_dir = 'models'

    try:
        if os.path.exists(os.path.join(models_dir, 'isolation_forest.joblib')):
            models['iso'] = joblib.load(os.path.join(models_dir, 'isolation_forest.joblib'))

        if (os.path.exists(os.path.join(models_dir, 'ocsvm.joblib')) and
                os.path.exists(os.path.join(models_dir, 'ocsvm_scaler.joblib'))):
            models['ocsvm'] = joblib.load(os.path.join(models_dir, 'ocsvm.joblib'))
            models['ocsvm_scaler'] = joblib.load(os.path.join(models_dir, 'ocsvm_scaler.joblib'))

        if (os.path.exists(os.path.join(models_dir, 'dbscan.joblib')) and
                os.path.exists(os.path.join(models_dir, 'dbscan_scaler.joblib'))):
            models['dbscan'] = joblib.load(os.path.join(models_dir, 'dbscan.joblib'))
            models['dbscan_scaler'] = joblib.load(os.path.join(models_dir, 'dbscan_scaler.joblib'))

        if os.path.exists(os.path.join(models_dir, 'model_metrics.json')):
            with open(os.path.join(models_dir, 'model_metrics.json'), 'r') as f:
                models['metrics'] = json.load(f)

    except Exception as e:
        st.error(f"加载模型失败: {e}")

    return models

# Sidebar
with st.sidebar:
    st.title("智能网络威胁检测平台")
    st.caption("网络入侵检测系统")

    start_capture_button = False
    stop_capture_button = False
    selected_interface = None
    live_bpf_filter = ""
    live_duration_s = 10
    live_packet_limit = 200
    attack_scenario = "spoof"
    attack_packet_count = 120
    attack_network_cidr = "192.168.56.0/24"
    mutation_scenario = "stealth_scan"
    mutation_packet_count = 120
    llm_provider_display = "自动"
    llm_model_name = "auto"
    llm_api_key_input = ""
    llm_base_url_input = ""

    with st.expander("数据来源与采集", expanded=True):
        data_source = st.radio(
            "选择数据来源",
            ["生成新数据", "上传 CSV", "上传 PCAP", "实时抓包", "ARP样本生成(离线)", "变异样本生成(离线)", "使用已保存数据"]
        )

        if data_source == "生成新数据":
            n_normal = st.slider("正常流量条数", 100, 10000, 2000, 100)
            n_attack = st.slider("攻击流量条数", 10, 5000, 500, 10)
            attack_types = st.multiselect(
                "攻击类型",
                ["port_scan", "brute_force", "data_exfiltration", "dos"],
                ["port_scan", "brute_force", "data_exfiltration", "dos"]
            )
        elif data_source == "上传 CSV":
            uploaded_file = st.file_uploader("上传网络流量 CSV 文件", type=["csv"])
        elif data_source == "上传 PCAP":
            uploaded_pcap_file = st.file_uploader("上传 PCAP 文件", type=["pcap", "pcapng"])
        elif data_source == "实时抓包":
            try:
                interfaces = list_interfaces()
                if interfaces:
                    selected_interface = st.selectbox("选择网卡接口", interfaces)
                else:
                    st.warning("未发现可用网卡接口。")
            except Exception as e:
                st.error(f"读取网卡接口失败：{e}")
                interfaces = []

            live_bpf_filter = st.text_input("BPF 过滤表达式（可选）", value="")
            live_duration_s = st.slider(
                "抓包时长（秒）",
                1,
                1800,
                10,
                1,
                help="建议实验取值：300~900 秒",
            )
            live_packet_limit = st.number_input(
                "最大抓包数量",
                min_value=1,
                max_value=500000,
                value=200,
                step=100,
                help="建议实验取值：50000~200000（按网络负载调整）",
            )

            start_capture_button = st.button(
                "开始抓包",
                width="stretch",
                disabled=st.session_state.get("live_capture_running", False)
            )
            stop_capture_button = st.button(
                "停止抓包",
                width="stretch",
                disabled=not st.session_state.get("live_capture_running", False)
            )
        elif data_source == "ARP样本生成(离线)":
            attack_scenario = st.selectbox(
                "攻击场景",
                ["spoof", "flood", "scan", "mitm", "abuse"],
                index=0
            )
            attack_packet_count = st.slider("攻击包数量", 20, 2000, 120, 20)
            attack_network_cidr = st.text_input("扫描网段（仅scan场景生效）", value="192.168.56.0/24")
            st.caption("仅生成离线实验样本，不执行在线攻击。")
        elif data_source == "变异样本生成(离线)":
            mutation_scenario = st.selectbox(
                "变异场景",
                ["stealth_scan", "dns_tunnel_lowrate", "jitter_beacon"],
                index=0
            )
            mutation_packet_count = st.slider("变异攻击包数量", 20, 2000, 120, 20)
            st.caption("用于生成低慢/变异行为离线回放样本。")

    with st.expander("模型与检测参数", expanded=False):
        iso_contamination = st.slider("IsolationForest 污染率", 0.01, 0.5, 0.1, 0.01)
        ocsvm_nu = st.slider("OneClassSVM Nu", 0.01, 0.5, 0.1, 0.01)
        dbscan_eps = st.slider("DBSCAN Epsilon", 0.1, 2.0, 0.5, 0.1)
        dbscan_min_samples = st.slider("DBSCAN 最小样本数", 2, 20, 5, 1)
        anomaly_threshold = st.slider("异常分数阈值", 0.0, 1.0, 0.8, 0.05)

        enable_simulation = st.checkbox("启用实时仿真", value=False)
        if enable_simulation:
            speed_factor = st.slider("仿真速度倍率", 1.0, 100.0, 10.0, 1.0)

    with st.expander("规则引擎参数", expanded=False):
        rule_window_s = st.slider("端口扫描时间窗口（秒）", 10, 600, 60, 10)
        rule_unique_port_threshold = st.slider("目标端口去重阈值", 1, 200, 20, 1)
        brute_force_attempt_threshold = st.slider("暴力破解尝试阈值", 3, 200, 15, 1)
        dns_unique_query_threshold = st.slider("DNS 查询去重阈值", 1, 200, 20, 1)
        dns_nxdomain_threshold = st.slider("DNS NXDOMAIN 阈值", 1, 200, 10, 1)
        arp_flood_threshold = st.slider("ARP 泛洪阈值（包数）", 5, 500, 50, 5)
        arp_scan_target_threshold = st.slider("ARP 扫描阈值（目标IP去重）", 3, 500, 20, 1)
        arp_spoof_mac_threshold = st.slider("ARP 欺骗阈值（同IP对应MAC数）", 2, 10, 2, 1)
        arp_mitm_ip_claim_threshold = st.slider("ARP MITM可疑阈值（同MAC宣称IP数）", 2, 10, 2, 1)
        arp_abuse_gratuitous_threshold = st.slider("ARP 滥用阈值（gratuitous计数）", 3, 500, 20, 1)
        stat_burst_multiplier = st.slider("突发流量倍数阈值", 1.5, 10.0, 3.0, 0.5)
        stat_burst_min_bytes = st.slider("突发流量最小字节阈值", 1000, 1000000, 50000, 1000)
        stat_beacon_min_events = st.slider("周期心跳最小事件数", 3, 50, 5, 1)
        stat_beacon_max_cv = st.slider("周期心跳最大IAT变异系数", 0.05, 1.0, 0.2, 0.05)
        incident_correlation_window_s = st.slider("事件聚合窗口（秒）", 30, 1800, 300, 30)

    with st.expander("LLM 研判参数", expanded=False):
        llm_provider_display = st.selectbox("LLM提供商", ["自动", "DeepSeek", "OpenAI"], index=0)
        llm_model_name = st.text_input("LLM模型名", value="auto")
        llm_api_key_input = st.text_input("API Key（可选，优先于环境变量）", type="password", value="")
        llm_base_url_input = st.text_input("自定义API地址（可选）", value="")
        st.caption("建议配置环境变量：DEEPSEEK_API_KEY 或 OPENAI_API_KEY")

    with st.expander("执行操作", expanded=True):
        run_rules_button = st.button("执行规则检测", width="stretch")
        generate_button = st.button("处理数据", width="stretch")

# Main content
st.title("智能网络威胁检测平台")
st.markdown(
    """
    <div class="hero-panel">
        <p class="hero-title">统一流量检测闭环</p>
        <p class="hero-sub">
            覆盖数据采集、规则检测、AI异常识别、事件聚合、自动响应与报告导出。
            左侧按模块展开配置，右侧按标签查看分析结果。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize the session state
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'simulation_running' not in st.session_state:
    st.session_state['simulation_running'] = False
if 'sim_thread' not in st.session_state:
    st.session_state['sim_thread'] = None
if 'sim_stop_event' not in st.session_state:
    st.session_state['sim_stop_event'] = None
if 'sim_total_source' not in st.session_state:
    st.session_state['sim_total_source'] = 0
if 'sim_total_emitted' not in st.session_state:
    st.session_state['sim_total_emitted'] = 0
if 'sim_auto_refresh' not in st.session_state:
    st.session_state['sim_auto_refresh'] = True
if 'sim_source_df' not in st.session_state:
    st.session_state['sim_source_df'] = None
if 'models' not in st.session_state:
    st.session_state['models'] = {}
if 'pcap_df' not in st.session_state:
    st.session_state['pcap_df'] = None
if 'pcap_file_path' not in st.session_state:
    st.session_state['pcap_file_path'] = None
if 'flows_df' not in st.session_state:
    st.session_state['flows_df'] = None
if 'rule_alerts_df' not in st.session_state:
    st.session_state['rule_alerts_df'] = None
if 'incident_df' not in st.session_state:
    st.session_state['incident_df'] = None
if 'response_actions_df' not in st.session_state:
    st.session_state['response_actions_df'] = None
if 'log_snapshot_paths' not in st.session_state:
    st.session_state['log_snapshot_paths'] = {}
if 'llm_assessment' not in st.session_state:
    st.session_state['llm_assessment'] = ""

def _render_status_cards(target, current_data_source: str) -> None:
    flow_df = st.session_state.get("flows_df")
    if flow_df is None:
        flow_df = st.session_state.get("df")
    flow_count = int(len(flow_df)) if isinstance(flow_df, pd.DataFrame) else 0
    packet_count = int(len(st.session_state.get("pcap_df"))) if isinstance(st.session_state.get("pcap_df"), pd.DataFrame) else 0

    capture_source = st.session_state.get("capture_source") or current_data_source
    source_name_map = {
        "synthetic": "生成新数据",
        "csv": "上传 CSV",
        "pcap": "上传 PCAP",
        "live": "实时抓包",
        "attack_lab": "ARP样本生成(离线)",
        "mutation_lab": "变异样本生成(离线)",
        "saved": "使用已保存数据",
    }
    summary_source = source_name_map.get(capture_source, str(capture_source))

    if st.session_state.get("live_capture_running", False):
        system_status = "采集中"
    elif capture_source == "live" and (packet_count > 0 or flow_count > 0):
        system_status = "已完成"
    else:
        system_status = "待机"

    target.markdown(
        f"""
        <div class="status-grid">
            <div class="status-card">
                <span class="status-label">系统状态</span>
                <span class="status-value {'status-online' if system_status in ['采集中','已完成'] else ''}">{system_status}</span>
            </div>
            <div class="status-card">
                <span class="status-label">报文数（Packet）</span>
                <span class="status-value">{packet_count}</span>
            </div>
            <div class="status-card">
                <span class="status-label">流量数（Flow）</span>
                <span class="status-value">{flow_count}</span>
            </div>
            <div class="status-card">
                <span class="status-label">当前数据源</span>
                <span class="status-value" style="font-size:1.05rem;">{summary_source}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


status_cards_placeholder = st.empty()
_render_status_cards(status_cards_placeholder, data_source)

# Process data when button is clicked
if stop_capture_button:
    stop_capture()
    st.session_state['live_capture_running'] = False
    st.info("已发送停止抓包指令。")

if generate_button or start_capture_button:
    st.session_state['rule_alerts_df'] = None
    st.session_state['incident_df'] = None
    st.session_state['response_actions_df'] = None
    st.session_state['log_snapshot_paths'] = {}
    if data_source == "生成新数据":
        with st.spinner("正在生成模拟网络流量..."):
            # Create data generator
            generator = DataGenerator()

            # Generate synthetic data
            df = generator.generate_synthetic_flows(n_normal, n_attack, attack_types)

            # Train models
            iso_model = generator.train_isolation_forest(df, iso_contamination)
            ocsvm_model, ocsvm_scaler = generator.train_ocsvm(df, ocsvm_nu)
            dbscan_model, dbscan_scaler = generator.train_dbscan(df, dbscan_eps, dbscan_min_samples)

            # Score flows
            df = generator.score_flows(df, iso_model, ocsvm_model, ocsvm_scaler, dbscan_model, dbscan_scaler)

            # Store models
            st.session_state['models'] = {
                'iso': iso_model,
                'ocsvm': ocsvm_model,
                'ocsvm_scaler': ocsvm_scaler,
                'dbscan': dbscan_model,
                'dbscan_scaler': dbscan_scaler
            }

            # Store dataframe
            st.session_state['df'] = df
            st.session_state['flows_df'] = None
            st.session_state['simulation_running'] = False
            st.session_state['capture_source'] = "synthetic"

            st.success(f"已生成 {len(df)} 条流量（正常 {n_normal}，攻击 {n_attack}）")

    elif data_source == "上传 CSV":
        if uploaded_file is not None:
            with st.spinner("正在处理上传数据..."):
                # Load CSV
                df = pd.read_csv(uploaded_file)

                # Check if required columns exist
                required_columns = ['timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port',
                                    'protocol', 'duration_ms', 'total_bytes', 'label']

                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    st.error(f"缺少必需列：{', '.join(missing_columns)}")
                else:
                    # Parse timestamp if needed
                    if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])

                    # Create data generator
                    generator = DataGenerator()

                    # Apply feature engineering if needed
                    features = generator.get_feature_columns()
                    missing_features = [col for col in features if col not in df.columns]

                    if missing_features:
                        st.info(f"正在执行特征工程，补齐 {len(missing_features)} 个缺失特征...")
                        df = generator._engineer_features(df)

                    # Check if model scores exist
                    if ('score_iso_norm' not in df.columns or
                            'score_ocsvm_norm' not in df.columns or
                            'score_dbscan_norm' not in df.columns):
                        st.info("正在训练模型并计算异常分数...")

                        # Train models
                        iso_model = generator.train_isolation_forest(df, iso_contamination)
                        ocsvm_model, ocsvm_scaler = generator.train_ocsvm(df, ocsvm_nu)
                        dbscan_model, dbscan_scaler = generator.train_dbscan(df, dbscan_eps, dbscan_min_samples)

                        # Score flows
                        df = generator.score_flows(df, iso_model, ocsvm_model, ocsvm_scaler, dbscan_model, dbscan_scaler)

                        # Store models
                        st.session_state['models'] = {
                            'iso': iso_model,
                            'ocsvm': ocsvm_model,
                            'ocsvm_scaler': ocsvm_scaler,
                            'dbscan': dbscan_model,
                            'dbscan_scaler': dbscan_scaler
                        }

                    # Store dataframe
                    st.session_state['df'] = df
                    st.session_state['flows_df'] = None
                    st.session_state['simulation_running'] = False
                    st.session_state['capture_source'] = "csv"

                    st.success(f"已处理 {len(df)} 条流量")
        else:
            st.warning("请先上传 CSV 文件再处理。")

    elif data_source == "上传 PCAP":
        if uploaded_pcap_file is not None:
            with st.spinner("正在解析上传的 PCAP..."):
                try:
                    os.makedirs('data', exist_ok=True)
                    timestamp_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    ext = os.path.splitext(uploaded_pcap_file.name)[1].lower()
                    if ext not in ['.pcap', '.pcapng']:
                        ext = '.pcap'

                    save_path = os.path.join('data', f'uploaded_{timestamp_suffix}{ext}')
                    with open(save_path, 'wb') as f:
                        f.write(uploaded_pcap_file.getbuffer())

                    records = parse_pcap(save_path)
                    pcap_df = pd.DataFrame(records)
                    flows_raw_df = build_flows(pcap_df)
                    flows_df = adapt_flows_for_model(flows_raw_df)

                    st.session_state['pcap_df'] = pcap_df
                    st.session_state['pcap_file_path'] = save_path
                    st.session_state['simulation_running'] = False
                    st.session_state['flows_df'] = flows_df
                    st.session_state['df'] = None
                    st.session_state['capture_source'] = "pcap"

                    if flows_df.empty:
                        st.warning(
                            f"已解析 {len(pcap_df)} 个数据包，但未构建出可用 IP 流量 "
                            f"（可能仅包含 ARP/非 IP 流量）。"
                        )
                    else:
                        generator = DataGenerator()
                        iso_model = generator.train_isolation_forest(flows_df, iso_contamination)
                        ocsvm_model, ocsvm_scaler = generator.train_ocsvm(flows_df, ocsvm_nu)
                        dbscan_model, dbscan_scaler = generator.train_dbscan(flows_df, dbscan_eps, dbscan_min_samples)

                        flows_df = generator.score_flows(
                            flows_df,
                            iso_model,
                            ocsvm_model,
                            ocsvm_scaler,
                            dbscan_model,
                            dbscan_scaler
                        )

                        st.session_state['models'] = {
                            'iso': iso_model,
                            'ocsvm': ocsvm_model,
                            'ocsvm_scaler': ocsvm_scaler,
                            'dbscan': dbscan_model,
                            'dbscan_scaler': dbscan_scaler
                        }
                        st.session_state['flows_df'] = flows_df

                        st.success(
                            f"已解析 {len(pcap_df)} 个数据包，并构建 {len(flows_df)} 条流量：{save_path}"
                        )
                except (FileNotFoundError, ValueError) as e:
                    st.error(f"PCAP 解析错误：{e}")
                except Exception as e:
                    st.error(f"处理 PCAP 时出现异常：{e}")
        else:
            st.warning("请先上传 PCAP/PCAPNG 文件再处理。")

    elif data_source == "实时抓包":
        if not selected_interface:
            st.warning("请先选择网卡接口。")
        else:
            st.session_state['live_capture_running'] = True
            os.makedirs("data", exist_ok=True)
            capture_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            jsonl_path = os.path.join("data", f"live_capture_{capture_ts}.jsonl")

            capture_status_placeholder = st.empty()
            capture_status_placeholder.info("正在实时抓包，请等待完成...")
            packet_table_placeholder = st.empty()
            metrics_placeholder = st.empty()

            def _on_record(_, records):
                # Reduce UI refresh pressure.
                if len(records) % 5 != 0:
                    return
                recent = pd.DataFrame(records[-200:])
                if recent.empty:
                    return

                protocol_counts = (
                    recent["protocol"].fillna("UNKNOWN").value_counts()
                    if "protocol" in recent.columns
                    else pd.Series(dtype=int)
                )
                arp_count = int((recent["protocol"] == "ARP").sum()) if "protocol" in recent.columns else 0

                with metrics_placeholder.container():
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("实时记录数", len(records))
                    with c2:
                        st.metric("协议类型数", int(len(protocol_counts)))
                    with c3:
                        st.metric("ARP 数量", arp_count)
                    if not protocol_counts.empty:
                        st.dataframe(
                            protocol_counts.rename_axis("protocol").reset_index(name="count"),
                            width="stretch"
                        )

                packet_table_placeholder.dataframe(
                    recent.sort_values("timestamp", ascending=False).reset_index(drop=True),
                    width="stretch"
                )

            try:
                records = capture_with_callback(
                    interface=selected_interface,
                    bpf_filter=live_bpf_filter if live_bpf_filter else None,
                    duration_s=int(live_duration_s),
                    packet_limit=int(live_packet_limit),
                    on_record=_on_record,
                    jsonl_path=jsonl_path,
                )
            except Exception as e:
                capture_status_placeholder.error(f"实时抓包失败：{e}")
                records = []

            pcap_df = pd.DataFrame(records)
            st.session_state["pcap_df"] = pcap_df
            st.session_state["pcap_file_path"] = jsonl_path
            st.session_state["df"] = None
            st.session_state["simulation_running"] = False
            st.session_state["capture_source"] = "live"
            st.session_state['live_capture_running'] = False

            if pcap_df.empty:
                st.session_state["flows_df"] = None
                capture_status_placeholder.warning(
                    "实时抓包未采集到可用数据。\n\n"
                    f"- 接口：`{selected_interface}`\n"
                    f"- 过滤：`{live_bpf_filter if live_bpf_filter else '无'}`\n"
                    "请检查接口是否为当前活跃网卡、是否具备抓包权限，以及过滤表达式是否过严。"
                )
            else:
                flows_raw_df = build_flows(pcap_df)
                flows_df = adapt_flows_for_model(flows_raw_df)
                st.session_state["flows_df"] = flows_df

                if flows_df.empty:
                    st.session_state["flows_df"] = None
                    st.warning("抓包已结束，但未构建出可用于检测的 IP 流量。")
                else:
                    generator = DataGenerator()
                    iso_model = generator.train_isolation_forest(flows_df, iso_contamination)
                    ocsvm_model, ocsvm_scaler = generator.train_ocsvm(flows_df, ocsvm_nu)
                    dbscan_model, dbscan_scaler = generator.train_dbscan(flows_df, dbscan_eps, dbscan_min_samples)

                    flows_df = generator.score_flows(
                        flows_df,
                        iso_model,
                        ocsvm_model,
                        ocsvm_scaler,
                        dbscan_model,
                        dbscan_scaler
                    )
                    st.session_state["models"] = {
                        "iso": iso_model,
                        "ocsvm": ocsvm_model,
                        "ocsvm_scaler": ocsvm_scaler,
                        "dbscan": dbscan_model,
                        "dbscan_scaler": dbscan_scaler
                    }
                    st.session_state["flows_df"] = flows_df

                    # Auto-run rules after live capture.
                    port_scan_alerts = detect_port_scan(
                        flows_df,
                        window_s=rule_window_s,
                        unique_port_threshold=rule_unique_port_threshold
                    )
                    brute_force_alerts = detect_brute_force(
                        flows_df,
                        window_s=rule_window_s,
                        attempt_threshold=brute_force_attempt_threshold
                    )
                    dns_alerts = detect_anomalous_dns(
                        pcap_df,
                        window_s=rule_window_s,
                        unique_query_threshold=dns_unique_query_threshold,
                        nxdomain_threshold=dns_nxdomain_threshold
                    )
                    arp_alerts = detect_arp_anomalies(
                        pcap_df,
                        window_s=rule_window_s,
                        arp_flood_threshold=arp_flood_threshold,
                        arp_scan_target_threshold=arp_scan_target_threshold,
                        arp_spoof_mac_threshold=arp_spoof_mac_threshold,
                        arp_mitm_ip_claim_threshold=arp_mitm_ip_claim_threshold,
                        arp_abuse_gratuitous_threshold=arp_abuse_gratuitous_threshold
                    )
                    stat_alerts = detect_statistical_anomalies(
                        flows_df,
                        window_s=rule_window_s,
                        burst_multiplier=stat_burst_multiplier,
                        burst_min_bytes=stat_burst_min_bytes,
                        beacon_min_events=stat_beacon_min_events,
                        beacon_max_iat_cv=stat_beacon_max_cv
                    )
                    st.session_state["rule_alerts_df"] = pd.concat(
                        [port_scan_alerts, brute_force_alerts, dns_alerts, arp_alerts, stat_alerts], ignore_index=True
                    )
                    st.session_state["incident_df"] = build_incidents(
                        st.session_state["rule_alerts_df"],
                        correlation_window_s=incident_correlation_window_s,
                    )
                    st.session_state["log_snapshot_paths"] = persist_snapshot(
                        source="live_capture",
                        packet_df=st.session_state.get("pcap_df"),
                        flow_df=st.session_state.get("flows_df"),
                        alert_df=st.session_state.get("rule_alerts_df"),
                        incident_df=st.session_state.get("incident_df"),
                        action_df=st.session_state.get("response_actions_df"),
                    )

                    st.success(
                        f"实时抓包完成：采集 {len(pcap_df)} 条报文，生成 {len(flows_df)} 条流量。"
                        f"日志已保存到 {jsonl_path}"
                    )
                    capture_status_placeholder.success(
                        f"实时抓包完成：采集 {len(pcap_df)} 条报文，生成 {len(flows_df)} 条流量。"
                    )

    elif data_source == "ARP样本生成(离线)":
        with st.spinner("正在生成ARP离线样本并执行检测..."):
            try:
                os.makedirs("data/samples", exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                sample_path = os.path.join("data", "samples", f"arp_{attack_scenario}_{ts}.pcap")

                meta = generate_arp_attack_pcap(
                    output_path=sample_path,
                    scenario=attack_scenario,
                    packet_count=int(attack_packet_count),
                    network_cidr=attack_network_cidr,
                )

                records = parse_pcap(sample_path)
                pcap_df = pd.DataFrame(records)
                st.session_state["pcap_df"] = pcap_df
                st.session_state["pcap_file_path"] = sample_path
                st.session_state["df"] = None
                st.session_state["simulation_running"] = False
                st.session_state["capture_source"] = "attack_lab"

                flows_raw_df = build_flows(pcap_df)
                flows_df = adapt_flows_for_model(flows_raw_df)
                st.session_state["flows_df"] = flows_df if not flows_df.empty else None

                if st.session_state["flows_df"] is not None:
                    generator = DataGenerator()
                    iso_model = generator.train_isolation_forest(flows_df, iso_contamination)
                    ocsvm_model, ocsvm_scaler = generator.train_ocsvm(flows_df, ocsvm_nu)
                    dbscan_model, dbscan_scaler = generator.train_dbscan(flows_df, dbscan_eps, dbscan_min_samples)
                    flows_df = generator.score_flows(
                        flows_df,
                        iso_model,
                        ocsvm_model,
                        ocsvm_scaler,
                        dbscan_model,
                        dbscan_scaler
                    )
                    st.session_state["models"] = {
                        "iso": iso_model,
                        "ocsvm": ocsvm_model,
                        "ocsvm_scaler": ocsvm_scaler,
                        "dbscan": dbscan_model,
                        "dbscan_scaler": dbscan_scaler
                    }
                    st.session_state["flows_df"] = flows_df

                base_df_for_rules = st.session_state.get("flows_df")
                if base_df_for_rules is None:
                    base_df_for_rules = pd.DataFrame(
                        columns=["timestamp", "src_ip", "dst_ip", "dst_port", "protocol", "total_bytes"]
                    )

                port_scan_alerts = detect_port_scan(
                    base_df_for_rules,
                    window_s=rule_window_s,
                    unique_port_threshold=rule_unique_port_threshold
                )
                brute_force_alerts = detect_brute_force(
                    base_df_for_rules,
                    window_s=rule_window_s,
                    attempt_threshold=brute_force_attempt_threshold
                )
                dns_alerts = detect_anomalous_dns(
                    pcap_df,
                    window_s=rule_window_s,
                    unique_query_threshold=dns_unique_query_threshold,
                    nxdomain_threshold=dns_nxdomain_threshold
                )
                arp_alerts = detect_arp_anomalies(
                    pcap_df,
                    window_s=rule_window_s,
                    arp_flood_threshold=arp_flood_threshold,
                    arp_scan_target_threshold=arp_scan_target_threshold,
                    arp_spoof_mac_threshold=arp_spoof_mac_threshold,
                    arp_mitm_ip_claim_threshold=arp_mitm_ip_claim_threshold,
                    arp_abuse_gratuitous_threshold=arp_abuse_gratuitous_threshold
                )
                stat_alerts = detect_statistical_anomalies(
                    base_df_for_rules,
                    window_s=rule_window_s,
                    burst_multiplier=stat_burst_multiplier,
                    burst_min_bytes=stat_burst_min_bytes,
                    beacon_min_events=stat_beacon_min_events,
                    beacon_max_iat_cv=stat_beacon_max_cv
                )
                st.session_state["rule_alerts_df"] = pd.concat(
                    [port_scan_alerts, brute_force_alerts, dns_alerts, arp_alerts, stat_alerts], ignore_index=True
                )
                st.session_state["incident_df"] = build_incidents(
                    st.session_state["rule_alerts_df"],
                    correlation_window_s=incident_correlation_window_s,
                )
                st.session_state["log_snapshot_paths"] = persist_snapshot(
                    source="attack_lab_arp",
                    packet_df=st.session_state.get("pcap_df"),
                    flow_df=st.session_state.get("flows_df"),
                    alert_df=st.session_state.get("rule_alerts_df"),
                    incident_df=st.session_state.get("incident_df"),
                    action_df=st.session_state.get("response_actions_df"),
                )

                st.success(
                    f"样本已生成并导入检测：{sample_path} | "
                    f"攻击场景={meta['scenario']} | 报文总数={meta['packet_count_total']} | "
                    f"规则告警={len(st.session_state['rule_alerts_df'])}"
                )
            except Exception as e:
                st.error(f"ARP样本生成或检测失败：{e}")
                st.session_state["incident_df"] = None

    elif data_source == "变异样本生成(离线)":
        with st.spinner("正在生成变异攻击样本并执行检测..."):
            try:
                os.makedirs("data/samples", exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                sample_path = os.path.join("data", "samples", f"mutation_{mutation_scenario}_{ts}.pcap")

                meta = generate_mutation_attack_pcap(
                    output_path=sample_path,
                    scenario=mutation_scenario,
                    packet_count=int(mutation_packet_count),
                )

                records = parse_pcap(sample_path)
                pcap_df = pd.DataFrame(records)
                st.session_state["pcap_df"] = pcap_df
                st.session_state["pcap_file_path"] = sample_path
                st.session_state["df"] = None
                st.session_state["simulation_running"] = False
                st.session_state["capture_source"] = "mutation_lab"

                flows_raw_df = build_flows(pcap_df)
                flows_df = adapt_flows_for_model(flows_raw_df)
                st.session_state["flows_df"] = flows_df if not flows_df.empty else None

                if st.session_state["flows_df"] is not None:
                    generator = DataGenerator()
                    iso_model = generator.train_isolation_forest(flows_df, iso_contamination)
                    ocsvm_model, ocsvm_scaler = generator.train_ocsvm(flows_df, ocsvm_nu)
                    dbscan_model, dbscan_scaler = generator.train_dbscan(flows_df, dbscan_eps, dbscan_min_samples)
                    flows_df = generator.score_flows(
                        flows_df,
                        iso_model,
                        ocsvm_model,
                        ocsvm_scaler,
                        dbscan_model,
                        dbscan_scaler
                    )
                    st.session_state["models"] = {
                        "iso": iso_model,
                        "ocsvm": ocsvm_model,
                        "ocsvm_scaler": ocsvm_scaler,
                        "dbscan": dbscan_model,
                        "dbscan_scaler": dbscan_scaler
                    }
                    st.session_state["flows_df"] = flows_df

                base_df_for_rules = st.session_state.get("flows_df")
                if base_df_for_rules is None:
                    base_df_for_rules = pd.DataFrame(
                        columns=["timestamp", "src_ip", "dst_ip", "dst_port", "protocol", "total_bytes"]
                    )

                port_scan_alerts = detect_port_scan(
                    base_df_for_rules,
                    window_s=rule_window_s,
                    unique_port_threshold=rule_unique_port_threshold
                )
                brute_force_alerts = detect_brute_force(
                    base_df_for_rules,
                    window_s=rule_window_s,
                    attempt_threshold=brute_force_attempt_threshold
                )
                dns_alerts = detect_anomalous_dns(
                    pcap_df,
                    window_s=rule_window_s,
                    unique_query_threshold=dns_unique_query_threshold,
                    nxdomain_threshold=dns_nxdomain_threshold
                )
                arp_alerts = detect_arp_anomalies(
                    pcap_df,
                    window_s=rule_window_s,
                    arp_flood_threshold=arp_flood_threshold,
                    arp_scan_target_threshold=arp_scan_target_threshold,
                    arp_spoof_mac_threshold=arp_spoof_mac_threshold,
                    arp_mitm_ip_claim_threshold=arp_mitm_ip_claim_threshold,
                    arp_abuse_gratuitous_threshold=arp_abuse_gratuitous_threshold
                )
                stat_alerts = detect_statistical_anomalies(
                    base_df_for_rules,
                    window_s=rule_window_s,
                    burst_multiplier=stat_burst_multiplier,
                    burst_min_bytes=stat_burst_min_bytes,
                    beacon_min_events=stat_beacon_min_events,
                    beacon_max_iat_cv=stat_beacon_max_cv
                )
                st.session_state["rule_alerts_df"] = pd.concat(
                    [port_scan_alerts, brute_force_alerts, dns_alerts, arp_alerts, stat_alerts], ignore_index=True
                )
                st.session_state["incident_df"] = build_incidents(
                    st.session_state["rule_alerts_df"],
                    correlation_window_s=incident_correlation_window_s,
                )
                st.session_state["log_snapshot_paths"] = persist_snapshot(
                    source="mutation_lab",
                    packet_df=st.session_state.get("pcap_df"),
                    flow_df=st.session_state.get("flows_df"),
                    alert_df=st.session_state.get("rule_alerts_df"),
                    incident_df=st.session_state.get("incident_df"),
                    action_df=st.session_state.get("response_actions_df"),
                )

                st.success(
                    f"变异样本已生成并导入检测：{sample_path} | "
                    f"场景={meta['scenario']} | 报文总数={meta['packet_count_total']} | "
                    f"规则告警={len(st.session_state['rule_alerts_df'])}"
                )
            except Exception as e:
                st.error(f"变异样本生成或检测失败：{e}")
                st.session_state["incident_df"] = None

    elif data_source == "使用已保存数据":
        with st.spinner("正在加载已保存数据..."):
            # Check if network_flows.csv exists
            if os.path.exists('network_flows.csv'):
                # Load CSV
                df = pd.read_csv('network_flows.csv')

                # Parse timestamp
                if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Load models
                models = load_models()

                if models:
                    st.session_state['models'] = models

                # Store dataframe
                st.session_state['df'] = df
                st.session_state['flows_df'] = None
                st.session_state['simulation_running'] = False
                st.session_state['capture_source'] = "saved"

                st.success(f"已从保存数据加载 {len(df)} 条流量")
            else:
                st.error("未找到已保存数据，请先生成数据。")

# Refresh top status cards after any state mutation in this run.
_render_status_cards(status_cards_placeholder, data_source)

# Show PCAP parsing results
if st.session_state.get('pcap_df') is not None:
    pcap_df = st.session_state['pcap_df']
    capture_source = st.session_state.get('capture_source')

    if capture_source == "live":
        st.header("实时抓包结果")
    elif capture_source == "attack_lab":
        st.header("ARP离线样本检测结果")
    elif capture_source == "mutation_lab":
        st.header("变异样本离线检测结果")
    else:
        st.header("PCAP 解析结果")
    if st.session_state.get('pcap_file_path'):
        st.caption(f"来源文件：{st.session_state['pcap_file_path']}")

    protocol_counts = pd.Series(dtype=int)
    arp_count = 0
    if 'protocol' in pcap_df.columns:
        protocol_counts = pcap_df['protocol'].fillna('UNKNOWN').value_counts()
        arp_count = int((pcap_df['protocol'] == 'ARP').sum())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总记录数", len(pcap_df))
    with col2:
        st.metric("协议类型数", int(len(protocol_counts)))
    with col3:
        st.metric("ARP 记录数", arp_count)

    st.subheader("协议分布")
    if protocol_counts.empty:
        st.info("无可用协议信息。")
    else:
        protocol_df = protocol_counts.rename_axis('protocol').reset_index(name='count')
        st.dataframe(protocol_df, width="stretch")

    st.subheader("解析记录（前 200 条）")
    packet_preview_cols = [
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
        "packet_length", "payload_size", "l7_protocol", "dns_query", "http_method"
    ]
    packet_preview_cols = [c for c in packet_preview_cols if c in pcap_df.columns]
    st.dataframe(pcap_df[packet_preview_cols].head(200) if packet_preview_cols else pcap_df.head(200), width="stretch")

if st.session_state.get('flows_df') is not None:
    st.subheader("构建流量结果（前 200 条）")
    flow_preview_df = st.session_state['flows_df']
    flow_preview_cols = [
        "flow_id", "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
        "duration_ms", "total_bytes", "packet_count", "score_iso_norm", "score_ensemble_norm", "label"
    ]
    flow_preview_cols = [c for c in flow_preview_cols if c in flow_preview_df.columns]
    st.dataframe(flow_preview_df[flow_preview_cols].head(200) if flow_preview_cols else flow_preview_df.head(200), width="stretch")

# If data is available, show the analysis interface
analysis_df = st.session_state.get('flows_df')
if analysis_df is None:
    analysis_df = st.session_state.get('df')

if analysis_df is not None and isinstance(analysis_df, pd.DataFrame) and not analysis_df.empty:
    df = analysis_df
    # Ensure minimum schema required by downstream tabs.
    if 'label' not in df.columns:
        df['label'] = 'normal'
    has_ground_truth = False
    y_true = None
    if 'label' in df.columns:
        label_lower = df['label'].astype(str).str.lower()
        has_ground_truth = label_lower.nunique() >= 2 and ('attack' in set(label_lower)) and ('normal' in set(label_lower))
        y_true = (label_lower == 'attack').astype(int)
    if run_rules_button:
        try:
            port_scan_alerts = detect_port_scan(
                df,
                window_s=rule_window_s,
                unique_port_threshold=rule_unique_port_threshold
            )
            brute_force_alerts = detect_brute_force(
                df,
                window_s=rule_window_s,
                attempt_threshold=brute_force_attempt_threshold
            )
            dns_source_df = st.session_state.get('pcap_df')
            if dns_source_df is None:
                dns_source_df = df
            dns_alerts = detect_anomalous_dns(
                dns_source_df,
                window_s=rule_window_s,
                unique_query_threshold=dns_unique_query_threshold,
                nxdomain_threshold=dns_nxdomain_threshold
            )
            arp_source_df = st.session_state.get('pcap_df')
            if arp_source_df is None:
                arp_source_df = pd.DataFrame(columns=['timestamp', 'protocol', 'src_ip', 'src_mac', 'dst_ip'])
            arp_alerts = detect_arp_anomalies(
                arp_source_df,
                window_s=rule_window_s,
                arp_flood_threshold=arp_flood_threshold,
                arp_scan_target_threshold=arp_scan_target_threshold,
                arp_spoof_mac_threshold=arp_spoof_mac_threshold,
                arp_mitm_ip_claim_threshold=arp_mitm_ip_claim_threshold,
                arp_abuse_gratuitous_threshold=arp_abuse_gratuitous_threshold
            )
            stat_alerts = detect_statistical_anomalies(
                df,
                window_s=rule_window_s,
                burst_multiplier=stat_burst_multiplier,
                burst_min_bytes=stat_burst_min_bytes,
                beacon_min_events=stat_beacon_min_events,
                beacon_max_iat_cv=stat_beacon_max_cv
            )
            st.session_state['rule_alerts_df'] = pd.concat(
                [port_scan_alerts, brute_force_alerts, dns_alerts, arp_alerts, stat_alerts], ignore_index=True
            )
            st.session_state['incident_df'] = build_incidents(
                st.session_state['rule_alerts_df'],
                correlation_window_s=incident_correlation_window_s,
            )
            st.session_state["log_snapshot_paths"] = persist_snapshot(
                source="manual_rule_run",
                packet_df=st.session_state.get("pcap_df"),
                flow_df=df,
                alert_df=st.session_state.get("rule_alerts_df"),
                incident_df=st.session_state.get("incident_df"),
                action_df=st.session_state.get("response_actions_df"),
            )
        except Exception as e:
            st.error(f"规则检测执行失败：{e}")
            st.session_state['rule_alerts_df'] = None
            st.session_state['incident_df'] = None

    # Get feature columns
    generator = DataGenerator()
    features = generator.get_feature_columns()

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "总览",
        "流量分析",
        "异常检测",
        "异常解释",
        "实时仿真",
        "规则检测",
        "事件中心",
        "ARP 专题",
        "日志中心",
        "报告导出"
    ])

    # Tab 1: Overview
    with tab1:
        # Summary statistics
        st.header("数据集概览")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("流量总数", len(df))
        with col2:
            st.metric("正常流量数", len(df[df['label'] == 'normal']))
        with col3:
            st.metric("攻击流量数", len(df[df['label'] == 'attack']))
        with col4:
            if has_ground_truth:
                detection_rate = len(df[(df['pred_iso'] == 1) & (df['label'] == 'attack')]) / max(1, len(df[df['label'] == 'attack']))
                st.metric("检测率（IF）", f"{detection_rate:.1%}")
            else:
                st.metric("检测率（IF）", "N/A")

        # Metrics comparison
        st.header("模型性能")
        if not has_ground_truth:
            st.info("当前数据缺少可用攻击真值标签（如实时抓包场景），已跳过监督评估指标（ROC/PR/混淆矩阵）。")
        else:
            metrics_list = []

            for name, pred_col, score_col in [
                ('IsolationForest', 'pred_iso', 'score_iso_norm'),
                ('OneClassSVM', 'pred_ocsvm', 'score_ocsvm_norm'),
                ('DBSCAN', 'pred_dbscan', 'score_dbscan_norm')
            ]:
                try:
                    roc = metrics.roc_auc_score(y_true, df[score_col])
                except Exception:
                    roc = np.nan

                try:
                    prec, rec, f1, _ = metrics.precision_recall_fscore_support(
                        y_true, df[pred_col], average='binary', zero_division=0
                    )
                    metrics_list.append({
                        'Model': name,
                        'ROC AUC': roc,
                        '精确率': prec,
                        '召回率': rec,
                        'F1 分数': f1
                    })
                except Exception:
                    metrics_list.append({
                        'Model': name,
                        'ROC AUC': roc,
                        '精确率': np.nan,
                        '召回率': np.nan,
                        'F1 分数': np.nan
                    })

            metrics_df = pd.DataFrame(metrics_list)

            st.dataframe(metrics_df.style.format({
                'ROC AUC': "{:.3f}", '精确率': "{:.3f}", '召回率': "{:.3f}", 'F1 分数': "{:.3f}"
            }))

            # Confusion matrices
            st.header("混淆矩阵")

            cols = st.columns(3)
            for i, (name, pred_col) in enumerate([
                ('IsolationForest', 'pred_iso'),
                ('OneClassSVM', 'pred_ocsvm'),
                ('DBSCAN', 'pred_dbscan')
            ]):
                with cols[i]:
                    st.subheader(name)
                    try:
                        cm = metrics.confusion_matrix(y_true, df[pred_col])
                        cm_df = pd.DataFrame(
                            cm,
                            index=['真实正常', '真实攻击'],
                            columns=['预测正常', '预测攻击']
                        )
                        st.dataframe(cm_df)

                        tn, fp, fn, tp = cm.ravel()
                        accuracy = (tp + tn) / (tp + tn + fp + fn)
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                        st.markdown(f"""
                        * **准确率**: {accuracy:.3f}
                        * **精确率**: {precision:.3f}
                        * **召回率**: {recall:.3f}
                        """)
                    except Exception:
                        st.warning(f"无法计算该模型混淆矩阵： {name}")

            # ROC and precision-recall curves
            st.header("模型对比曲线")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ROC 曲线")
                y_scores_dict = {
                    'IsolationForest': df['score_iso_norm'],
                    'OneClassSVM': df['score_ocsvm_norm'],
                    'DBSCAN': df['score_dbscan_norm']
                }
                fig_roc = plot_roc_curve(y_true, y_scores_dict)
                st.pyplot(fig_roc)

            with col2:
                st.subheader("PR 曲线")
                fig_pr = plot_precision_recall_curve(y_true, y_scores_dict)
                st.pyplot(fig_pr)

        # 攻击类型分布
        st.header("攻击类型分布")
        if 'attack_type' in df.columns and 'label' in df.columns:
            attack_counts = df[df['label'].astype(str).str.lower() == 'attack']['attack_type'].value_counts().reset_index()
            attack_counts.columns = ['attack_type', 'count']
            if attack_counts.empty:
                st.info("当前数据中未标注攻击类型，无法绘制攻击类型分布（实时抓包场景通常无真值标签）。")
            else:
                bar = alt.Chart(attack_counts).mark_bar().encode(
                    x=alt.X('attack_type:N', title='攻击类型'),
                    y=alt.Y('count:Q', title='数量'),
                    color=alt.Color('attack_type:N', legend=None)
                ).properties(
                    width=600,
                    height=300
                )
                st.altair_chart(bar, width="stretch")
        else:
            st.info("当前数据不包含攻击类型字段，已跳过该图表。")

        # Download data
        st.header("下载数据集")
        st.markdown(get_download_link(df), unsafe_allow_html=True)

    # Tab 2: Flow Analysis
    with tab2:
        st.header("网络流量分析")

        # Flow filtering
        col1, col2 = st.columns(2)
        with col1:
            filter_label = st.selectbox("按标签筛选", ["全部", "正常", "攻击"])
        with col2:
            if filter_label == "攻击" and 'attack_type' in df.columns:
                filter_attack = st.selectbox("按攻击类型筛选", ["全部"] + sorted(df['attack_type'].dropna().unique().tolist()))
            else:
                filter_attack = "全部"

        # Apply filters
        filtered_df = df.copy()
        if filter_label == "正常":
            filtered_df = filtered_df[filtered_df['label'].str.lower() == "normal"]
        elif filter_label == "攻击":
            filtered_df = filtered_df[filtered_df['label'].str.lower() == "attack"]

        if filter_attack != "全部" and filter_label == "攻击" and 'attack_type' in df.columns:
            filtered_df = filtered_df[filtered_df['attack_type'] == filter_attack]

        # Feature distributions
        st.subheader("特征分布")

        dist_feature = st.selectbox(
            "选择要可视化的特征",
            ["duration_ms", "total_bytes", "packet_count", "byte_rate", "packets_per_second", "bytes_per_packet"]
        )

        # Scale feature for better visualization
        scale_log = st.checkbox("对数坐标", value=True)

        if scale_log and (filtered_df[dist_feature] > 0).all():
            filtered_df[f"{dist_feature}_log"] = np.log1p(filtered_df[dist_feature])
            plot_feature = f"{dist_feature}_log"
            title_suffix = "（对数）"
        else:
            plot_feature = dist_feature
            title_suffix = ""

        # Create histogram
        hist = alt.Chart(filtered_df).mark_bar().encode(
            alt.X(f"{plot_feature}:Q", title=f"{dist_feature}{title_suffix}", bin=alt.Bin(maxbins=50)),
            alt.Y('count()', title='数量'),
            alt.Color('label:N', title='标签')
        ).properties(
            width=600,
            height=300,
            title=f"{dist_feature}{title_suffix} 分布"
        )

        st.altair_chart(hist, width="stretch")

        # Protocol distribution
        st.subheader("协议分布")

        protocol_counts = filtered_df['protocol'].value_counts().reset_index()
        protocol_counts.columns = ['protocol', 'count']

        pie = alt.Chart(protocol_counts).mark_arc().encode(
            theta=alt.Theta('count:Q'),
            color=alt.Color('protocol:N', scale=alt.Scale(scheme='category10')),
            tooltip=['protocol', 'count']
        ).properties(
            width=400,
            height=400,
            title="协议分布"
        )

        st.altair_chart(pie, width="stretch")

        # Time series
        st.subheader("流量时间趋势")

        # Resample by minute
        filtered_df['minute'] = filtered_df['timestamp'].dt.floor('min')
        time_series = filtered_df.groupby(['minute', 'label']).size().reset_index(name='count')

        line = alt.Chart(time_series).mark_line().encode(
            x=alt.X('minute:T', title='时间'),
            y=alt.Y('count:Q', title='流量数'),
            color=alt.Color('label:N', title='标签')
        ).properties(
            width=600,
            height=300,
            title="流量随时间变化"
        )

        st.altair_chart(line, width="stretch")

        # Port heatmap
        st.subheader("目标端口分析")

        # Get top ports
        top_ports = filtered_df['dst_port'].value_counts().head(20).index.tolist()
        port_df = filtered_df[filtered_df['dst_port'].isin(top_ports)]

        heatmap = alt.Chart(port_df).mark_rect().encode(
            x=alt.X('dst_port:O', title='目标端口'),
            y=alt.Y('label:N', title='标签'),
            color=alt.Color('count()', title='数量', scale=alt.Scale(scheme='viridis')),
            tooltip=['dst_port', 'label', 'count()']
        ).properties(
            width=600,
            height=200,
            title="按标签统计的目标端口使用情况"
        )

        st.altair_chart(heatmap, width="stretch")

    # Tab 3: Anomaly Detection
    with tab3:
        st.header("异常检测结果")
        if st.session_state.get("capture_source") == "live" and len(df) < 50:
            st.warning(
                f"当前实时样本较少（{len(df)} 条 flow），异常检测结果可能波动较大，"
                "建议延长抓包时长或提高最大抓包数量后再复核。"
            )

        # Model selection
        model_display = st.selectbox(
            "选择异常检测模型",
            ["孤立森林", "单类 SVM", "DBSCAN", "集成模型（多数投票）"]
        )
        model_choice_map = {
            "孤立森林": "IsolationForest",
            "单类 SVM": "OneClassSVM",
            "DBSCAN": "DBSCAN",
            "集成模型（多数投票）": "Ensemble (Majority Vote)"
        }
        model_choice = model_choice_map[model_display]

        if model_choice == "IsolationForest":
            score_col = 'score_iso_norm'
            pred_col = 'pred_iso'
            model = st.session_state['models'].get('iso')
        elif model_choice == "OneClassSVM":
            score_col = 'score_ocsvm_norm'
            pred_col = 'pred_ocsvm'
            model = st.session_state['models'].get('ocsvm')
        elif model_choice == "DBSCAN":
            score_col = 'score_dbscan_norm'
            pred_col = 'pred_dbscan'
            model = st.session_state['models'].get('dbscan')
        else:  # Ensemble
            # Create ensemble prediction (majority vote)
            df['pred_ensemble'] = (
                (df['pred_iso'] + df['pred_ocsvm'] + df['pred_dbscan'] >= 2)
            ).astype(int)

            # Create ensemble score (average of normalized scores)
            df['score_ensemble_norm'] = (
                                                df['score_iso_norm'] + df['score_ocsvm_norm'] + df['score_dbscan_norm']
                                        ) / 3

            score_col = 'score_ensemble_norm'
            pred_col = 'pred_ensemble'
            model = None  # No single model for ensemble

        # 异常阈值ing
        threshold = st.slider(
            "用于告警的异常分数阈值",
            min_value=0.0,
            max_value=1.0,
            value=anomaly_threshold,
            step=0.05,
            key="anomaly_threshold_slider"
        )

        # Apply threshold to create alert flag
        df['is_alert'] = (df[score_col] >= threshold).astype(int)

        # Show metrics at current threshold
        st.subheader(f"阈值下性能 {threshold:.2f}")

        col1, col2, col3, col4 = st.columns(4)

        # Calculate metrics
        y_pred = df['is_alert']
        if has_ground_truth:
            try:
                prec = metrics.precision_score(y_true, y_pred)
                rec = metrics.recall_score(y_true, y_pred)
                f1 = metrics.f1_score(y_true, y_pred)
                acc = metrics.accuracy_score(y_true, y_pred)

                with col1:
                    st.metric("精确率", f"{prec:.3f}")
                with col2:
                    st.metric("召回率", f"{rec:.3f}")
                with col3:
                    st.metric("F1 分数", f"{f1:.3f}")
                with col4:
                    st.metric("准确率", f"{acc:.3f}")
            except Exception:
                st.warning("无法在当前阈值下计算指标")
        else:
            st.info("当前为无标签数据，仅展示异常告警数量与分布，不计算监督指标。")
            with col1:
                st.metric("触发告警数", int(y_pred.sum()))
            with col2:
                st.metric("告警占比", f"{(float(y_pred.mean()) if len(y_pred) else 0):.1%}")
            with col3:
                st.metric("平均异常分数", f"{float(df[score_col].mean()):.3f}")
            with col4:
                st.metric("最大异常分数", f"{float(df[score_col].max()):.3f}")

        # Show anomalies
        st.subheader("检测到的异常")

        # Apply threshold
        anomalies = df[df[score_col] >= threshold].sort_values(score_col, ascending=False)

        if anomalies.empty:
            st.info("当前阈值下未检测到异常。")
        else:
            # Style anomalies by severity
            def color_anomaly_score(val):
                if val >= 0.9:
                    return 'background-color: rgba(255,0,0,0.2); color: #B30000; font-weight: bold'
                elif val >= 0.7:
                    return 'background-color: rgba(255,165,0,0.2); color: #CC5500; font-weight: bold'
                else:
                    return 'background-color: rgba(255,255,0,0.1); color: #999900'

            # Show high-level anomaly table
            display_cols = ['flow_id', 'timestamp', 'src_ip', 'dst_ip', 'dst_port', 'protocol',
                            'duration_ms', 'total_bytes', 'label']

            if 'attack_type' in anomalies.columns:
                display_cols.append('attack_type')

            display_cols.append(score_col)

            st.dataframe(
                anomalies[display_cols]
                .rename(columns={score_col: 'anomaly_score'})
                .style.format({'anomaly_score': '{:.3f}'})
                .map(color_anomaly_score, subset=['anomaly_score'])
            )

            # Alert summary
            st.subheader("告警摘要")

            col1, col2 = st.columns(2)

            with col1:
                if has_ground_truth:
                    true_pos = len(anomalies[anomalies['label'] == 'attack'])
                    false_pos = len(anomalies[anomalies['label'] == 'normal'])
                    st.metric("真正告警", true_pos)
                    st.metric("误报告警", false_pos)
                else:
                    st.metric("告警总数", len(anomalies))
                    st.metric("告警占比", f"{(len(anomalies)/max(1, len(df))):.1%}")

            with col2:
                # 告警中的攻击类型分布
                if 'attack_type' in anomalies.columns:
                    attack_distribution = anomalies[anomalies['label'] == 'attack']['attack_type'].value_counts()
                    st.markdown("**检测到的攻击类型：**")
                    for attack_type, count in attack_distribution.items():
                        st.markdown(f"- {attack_type}: {count}")

        # Anomaly score distribution
        st.subheader("异常分数分布")

        hist = alt.Chart(df).mark_bar().encode(
            alt.X(f"{score_col}:Q", title="异常分数", bin=alt.Bin(maxbins=50)),
            alt.Y('count()', title="数量"),
            alt.Color('label:N', title="标签")
        ).properties(
            width=600,
            height=300,
            title=f"{model_choice} 异常分数分布"
        )

        # Add threshold line
        threshold_line = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_rule(
            color='red',
            strokeDash=[3, 3]
        ).encode(
            x='threshold:Q'
        )

        st.altair_chart(hist + threshold_line, width="stretch")

        # Scatter plot of key features
        st.subheader("特征关系")

        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X 轴特征", features, index=features.index('byte_rate'))
        with col2:
            y_feature = st.selectbox("Y 轴特征", features, index=features.index('duration_ms'))

        scatter = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X(f"{x_feature}:Q", title=x_feature),
            y=alt.Y(f"{y_feature}:Q", title=y_feature),
            color=alt.Color(f"{score_col}:Q", title="异常分数", scale=alt.Scale(scheme='inferno')),
            tooltip=['flow_id', 'src_ip', 'dst_ip', 'protocol', x_feature, y_feature, score_col, 'label']
        ).properties(
            width=700,
            height=400,
            title=f"{x_feature} 与 {y_feature} 的关系（按异常分数着色）"
        )

        st.altair_chart(scatter, width="stretch")

    # Tab 4: Explanations
    with tab4:
        st.header("异常解释")

        # 选择解释模型
        model_choice_exp = st.selectbox(
            "选择解释模型",
            ["IsolationForest", "OneClassSVM", "DBSCAN"],
            key="explanation_model"
        )

        if model_choice_exp == "IsolationForest":
            score_col_exp = 'score_iso_norm'
            model_exp = st.session_state['models'].get('iso')
        elif model_choice_exp == "OneClassSVM":
            score_col_exp = 'score_ocsvm_norm'
            model_exp = st.session_state['models'].get('ocsvm')
        else:  # DBSCAN
            score_col_exp = 'score_dbscan_norm'
            model_exp = st.session_state['models'].get('dbscan')

        # Get top anomalies
        top_anomalies = df.sort_values(score_col_exp, ascending=False).head(10)

        if top_anomalies.empty:
            st.info("暂无可解释异常。")
        else:
            # Select an anomaly to explain
            selected_flow_id = st.selectbox(
                "选择要解释的流量",
                top_anomalies['flow_id'].tolist(),
                format_func=lambda x: f"Flow {x} (Score: {float(top_anomalies[top_anomalies['flow_id'] == x][score_col_exp].iloc[0]):.3f})"
            )

            # Get the selected flow
            selected_flow = df[df['flow_id'] == selected_flow_id].iloc[0]

            # Display flow details
            st.subheader("流量详情")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**源：** {selected_flow['src_ip']}:{selected_flow['src_port']}")
                st.markdown(f"**目标：** {selected_flow['dst_ip']}:{selected_flow['dst_port']}")
                st.markdown(f"**协议：** {selected_flow['protocol']}")

            with col2:
                st.markdown(f"**持续时间：** {selected_flow['duration_ms']:.2f} ms")
                st.markdown(f"**字节数：** {selected_flow['total_bytes']} 字节")
                if 'packet_count' in selected_flow:
                    st.markdown(f"**包数：** {selected_flow['packet_count']} 包")

            with col3:
                st.markdown(f"**字节速率：** {selected_flow['byte_rate']:.2f} 字节/毫秒")
                if 'packets_per_second' in selected_flow:
                    st.markdown(f"**包速率：** {selected_flow['packets_per_second']:.2f} 包/秒")
                st.markdown(f"**异常分数：** {selected_flow[score_col_exp]:.3f}")

            # Generate explanation
            if model_exp is not None:
                explanation = explain_anomaly(model_exp, selected_flow, features)

                # Extract insights
                insights = extract_key_insights(explanation, top_n=5)

                # Display insights
                st.subheader("关键洞察")

                for i, insight in enumerate(insights):
                    st.markdown(f"{i+1}. {insight}")

                # Feature importance visualization
                st.subheader("特征贡献")

                # Get top contributing features
                top_features = explanation.head(10)

                # Create bar chart
                bar = alt.Chart(top_features).mark_bar().encode(
                    x=alt.X('contribution:Q', title='异常贡献度'),
                    y=alt.Y('feature:N', title='特征', sort='-x'),
                    color=alt.Color('contribution:Q', scale=alt.Scale(scheme='reds')),
                    tooltip=['feature', 'value', 'contribution']
                ).properties(
                    width=700,
                    height=400,
                    title="异常贡献最高的特征"
                )

                st.altair_chart(bar, width="stretch")

                # Comparison to normal traffic
                st.subheader("与正常流量对比")

                # Select top features to compare
                compare_features = explanation.head(5)['feature'].tolist()

                # Prepare data for comparison
                compare_data = []
                for feature in compare_features:
                    # Get stats for normal traffic
                    normal_mean = df[df['label'] == 'normal'][feature].mean()
                    normal_std = df[df['label'] == 'normal'][feature].std()

                    # Get value for anomaly
                    anomaly_value = selected_flow[feature]

                    # Calculate z-score
                    z_score = (anomaly_value - normal_mean) / max(normal_std, 1e-5)  # Avoid division by zero

                    compare_data.append({
                        'feature': feature,
                        'normal_mean': normal_mean,
                        'anomaly_value': anomaly_value,
                        'z_score': z_score
                    })

                compare_df = pd.DataFrame(compare_data)

                # Create comparison chart
                comparison = alt.Chart(compare_df).mark_bar().encode(
                    x=alt.X('z_score:Q', title='偏离均值的标准差'),
                    y=alt.Y('feature:N', title='特征', sort='-x'),
                    color=alt.Color('z_score:Q', scale=alt.Scale(domain=[-3, 3], scheme='redblue')),
                    tooltip=['feature', 'normal_mean', 'anomaly_value', 'z_score']
                ).properties(
                    width=700,
                    height=300,
                    title="异常流量与正常流量的 Z-Score 对比"
                )

                st.altair_chart(comparison, width="stretch")
            else:
                st.warning("模型不可用，请重新训练。")

    # Tab 5: Simulation
    with tab5:
        st.header("实时网络仿真")
        if st.session_state.get('capture_source') == "live":
            st.caption("提示：实时网络仿真是对已处理流量的离线回放，不会直接继续抓取网卡新报文。")

        if enable_simulation:
            # Initialize session state for real-time data.
            if 'real_time_flows' not in st.session_state:
                st.session_state['real_time_flows'] = []
            if 'alert_count' not in st.session_state:
                st.session_state['alert_count'] = 0
            if df is None or not isinstance(df, pd.DataFrame) or df.empty or 'timestamp' not in df.columns:
                st.warning("当前数据不支持仿真回放。请先点击“处理数据”，并确保存在 `timestamp` 字段。")
            else:
                # Control actions for simulation lifecycle.
                ctl1, ctl2, ctl3, ctl4 = st.columns(4)
                with ctl1:
                    start_sim_replay = st.button("启动仿真回放", width="stretch", key="sim_start_btn")
                with ctl2:
                    pause_sim_replay = st.button("暂停仿真", width="stretch", key="sim_pause_btn")
                with ctl3:
                    refresh_sim_frame = st.button("刷新仿真帧", width="stretch", key="sim_refresh_btn")
                with ctl4:
                    reset_sim_replay = st.button("重置仿真回放", width="stretch", key="sim_reset_btn")

                if start_sim_replay:
                    # Stop previous simulator if still alive.
                    old_stop_event = st.session_state.get('sim_stop_event')
                    if old_stop_event is not None:
                        old_stop_event.set()

                    while not flow_queue.empty():
                        try:
                            flow_queue.get_nowait()
                        except queue.Empty:
                            break

                    st.session_state['real_time_flows'] = []
                    st.session_state['alert_count'] = 0
                    st.session_state['sim_total_emitted'] = 0
                    sim_source_df = df.copy()
                    st.session_state['sim_source_df'] = sim_source_df
                    st.session_state['sim_total_source'] = int(len(sim_source_df))

                    stop_event = threading.Event()
                    simulator = create_flow_simulator(sim_source_df, speed_factor, flow_queue, stop_event=stop_event)
                    simulator.daemon = True
                    simulator.start()

                    st.session_state['simulation_running'] = True
                    st.session_state['sim_thread'] = simulator
                    st.session_state['sim_stop_event'] = stop_event
                    st.success("仿真已启动，正在展示实时流量。")

                if pause_sim_replay:
                    stop_event = st.session_state.get('sim_stop_event')
                    if stop_event is not None:
                        stop_event.set()
                    st.session_state['simulation_running'] = False
                    st.session_state['sim_thread'] = None
                    st.info("仿真已暂停。")

                if reset_sim_replay:
                    stop_event = st.session_state.get('sim_stop_event')
                    if stop_event is not None:
                        stop_event.set()
                    st.session_state['simulation_running'] = False
                    st.session_state['sim_thread'] = None
                    st.session_state['sim_stop_event'] = None
                    st.session_state['real_time_flows'] = []
                    st.session_state['alert_count'] = 0
                    st.session_state['sim_total_emitted'] = 0
                    st.session_state['sim_total_source'] = int(len(df))
                    st.session_state['sim_source_df'] = None
                    while not flow_queue.empty():
                        try:
                            flow_queue.get_nowait()
                        except queue.Empty:
                            break
                    st.success("已重置仿真回放状态。")

                sim_thread = st.session_state.get('sim_thread')
                if st.session_state.get('simulation_running', False) and sim_thread is not None:
                    if (not sim_thread.is_alive()) and flow_queue.empty():
                        st.session_state['simulation_running'] = False
                        st.session_state['sim_thread'] = None
                        st.info("仿真回放已结束。可点击“启动仿真回放”再次开始。")

                # Container for real-time updates
                flow_container = st.empty()
                alert_container = st.empty()
                chart_container = st.empty()

                # Simulation settings
                col1, col2 = st.columns(2)
                with col1:
                    auto_scroll = st.checkbox(
                        "自动滚动到最新",
                        value=st.session_state.get('sim_auto_refresh', True),
                        key="sim_auto_refresh_checkbox"
                    )
                    st.session_state['sim_auto_refresh'] = auto_scroll
                with col2:
                    max_flows = st.number_input("最大显示流量数", min_value=10, max_value=1000, value=50, step=10)

                # Get new flows from queue
                new_flows = []
                while not flow_queue.empty():
                    try:
                        flow = flow_queue.get_nowait()
                        new_flows.append(flow)
                    except queue.Empty:
                        break

                st.session_state['sim_total_emitted'] = st.session_state.get('sim_total_emitted', 0) + len(new_flows)

                # Add to session state
                st.session_state['real_time_flows'].extend(new_flows)

                # Keep only the latest flows
                if len(st.session_state['real_time_flows']) > max_flows:
                    st.session_state['real_time_flows'] = st.session_state['real_time_flows'][-max_flows:]

                # Convert to DataFrame
                if st.session_state['real_time_flows']:
                    live_df = pd.DataFrame(st.session_state['real_time_flows'])
                    sim_ref_df = st.session_state.get('sim_source_df')
                    if not isinstance(sim_ref_df, pd.DataFrame) or sim_ref_df.empty:
                        sim_ref_df = df

                    # Score the flows using the selected model
                    if 'iso' in st.session_state['models'] and model_choice == "IsolationForest":
                        iso_model = st.session_state['models']['iso']
                        live_features = live_df[features]
                        live_df['score'] = -iso_model.decision_function(live_features)

                        # 正常ize score
                        score_ref_col = 'score_iso' if 'score_iso' in sim_ref_df.columns else None
                        if score_ref_col is None and 'score_ensemble_norm' in sim_ref_df.columns:
                            score_ref_col = 'score_ensemble_norm'
                        if score_ref_col is not None:
                            min_val = sim_ref_df[score_ref_col].min()
                            max_val = sim_ref_df[score_ref_col].max()
                        else:
                            min_val = float(live_df['score'].min())
                            max_val = float(live_df['score'].max())
                        if max_val > min_val:
                            live_df['score_norm'] = (live_df['score'] - min_val) / (max_val - min_val)
                        else:
                            live_df['score_norm'] = 0

                        # Apply threshold
                        live_df['is_alert'] = (live_df['score_norm'] >= threshold).astype(int)

                    elif 'ocsvm' in st.session_state['models'] and 'ocsvm_scaler' in st.session_state['models'] and model_choice == "OneClassSVM":
                        ocsvm_model = st.session_state['models']['ocsvm']
                        ocsvm_scaler = st.session_state['models']['ocsvm_scaler']

                        live_features = live_df[features]
                        X_scaled = ocsvm_scaler.transform(live_features)
                        live_df['score'] = -ocsvm_model.decision_function(X_scaled)

                        # 正常ize score
                        score_ref_col = 'score_ocsvm' if 'score_ocsvm' in sim_ref_df.columns else None
                        if score_ref_col is None and 'score_ensemble_norm' in sim_ref_df.columns:
                            score_ref_col = 'score_ensemble_norm'
                        if score_ref_col is not None:
                            min_val = sim_ref_df[score_ref_col].min()
                            max_val = sim_ref_df[score_ref_col].max()
                        else:
                            min_val = float(live_df['score'].min())
                            max_val = float(live_df['score'].max())
                        if max_val > min_val:
                            live_df['score_norm'] = (live_df['score'] - min_val) / (max_val - min_val)
                        else:
                            live_df['score_norm'] = 0

                        # Apply threshold
                        live_df['is_alert'] = (live_df['score_norm'] >= threshold).astype(int)

                    elif 'dbscan' in st.session_state['models'] and 'dbscan_scaler' in st.session_state['models'] and model_choice == "DBSCAN":
                        dbscan_model = st.session_state['models']['dbscan']
                        dbscan_scaler = st.session_state['models']['dbscan_scaler']

                        live_features = live_df[features]
                        X_scaled = dbscan_scaler.transform(live_features)
                        live_df['pred'] = dbscan_model.fit_predict(X_scaled)
                        live_df['score_norm'] = (live_df['pred'] == -1).astype(float)

                        # Apply threshold
                        live_df['is_alert'] = (live_df['score_norm'] >= threshold).astype(int)

                    else:  # Ensemble or fallback
                        # Use a simple heuristic if models aren't available
                        if (
                            'byte_rate' in live_df.columns and 'duration_ms' in live_df.columns
                            and 'byte_rate' in sim_ref_df.columns and 'duration_ms' in sim_ref_df.columns
                        ):
                            # Flag very high byte rates or very short/long durations
                            byte_rate_threshold = sim_ref_df['byte_rate'].quantile(0.95)
                            duration_low = sim_ref_df['duration_ms'].quantile(0.05)
                            duration_high = sim_ref_df['duration_ms'].quantile(0.95)

                            live_df['score_norm'] = 0.0
                            # Set high score for unusual byte rates or durations
                            live_df.loc[live_df['byte_rate'] > byte_rate_threshold, 'score_norm'] = 0.8
                            live_df.loc[live_df['duration_ms'] < duration_low, 'score_norm'] = 0.8
                            live_df.loc[live_df['duration_ms'] > duration_high, 'score_norm'] = 0.7

                            live_df['is_alert'] = (live_df['score_norm'] >= threshold).astype(int)
                        else:
                            # If no features available, don't score
                            live_df['score_norm'] = 0.0
                            live_df['is_alert'] = 0

                    # Update alert count
                    new_alerts = int(live_df['is_alert'].sum()) if 'is_alert' in live_df.columns else 0
                    st.session_state['alert_count'] = new_alerts

                    # Display alerts
                    with alert_container:
                        alert_col1, alert_col2 = st.columns(2)
                        with alert_col1:
                            st.metric("总告警数", st.session_state['alert_count'])
                            total_source = st.session_state.get('sim_total_source', int(len(df)))
                            emitted = st.session_state.get('sim_total_emitted', len(live_df))
                            st.caption(f"回放进度：{emitted}/{total_source}")
                        with alert_col2:
                            # Get most recent alert
                            if new_alerts > 0:
                                last_alert = live_df[live_df['is_alert'] == 1].iloc[-1]
                                st.markdown(f"""
                            <div class="alert-box">
                                <strong>鈿狅笍 新告警！</strong><br>
                                源： {last_alert['src_ip']}:{last_alert['src_port']}<br>
                                目标： {last_alert['dst_ip']}:{last_alert['dst_port']}<br>
                                Score: {last_alert['score_norm']:.3f}
                            </div>
                            """, unsafe_allow_html=True)

                    # Display flows
                    with flow_container:
                        st.subheader("实时网络流量")

                        # Style function for highlighting anomalies
                        def highlight_anomalies(row):
                            is_alert_val = row.get('is_alert', 0)
                            score_val = row.get('score_norm', 0.0)
                            if is_alert_val == 1:
                                if score_val >= 0.9:
                                    return ['background-color: rgba(255,0,0,0.2)'] * len(row)
                                elif score_val >= 0.7:
                                    return ['background-color: rgba(255,165,0,0.2)'] * len(row)
                                else:
                                    return ['background-color: rgba(255,255,0,0.1)'] * len(row)
                            return [''] * len(row)

                        # Format score
                        def format_score(val):
                            if val >= 0.9:
                                return f'<span class="anomaly-high">{val:.3f}</span>'
                            elif val >= 0.7:
                                return f'<span class="anomaly-medium">{val:.3f}</span>'
                            else:
                                return f'{val:.3f}'

                        # Display dataframe with styling
                        display_cols = ['timestamp', 'src_ip', 'dst_ip', 'dst_port', 'protocol',
                                        'duration_ms', 'total_bytes']

                        if 'packet_count' in live_df.columns:
                            display_cols.append('packet_count')

                        display_cols.extend([c for c in ['score_norm', 'label', 'is_alert'] if c in live_df.columns])

                        if 'attack_type' in live_df.columns:
                            display_cols.append('attack_type')

                        st.dataframe(
                            live_df[display_cols]
                            .sort_values('timestamp', ascending=False)
                            .reset_index(drop=True)
                            .style
                            .apply(highlight_anomalies, axis=1)
                            .format({'score_norm': '{:.3f}', 'duration_ms': '{:.2f}'})
                        )

                    # Plot real-time metrics
                    with chart_container:
                        st.subheader("实时指标")

                        # Prepare time series data
                        live_df['minute'] = pd.to_datetime(live_df['timestamp']).dt.floor('min')
                        time_series = live_df.groupby('minute').agg({
                            'flow_id': 'count',
                            'is_alert': 'sum'
                        }).reset_index()
                        time_series.columns = ['minute', 'total_flows', 'alerts']

                        # Melt for plotting
                        plot_df = pd.melt(
                            time_series,
                            id_vars=['minute'],
                            value_vars=['total_flows', 'alerts'],
                            var_name='metric',
                            value_name='count'
                        )

                        # Create line chart
                        line = alt.Chart(plot_df).mark_line().encode(
                            x=alt.X('minute:T', title='时间'),
                            y=alt.Y('count:Q', title='数量'),
                            color=alt.Color('metric:N', title='指标')
                        ).properties(
                            width=700,
                            height=300,
                            title="流量与告警趋势"
                        )

                        st.altair_chart(line, width="stretch")

                else:
                    if st.session_state.get('simulation_running', False):
                        total_source = st.session_state.get('sim_total_source', int(len(df)))
                        emitted = st.session_state.get('sim_total_emitted', 0)
                        st.info(f"正在等待网络流量...（已回放 {emitted}/{total_source}，可点击“刷新仿真帧”）")
                    else:
                        st.info("当前无仿真流量。可点击“启动仿真回放”开始。")

                # Auto refresh while simulation is running.
                if (
                    st.session_state.get('simulation_running', False)
                    and st.session_state.get('sim_auto_refresh', True)
                    and (not refresh_sim_frame)
                ):
                    time.sleep(0.4)
                    st.rerun()

        else:
            st.info("请在侧栏启用实时仿真以查看实时流量。")

        # Alert configuration
        st.header("告警配置")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("告警阈值")

            st.markdown("""
            为不同严重级别配置异常分数阈值：
            - **高危**：需立即处理，可能存在进行中的威胁
            - **中危**：可疑行为，建议进一步排查
            - **低危**：存在异常，但可能为正常波动
            """)

            high_threshold = st.slider("高危阈值", 0.0, 1.0, 0.9, 0.05)
            medium_threshold = st.slider("中危阈值", 0.0, high_threshold, 0.7, 0.05)
            low_threshold = st.slider("低危阈值", 0.0, medium_threshold, 0.5, 0.05)

        with col2:
            st.subheader("告警动作")

            st.markdown("配置告警触发后的动作：")

            log_all_alerts_to_file = st.checkbox("记录全部告警到文件", value=True, key="resp_log_all")
            send_high_alert_email = st.checkbox("高危告警发送邮件", value=False, key="resp_email_enable")
            send_webhook_notify = st.checkbox("发送 Webhook 通知", value=False, key="resp_webhook_enable")
            webhook_url = st.text_input("Webhook 地址（启用时）", key="resp_webhook_url")
            email_target = st.text_input("邮件接收人（可选，留空用 SMTP_TO）", key="resp_email_to")
            email_min_severity = st.selectbox("邮件触发最小严重级别", ["high", "medium", "low"], index=0, key="resp_email_min_sev")
            execute_response_button = st.button("执行响应动作", width="stretch", key="resp_execute_btn")

            st.markdown("告警保留策略：")
            st.radio(
                "保留周期",
                ["24 小时", "7 天", "30 天", "90 天", "永久"],
                index=2
            )

            if execute_response_button:
                current_alerts = st.session_state.get("rule_alerts_df")
                action_df = execute_responses(
                    alerts_df=current_alerts,
                    log_all_alerts=log_all_alerts_to_file,
                    enable_webhook=send_webhook_notify,
                    webhook_url=webhook_url,
                    enable_email=send_high_alert_email,
                    email_min_severity=email_min_severity,
                    smtp_to=email_target,
                    output_dir="data",
                )
                st.session_state["response_actions_df"] = action_df
                st.session_state["log_snapshot_paths"] = persist_snapshot(
                    source="response_execute",
                    packet_df=st.session_state.get("pcap_df"),
                    flow_df=st.session_state.get("flows_df"),
                    alert_df=st.session_state.get("rule_alerts_df"),
                    incident_df=st.session_state.get("incident_df"),
                    action_df=st.session_state.get("response_actions_df"),
                )
                st.success("响应动作执行完成。")

            actions_df = st.session_state.get("response_actions_df")
            if actions_df is not None and not actions_df.empty:
                st.subheader("响应执行结果")
                st.dataframe(actions_df.sort_values("timestamp", ascending=False), width="stretch")
                st.caption("响应审计日志已写入 data/response_audit_YYYYMMDD.jsonl")

    # Tab 6: Rules Detection
    with tab6:
        st.header("基于规则的检测结果")
        st.markdown(
            f"当前参数：时间窗口 `{rule_window_s}` 秒，"
            f"端口去重阈值 `{rule_unique_port_threshold}`，"
            f"暴力破解阈值 `{brute_force_attempt_threshold}`，"
            f"DNS 查询去重阈值 `{dns_unique_query_threshold}`，"
            f"DNS NXDOMAIN 阈值 `{dns_nxdomain_threshold}`，"
            f"ARP 泛洪阈值 `{arp_flood_threshold}`，"
            f"ARP 扫描阈值 `{arp_scan_target_threshold}`，"
            f"ARP 欺骗阈值 `{arp_spoof_mac_threshold}`，"
            f"ARP MITM可疑阈值 `{arp_mitm_ip_claim_threshold}`，"
            f"ARP 滥用阈值 `{arp_abuse_gratuitous_threshold}`，"
            f"突发流量倍数阈值 `{stat_burst_multiplier}`，"
            f"突发最小字节阈值 `{stat_burst_min_bytes}`，"
            f"周期最小事件数 `{stat_beacon_min_events}`，"
            f"周期最大IAT变异系数 `{stat_beacon_max_cv}`"
        )
        st.caption("使用侧栏“执行规则检测”按钮刷新结果。")

        alerts_df = st.session_state.get('rule_alerts_df')
        if alerts_df is None:
            st.info("暂无规则检测结果，请点击侧栏“执行规则检测”。")
        elif alerts_df.empty:
            st.success("当前阈值下未命中规则告警。")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("规则告警总数", int(len(alerts_df)))
            with col2:
                st.metric("源 IP 去重数", int(alerts_df['src_ip'].nunique()))

            st.subheader("按规则统计告警")
            by_rule = (
                alerts_df.groupby('rule_name', as_index=False)
                .agg(alert_count=('rule_name', 'count'))
                .sort_values('alert_count', ascending=False)
            )
            st.dataframe(by_rule, width="stretch")

            st.subheader("按源 IP 统计告警")
            by_src = (
                alerts_df.groupby('src_ip', as_index=False)
                .agg(alert_count=('rule_name', 'count'), max_unique_ports=('unique_ports', 'max'))
                .sort_values(['alert_count', 'max_unique_ports'], ascending=[False, False])
            )
            st.dataframe(by_src, width="stretch")

            st.subheader("告警明细")
            rule_options = ["全部"] + sorted(alerts_df['rule_name'].astype(str).unique().tolist())
            selected_rule = st.selectbox("按规则名筛选", rule_options, key="rule_name_filter")
            src_options = ["全部"] + sorted(alerts_df['src_ip'].astype(str).unique().tolist())
            selected_src = st.selectbox("按源 IP 筛选", src_options, key="rule_src_filter")

            details_df = alerts_df.copy()
            if selected_rule != "全部":
                details_df = details_df[details_df['rule_name'].astype(str) == selected_rule]
            if selected_src != "全部":
                details_df = details_df[details_df['src_ip'].astype(str) == selected_src]

            details_df = details_df.sort_values(by=["timestamp", "unique_ports"], ascending=[False, False])
            st.dataframe(details_df, width="stretch")

    # Tab 7: 事件中心
    with tab7:
        st.header("事件中心（告警聚合与攻击链）")
        st.caption(
            f"当前事件聚合窗口：`{incident_correlation_window_s}` 秒。"
            "同一源IP在窗口内的连续告警将聚合为同一事件。"
        )

        incident_df = st.session_state.get("incident_df")
        if incident_df is None:
            st.info("暂无事件聚合结果，请先执行规则检测。")
        elif incident_df.empty:
            st.success("当前告警未形成可聚合事件。")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("事件总数", int(len(incident_df)))
            with c2:
                st.metric("高危事件数", int((incident_df["max_severity"] == "high").sum()))
            with c3:
                st.metric("源IP去重数", int(incident_df["src_ip"].nunique()))
            with c4:
                st.metric("平均风险分", f"{incident_df['risk_score'].mean():.1f}")

            st.subheader("事件时间线")
            timeline_df = incident_df.sort_values("start_time", ascending=False).copy()
            st.dataframe(timeline_df, width="stretch")

            st.subheader("按源IP聚合")
            by_src = (
                incident_df.groupby("src_ip", as_index=False)
                .agg(
                    incident_count=("incident_id", "count"),
                    total_alerts=("alert_count", "sum"),
                    max_risk=("risk_score", "max"),
                )
                .sort_values(["max_risk", "incident_count"], ascending=[False, False])
            )
            st.dataframe(by_src, width="stretch")

    # Tab 8: ARP 专题
    with tab8:
        st.header("ARP 攻击检测专题")
        alerts_df = st.session_state.get('rule_alerts_df')
        if alerts_df is None:
            st.info("暂无规则检测结果，请先在侧栏点击“执行规则检测”。")
        else:
            arp_rules = ["ARP_FLOOD", "ARP_SCAN", "ARP_SPOOF"]
            arp_alerts = alerts_df[alerts_df["rule_name"].isin(arp_rules)].copy()

            if arp_alerts.empty:
                st.info("当前结果中未检测到 ARP 相关告警。可尝试上传包含 ARP 流量的 PCAP。")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ARP 告警总数", int(len(arp_alerts)))
                with col2:
                    st.metric("ARP 源 IP 去重数", int(arp_alerts["src_ip"].nunique()))
                with col3:
                    st.metric("ARP 规则类型数", int(arp_alerts["rule_name"].nunique()))

                st.subheader("按 ARP 规则统计")
                by_rule = (
                    arp_alerts.groupby("rule_name", as_index=False)
                    .agg(alert_count=("rule_name", "count"))
                    .sort_values("alert_count", ascending=False)
                )
                st.dataframe(by_rule, width="stretch")

                st.subheader("ARP 告警明细")
                arp_rule_options = ["全部"] + sorted(arp_alerts["rule_name"].unique().tolist())
                selected_arp_rule = st.selectbox("筛选 ARP 规则", arp_rule_options, key="arp_rule_filter")

                arp_details = arp_alerts.copy()
                if selected_arp_rule != "全部":
                    arp_details = arp_details[arp_details["rule_name"] == selected_arp_rule]
                arp_details = arp_details.sort_values(by=["timestamp", "rule_name"], ascending=[False, True])
                st.dataframe(arp_details, width="stretch")

    # Tab 9: 日志中心
    with tab9:
        st.header("结构化日志中心")
        st.markdown("统一查看并下载 packet/flow/alert/incident/action 五类 JSONL 日志。")

        latest_paths = st.session_state.get("log_snapshot_paths") or {}
        if latest_paths:
            st.subheader("最近一次快照路径")
            latest_df = pd.DataFrame(
                [{"log_type": k, "path": v} for k, v in latest_paths.items() if k != "manifest"]
            )
            if not latest_df.empty:
                st.dataframe(latest_df, width="stretch")

        manifest_path = os.path.join("data", "logs", "manifest.jsonl")
        if os.path.exists(manifest_path):
            st.subheader("快照清单（最近20条）")
            manifest_rows = []
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        manifest_rows.append(json.loads(line))
                    except Exception:
                        continue
            manifest_rows = manifest_rows[-20:]
            if manifest_rows:
                manifest_df = pd.DataFrame(manifest_rows).sort_values("timestamp", ascending=False)
                st.dataframe(manifest_df, width="stretch")
        else:
            st.info("尚未生成日志快照。请先处理数据/执行规则检测/执行响应动作。")

        st.subheader("日志文件下载")
        log_files = sorted(glob.glob(os.path.join("data", "logs", "**", "*.jsonl"), recursive=True), reverse=True)
        if not log_files:
            st.caption("未找到日志文件。")
        else:
            selected_file = st.selectbox("选择日志文件", log_files, index=0)
            if selected_file and os.path.exists(selected_file):
                with open(selected_file, "rb") as f:
                    content = f.read()
                st.download_button(
                    "下载选中文件",
                    data=content,
                    file_name=os.path.basename(selected_file),
                    mime="application/x-ndjson",
                    width="stretch",
                )
                try:
                    preview_lines = content.decode("utf-8", errors="ignore").splitlines()[:20]
                    st.code("\n".join(preview_lines), language="json")
                except Exception:
                    st.caption("预览失败，可直接下载查看。")

    # Tab 10: 报告导出
    with tab10:
        st.header("检测报告导出")
        st.markdown("可基于当前页面数据导出 Markdown 安全报告。")

        report_markdown = build_markdown_report(
            analysis_df=df,
            rule_alerts_df=st.session_state.get("rule_alerts_df"),
            pcap_df=st.session_state.get("pcap_df"),
            incident_df=st.session_state.get("incident_df"),
            action_df=st.session_state.get("response_actions_df"),
            llm_assessment=st.session_state.get("llm_assessment", ""),
        )
        md_filename = build_report_filename("sentinel_report", ext="md")
        pdf_bytes, pdf_error = markdown_to_pdf_bytes(report_markdown)
        pdf_filename = build_report_filename("sentinel_report", ext="pdf")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="下载 Markdown 报告",
                data=report_markdown,
                file_name=md_filename,
                mime="text/markdown",
                width="stretch",
            )
        with col2:
            if pdf_bytes is not None:
                st.download_button(
                    label="下载 PDF 报告",
                    data=pdf_bytes,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    width="stretch",
                )
            else:
                st.caption(pdf_error or "PDF 导出不可用。")
        with col3:
            st.metric("报告字符数", len(report_markdown))

        st.subheader("报告预览")
        st.code(report_markdown[:4000], language="markdown")

        st.subheader("LLM 威胁研判")
        if st.button("生成LLM研判", width="stretch"):
            anomalies_for_assess = None
            if "is_alert" in df.columns:
                anomalies_for_assess = df[df["is_alert"] == 1].copy()
            elif "score_ensemble_norm" in df.columns:
                anomalies_for_assess = df.sort_values("score_ensemble_norm", ascending=False).head(50)
            else:
                anomalies_for_assess = df.head(50)

            st.session_state["llm_assessment"] = generate_threat_assessment(
                alerts_df=st.session_state.get("rule_alerts_df"),
                anomalies_df=anomalies_for_assess,
                provider={
                    "自动": "auto",
                    "DeepSeek": "deepseek",
                    "OpenAI": "openai",
                }.get(llm_provider_display, "auto"),
                model=(llm_model_name.strip() if llm_model_name else "auto"),
                api_key=(llm_api_key_input.strip() if llm_api_key_input else None),
                base_url=(llm_base_url_input.strip() if llm_base_url_input else None),
            )

        if st.session_state.get("llm_assessment"):
            st.text_area("研判输出", st.session_state["llm_assessment"], height=240)

# Footer (fixed bottom)
st.markdown(
    """
    <div class="app-footer">智能网络威胁检测平台</div>
    """,
    unsafe_allow_html=True,
)



