# /home/avi/NSSON_V02/qos/qos_metrics_builder.py

import os
import json
import yaml
import pandas as pd
import numpy as np

NSSON_ROOT = "/home/avi/NSSON_V02"

DECISION_LOG = os.path.join(NSSON_ROOT, "offload", "decision_log.json")
LATENCY_LOG = os.path.join(NSSON_ROOT, "offload", "latency_log.csv")
POWER_LOG = os.path.join(NSSON_ROOT, "offload", "power_log.csv")
RYU_FLOW_LOG = os.path.join(NSSON_ROOT, "vmcontroller_runs", "run1", "ryu_flow_log.jsonl")
LINK_STATS_CSV = os.path.join(NSSON_ROOT, "logs", "linkstats.csv")

CONFIG_PATH = os.path.join(NSSON_ROOT, "configs", "qos_config.yaml")

QOS_METRICS_CSV = os.path.join(NSSON_ROOT, "qos_metrics.csv")
ROUTING_PATHS_CSV = os.path.join(NSSON_ROOT, "routing_paths.csv")
OFFLOAD_FEEDBACK_CSV = os.path.join(NSSON_ROOT, "offload_feedback.csv")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_latency_power():
    lat_df = pd.read_csv(LATENCY_LOG)
    pow_df = pd.read_csv(POWER_LOG)
    lat_df["cycle"] = np.arange(1, len(lat_df) + 1)
    pow_df["cycle"] = np.arange(1, len(pow_df) + 1)
    df = pd.merge(lat_df, pow_df, on="cycle", how="inner")
    df.rename(columns={"total_sec": "latency_sec", "total_power_wh": "power_wh"}, inplace=True)
    return df

def parse_decision_log():
    records = []
    with open(DECISION_LOG, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "cycle" not in rec:
                rec["cycle"] = len(records) + 1
            records.append(rec)
    return pd.DataFrame(records)

def parse_ryu_flow_log():
    records = []
    if not os.path.exists(RYU_FLOW_LOG):
        return pd.DataFrame()
    with open(RYU_FLOW_LOG, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "cycle" not in rec:
                rec["cycle"] = len(records) + 1
            records.append(rec)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)

def build_offload_feedback(decisions_df, lat_pow_df):
    agg = decisions_df.groupby("cycle").agg(
        offload_ratio=("decision", lambda s: np.mean(s == "offload")),
        threshold=("threshold", "last")
    ).reset_index()
    merged = pd.merge(agg, lat_pow_df[["cycle", "latency_sec", "power_wh"]], on="cycle", how="left")
    merged["latency_ms"] = merged["latency_sec"] * 1000.0
    return merged

def build_routing_paths(flow_df, linkstats_df):
    if flow_df.empty:
        return pd.DataFrame()
    cols_needed = ["cycle", "path_id", "selected", "slice"]
    for c in cols_needed:
        if c not in flow_df.columns:
            flow_df[c] = None
    if linkstats_df is not None and not linkstats_df.empty:
        merged = pd.merge(flow_df[["cycle", "path_id", "selected", "slice"]],
                          linkstats_df,
                          on=["cycle", "path_id"],
                          how="left")
    else:
        merged = flow_df[["cycle", "path_id", "selected", "slice"]].copy()
        merged["latency_ms"] = np.nan
        merged["loss_rate"] = np.nan
        merged["snr_db"] = np.nan
    return merged

def compute_jitter(latency_series):
    return latency_series.diff().abs()

def build_qos_metrics(config, lat_pow_df, offload_feedback_df, flow_df):
    qos_records = []
    lat_pow_df["jitter_sec"] = compute_jitter(lat_pow_df["latency_sec"])
    lat_pow_df["jitter_ms"] = lat_pow_df["jitter_sec"] * 1000.0

    lat_pow_df["pdr"] = 0.95
    lat_pow_df["drop_rate"] = 0.02
    lat_pow_df["routing_overhead"] = 0.5
    lat_pow_df["throughput_kbps"] = 200.0

    df = lat_pow_df.merge(offload_feedback_df[["cycle", "offload_ratio", "threshold"]],
                          on="cycle", how="left")

    if not flow_df.empty and "slice" in flow_df.columns:
        slice_per_cycle = flow_df.groupby("cycle")["slice"].agg(lambda s: s.iloc[0]).reset_index()
        df = df.merge(slice_per_cycle, on="cycle", how="left")
    else:
        df["slice"] = "control"

    for _, row in df.iterrows():
        cycle = int(row["cycle"])
        slice_name = row["slice"]
        mode = "nsson"

        slice_cfg = config["slices"].get(slice_name, config["slices"]["control"])
        sla = slice_cfg["sla"]

        latency_ms = row["latency_sec"] * 1000.0
        jitter_ms = row["jitter_ms"]
        pdr = row["pdr"]
        drop_rate = row["drop_rate"]
        throughput_kbps = row["throughput_kbps"]
        routing_overhead = row["routing_overhead"]

        sla_met_latency = 1 if latency_ms <= sla["latency_ms_max"] else 0
        sla_met_jitter = 1 if jitter_ms <= sla["jitter_ms_max"] else 0
        sla_met_pdr = 1 if pdr >= sla["pdr_min"] else 0
        sla_met_drop = 1 if drop_rate <= sla["drop_rate_max"] else 0
        sla_met_throughput = 1 if throughput_kbps >= sla["throughput_kbps_min"] else 0

        qos_records.append({
            "cycle": cycle,
            "mode": mode,
            "slice": slice_name,
            "app_class": slice_cfg["app_class"],
            "latency_ms": latency_ms,
            "jitter_ms": jitter_ms,
            "pdr": pdr,
            "drop_rate": drop_rate,
            "routing_overhead": routing_overhead,
            "throughput_kbps": throughput_kbps,
            "sla_latency_ms": sla["latency_ms_max"],
            "sla_pdr": sla["pdr_min"],
            "sla_met_latency": sla_met_latency,
            "sla_met_jitter": sla_met_jitter,
            "sla_met_pdr": sla_met_pdr,
            "sla_met_drop": sla_met_drop,
            "sla_met_throughput": sla_met_throughput,
        })

    return pd.DataFrame(qos_records)

def main():
    config = load_config()
    lat_pow_df = load_latency_power()
    decisions_df = parse_decision_log()
    flow_df = parse_ryu_flow_log()

    linkstats_df = pd.read_csv(LINK_STATS_CSV) if os.path.exists(LINK_STATS_CSV) else None

    offload_feedback_df = build_offload_feedback(decisions_df, lat_pow_df)
    offload_feedback_df.to_csv(OFFLOAD_FEEDBACK_CSV, index=False)

    routing_paths_df = build_routing_paths(flow_df, linkstats_df)
    routing_paths_df.to_csv(ROUTING_PATHS_CSV, index=False)

    qos_df = build_qos_metrics(config, lat_pow_df, offload_feedback_df, flow_df)
    qos_df.to_csv(QOS_METRICS_CSV, index=False)

    print("Written:")
    print("  ", QOS_METRICS_CSV)
    print("  ", ROUTING_PATHS_CSV)
    print("  ", OFFLOAD_FEEDBACK_CSV)

if __name__ == "__main__":
    main()
