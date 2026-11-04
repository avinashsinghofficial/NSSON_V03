# /home/avi/NSSON_V02/generate_qos_data.py
#
# Build:
#   - qos_metrics.csv      (multi-dimensional QoS + SLA flags per cycle/mode)
#   - routing_paths.csv    (per-cycle path metrics per mode for SDN runs)
#   - offload_feedback.csv (per-cycle offload ratio + theta + latency + power per mode)
#
# Modes (scenarios) match Section V-B: Local Only, Offload Only, Adaptive no SDN, NSSON Full,
# plus baseline SDN-IoT for QoS comparison.

import os
import json
import yaml
import pandas as pd
import numpy as np

NSSON_ROOT = "/home/avi/NSSON_V02"

CONFIG_PATH = os.path.join(NSSON_ROOT, "configs", "qos_config.yaml")

SCENARIOS = [
    "local_only",
    "offload_only",
    "adaptive_no_sdn",
    "nsson_full",
    "baseline_sdn_iot",  # for extended QoS baseline vs NSSON
]

def scenario_paths(mode):
    base = os.path.join(NSSON_ROOT, "offload", mode)
    return {
        "latency": os.path.join(base, "latency_log.csv"),
        "power": os.path.join(base, "power_log.csv"),
        "decision": os.path.join(base, "decision_log.json"),
        "ryu_flow": os.path.join(NSSON_ROOT, "vmcontroller_runs", mode, "ryu_flow_log.jsonl"),
    }

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_latency_power(lat_path, pow_path, mode):
    lat_df = pd.read_csv(lat_path)
    pow_df = pd.read_csv(pow_path)
    # per paper: sense_sec, compute_sec, transfer_sec, idle_sec, total_sec
    lat_df.rename(columns={
        "sense_sec": "sense_sec",
        "compute_sec": "compute_sec",
        "transfer_sec": "transfer_sec",
        "idle_sec": "idle_sec",
        "total_sec": "total_sec",
    }, inplace=True)
    pow_df.rename(columns={
        "data_mb": "data_mb",
        "compute_sec": "compute_sec",
        "idle_sec": "idle_sec",
        "compute_power_w": "compute_power_w",
        "transfer_power_w": "transfer_power_w",
        "idle_power_w": "idle_power_w",
        "total_power_wh": "total_power_wh",
    }, inplace=True)
    lat_df["cycle"] = np.arange(1, len(lat_df) + 1)
    pow_df["cycle"] = np.arange(1, len(pow_df) + 1)
    df = pd.merge(lat_df, pow_df, on="cycle", how="inner")
    df["mode"] = mode
    return df

def parse_decision_log(decision_path, mode):
    records = []
    if not os.path.exists(decision_path):
        return pd.DataFrame()
    with open(decision_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Telemetry fields per paper: sample_id, confidence, decision, threshold/theta, timestamp
            if "cycle" not in rec:
                rec["cycle"] = rec.get("sample_id", len(records) + 1)
            rec["mode"] = mode
            records.append(rec)
    df = pd.DataFrame(records)
    # Normalize threshold name: use 'theta' if present, else 'threshold'
    if "theta" in df.columns and "threshold" not in df.columns:
        df = df.rename(columns={"theta": "threshold"})
    return df

def build_offload_feedback(decisions_df, lat_pow_df):
    if decisions_df.empty or lat_pow_df.empty:
        return pd.DataFrame()
    # Ensure threshold column exists
    if "threshold" not in decisions_df.columns:
        decisions_df = decisions_df.copy()
        decisions_df["threshold"] = 0.6  # default fallback

    agg = decisions_df.groupby(["mode", "cycle"]).agg(
        offload_ratio=("decision", lambda s: np.mean(np.array(s) == "offload")),
        threshold=("threshold", "last"),
    ).reset_index()
    merged = pd.merge(
        agg,
        lat_pow_df[["mode", "cycle", "total_sec", "total_power_wh"]],
        on=["mode", "cycle"],
        how="left",
    )
    merged["latency_ms"] = merged["total_sec"] * 1000.0
    merged.rename(columns={"total_power_wh": "power_wh"}, inplace=True)
    return merged

def parse_ryu_flow_log(flow_path, mode):
    if not os.path.exists(flow_path):
        return pd.DataFrame()
    records = []
    with open(flow_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["mode"] = mode
            records.append(rec)
    df = pd.DataFrame(records)
    if "cycle" not in df.columns:
        df["cycle"] = np.arange(1, len(df) + 1)
    # Controller should write: path_id, slice, selected (1/0 or True/False)
    return df

def build_routing_paths(flow_df):
    if flow_df.empty:
        return pd.DataFrame()
    cols = ["mode", "cycle", "path_id", "selected", "slice"]
    for c in cols:
        if c not in flow_df.columns:
            flow_df[c] = None
    routing = flow_df[["mode", "cycle", "path_id", "selected", "slice"]].copy()
    routing["latency_ms"] = np.nan
    routing["loss_rate"] = np.nan
    routing["snr_db"] = np.nan
    routing["throughput_kbps"] = np.nan
    routing["routing_overhead"] = np.nan
    return routing

def compute_jitter(latency_series):
    return latency_series.diff().abs()

def build_qos_metrics(config, lat_pow_df, offload_df, routing_df):
    if lat_pow_df.empty:
        return pd.DataFrame()
    lat_pow_df = lat_pow_df.copy()
    lat_pow_df["jitter_sec"] = compute_jitter(lat_pow_df["total_sec"])
    lat_pow_df["jitter_ms"] = lat_pow_df["jitter_sec"] * 1000.0
    # Placeholder QoS metrics: replace with values from your extended QoS runs
    lat_pow_df["pdr"] = 0.95
    lat_pow_df["drop_rate"] = 0.02
    lat_pow_df["routing_overhead"] = 0.5
    lat_pow_df["throughput_kbps"] = 200.0

    if offload_df.empty:
        df = lat_pow_df.copy()
        df["offload_ratio"] = np.nan
        df["threshold"] = np.nan
    else:
        df = lat_pow_df.merge(
            offload_df[["mode", "cycle", "offload_ratio", "threshold"]],
            on=["mode", "cycle"],
            how="left",
        )

    # If routing_df has slice per cycle/mode, merge it; otherwise assign "control"
    if not routing_df.empty and "slice" in routing_df.columns:
        slice_per_cycle = routing_df.groupby(["mode", "cycle"])["slice"].agg(lambda s: s.iloc[0]).reset_index()
        df = df.merge(slice_per_cycle, on=["mode", "cycle"], how="left")
    else:
        df["slice"] = "control"

    records = []
    for _, row in df.iterrows():
        mode = row["mode"]
        cycle = int(row["cycle"])
        slice_name = row["slice"] if pd.notna(row["slice"]) else "control"

        slice_cfg = config["slices"].get(slice_name, config["slices"]["control"])
        sla = slice_cfg["sla"]

        latency_ms = row["total_sec"] * 1000.0
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

        records.append({
            "mode": mode,
            "cycle": cycle,
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

    return pd.DataFrame(records)

def main():
    config = load_config()

    all_qos = []
    all_routing = []
    all_offload = []

    for mode in SCENARIOS:
        paths = scenario_paths(mode)
        if not os.path.exists(paths["latency"]) or not os.path.exists(paths["power"]):
            print(f"[WARN] Missing latency/power logs for mode {mode}, skipping")
            continue

        lat_pow_df = load_latency_power(paths["latency"], paths["power"], mode)
        decisions_df = parse_decision_log(paths["decision"], mode)
        offload_df = build_offload_feedback(decisions_df, lat_pow_df)
        if not offload_df.empty:
            all_offload.append(offload_df)

        flow_df = parse_ryu_flow_log(paths["ryu_flow"], mode)
        routing_df = build_routing_paths(flow_df)
        if not routing_df.empty:
            all_routing.append(routing_df)

        qos_df = build_qos_metrics(config, lat_pow_df, offload_df, routing_df)
        if not qos_df.empty:
            all_qos.append(qos_df)

    qos_metrics_df = pd.concat(all_qos, ignore_index=True) if all_qos else pd.DataFrame()
    routing_paths_df = pd.concat(all_routing, ignore_index=True) if all_routing else pd.DataFrame()
    offload_feedback_df = pd.concat(all_offload, ignore_index=True) if all_offload else pd.DataFrame()

    if qos_metrics_df.empty:
        print("[WARN] No QoS data generated; check that at least one mode has logs.")
    else:
        qos_metrics_df.to_csv(os.path.join(NSSON_ROOT, "qos_metrics.csv"), index=False)
        print("Written qos_metrics.csv in", NSSON_ROOT)

    if routing_paths_df.empty:
        print("[WARN] No routing data generated; SDN flow logs may be missing.")
    else:
        routing_paths_df.to_csv(os.path.join(NSSON_ROOT, "routing_paths.csv"), index=False)
        print("Written routing_paths.csv in", NSSON_ROOT)

    if offload_feedback_df.empty:
        print("[WARN] No offload feedback data generated.")
    else:
        offload_feedback_df.to_csv(os.path.join(NSSON_ROOT, "offload_feedback.csv"), index=False)
        print("Written offload_feedback.csv in", NSSON_ROOT)

if __name__ == "__main__":
    main()
