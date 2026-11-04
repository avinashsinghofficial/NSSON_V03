#!/usr/bin/env python3
"""Telemetry logger for NSSON experiments"""
import csv
import json
import os
from datetime import datetime
from pathlib import Path

class TelemetryLogger:
    def __init__(self, run_id, model, policy="nsson"):
        self.run_id = run_id
        self.model = model
        self.policy = policy
        self.telemetry_file = Path(f"logs/telemetry/{run_id}_{model}_{policy}.csv")
        self.meta_file = Path(f"logs/meta/{run_id}_{model}_{policy}.json")
        
        self.fieldnames = [
            "cycle_id", "node_id", "model", "policy", "decision", "threshold",
            "decision_score", "rtt_ms", "uplink_ms", "edge_exec_ms", "local_exec_ms",
            "total_latency_ms", "local_power_mw", "offload_power_mw", "success", "timestamp"
        ]
        
        self.telemetry_file.parent.mkdir(parents=True, exist_ok=True)
        self.write_header = not self.telemetry_file.exists()
        
        self.start_time = datetime.now().isoformat()
        
    def log_cycle(self, cycle_id, node_id, decision, threshold, score,
                  rtt_ms, uplink_ms, edge_exec_ms, local_exec_ms,
                  total_latency_ms, local_power_mw, offload_power_mw, success=True):
        with open(self.telemetry_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if self.write_header:
                writer.writeheader()
                self.write_header = False
            writer.writerow({
                "cycle_id": cycle_id,
                "node_id": node_id,
                "model": self.model,
                "policy": self.policy,
                "decision": decision,
                "threshold": threshold,
                "decision_score": score,
                "rtt_ms": rtt_ms,
                "uplink_ms": uplink_ms,
                "edge_exec_ms": edge_exec_ms,
                "local_exec_ms": local_exec_ms,
                "total_latency_ms": total_latency_ms,
                "local_power_mw": local_power_mw,
                "offload_power_mw": offload_power_mw,
                "success": int(success),
                "timestamp": datetime.now().isoformat()
            })
    
    def save_summary(self, nodes, aps, ovs_switches, 
                     accuracy, avg_latency, avg_power, avg_throughput):
        summary = {
            "run_id": self.run_id,
            "model": self.model,
            "policy": self.policy,
            "nodes": nodes,
            "aps": aps,
            "ovs_switches": ovs_switches,
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency,
            "avg_power_mw": avg_power,
            "avg_throughput_fps": avg_throughput,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat()
        }
        with open(self.meta_file, "w") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    # Test
    logger = TelemetryLogger("test_run", "snn")
    logger.log_cycle(
        cycle_id=1, node_id=1, decision="local", threshold=0.7, score=0.85,
        rtt_ms=12.3, uplink_ms=5.2, edge_exec_ms=8.1, local_exec_ms=10.5,
        total_latency_ms=15.6, local_power_mw=320.0, offload_power_mw=450.0
    )
    logger.save_summary(nodes=10, aps=2, ovs_switches=1,
                       accuracy=90.3, avg_latency=15.6, 
                       avg_power=320.0, avg_throughput=64.1)
    print(f"✓ Test logged to {logger.telemetry_file}")
    print(f"✓ Summary saved to {logger.meta_file}")
