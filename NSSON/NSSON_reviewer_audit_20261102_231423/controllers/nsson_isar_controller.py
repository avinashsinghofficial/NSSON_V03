#!/usr/bin/env python3
"""
NSSON-ISAR:
Inference-aware Slicing and Adaptive Routing controller.

Modes:
  baseline
  nsson_priority
  nsson_slice_sla
  nsson_isar
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import yaml

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ethernet, ipv4, packet, tcp, udp
from ryu.ofproto import ofproto_v1_3


ROOT = Path(os.environ.get("NSSON_ROOT", "/home/avi/NSSON_V02"))
CONFIG_PATH = ROOT / "configs" / "nsson_qos_sla.yaml"
LOG_DIR = ROOT / "logs" / "qos_sla"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def now():
    return time.time()


class NssonIsarController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ensure_dir(LOG_DIR)

        self.mode = os.environ.get("NSSON_MODE", "nsson_isar")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        if self.mode not in self.cfg["modes"]:
            raise ValueError(f"Unknown NSSON_MODE={self.mode}")

        self.mode_cfg = self.cfg["modes"][self.mode]
        self.slices = self.cfg["slices"]
        self.routing_cfg = self.cfg["routing"]
        self.period = float(self.cfg["experiment"]["telemetry_period_s"])
        self.window_size = int(self.cfg["experiment"]["window_size"])
        self.persist_n = int(self.cfg["experiment"]["reroute_persistence_windows"])
        self.hysteresis = float(self.cfg["experiment"]["reroute_hysteresis_score"])

        self.datapaths = {}
        self.mac_to_port = defaultdict(dict)
        self.path_metrics = {
            "path_primary": {
                "rtt_ms": 15.0, "jitter_ms": 2.0, "loss_pct": 0.0,
                "snr_db": 25.0, "available_bw_mbps": 30.0
            },
            "path_backup": {
                "rtt_ms": 22.0, "jitter_ms": 3.0, "loss_pct": 0.0,
                "snr_db": 22.0, "available_bw_mbps": 25.0
            },
        }
        self.slice_history = defaultdict(lambda: deque(maxlen=self.window_size))
        self.violation_count = defaultdict(int)
        self.current_path = defaultdict(lambda: "path_primary")
        self.cycle = 0

        self.flow_log = LOG_DIR / f"flow_slice_log_{self.mode}.jsonl"
        self.path_log = LOG_DIR / f"path_telemetry_{self.mode}.csv"
        self.reroute_log = LOG_DIR / f"reroute_events_{self.mode}.csv"
        self.qos_log = LOG_DIR / f"qos_slice_metrics_{self.mode}.csv"

        self._init_logs()
        self.monitor_thread = hub.spawn(self._monitor)
        self.logger.info(
            "NSSON-ISAR started: mode=%s priority=%s slicing=%s sla=%s adaptive=%s",
            self.mode,
            self.mode_cfg["enable_priority"],
            self.mode_cfg["enable_slicing"],
            self.mode_cfg["enable_sla"],
            self.mode_cfg["enable_adaptive_routing"],
        )

    def _init_logs(self):
        if not self.path_log.exists():
            with self.path_log.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "cycle", "mode", "path_id",
                    "rtt_ms", "jitter_ms", "loss_pct", "snr_db",
                    "available_bw_mbps"
                ])
        if not self.reroute_log.exists():
            with self.reroute_log.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "cycle", "mode", "slice",
                    "old_path", "new_path", "reason",
                    "old_score", "new_score"
                ])
        if not self.qos_log.exists():
            with self.qos_log.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "cycle", "mode", "slice", "path_id",
                    "rtt_ms", "jitter_ms", "loss_pct", "snr_db",
                    "available_bw_mbps", "throughput_mbps",
                    "sla_rtt_ok", "sla_jitter_ok", "sla_loss_ok",
                    "sla_bw_ok", "sla_all_ok"
                ])

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        parser = dp.ofproto_parser
        ofp = dp.ofproto

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=0, match=match, instructions=inst)
        dp.send_msg(mod)
        self.logger.info("Installed table-miss on dpid=%s", dp.id)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)

    def _slice_for_flow(self, tcp_hdr, udp_hdr):
        if tcp_hdr:
            port = tcp_hdr.dst_port
        elif udp_hdr:
            port = udp_hdr.dst_port
        else:
            return "background"

        for slice_name, profile in self.slices.items():
            if int(profile["port"]) == int(port):
                return slice_name
        return "background"

    def _priority_for(self, slice_name):
        if not self.mode_cfg["enable_priority"]:
            return 10
        if not self.mode_cfg["enable_slicing"]:
            return 200 if slice_name != "background" else 10
        return int(self.slices[slice_name]["priority"])

    def _queue_for(self, slice_name):
        if not self.mode_cfg["enable_slicing"]:
            return 1 if slice_name != "background" else 0
        return int(self.slices[slice_name]["queue_id"])

    def _score_path(self, slice_name, path_id):
        profile = self.slices[slice_name]
        sla = profile["sla"]
        p = self.path_metrics[path_id]
        w = self.routing_cfg["score_weights"]

        deadline = float(profile["deadline_ms"])
        predicted_e2e = p["rtt_ms"] + max(1.0, 8.0 / max(p["available_bw_mbps"], 0.1))

        rtt_penalty = p["rtt_ms"] / float(sla["rtt_ms_max"])
        jitter_penalty = p["jitter_ms"] / float(sla["jitter_ms_max"])
        loss_penalty = p["loss_pct"] / float(sla["loss_pct_max"])
        bw_penalty = max(
            0.0,
            (float(sla["throughput_mbps_min"]) - p["available_bw_mbps"])
            / float(sla["throughput_mbps_min"])
        )
        snr_penalty = max(
            0.0,
            (float(profile["min_snr_db"]) - p["snr_db"])
            / max(float(profile["min_snr_db"]), 1.0)
        )
        deadline_penalty = max(0.0, (predicted_e2e - deadline) / deadline)

        return (
            w["rtt"] * rtt_penalty
            + w["jitter"] * jitter_penalty
            + w["loss"] * loss_penalty
            + w["bandwidth"] * bw_penalty
            + w["snr"] * snr_penalty
            + w["deadline"] * deadline_penalty
        )

    def _sla_flags(self, slice_name, path_id):
        sla = self.slices[slice_name]["sla"]
        p = self.path_metrics[path_id]

        rtt_ok = int(p["rtt_ms"] <= float(sla["rtt_ms_max"]))
        jitter_ok = int(p["jitter_ms"] <= float(sla["jitter_ms_max"]))
        loss_ok = int(p["loss_pct"] <= float(sla["loss_pct_max"]))
        bw_ok = int(p["available_bw_mbps"] >= float(sla["throughput_mbps_min"]))
        all_ok = int(rtt_ok and jitter_ok and loss_ok and bw_ok)
        return rtt_ok, jitter_ok, loss_ok, bw_ok, all_ok

    def _choose_path(self, slice_name):
        if not self.mode_cfg["enable_adaptive_routing"]:
            return "path_primary"

        scores = {
            path_id: self._score_path(slice_name, path_id)
            for path_id in self.path_metrics
        }
        return min(scores, key=scores.get)

    def _map_path_to_output_port(self, datapath, path_id):
        # IMPORTANT:
        # Change these mappings after `ovs-ofctl -O OpenFlow13 show <switch>`.
        # The topology script provided below defines sw1 output port 2=primary,
        # port 3=backup. Other switches use learning-switch fallback.
        if datapath.id == 1:
            return 2 if path_id == "path_primary" else 3
        return datapath.ofproto.OFPP_FLOOD

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        ip = pkt.get_protocol(ipv4.ipv4)
        tcp_hdr = pkt.get_protocol(tcp.tcp)
        udp_hdr = pkt.get_protocol(udp.udp)

        if eth is None:
            return

        self.mac_to_port[dp.id][eth.src] = in_port
        if ip is None:
            out_port = self.mac_to_port[dp.id].get(eth.dst, ofp.OFPP_FLOOD)
            actions = [parser.OFPActionOutput(out_port)]
            out = parser.OFPPacketOut(
                datapath=dp, buffer_id=msg.buffer_id, in_port=in_port,
                actions=actions, data=msg.data
            )
            dp.send_msg(out)
            return

        slice_name = self._slice_for_flow(tcp_hdr, udp_hdr)
        selected_path = self._choose_path(slice_name)
        priority = self._priority_for(slice_name)
        queue_id = self._queue_for(slice_name)

        if self.mode_cfg["enable_adaptive_routing"] and dp.id == 1 and slice_name != "background":
            out_port = self._map_path_to_output_port(dp, selected_path)
        else:
            out_port = self.mac_to_port[dp.id].get(eth.dst, ofp.OFPP_FLOOD)

        actions = []
        if self.mode_cfg["enable_slicing"] and slice_name != "background":
            actions.append(parser.OFPActionSetQueue(queue_id))
        actions.append(parser.OFPActionOutput(out_port))

        match_fields = {
            "in_port": in_port,
            "eth_type": 0x0800,
            "ipv4_src": ip.src,
            "ipv4_dst": ip.dst,
            "ip_proto": ip.proto,
        }
        if tcp_hdr:
            match_fields["tcp_dst"] = tcp_hdr.dst_port
        if udp_hdr:
            match_fields["udp_dst"] = udp_hdr.dst_port

        match = parser.OFPMatch(**match_fields)
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=10,
            hard_timeout=30,
        )
        dp.send_msg(mod)

        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None,
        )
        dp.send_msg(out)

        record = {
            "timestamp": now(),
            "cycle": self.cycle,
            "mode": self.mode,
            "dpid": dp.id,
            "slice": slice_name,
            "src_ip": ip.src,
            "dst_ip": ip.dst,
            "dst_port": tcp_hdr.dst_port if tcp_hdr else (udp_hdr.dst_port if udp_hdr else None),
            "priority": priority,
            "queue_id": queue_id,
            "selected_path": selected_path,
            "out_port": out_port,
        }
        with self.flow_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _request_port_stats(self, dp):
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        req = parser.OFPPortStatsRequest(dp, 0, ofp.OFPP_ANY)
        dp.send_msg(req)

    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                self._request_port_stats(dp)
            self._emit_telemetry()
            hub.sleep(self.period)

    def _emit_telemetry(self):
        self.cycle += 1

        # Path metrics can be updated from real probe scripts through:
        # logs/qos_sla/live_path_metrics.json
        live_file = LOG_DIR / "live_path_metrics.json"
        if live_file.exists():
            try:
                with live_file.open("r", encoding="utf-8") as f:
                    live = json.load(f)
                for pid, metrics in live.items():
                    if pid in self.path_metrics:
                        self.path_metrics[pid].update(metrics)
            except Exception as exc:
                self.logger.warning("Unable to load live path metrics: %s", exc)

        for path_id, p in self.path_metrics.items():
            with self.path_log.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    now(), self.cycle, self.mode, path_id,
                    p["rtt_ms"], p["jitter_ms"], p["loss_pct"],
                    p["snr_db"], p["available_bw_mbps"]
                ])

        for slice_name in self.slices:
            if slice_name == "background":
                continue

            old_path = self.current_path[slice_name]
            new_path = self._choose_path(slice_name)
            old_score = self._score_path(slice_name, old_path)
            new_score = self._score_path(slice_name, new_path)
            flags = self._sla_flags(slice_name, old_path)
            violation = not bool(flags[-1])

            if violation:
                self.violation_count[slice_name] += 1
            else:
                self.violation_count[slice_name] = 0

            if (
                self.mode_cfg["enable_adaptive_routing"]
                and self.violation_count[slice_name] >= self.persist_n
                and new_path != old_path
                and new_score < old_score - self.hysteresis
            ):
                self.current_path[slice_name] = new_path
                with self.reroute_log.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        now(), self.cycle, self.mode, slice_name,
                        old_path, new_path, "persistent_sla_violation",
                        old_score, new_score
                    ])
                self.logger.info(
                    "REROUTE slice=%s %s->%s old_score=%.3f new_score=%.3f",
                    slice_name, old_path, new_path, old_score, new_score
                )

            selected = self.current_path[slice_name]
            p = self.path_metrics[selected]
            rtt_ok, jitter_ok, loss_ok, bw_ok, all_ok = self._sla_flags(slice_name, selected)
            throughput = min(
                p["available_bw_mbps"],
                max(0.1, p["available_bw_mbps"] * (1.0 - p["loss_pct"] / 100.0))
            )

            self.slice_history[slice_name].append({
                "rtt_ms": p["rtt_ms"],
                "jitter_ms": p["jitter_ms"],
                "loss_pct": p["loss_pct"],
                "throughput_mbps": throughput,
                "sla_all_ok": all_ok,
            })

            with self.qos_log.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    now(), self.cycle, self.mode, slice_name, selected,
                    p["rtt_ms"], p["jitter_ms"], p["loss_pct"], p["snr_db"],
                    p["available_bw_mbps"], throughput,
                    rtt_ok, jitter_ok, loss_ok, bw_ok, all_ok
                ])


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", default=os.environ.get("NSSON_MODE", "nsson_isar"))
    args, _ = parser.parse_known_args()
    os.environ["NSSON_MODE"] = args.mode


if __name__ == "__main__":
    main()
