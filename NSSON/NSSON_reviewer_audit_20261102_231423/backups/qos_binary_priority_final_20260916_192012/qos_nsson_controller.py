#!/usr/bin/env python3
"""
NSSON QoS-aware Ryu controller for NSSON_V02.

Implemented and logged behaviour:
  * OpenFlow 1.3 learning-switch forwarding.
  * Inference-aware traffic classification using TCP/UDP ports 8090/8091.
  * Offload inference traffic: priority 100, slice=control.
  * Ordinary/background traffic: priority 10, slice=monitoring.
  * JSONL audit log for each learned unicast flow.

Scope note:
  The current topology has one OVS switch and one physical path. Therefore
  path_id is logged as "single_path"; this controller does not claim
  actual adaptive multi-path routing until a multi-path topology is used.
"""

import json
import os
import time

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib.packet import ethernet, ipv4, packet, tcp, udp
from ryu.ofproto import ofproto_v1_3

ROOT = "/home/avi/NSSON_V02"
LOG_DIR = os.path.join(ROOT, "logs")
FLOW_LOG = os.path.join(LOG_DIR, "ryuflowlog.jsonl")

OFFLOAD_PORTS = {8090, 8091}
PRIORITY_BACKGROUND = 10
PRIORITY_OFFLOAD = 100


class NSSONController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(NSSONController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.flow_counter = 0
        os.makedirs(LOG_DIR, exist_ok=True)
        self.logger.info("NSSON QoS-aware controller started")

    def add_flow(self, datapath, priority, match, actions,
                 idle_timeout=20, hard_timeout=60):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        instructions = [
            parser.OFPInstructionActions(
                ofproto.OFPIT_APPLY_ACTIONS,
                actions
            )
        ]
        flow_mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=instructions,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
        )
        datapath.send_msg(flow_mod)

    def classify_flow(self, pkt):
        """
        Infer controller QoS class from transport port.
        TCP/UDP 8090 and 8091 correspond to offload inference endpoints.
        """
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)

        if tcp_pkt is not None:
            if tcp_pkt.src_port in OFFLOAD_PORTS or tcp_pkt.dst_port in OFFLOAD_PORTS:
                return "control", "offload_inference", PRIORITY_OFFLOAD

        if udp_pkt is not None:
            if udp_pkt.src_port in OFFLOAD_PORTS or udp_pkt.dst_port in OFFLOAD_PORTS:
                return "control", "offload_inference", PRIORITY_OFFLOAD

        return "monitoring", "best_effort", PRIORITY_BACKGROUND

    def write_flow_log(self, record):
        record.update(
            {
                "timestamp": time.time(),
                "controller": "nsson_qos",
                "flow_id": self.flow_counter,
                "policy": "nsson_inference_aware_priority",
                "path_id": "single_path",
                "selected": 1,
            }
        )
        with open(FLOW_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [
            parser.OFPActionOutput(
                ofproto.OFPP_CONTROLLER,
                ofproto.OFPCML_NO_BUFFER
            )
        ]
        self.add_flow(datapath, 0, match, actions)
        self.logger.info("Installed table-miss rule: dpid=%s", datapath.id)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth is None or eth.ethertype == 0x88CC:
            return

        dpid = datapath.id
        src_mac = eth.src
        dst_mac = eth.dst

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src_mac] = in_port

        out_port = self.mac_to_port[dpid].get(dst_mac, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        slice_name, traffic_class, priority = self.classify_flow(pkt)

        if out_port != ofproto.OFPP_FLOOD:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            tcp_pkt = pkt.get_protocol(tcp.tcp)
            udp_pkt = pkt.get_protocol(udp.udp)

            fields = {
                "in_port": in_port,
                "eth_src": src_mac,
                "eth_dst": dst_mac,
            }

            if ip_pkt is not None:
                fields.update(
                    {
                        "eth_type": 0x0800,
                        "ipv4_src": ip_pkt.src,
                        "ipv4_dst": ip_pkt.dst,
                        "ip_proto": ip_pkt.proto,
                    }
                )
                if tcp_pkt is not None:
                    fields.update(
                        {
                            "tcp_src": tcp_pkt.src_port,
                            "tcp_dst": tcp_pkt.dst_port,
                        }
                    )
                elif udp_pkt is not None:
                    fields.update(
                        {
                            "udp_src": udp_pkt.src_port,
                            "udp_dst": udp_pkt.dst_port,
                        }
                    )

            match = parser.OFPMatch(**fields)
            self.add_flow(datapath, priority, match, actions)

            self.flow_counter += 1
            self.write_flow_log(
                {
                    "datapath_id": int(dpid),
                    "in_port": int(in_port),
                    "out_port": int(out_port),
                    "src_mac": src_mac,
                    "dst_mac": dst_mac,
                    "src_ip": ip_pkt.src if ip_pkt else None,
                    "dst_ip": ip_pkt.dst if ip_pkt else None,
                    "protocol": int(ip_pkt.proto) if ip_pkt else None,
                    "src_port": (
                        tcp_pkt.src_port if tcp_pkt is not None else
                        udp_pkt.src_port if udp_pkt is not None else None
                    ),
                    "dst_port": (
                        tcp_pkt.dst_port if tcp_pkt is not None else
                        udp_pkt.dst_port if udp_pkt is not None else None
                    ),
                    "slice": slice_name,
                    "traffic_class": traffic_class,
                    "priority": int(priority),
                    "decision": (
                        "inference_aware_priority"
                        if priority == PRIORITY_OFFLOAD
                        else "background_best_effort"
                    ),
                }
            )

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        packet_out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )
        datapath.send_msg(packet_out)
