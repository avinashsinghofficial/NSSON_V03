#!/usr/bin/env python3
"""
Conventional SDN-IoT baseline for NSSON_V02.

This is intentionally a non-NSSON controller:
- Best-effort MAC-learning forwarding only.
- No inference confidence.
- No dynamic threshold control.
- No SLA enforcement.
- No network slicing.
- No adaptive routing.

It provides a reproducible conventional SDN baseline against which
the QoS-aware NSSON controller can be evaluated.
"""

import json
import os
import time

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib.packet import ethernet, packet
from ryu.ofproto import ofproto_v1_3

ROOT = "/home/avi/NSSON_V02"
LOG_DIR = os.path.join(ROOT, "logs")
FLOW_LOG = os.path.join(LOG_DIR, "ryuflowlog.jsonl")


class BaselineSdnIotController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(BaselineSdnIotController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        os.makedirs(LOG_DIR, exist_ok=True)
        self.logger.info("Baseline SDN-IoT controller started")

    def write_flow_log(self, record):
        record.update({
            "timestamp": time.time(),
            "controller": "baseline_sdn_iot",
            "policy": "best_effort_static",
            "slice": "best_effort",
            "path_id": "static_path",
            "selected": 1,
        })

        with open(FLOW_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        match = parser.OFPMatch()
        actions = [
            parser.OFPActionOutput(
                ofproto.OFPP_CONTROLLER,
                ofproto.OFPCML_NO_BUFFER
            )
        ]
        instructions = [
            parser.OFPInstructionActions(
                ofproto.OFPIT_APPLY_ACTIONS,
                actions
            )
        ]

        flow_mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=0,
            match=match,
            instructions=instructions
        )
        datapath.send_msg(flow_mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth is None or eth.ethertype == 0x88CC:
            return

        dpid = datapath.id
        src = eth.src
        dst = eth.dst

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(
                in_port=in_port,
                eth_src=src,
                eth_dst=dst
            )
            instructions = [
                parser.OFPInstructionActions(
                    ofproto.OFPIT_APPLY_ACTIONS,
                    actions
                )
            ]

            flow_mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=10,
                match=match,
                instructions=instructions,
                idle_timeout=20,
                hard_timeout=60
            )
            datapath.send_msg(flow_mod)

            self.write_flow_log({
                "datapath_id": int(dpid),
                "in_port": int(in_port),
                "out_port": int(out_port),
                "src_mac": src,
                "dst_mac": dst,
                "priority": 10,
                "decision": "not_inference_aware"
            })

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None

        packet_out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )
        datapath.send_msg(packet_out)
