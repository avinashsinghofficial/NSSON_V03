#!/usr/bin/env python3
"""
NSSON multi-slice SLA-aware Ryu controller.

Corrected behavior:
  * TCP/8090 and UDP/9090 -> autonomous -> priority 400 -> queue 1
  * TCP/8091 and UDP/9091 -> healthcare -> priority 300 -> queue 2
  * TCP/8092 and UDP/9092 -> industrial -> priority 200 -> queue 3
  * UDP/5001              -> background -> priority 10 -> queue 0

The controller classifies both forward and reverse packets using both source and
destination ports. This is required because an iperf client request uses the
well-known service/probe port as destination port, whereas server responses use
the same service/probe port as source port.
"""

import json
import os
import time

import yaml

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib.packet import ethernet, ipv4, packet, tcp, udp
from ryu.ofproto import ofproto_v1_3


ROOT = "/home/avi/NSSON_V02"
CONFIG_FILE = os.path.join(ROOT, "configs", "qos_multislice_sla.yaml")
LOG_DIR = os.path.join(ROOT, "logs", "qos_multislice_raw")
FLOW_LOG = os.path.join(LOG_DIR, "ryu_multislice_flows.jsonl")


class NSSONMultiSliceSLAController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(NSSONMultiSliceSLAController, self).__init__(*args, **kwargs)

        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        self.slices = self.config["slices"]
        self.mac_to_port = {}
        self.flow_counter = 0

        os.makedirs(LOG_DIR, exist_ok=True)

        self.tcp_port_to_slice = {}
        self.udp_port_to_slice = {}

        for slice_name, slice_cfg in self.slices.items():
            protocol = str(slice_cfg.get("protocol", "")).lower()

            if protocol == "tcp":
                self.tcp_port_to_slice[int(slice_cfg["port"])] = slice_name

                probe_port = slice_cfg.get("udp_probe_port")
                if probe_port is not None:
                    self.udp_port_to_slice[int(probe_port)] = slice_name

            elif protocol == "udp":
                self.udp_port_to_slice[int(slice_cfg["port"])] = slice_name

        self.logger.info(
            "NSSON multi-slice SLA controller started; TCP map=%s; UDP map=%s",
            self.tcp_port_to_slice,
            self.udp_port_to_slice,
        )

    def slice_record(self, slice_name, protocol, src_port, dst_port, matched_port):
        slice_cfg = self.slices[slice_name]

        return {
            "slice": slice_name,
            "traffic_class": (
                "inference_slice"
                if slice_name != "background"
                else "best_effort"
            ),
            "priority": int(slice_cfg["priority"]),
            "queue_id": int(slice_cfg["queue_id"]),
            "protocol": protocol,
            "src_port": src_port,
            "dst_port": dst_port,
            "matched_port": matched_port,
        }

    @staticmethod
    def _lookup_slice(port_map, src_port, dst_port):
        """
        Return the slice associated with a known destination port first.
        For reverse traffic, return the slice associated with the source port.
        """
        if dst_port in port_map:
            return port_map[dst_port], dst_port

        if src_port in port_map:
            return port_map[src_port], src_port

        return "background", None

    def classify_flow(self, pkt):
        tcp_packet = pkt.get_protocol(tcp.tcp)
        udp_packet = pkt.get_protocol(udp.udp)

        if tcp_packet is not None:
            src_port = int(tcp_packet.src_port)
            dst_port = int(tcp_packet.dst_port)

            slice_name, matched_port = self._lookup_slice(
                self.tcp_port_to_slice,
                src_port,
                dst_port,
            )

            return self.slice_record(
                slice_name=slice_name,
                protocol=6,
                src_port=src_port,
                dst_port=dst_port,
                matched_port=matched_port,
            )

        if udp_packet is not None:
            src_port = int(udp_packet.src_port)
            dst_port = int(udp_packet.dst_port)

            slice_name, matched_port = self._lookup_slice(
                self.udp_port_to_slice,
                src_port,
                dst_port,
            )

            return self.slice_record(
                slice_name=slice_name,
                protocol=17,
                src_port=src_port,
                dst_port=dst_port,
                matched_port=matched_port,
            )

        return self.slice_record(
            slice_name="background",
            protocol=None,
            src_port=None,
            dst_port=None,
            matched_port=None,
        )

    def add_flow(self, datapath, priority, match, actions, idle_timeout=30):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        instructions = [
            parser.OFPInstructionActions(
                ofproto.OFPIT_APPLY_ACTIONS,
                actions,
            )
        ]

        flow_mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=instructions,
            idle_timeout=idle_timeout,
        )

        datapath.send_msg(flow_mod)

    def write_flow_log(self, record):
        self.flow_counter += 1

        record.update(
            {
                "timestamp_epoch": time.time(),
                "flow_id": self.flow_counter,
                "controller": "nsson_multislice_sla",
                "policy": "bidirectional_per_slice_queue_sla_udp_probe",
                "path_id": "single_path",
                "selected": 1,
            }
        )

        with open(FLOW_LOG, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, event):
        datapath = event.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [
            parser.OFPActionOutput(
                ofproto.OFPP_CONTROLLER,
                ofproto.OFPCML_NO_BUFFER,
            )
        ]

        self.add_flow(
            datapath=datapath,
            priority=0,
            match=match,
            actions=actions,
        )

        self.logger.info(
            "Installed table-miss rule on datapath=%s",
            datapath.id,
        )

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, event):
        message = event.msg
        datapath = message.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        in_port = message.match["in_port"]

        pkt = packet.Packet(message.data)
        eth_packet = pkt.get_protocol(ethernet.ethernet)
        ip_packet = pkt.get_protocol(ipv4.ipv4)

        if eth_packet is None:
            return

        dpid = datapath.id
        src_mac = eth_packet.src
        dst_mac = eth_packet.dst

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src_mac] = in_port

        out_port = self.mac_to_port[dpid].get(
            dst_mac,
            ofproto.OFPP_FLOOD,
        )

        classification = self.classify_flow(pkt)

        self.logger.info(
            "PacketIn dpid=%s in=%s out=%s proto=%s src_port=%s dst_port=%s "
            "matched_port=%s slice=%s queue=%s priority=%s",
            dpid,
            in_port,
            out_port,
            classification["protocol"],
            classification["src_port"],
            classification["dst_port"],
            classification["matched_port"],
            classification["slice"],
            classification["queue_id"],
            classification["priority"],
        )

        actions = []

        # Queue assignment is only meaningful for a resolved output port.
        # Flooded packets are forwarded without queue binding so ARP/MAC learning
        # can complete normally.
        if out_port != ofproto.OFPP_FLOOD:
            actions.append(
                parser.OFPActionSetQueue(classification["queue_id"])
            )

        actions.append(parser.OFPActionOutput(out_port))

        if ip_packet is not None and out_port != ofproto.OFPP_FLOOD:
            match_fields = {
                "in_port": in_port,
                "eth_type": 0x0800,
                "ipv4_src": ip_packet.src,
                "ipv4_dst": ip_packet.dst,
                "ip_proto": classification["protocol"],
            }

            # Match the complete observed transport direction. This prevents the
            # first slice's UDP rule from accidentally matching other UDP slices.
            if classification["protocol"] == 6:
                match_fields.update(
                    {
                        "tcp_src": classification["src_port"],
                        "tcp_dst": classification["dst_port"],
                    }
                )

            elif classification["protocol"] == 17:
                match_fields.update(
                    {
                        "udp_src": classification["src_port"],
                        "udp_dst": classification["dst_port"],
                    }
                )

            match = parser.OFPMatch(**match_fields)

            self.add_flow(
                datapath=datapath,
                priority=classification["priority"],
                match=match,
                actions=actions,
            )

            self.logger.info(
                "Installed flow dpid=%s slice=%s priority=%s queue=%s "
                "match=%s out=%s",
                dpid,
                classification["slice"],
                classification["priority"],
                classification["queue_id"],
                match_fields,
                out_port,
            )

        self.write_flow_log(
            {
                "datapath_id": dpid,
                "in_port": in_port,
                "out_port": out_port,
                "src_mac": src_mac,
                "dst_mac": dst_mac,
                "src_ip": ip_packet.src if ip_packet else None,
                "dst_ip": ip_packet.dst if ip_packet else None,
                **classification,
            }
        )

        data = None
        if message.buffer_id == ofproto.OFP_NO_BUFFER:
            data = message.data

        packet_out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=message.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )

        datapath.send_msg(packet_out)


if __name__ == "__main__":
    pass
