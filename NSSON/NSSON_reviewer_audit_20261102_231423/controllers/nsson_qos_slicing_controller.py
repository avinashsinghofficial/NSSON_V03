#!/usr/bin/env python3

import csv
import os
import time
from pathlib import Path

import yaml

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER
from ryu.controller.handler import MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib.packet import ethernet
from ryu.lib.packet import ipv4
from ryu.lib.packet import packet
from ryu.lib.packet import tcp
from ryu.lib.packet import udp
from ryu.ofproto import ofproto_v1_3


NSSON_ROOT = "/home/avi/NSSON_V02"
CONFIG_FILE = f"{NSSON_ROOT}/configs/qos_slicing_config.yaml"
FLOW_LOG = f"{NSSON_ROOT}/logs/qos_slicing/ryu_slice_flow_log.csv"


class NssonQosSlicingController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(NssonQosSlicingController, self).__init__(*args, **kwargs)

        with open(CONFIG_FILE, "r") as config_file:
            self.config = yaml.safe_load(config_file)

        self.mac_to_port = {}

        self.slice_by_port = {
            int(slice_config["traffic_port"]): slice_name
            for slice_name, slice_config in self.config["slices"].items()
        }

        Path(FLOW_LOG).parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "NSSON QoS slicing controller started with %d slice port mappings",
            len(self.slice_by_port),
        )

    def classify_slice(self, tcp_packet, udp_packet):
        if tcp_packet and tcp_packet.dst_port in self.slice_by_port:
            return self.slice_by_port[tcp_packet.dst_port]

        if udp_packet and udp_packet.dst_port in self.slice_by_port:
            return self.slice_by_port[udp_packet.dst_port]

        return "background"

    def write_flow_log(self, record):
        needs_header = not os.path.exists(FLOW_LOG)

        with open(FLOW_LOG, "a", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=record.keys(),
            )

            if needs_header:
                writer.writeheader()

            writer.writerow(record)

    def add_flow(self, datapath, priority, match, actions, idle_timeout=20):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        instructions = [
            parser.OFPInstructionActions(
                ofproto.OFPIT_APPLY_ACTIONS,
                actions,
            )
        ]

        flow_modification = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=instructions,
            idle_timeout=idle_timeout,
        )

        datapath.send_msg(flow_modification)

    @set_ev_cls(
        ofp_event.EventOFPSwitchFeatures,
        CONFIG_DISPATCHER,
    )
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

    @set_ev_cls(
        ofp_event.EventOFPPacketIn,
        MAIN_DISPATCHER,
    )
    def packet_in_handler(self, event):
        message = event.msg
        datapath = message.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        input_port = message.match["in_port"]

        packet_data = packet.Packet(message.data)
        ethernet_packet = packet_data.get_protocol(ethernet.ethernet)
        ip_packet = packet_data.get_protocol(ipv4.ipv4)
        tcp_packet = packet_data.get_protocol(tcp.tcp)
        udp_packet = packet_data.get_protocol(udp.udp)

        if ethernet_packet is None:
            return

        datapath_id = datapath.id
        source_mac = ethernet_packet.src
        destination_mac = ethernet_packet.dst

        self.mac_to_port.setdefault(datapath_id, {})
        self.mac_to_port[datapath_id][source_mac] = input_port

        output_port = self.mac_to_port[datapath_id].get(
            destination_mac,
            ofproto.OFPP_FLOOD,
        )

        if ip_packet is None:
            actions = [parser.OFPActionOutput(output_port)]

            packet_out = parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=message.buffer_id,
                in_port=input_port,
                actions=actions,
                data=message.data,
            )

            datapath.send_msg(packet_out)
            return

        slice_name = self.classify_slice(tcp_packet, udp_packet)
        slice_config = self.config["slices"][slice_name]

        flow_priority = int(slice_config["priority"])
        queue_id = int(slice_config["queue_id"])

        actions = [
            parser.OFPActionSetQueue(queue_id),
            parser.OFPActionOutput(output_port),
        ]

        match_fields = {
            "in_port": input_port,
            "eth_type": 0x0800,
            "ipv4_src": ip_packet.src,
            "ipv4_dst": ip_packet.dst,
        }

        if tcp_packet:
            match_fields["ip_proto"] = 6
            match_fields["tcp_dst"] = tcp_packet.dst_port

        if udp_packet:
            match_fields["ip_proto"] = 17
            match_fields["udp_dst"] = udp_packet.dst_port

        match = parser.OFPMatch(**match_fields)

        self.add_flow(
            datapath=datapath,
            priority=flow_priority,
            match=match,
            actions=actions,
        )

        self.write_flow_log(
            {
                "timestamp_epoch": time.time(),
                "datapath_id": datapath_id,
                "slice": slice_name,
                "priority": flow_priority,
                "queue_id": queue_id,
                "source_ip": ip_packet.src,
                "destination_ip": ip_packet.dst,
                "tcp_destination_port": (
                    tcp_packet.dst_port if tcp_packet else ""
                ),
                "udp_destination_port": (
                    udp_packet.dst_port if udp_packet else ""
                ),
                "input_port": input_port,
                "output_port": output_port,
            }
        )

        packet_payload = None
        if message.buffer_id == ofproto.OFP_NO_BUFFER:
            packet_payload = message.data

        packet_out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=message.buffer_id,
            in_port=input_port,
            actions=actions,
            data=packet_payload,
        )

        datapath.send_msg(packet_out)
