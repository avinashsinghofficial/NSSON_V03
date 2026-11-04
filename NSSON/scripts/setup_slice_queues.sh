#!/usr/bin/env bash
set -euo pipefail

SWITCH_NAME="${1:-s1}"
PORT_NAME="${2:-s1-eth1}"
LINK_RATE_BPS="${3:-25000000}"

echo "================================================"
echo "Configuring OVS queues"
echo "Switch        : ${SWITCH_NAME}"
echo "Port          : ${PORT_NAME}"
echo "Link rate     : ${LINK_RATE_BPS} bit/s"
echo "================================================"

sudo ovs-vsctl -- \
  --id=@q0 create Queue \
    other-config:min-rate=1000000 \
    other-config:max-rate=3000000 \
  --id=@q1 create Queue \
    other-config:min-rate=8000000 \
    other-config:max-rate=15000000 \
  --id=@q2 create Queue \
    other-config:min-rate=6000000 \
    other-config:max-rate=10000000 \
  --id=@q3 create Queue \
    other-config:min-rate=3000000 \
    other-config:max-rate=6000000 \
  --id=@qos create QoS \
    type=linux-htb \
    other-config:max-rate="${LINK_RATE_BPS}" \
    queues:0=@q0 \
    queues:1=@q1 \
    queues:2=@q2 \
    queues:3=@q3 \
  set Port "${PORT_NAME}" qos=@qos

echo "Configured queues:"
echo "  Queue 0 -> background best-effort"
echo "  Queue 1 -> autonomous slice"
echo "  Queue 2 -> healthcare slice"
echo "  Queue 3 -> industrial slice"

echo
echo "Current QoS configuration:"
sudo ovs-vsctl list qos

echo
echo "Current queue configuration:"
sudo ovs-vsctl list queue
