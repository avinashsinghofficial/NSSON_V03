#!/usr/bin/env bash
set -euo pipefail

NSSON_ROOT="/home/avi/NSSON_V02"

CONFIG_FILE="${NSSON_ROOT}/configs/qos_slicing_config.yaml"
PROBE_SCRIPT="${NSSON_ROOT}/experiments/qos_slicing/qos_probe.py"
ROUTING_SCRIPT="${NSSON_ROOT}/experiments/qos_slicing/adaptive_routing.py"

RESULTS_DIR="${NSSON_ROOT}/results/qos_slicing"
LOGS_DIR="${NSSON_ROOT}/logs/qos_slicing"

METRICS_CSV="${RESULTS_DIR}/qos_slice_metrics.csv"
PATH_METRICS_CSV="${RESULTS_DIR}/path_metrics.csv"
REROUTE_EVENTS_CSV="${LOGS_DIR}/reroute_events.csv"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

rm -f "${METRICS_CSV}" "${PATH_METRICS_CSV}" "${REROUTE_EVENTS_CSV}"

# -------------------------------------------------------------------
# IMPORTANT: Modify these values if your Mininet-WiFi topology differs.
# The command passed to --src-host must be valid in the terminal where
# this script runs. If h1/h2 are Mininet CLI names, run this experiment
# from the Mininet CLI or replace h1 with the proper namespace command.
# -------------------------------------------------------------------
SRC_HOST="h1"
DST_IP="10.0.0.2"
LINK_CAPACITY_MBPS="25"

declare -A PORTS=(
  [autonomous]=8090
  [healthcare]=8091
  [industrial]=8092
)

declare -A PATH_A_SNR=(
  [light]=30
  [moderate]=22
  [heavy]=12
)

declare -A PATH_B_SNR=(
  [light]=24
  [moderate]=21
  [heavy]=18
)

declare -A BACKGROUND_RATE=(
  [light]=2M
  [moderate]=12M
  [heavy]=19M
)

echo "================================================"
echo "NSSON QoS slicing experiment"
echo "Configuration: ${CONFIG_FILE}"
echo "Results      : ${RESULTS_DIR}"
echo "================================================"

for MODE in baseline priority slicing adaptive; do
  echo
  echo "################################################"
  echo "MODE: ${MODE}"
  echo "################################################"

  for LOAD in light moderate heavy; do
    echo
    echo "------------------------------------------------"
    echo "MODE=${MODE}, LOAD=${LOAD}, BACKGROUND=${BACKGROUND_RATE[${LOAD}]}"
    echo "------------------------------------------------"

    # ---------------------------------------------------------------
    # Start real background traffic before or during each load case.
    #
    # Example command inside Mininet CLI, assuming h3 is background
    # source and h2 is the edge server:
    #
    # h3 iperf3 -c 10.0.0.2 -u -b ${BACKGROUND_RATE[${LOAD}]} -t 240 &
    #
    # Use your actual topology host names/IPs. Do not fabricate loads.
    # ---------------------------------------------------------------

    for CYCLE in $(seq 1 10); do
      echo "[INFO] MODE=${MODE}, LOAD=${LOAD}, CYCLE=${CYCLE}"

      for SLICE_NAME in autonomous healthcare industrial; do
        PORT="${PORTS[${SLICE_NAME}]}"
        CURRENT_PATH="path_A"
        CURRENT_SNR="${PATH_A_SNR[${LOAD}]}"

        python "${PROBE_SCRIPT}" \
          --config "${CONFIG_FILE}" \
          --mode "${MODE}" \
          --load "${LOAD}" \
          --slice "${SLICE_NAME}" \
          --cycle "${CYCLE}" \
          --src-host "${SRC_HOST}" \
          --dst-ip "${DST_IP}" \
          --dst-port "${PORT}" \
          --path-id "${CURRENT_PATH}" \
          --snr-db "${CURRENT_SNR}" \
          --link-capacity-mbps "${LINK_CAPACITY_MBPS}" \
          --csv "${METRICS_CSV}" \
          --ping-count 20 \
          --iperf-seconds 5

        # Copy all collected path telemetry. For a true two-path setup,
        # run a probe for path_A and path_B through physically/logically
        # distinct routes, then retain both records in path_metrics.csv.
        cp "${METRICS_CSV}" "${PATH_METRICS_CSV}"

        if [[ "${MODE}" == "adaptive" ]]; then
          ROUTE_JSON="${LOGS_DIR}/route_${MODE}_${LOAD}_${SLICE_NAME}_${CYCLE}.json"

          python "${ROUTING_SCRIPT}" \
            --config "${CONFIG_FILE}" \
            --slice "${SLICE_NAME}" \
            --path-metrics-csv "${PATH_METRICS_CSV}" \
            --current-path "${CURRENT_PATH}" \
            --output-json "${ROUTE_JSON}"

          python - "${ROUTE_JSON}" "${REROUTE_EVENTS_CSV}" "${MODE}" "${LOAD}" "${CYCLE}" <<'PY'
import csv
import json
import sys
from pathlib import Path

route_json, output_csv, mode, load, cycle = sys.argv[1:]

decision = json.loads(Path(route_json).read_text())

record = {
    "cycle": cycle,
    "mode": mode,
    "load": load,
    "slice": decision["slice"],
    "current_path": decision["current_path"],
    "selected_path": decision["selected_path"],
    "reroute": decision["reroute"],
    "reason": decision["reason"],
    "selected_score": decision["selected_score"],
}

path = Path(output_csv)
new_file = not path.exists()

with path.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=record.keys())
    if new_file:
        writer.writeheader()
    writer.writerow(record)
PY
        fi
      done
    done
  done
done

echo
echo "================================================"
echo "Experiment completed."
echo "QoS telemetry: ${METRICS_CSV}"
echo "Path telemetry: ${PATH_METRICS_CSV}"
echo "Reroute events: ${REROUTE_EVENTS_CSV}"
echo "================================================"
