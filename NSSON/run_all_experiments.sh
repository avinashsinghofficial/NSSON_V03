#!/usr/bin/env bash
#
# NSSON_V02 full-fetch / paper-aligned local experiment pipeline.
#
# This script deliberately does NOT use:
# - a virtual machine,
# - Flask camera demo workers,
# - apps/edge_worker.py,
# - apps/remote_worker.py,
# - ports 8090 or 8091.
#
# Everything runs locally under /home/avi/NSSON_V02 using:
#   offloadengine.py
#   latencytracker.py
#   powermodel.py
#   feedbackloop.py
#   simrunner.py
#
# Modes:
#   local_only
#   offload_only
#   adaptive_no_sdn
#   baseline_sdn_iot
#   nsson_full
#
# SDN controller processes are optional. The script will use them only if:
#   .venv_ryu39 exists,
#   controllers/baseline_sdn_iot_controller.py exists,
#   controllers/qos_nsson_controller.py exists.
#
# The simulation still runs even if Ryu is unavailable, but it labels those
# runs as simulation-level SDN scenarios. Do not claim physical OpenFlow/
# Mininet-WiFi results unless controller/link telemetry was actually recorded.
#

set -Eeuo pipefail

ROOT="/home/avi/NSSON_V02"
GPU_VENV="${ROOT}/.venv_nsson_gpu"
RYU_VENV="${ROOT}/.venv_ryu39"

CYCLES="${CYCLES:-300}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT}/results/run_${RUN_ID}"

RYU_PID=""
RYU_AVAILABLE=0

export PYTHONPATH="${ROOT}"
export MPLBACKEND=Agg
export EVENTLET_NO_GREENDNS=yes

log() {
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] $*"
    echo "================================================================"
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || die "Required file not found: $1"
}

copy_if_exists() {
    local source="$1"
    local destination="$2"

    if [[ -f "${source}" ]]; then
        cp -f "${source}" "${destination}"
        echo "[OK] Saved $(basename "${destination}")"
    fi
}

stop_ryu_controller() {
    if [[ -n "${RYU_PID}" ]] && kill -0 "${RYU_PID}" 2>/dev/null; then
        kill "${RYU_PID}" 2>/dev/null || true
        wait "${RYU_PID}" 2>/dev/null || true
    fi

    RYU_PID=""
    pkill -f "ryu.cmd.manager" 2>/dev/null || true
    sleep 1
}

cleanup() {
    stop_ryu_controller
}

trap cleanup EXIT INT TERM

wait_for_port() {
    local port="$1"
    local label="$2"
    local retries=20
    local i

    for ((i = 1; i <= retries; i++)); do
        if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"; then
            echo "[OK] ${label} listening on port ${port}"
            return 0
        fi
        sleep 1
    done

    return 1
}

clear_runtime_logs() {
    mkdir -p "${ROOT}/logs"

    rm -f \
        "${ROOT}/logs/metrics.csv" \
        "${ROOT}/logs/offloaddecisionlog.json" \
        "${ROOT}/logs/decisionlog.json" \
        "${ROOT}/logs/offloadlatencylog.csv" \
        "${ROOT}/logs/latencylog.csv" \
        "${ROOT}/logs/offloadpowerlog.csv" \
        "${ROOT}/logs/powerlog.csv" \
        "${ROOT}/logs/thresholdlog.csv" \
        "${ROOT}/logs/ryuflowlog.jsonl" \
        "${ROOT}/logs/linkstats.csv"
}

snapshot_mode() {
    local mode="$1"
    local source="${ROOT}/logs"
    local destination="${RUN_DIR}/${mode}"

    mkdir -p "${destination}"

    copy_if_exists "${source}/metrics.csv" \
        "${destination}/metrics.csv"

    copy_if_exists "${source}/offloaddecisionlog.json" \
        "${destination}/decision_log.json"

    copy_if_exists "${source}/decisionlog.json" \
        "${destination}/decision_log.json"

    copy_if_exists "${source}/offloadlatencylog.csv" \
        "${destination}/latency_log.csv"

    copy_if_exists "${source}/latencylog.csv" \
        "${destination}/latency_log.csv"

    copy_if_exists "${source}/offloadpowerlog.csv" \
        "${destination}/power_log.csv"

    copy_if_exists "${source}/powerlog.csv" \
        "${destination}/power_log.csv"

    copy_if_exists "${source}/thresholdlog.csv" \
        "${destination}/threshold_log.csv"

    copy_if_exists "${source}/ryuflowlog.jsonl" \
        "${destination}/ryu_flow_log.jsonl"

    copy_if_exists "${source}/linkstats.csv" \
        "${destination}/linkstats.csv"

    printf '%s\n' "${mode}" > "${destination}/mode.txt"
}

publish_for_analysis() {
    local mode="$1"
    local source="${RUN_DIR}/${mode}"

    mkdir -p \
        "${ROOT}/offload/${mode}" \
        "${ROOT}/controller_runs/${mode}"

    copy_if_exists "${source}/latency_log.csv" \
        "${ROOT}/offload/${mode}/latency_log.csv"

    copy_if_exists "${source}/power_log.csv" \
        "${ROOT}/offload/${mode}/power_log.csv"

    copy_if_exists "${source}/decision_log.json" \
        "${ROOT}/offload/${mode}/decision_log.json"

    copy_if_exists "${source}/metrics.csv" \
        "${ROOT}/offload/${mode}/metrics.csv"

    copy_if_exists "${source}/ryu_flow_log.jsonl" \
        "${ROOT}/controller_runs/${mode}/ryu_flow_log.jsonl"

    copy_if_exists "${source}/linkstats.csv" \
        "${ROOT}/controller_runs/${mode}/linkstats.csv"
}

check_ryu_availability() {
    if [[ -f "${RYU_VENV}/bin/activate" ]] && \
       [[ -f "${ROOT}/controllers/baseline_sdn_iot_controller.py" ]] && \
       [[ -f "${ROOT}/controllers/qos_nsson_controller.py" ]]; then
        RYU_AVAILABLE=1
        echo "[INFO] Ryu controllers detected. Controller processes will be used."
    else
        RYU_AVAILABLE=0
        echo "[WARN] Ryu controller environment or controller files unavailable."
        echo "[WARN] baseline_sdn_iot and nsson_full will still run at simulation level."
        echo "[WARN] Do not report actual SDN routing/PDR/SLA claims without real controller telemetry."
    fi
}

start_ryu_controller() {
    local controller="$1"
    local label="$2"
    local logfile="$3"

    if [[ "${RYU_AVAILABLE}" -ne 1 ]]; then
        return 0
    fi

    stop_ryu_controller

    log "Starting ${label}"

    (
        cd "${ROOT}"
        source "${RYU_VENV}/bin/activate"
        export PYTHONPATH="${ROOT}"
        export EVENTLET_NO_GREENDNS=yes

        exec python -m ryu.cmd.manager \
            "${controller}" \
            --wsapi-port 8080
    ) > "${logfile}" 2>&1 &

    RYU_PID="$!"

    if ! wait_for_port 8080 "${label}"; then
        echo "[WARN] ${label} did not open port 8080."
        echo "[WARN] Continuing with a simulation-only run."
        stop_ryu_controller
        RYU_AVAILABLE=0
    fi
}

run_mode() {
    local mode="$1"

    log "Running mode: ${mode}"
    clear_runtime_logs

    (
        cd "${ROOT}"
        source "${GPU_VENV}/bin/activate"
        export PYTHONPATH="${ROOT}"
        export MPLBACKEND=Agg

        python "${ROOT}/simrunner.py" \
            --mode "${mode}" \
            --cycles "${CYCLES}" \
            --output-dir "${ROOT}/logs"
    ) 2>&1 | tee "${RUN_DIR}/${mode}/simrunner.log"

    snapshot_mode "${mode}"
    publish_for_analysis "${mode}"
}

validate_project() {
    require_file "${GPU_VENV}/bin/activate"
    require_file "${ROOT}/simrunner.py"
    require_file "${ROOT}/offloadengine.py"
    require_file "${ROOT}/latencytracker.py"
    require_file "${ROOT}/powermodel.py"
    require_file "${ROOT}/feedbackloop.py"
    require_file "${ROOT}/generate_qos_data.py"
    require_file "${ROOT}/nsson_qos_plots.py"
    require_file "${ROOT}/configs/qos_config.yaml"
}

validate_per_mode_outputs() {
    local mode="$1"
    local mode_dir="${RUN_DIR}/${mode}"

    require_file "${mode_dir}/decision_log.json"
    require_file "${mode_dir}/latency_log.csv"
    require_file "${mode_dir}/power_log.csv"
}

main() {
    cd "${ROOT}"
    validate_project

    mkdir -p \
        "${RUN_DIR}" \
        "${ROOT}/logs" \
        "${ROOT}/results" \
        "${ROOT}/offload" \
        "${ROOT}/controller_runs" \
        "${ROOT}/plots_qos"

    for mode in \
        local_only \
        offload_only \
        adaptive_no_sdn \
        baseline_sdn_iot \
        nsson_full
    do
        mkdir -p "${RUN_DIR}/${mode}"
    done

    log "NSSON_V02 full-fetch experiment suite"
    echo "Project root : ${ROOT}"
    echo "Run directory: ${RUN_DIR}"
    echo "Cycles/mode  : ${CYCLES}"
    echo "Execution    : one local working directory; no VM; no Flask demo."

    check_ryu_availability

    # Three non-SDN scenarios.
    run_mode "local_only"
    validate_per_mode_outputs "local_only"

    run_mode "offload_only"
    validate_per_mode_outputs "offload_only"

    run_mode "adaptive_no_sdn"
    validate_per_mode_outputs "adaptive_no_sdn"

    # Conventional SDN-IoT baseline.
    start_ryu_controller \
        "${ROOT}/controllers/baseline_sdn_iot_controller.py" \
        "Baseline SDN-IoT controller" \
        "${RUN_DIR}/baseline_sdn_iot/ryu_controller.log"

    run_mode "baseline_sdn_iot"
    validate_per_mode_outputs "baseline_sdn_iot"
    stop_ryu_controller

    # Proposed QoS-aware NSSON mode.
    start_ryu_controller \
        "${ROOT}/controllers/qos_nsson_controller.py" \
        "NSSON QoS-aware controller" \
        "${RUN_DIR}/nsson_full/ryu_controller.log"

    run_mode "nsson_full"
    validate_per_mode_outputs "nsson_full"
    stop_ryu_controller

    log "Generating unified analysis CSV files"

    (
        cd "${ROOT}"
        source "${GPU_VENV}/bin/activate"
        export PYTHONPATH="${ROOT}"

        python "${ROOT}/generate_qos_data.py"
    ) 2>&1 | tee "${RUN_DIR}/generate_qos_data.log"

    log "Generating plots"

    (
        cd "${ROOT}"
        source "${GPU_VENV}/bin/activate"
        export PYTHONPATH="${ROOT}"
        export MPLBACKEND=Agg

        python "${ROOT}/nsson_qos_plots.py"
    ) 2>&1 | tee "${RUN_DIR}/nsson_qos_plots.log"

    mkdir -p "${RUN_DIR}/final_csv" "${RUN_DIR}/final_plots"

    copy_if_exists "${ROOT}/qos_metrics.csv" \
        "${RUN_DIR}/final_csv/qos_metrics.csv"

    copy_if_exists "${ROOT}/routing_paths.csv" \
        "${RUN_DIR}/final_csv/routing_paths.csv"

    copy_if_exists "${ROOT}/offload_feedback.csv" \
        "${RUN_DIR}/final_csv/offload_feedback.csv"

    cp -af "${ROOT}/plots_qos/." "${RUN_DIR}/final_plots/" 2>/dev/null || true

    log "NSSON_V02 experiment suite completed"

    echo "All results are saved in:"
    echo "  ${RUN_DIR}"

    echo
    echo "Final CSV outputs:"
    find "${RUN_DIR}/final_csv" -maxdepth 1 -type f -printf "  %f\n" 2>/dev/null | sort || true

    echo
    echo "Final plot outputs:"
    find "${RUN_DIR}/final_plots" \
        -maxdepth 1 \
        -type f \
        \( -name "*.png" -o -name "*.jpg" -o -name "*.pdf" \) \
        -printf "  %f\n" \
        2>/dev/null | sort || true
}

main "$@"
