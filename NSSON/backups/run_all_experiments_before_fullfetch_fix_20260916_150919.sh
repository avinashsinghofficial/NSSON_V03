#!/usr/bin/env bash
#
# NSSON_V02 complete local experiment pipeline.
#
# Everything runs locally under /home/avi/NSSON_V02:
# - local_only
# - offload_only
# - adaptive_no_sdn
# - baseline_sdn_iot
# - nsson_full
#
# Outputs:
#   results/<run_id>/<mode>/
#   offload/<mode>/
#   controller_runs/<mode>/
#   qos_metrics.csv
#   routing_paths.csv
#   offload_feedback.csv
#   plots_qos/
#

set -Eeuo pipefail

ROOT="/home/avi/NSSON_V02"
GPU_VENV="${ROOT}/.venv_nsson_gpu"
RYU_VENV="${ROOT}/.venv_ryu39"

CYCLES="${CYCLES:-300}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT}/results/run_${RUN_ID}"

EDGE_PID=""
REMOTE_PID=""
RYU_PID=""

export PYTHONPATH="${ROOT}"
export EVENTLET_NO_GREENDNS=yes
export MPLBACKEND=Agg

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

stop_pid() {
    local pid="${1:-}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
}

cleanup() {
    log "Stopping local NSSON services"

    stop_pid "${RYU_PID}"
    stop_pid "${EDGE_PID}"
    stop_pid "${REMOTE_PID}"

    pkill -f "apps/edge_worker.py" 2>/dev/null || true
    pkill -f "apps/remote_worker.py" 2>/dev/null || true
    pkill -f "ryu.cmd.manager" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

wait_for_port() {
    local port="$1"
    local label="$2"
    local retries=30
    local i

    for ((i = 1; i <= retries; i++)); do
        if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"; then
            echo "[OK] ${label} listening on port ${port}"
            return 0
        fi
        sleep 1
    done

    die "${label} failed to open port ${port}. Inspect ${RUN_DIR}/services/"
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

copy_if_exists() {
    local source="$1"
    local destination="$2"

    if [[ -f "${source}" ]]; then
        cp -f "${source}" "${destination}"
        echo "[OK] Saved $(basename "${destination}")"
    fi
}

snapshot_mode() {
    local mode="$1"
    local mode_dir="${RUN_DIR}/${mode}"

    mkdir -p "${mode_dir}"

    copy_if_exists "${ROOT}/logs/metrics.csv" \
        "${mode_dir}/metrics.csv"

    copy_if_exists "${ROOT}/logs/offloaddecisionlog.json" \
        "${mode_dir}/decision_log.json"

    copy_if_exists "${ROOT}/logs/decisionlog.json" \
        "${mode_dir}/decision_log.json"

    copy_if_exists "${ROOT}/logs/offloadlatencylog.csv" \
        "${mode_dir}/latency_log.csv"

    copy_if_exists "${ROOT}/logs/latencylog.csv" \
        "${mode_dir}/latency_log.csv"

    copy_if_exists "${ROOT}/logs/offloadpowerlog.csv" \
        "${mode_dir}/power_log.csv"

    copy_if_exists "${ROOT}/logs/powerlog.csv" \
        "${mode_dir}/power_log.csv"

    copy_if_exists "${ROOT}/logs/thresholdlog.csv" \
        "${mode_dir}/threshold_log.csv"

    copy_if_exists "${ROOT}/logs/ryuflowlog.jsonl" \
        "${mode_dir}/ryu_flow_log.jsonl"

    copy_if_exists "${ROOT}/logs/linkstats.csv" \
        "${mode_dir}/linkstats.csv"

    printf '%s\n' "${mode}" > "${mode_dir}/mode.txt"
}

publish_for_analysis() {
    local mode="$1"
    local source_dir="${RUN_DIR}/${mode}"

    mkdir -p \
        "${ROOT}/offload/${mode}" \
        "${ROOT}/controller_runs/${mode}"

    copy_if_exists "${source_dir}/latency_log.csv" \
        "${ROOT}/offload/${mode}/latency_log.csv"

    copy_if_exists "${source_dir}/power_log.csv" \
        "${ROOT}/offload/${mode}/power_log.csv"

    copy_if_exists "${source_dir}/decision_log.json" \
        "${ROOT}/offload/${mode}/decision_log.json"

    copy_if_exists "${source_dir}/metrics.csv" \
        "${ROOT}/offload/${mode}/metrics.csv"

    copy_if_exists "${source_dir}/ryu_flow_log.jsonl" \
        "${ROOT}/controller_runs/${mode}/ryu_flow_log.jsonl"

    copy_if_exists "${source_dir}/linkstats.csv" \
        "${ROOT}/controller_runs/${mode}/linkstats.csv"
}

start_edge_worker() {
    log "Starting local edge / Jetson-emulator worker"

    (
        cd "${ROOT}"
        source "${GPU_VENV}/bin/activate"
        export PYTHONPATH="${ROOT}"

        exec python "${ROOT}/apps/edge_worker.py" \
            --config "${ROOT}/configs/edge_emulator.yaml"
    ) > "${RUN_DIR}/services/edge_worker.log" 2>&1 &

    EDGE_PID="$!"
    wait_for_port 8090 "Edge worker"
}

start_remote_worker() {
    log "Starting local remote-verifier worker"

    (
        cd "${ROOT}"
        source "${GPU_VENV}/bin/activate"
        export PYTHONPATH="${ROOT}"

        exec python "${ROOT}/apps/remote_worker.py" \
            --config "${ROOT}/configs/remote_worker.yaml"
    ) > "${RUN_DIR}/services/remote_worker.log" 2>&1 &

    REMOTE_PID="$!"
    wait_for_port 8091 "Remote worker"
}

stop_ryu_controller() {
    stop_pid "${RYU_PID}"
    RYU_PID=""

    pkill -f "ryu.cmd.manager" 2>/dev/null || true
    sleep 2
}

start_ryu_controller() {
    local controller="$1"
    local label="$2"
    local logfile="$3"

    require_file "${controller}"
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
    wait_for_port 8080 "${label}"
}

run_mode() {
    local mode="$1"

    log "Running experiment mode: ${mode}"
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

    require_file "${ROOT}/apps/edge_worker.py"
    require_file "${ROOT}/apps/remote_worker.py"
    require_file "${ROOT}/configs/edge_emulator.yaml"
    require_file "${ROOT}/configs/remote_worker.yaml"

    require_file "${RYU_VENV}/bin/activate"
    require_file "${ROOT}/controllers/baseline_sdn_iot_controller.py"
    require_file "${ROOT}/controllers/qos_nsson_controller.py"
}

main() {
    cd "${ROOT}"
    validate_project

    mkdir -p \
        "${RUN_DIR}/services" \
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

    log "Starting NSSON_V02 experiment suite"
    echo "Project root : ${ROOT}"
    echo "Run directory: ${RUN_DIR}"
    echo "Cycles/mode  : ${CYCLES}"

    start_edge_worker
    start_remote_worker

    # Non-SDN baseline scenarios.
    run_mode "local_only"
    run_mode "offload_only"
    run_mode "adaptive_no_sdn"

    # Conventional static/best-effort SDN baseline.
    start_ryu_controller \
        "${ROOT}/controllers/baseline_sdn_iot_controller.py" \
        "Baseline SDN-IoT controller" \
        "${RUN_DIR}/services/baseline_sdn_iot_controller.log"

    run_mode "baseline_sdn_iot"
    stop_ryu_controller

    # Proposed full NSSON scenario.
    start_ryu_controller \
        "${ROOT}/controllers/qos_nsson_controller.py" \
        "NSSON QoS-aware controller" \
        "${RUN_DIR}/services/nsson_full_controller.log"

    run_mode "nsson_full"
    stop_ryu_controller

    log "Generating combined QoS CSV files"

    (
        cd "${ROOT}"
        source "${GPU_VENV}/bin/activate"
        export PYTHONPATH="${ROOT}"

        python "${ROOT}/generate_qos_data.py"
    ) 2>&1 | tee "${RUN_DIR}/generate_qos_data.log"

    log "Generating final paper plots"

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

    log "NSSON_V02 suite completed"

    echo "All raw logs, CSV outputs, service logs, and plots are saved in:"
    echo "  ${RUN_DIR}"

    echo
    echo "Generated plots:"
    find "${RUN_DIR}/final_plots" \
        -maxdepth 1 \
        -type f \
        \( -name "*.png" -o -name "*.jpg" -o -name "*.pdf" \) \
        -printf "  %f\n" \
        2>/dev/null | sort || true
}

main "$@"
