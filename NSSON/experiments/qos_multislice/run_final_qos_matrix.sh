#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/avi/NSSON_V02"
VENV="/home/avi/NSSON/.venv_nsson_gpu"
SYSTEM_PYTHON="/usr/bin/python3"
TOPOLOGY="${ROOT}/qos_multislice/qos_multislice_sla_experiment.py"
RAW_DIR="${ROOT}/logs/qos_multislice_raw"
RESULTS_DIR="${ROOT}/results/qos_multislice_final"
INVALID_DIR="${RESULTS_DIR}/invalid"
RUN_LOG_DIR="${RESULTS_DIR}/run_logs"

export NSSON_ROOT="${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

mkdir -p "${RAW_DIR}" "${RESULTS_DIR}" "${INVALID_DIR}" "${RUN_LOG_DIR}"

MODE_TO_CONTROLLER() {
  case "$1" in
    baseline_sdn_iot)
      echo "${ROOT}/controllers/baseline_sdn_iot_controller.py"
      ;;
    priority_only)
      echo "${ROOT}/controllers/qos_nsson_controller.py"
      ;;
    slicing_sla)
      echo "${ROOT}/controllers/nsson_multislice_sla_controller.py"
      ;;
    *)
      echo "Unknown mode: $1" >&2
      exit 2
      ;;
  esac
}

cleanup_environment() {
  echo "[CLEANUP] Killing stale Ryu/iperf/Mininet processes..."
  sudo pkill -9 -f ryu-manager 2>/dev/null || true
  sudo pkill -9 -f iperf 2>/dev/null || true
  sudo pkill -9 -f iperf3 2>/dev/null || true
  sudo pkill -9 -f mnexec 2>/dev/null || true
  sudo pkill -9 -f mininet 2>/dev/null || true
  sudo mn -c >/dev/null 2>&1 || true
  sleep 4
}

wait_for_ryu() {
  local timeout=20
  local elapsed=0

  while (( elapsed < timeout )); do
    if nc -z 127.0.0.1 6633 >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  return 1
}

validate_csv() {
  local csv_path="$1"

  "${VENV}/bin/python" - "${csv_path}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

p = Path(sys.argv[1])

if not p.exists():
    raise SystemExit(f"CSV_MISSING: {p}")

df = pd.read_csv(p)

if len(df) != 3:
    raise SystemExit(f"INVALID_ROW_COUNT: expected=3 actual={len(df)}")

required_slices = {"autonomous", "healthcare", "industrial"}
actual_slices = set(df["slice"].astype(str))

if actual_slices != required_slices:
    raise SystemExit(
        f"INVALID_SLICES: expected={sorted(required_slices)} "
        f"actual={sorted(actual_slices)}"
    )

bad = df[
    (df["tcp_valid_measurement"] != 1)
    | (df["udp_valid_measurement"] != 1)
]

if len(bad) > 0:
    print(bad.to_string(index=False))
    raise SystemExit(f"INVALID_MEASUREMENTS: invalid_rows={len(bad)}")

if df["tcp_throughput_mbps"].isna().any():
    raise SystemExit("INVALID_TCP_METRIC: NaN throughput found")

if df["udp_jitter_ms"].isna().any():
    raise SystemExit("INVALID_UDP_METRIC: NaN jitter found")

print("CSV_VALID")
print(
    df[
        [
            "slice",
            "tcp_throughput_mbps",
            "udp_jitter_ms",
            "udp_loss_pct",
            "udp_pdr",
            "sla_all_met",
        ]
    ].to_string(index=False)
)
PY
}

run_one() {
  local mode="$1"
  local load="$2"
  local rep="$3"

  local controller
  controller="$(MODE_TO_CONTROLLER "${mode}")"

  local tag="${mode}_${load}_rep${rep}"
  local csv="${RAW_DIR}/multislice_${tag}.csv"
  local txt="${RAW_DIR}/multislice_${tag}.txt"
  local ryu_log="${RUN_LOG_DIR}/ryu_${tag}.log"
  local topology_log="${RUN_LOG_DIR}/topology_${tag}.log"
  local manifest="${RESULTS_DIR}/${tag}_manifest.json"

  echo
  echo "============================================================"
  echo "[RUN] ${tag}"
  echo "[RUN] Controller: ${controller}"
  echo "============================================================"

  cleanup_environment

  rm -f "${csv}" "${txt}"

  echo "[RUN] Starting Ryu controller..."
  source "${VENV}/bin/activate"

  ryu-manager \
    "${controller}" \
    --ofp-tcp-listen-port 6633 \
    > "${ryu_log}" 2>&1 &
  local ryu_pid=$!

  if ! wait_for_ryu; then
    echo "[ERROR] Ryu did not listen on port 6633."
    cp "${ryu_log}" "${INVALID_DIR}/ryu_${tag}_startup_failure.log" 2>/dev/null || true
    kill "${ryu_pid}" 2>/dev/null || true
    return 1
  fi

  echo "[RUN] Ryu is listening on 6633. Launching topology..."

  set +e
  sudo -E env \
    PYTHONPATH="${ROOT}" \
    "${SYSTEM_PYTHON}" \
    "${TOPOLOGY}" \
    --mode "${mode}" \
    --load "${load}" \
    --rep "${rep}" \
    --controller-port 6633 \
    > "${topology_log}" 2>&1
  local topology_status=$?
  set -e

  kill "${ryu_pid}" 2>/dev/null || true
  wait "${ryu_pid}" 2>/dev/null || true

  if [[ "${topology_status}" -ne 0 ]]; then
    echo "[ERROR] Topology process failed for ${tag}."
    cp "${topology_log}" "${INVALID_DIR}/topology_${tag}_failure.log" 2>/dev/null || true
    cleanup_environment
    return 1
  fi

  if ! validation_output="$(validate_csv "${csv}" 2>&1)"; then
    echo "[ERROR] Invalid measurement run: ${tag}"
    echo "${validation_output}"

    [[ -f "${csv}" ]] && cp "${csv}" "${INVALID_DIR}/${tag}_INVALID.csv"
    [[ -f "${txt}" ]] && cp "${txt}" "${INVALID_DIR}/${tag}_INVALID.txt"
    cp "${ryu_log}" "${INVALID_DIR}/ryu_${tag}_INVALID.log" 2>/dev/null || true
    cp "${topology_log}" "${INVALID_DIR}/topology_${tag}_INVALID.log" 2>/dev/null || true

    cleanup_environment
    return 1
  fi

  echo "${validation_output}"

  cp "${csv}" "${RESULTS_DIR}/${tag}_VALID.csv"
  cp "${txt}" "${RESULTS_DIR}/${tag}_VALID.txt"
  cp "${ryu_log}" "${RESULTS_DIR}/ryu_${tag}_VALID.log"
  cp "${topology_log}" "${RESULTS_DIR}/topology_${tag}_VALID.log"

  cat > "${manifest}" <<EOF
{
  "status": "valid",
  "mode": "${mode}",
  "load": "${load}",
  "rep": ${rep},
  "controller": "$(basename "${controller}")",
  "controller_port": 6633,
  "tcp_metric": "median steady-state classic-iperf throughput; excludes first second and final two seconds",
  "udp_metric": "controlled UDP probe before long TCP/background workload",
  "validation": "three required slices; valid TCP and UDP measurements; non-NaN metrics"
}
EOF

  echo "[SUCCESS] Valid result saved: ${RESULTS_DIR}/${tag}_VALID.csv"

  cleanup_environment
  return 0
}

# Final 3 x 3 experimental matrix.
declare -a MODES=(
  "baseline_sdn_iot"
  "priority_only"
  "slicing_sla"
)

declare -a LOADS=(
  "light"
  "moderate"
  "heavy"
)

declare -A REP_BY_LOAD=(
  [light]=11
  [moderate]=12
  [heavy]=13
)

success_count=0
failure_count=0

for load in "${LOADS[@]}"; do
  for mode in "${MODES[@]}"; do
    rep="${REP_BY_LOAD[${load}]}"

    if run_one "${mode}" "${load}" "${rep}"; then
      success_count=$((success_count + 1))
    else
      failure_count=$((failure_count + 1))
      echo "[WARNING] Continuing after failure: ${mode}/${load}/rep${rep}"
    fi
  done
done

echo
echo "============================================================"
echo "FINAL MATRIX COMPLETE"
echo "Valid runs:   ${success_count}"
echo "Invalid runs: ${failure_count}"
echo "Valid output: ${RESULTS_DIR}"
echo "Invalid logs: ${INVALID_DIR}"
echo "============================================================"

"${VENV}/bin/python" - "${RESULTS_DIR}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
files = sorted(root.glob("*_VALID.csv"))

if not files:
    print("No valid CSV files found.")
    raise SystemExit(0)

df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)

summary = (
    df.groupby(["mode", "load", "slice"], as_index=False)
      .agg(
          tcp_throughput_mbps=("tcp_throughput_mbps", "mean"),
          udp_jitter_ms=("udp_jitter_ms", "mean"),
          udp_loss_pct=("udp_loss_pct", "mean"),
          udp_pdr=("udp_pdr", "mean"),
          sla_all_met=("sla_all_met", "mean"),
      )
)

summary_path = root / "qos_multislice_final_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nFINAL SUMMARY")
print(summary.to_string(index=False))
print(f"\nSaved: {summary_path}")
PY
