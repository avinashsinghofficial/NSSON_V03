import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from energy_model import STATES, self_test

BASE = Path(os.environ["NSSON_ENERGY_INPUT"])
OUT = Path(os.environ["NSSON_ENERGY_OUTPUT"])


def write_json(name, value):
    (OUT / name).write_text(
        json.dumps(value, indent=2) + "\n"
    )


def read_csv(name):
    with (BASE / name).open() as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    if not rows:
        raise RuntimeError(f"No rows for {name}")

    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity(row):
    return (
        row["run_id"],
        row["session_id"],
        row["request_id"],
    )


def nonnegative_number(row, name):
    value = float(row[name])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {name}: {value}")
    return value


clients = read_csv("client_requests.csv")
servers = read_csv("server_requests.csv")

if not clients:
    raise RuntimeError("Client log is empty")

server_lookup = {identity(row): row for row in servers}
if len(server_lookup) != len(servers):
    raise RuntimeError("Duplicate server request identity")

expected_remote = [
    row for row in clients if row["decision"] == "offload"
]

expected_identities = [identity(row) for row in expected_remote]

if len(set(expected_identities)) != len(expected_identities):
    raise RuntimeError("Duplicate remote client request identity")

if set(expected_identities) != set(server_lookup):
    raise RuntimeError("Client/server identity sets differ")

for row in expected_remote:
    observed = server_lookup[identity(row)]

    for field in (
        "sample_id",
        "image_sha256",
        "client_ip",
        "client_port",
        "server_ip",
        "server_port",
    ):
        if row[field] != observed[field]:
            raise RuntimeError(
                f"Association mismatch in {field}: {identity(row)}"
            )

    if row["final_prediction"] != observed["prediction"]:
        raise RuntimeError("Client/server prediction mismatch")

basis_rows = []
seen_samples = set()
maximum_residual_mismatch = 0.0
negative_roundoff_count = 0

for row in clients:
    key = (row["policy"], row["sample_id"])
    if key in seen_samples:
        raise RuntimeError(f"Duplicate policy/sample record: {key}")
    seen_samples.add(key)

    total = nonnegative_number(row, "request_sec")
    local = nonnegative_number(row, "local_phase_sec")
    rpc = nonnegative_number(row, "remote_rpc_sec")

    if total <= 0:
        raise RuntimeError("Nonpositive request interval")

    raw_other = total - local - rpc
    recorded_other = float(row["residual_overhead_sec"])

    if not math.isfinite(recorded_other):
        raise RuntimeError("Non-finite recorded residual")

    maximum_residual_mismatch = max(
        maximum_residual_mismatch,
        abs(raw_other - recorded_other),
    )

    if abs(raw_other - recorded_other) > 1e-9:
        raise RuntimeError("Recorded timing residual is inconsistent")

    if raw_other < -1e-9:
        raise RuntimeError(
            "Timing components overlap or exceed request interval"
        )

    if raw_other < 0:
        negative_roundoff_count += 1

    other = max(raw_other, 0.0)

    if row["decision"] == "local" and rpc != 0:
        raise RuntimeError("Local request contains remote RPC duration")

    if row["decision"] == "offload" and rpc <= 0:
        raise RuntimeError("Remote request has no RPC duration")

    if row["policy"] == "remote_cnn" and local != 0:
        raise RuntimeError("Remote-only request unexpectedly ran locally")

    local_cnn = local if row["policy"] == "local_cnn" else 0.0
    local_snn = (
        local if row["policy"] in ("local_snn", "adaptive_fixed")
        else 0.0
    )

    if local > 0 and local_cnn == 0 and local_snn == 0:
        raise RuntimeError("Unclassified local execution duration")

    basis_rows.append({
        "run_id": row["run_id"],
        "policy": row["policy"],
        "sample_id": int(row["sample_id"]),
        "decision": row["decision"],
        "t_local_cnn_s": local_cnn,
        "t_local_snn_s": local_snn,
        "t_rpc_s": rpc,
        "t_other_s": other,
        "request_sec": total,
        "raw_other_s": raw_other,
        "energy_j": "",
        "energy_status": "not_computed_missing_justified_power_inputs",
    })

write_csv("energy_duration_basis.csv", basis_rows)

policies = sorted({row["policy"] for row in basis_rows})
policy_summary = {}

for policy in policies:
    rows = [r for r in basis_rows if r["policy"] == policy]

    policy_summary[policy] = {
        "requests": len(rows),
        "offloaded": sum(r["decision"] == "offload" for r in rows),
        "mean_request_sec": statistics.mean(
            r["request_sec"] for r in rows
        ),
        "mean_local_cnn_sec": statistics.mean(
            r["t_local_cnn_s"] for r in rows
        ),
        "mean_local_snn_sec": statistics.mean(
            r["t_local_snn_s"] for r in rows
        ),
        "mean_rpc_sec": statistics.mean(
            r["t_rpc_s"] for r in rows
        ),
        "mean_other_sec": statistics.mean(
            r["t_other_s"] for r in rows
        ),
    }

# Analytical sensitivity only.
# Assumption: CNN, SNN and other client states share reference power P_ref.
# RPC-state power = rho * P_ref.
#
# Report E/P_ref, with units J/W = seconds.
# No absolute device power or measured energy is claimed.
ratios = (0.0, 0.25, 0.5, 1.0, 2.0)
sensitivity = []

for rho in ratios:
    equivalents = {}

    for policy, summary in policy_summary.items():
        equivalents[policy] = (
            summary["mean_local_cnn_sec"]
            + summary["mean_local_snn_sec"]
            + summary["mean_other_sec"]
            + rho * summary["mean_rpc_sec"]
        )

    for policy, equivalent in equivalents.items():
        record = {
            "policy": policy,
            "rpc_to_reference_power_ratio": rho,
            "energy_over_reference_power_s": equivalent,
            "scenario_type": "analytical_assumption_not_measurement",
            "zero_ratio_interpretation": (
                "idealized_zero_RPC_power_bound"
                if rho == 0 else "not_applicable"
            ),
        }

        for baseline in ("local_cnn", "local_snn"):
            reference = equivalents[baseline]
            record[f"conditional_reduction_vs_{baseline}_percent"] = (
                100.0 * (1.0 - equivalent / reference)
            )

        sensitivity.append(record)

write_csv("conditional_power_ratio_sensitivity.csv", sensitivity)

remote = policy_summary["remote_cnn"]
remote_nonrpc = (
    remote["mean_local_cnn_sec"]
    + remote["mean_local_snn_sec"]
    + remote["mean_other_sec"]
)

break_even = {}
for baseline in ("local_cnn", "local_snn"):
    break_even[baseline] = (
        (
            policy_summary[baseline]["mean_request_sec"]
            - remote_nonrpc
        ) / remote["mean_rpc_sec"]
    )

summary = {
    "status": "completed",
    "input_directory": str(BASE),
    "audit_directory": str(OUT),
    "client_requests": len(clients),
    "matched_remote_requests": len(expected_remote),
    "observed_cross_log_mismatches": 0,
    "maximum_residual_mismatch_sec": maximum_residual_mismatch,
    "negative_roundoff_residuals_clipped": negative_roundoff_count,
    "energy_model_tests": self_test(),
    "policy_duration_summary": policy_summary,
    "remote_only_break_even_rpc_power_ratio": break_even,
    "break_even_assumption": (
        "Local CNN, local SNN, and other states share one reference "
        "power; RPC-state power is rho times that reference power. "
        "These are conditional bounds, not calibrated hardware results."
    ),
    "absolute_energy_results": "not_computed",
    "reason": "No justified state-power inputs have been supplied.",
    "limitations": [
        "20 requests per policy; smoke-test data only.",
        "Sequential application association, not concurrent SDN validation.",
        "RPC aggregates communication, server waiting, and response handling.",
        "Server energy is excluded from this originating-client model.",
        "The same physical workstation/GPU serves both namespaces.",
        "No Jetson hardware energy or latency is established.",
        "Sensitivity ratios are analytical assumptions, not confidence intervals.",
        "No feedback-controlled decision or QoS improvement is established.",
    ],
    "input_sha256": {
        name: hashlib.sha256((BASE / name).read_bytes()).hexdigest()
        for name in (
            "client_requests.csv",
            "server_requests.csv",
            "summary.json",
        )
    },
}

write_json("audit_summary.json", summary)
print(json.dumps(summary, indent=2))

print("\n===== CONDITIONAL SENSITIVITY: NOT MEASURED ENERGY =====")
for row in sensitivity:
    print(
        f"rho={row['rpc_to_reference_power_ratio']:.2f} "
        f"policy={row['policy']} "
        f"E/P_ref={row['energy_over_reference_power_s']:.6f} s "
        f"conditional_change_vs_local_snn="
        f"{row['conditional_reduction_vs_local_snn_percent']:.2f}%"
    )

print(
    "\nDo not insert the conditional percentages into the abstract "
    "as measured or calibrated energy savings."
)
