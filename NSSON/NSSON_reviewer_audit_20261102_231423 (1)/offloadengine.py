import os
import json
import time
import numpy as np

NSSON_ROOT = "/home/avi/NSSON_V02"
DECISION_LOG = os.path.join(NSSON_ROOT, "logs", "offloaddecisionlog.json")

# Paper values
CNN_ACC = 0.9484
SNN_ACC = 0.7766

def generate_confidences(label, mode="adaptive_no_sdn"):
    # Simple synthetic confidences consistent with paper accuracies:
    # - On average, correct samples get higher confidence.
    # - SNN has lower accuracy and slightly lower max confidence than CNN.
    correct = np.random.rand() < SNN_ACC
    if correct:
        conf_snn = 0.6 + 0.4 * np.random.rand()  # [0.6, 1.0]
    else:
        conf_snn = 0.2 + 0.4 * np.random.rand()  # [0.2, 0.6]

    correct_cnn = np.random.rand() < CNN_ACC
    if correct_cnn:
        conf_cnn = 0.7 + 0.3 * np.random.rand()  # [0.7, 1.0]
    else:
        conf_cnn = 0.3 + 0.4 * np.random.rand()  # [0.3, 0.7]

    return float(conf_snn), float(conf_cnn)

def run_offload(cycles=300, base_theta=0.6, mode="adaptive_no_sdn"):
    np.random.seed(42)  # reproducibility

    with open(DECISION_LOG, "w") as f:
        for cycle in range(cycles):
            label = np.random.randint(0, 10)
            conf_snn, conf_cnn = generate_confidences(label, mode)

            if mode == "local_only":
                decision = "local"
            elif mode == "offload_only":
                decision = "offload"
            else:
                decision = "offload" if conf_snn < base_theta else "local"

            rec = {
                "sampleid": int(cycle),
                "label": int(label),
                "conf_snn": conf_snn,
                "conf_cnn": conf_cnn,
                "decision": decision,
                "theta": float(base_theta),
                "timestamp": time.time(),
                "cycle": int(cycle),
                "mode": mode,
            }
            f.write(json.dumps(rec) + "\n")

    print("Offload decisions written to", DECISION_LOG)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["local_only", "offload_only", "adaptive_no_sdn", "nsson_full"])
    args = p.parse_args()
    run_offload(cycles=300, base_theta=0.6, mode=args.mode)
