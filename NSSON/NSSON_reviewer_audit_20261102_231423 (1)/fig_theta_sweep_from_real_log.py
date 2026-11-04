#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT = Path("results/run_20260916_150942/nsson_full/decision_log.json")
OUT_DIR = Path("figures_ieee")
OUT_DIR.mkdir(parents=True, exist_ok=True)

records = []
with INPUT.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

df = pd.DataFrame(records)

required = {"conf_snn", "sampleid"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

confidence = pd.to_numeric(df["conf_snn"], errors="coerce").dropna().to_numpy()

if len(confidence) == 0:
    raise ValueError("No valid conf_snn values found.")

thresholds = np.arange(0.30, 0.901, 0.05)

rows = []
for theta in thresholds:
    decisions = confidence < theta
    rows.append({
        "theta": theta,
        "offloaded_samples": int(decisions.sum()),
        "total_samples": int(len(confidence)),
        "offload_ratio": float(decisions.mean()),
    })

out = pd.DataFrame(rows)
out.to_csv(OUT_DIR / "theta_offload_ratio_sweep.csv", index=False)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
})

fig, ax = plt.subplots(figsize=(3.45, 2.45))
ax.plot(
    out["theta"],
    out["offload_ratio"],
    marker="o",
    markersize=4.5,
    linewidth=1.7,
    color="#1f77b4",
    label="Measured empirical offload ratio",
)

for _, r in out.iterrows():
    ax.annotate(
        f"{int(r['offloaded_samples'])}/{int(r['total_samples'])}",
        (r["theta"], r["offload_ratio"]),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=6.5,
    )

ax.axvline(
    0.60,
    color="#d62728",
    linestyle="--",
    linewidth=1.0,
    label=r"Original operating point: $\theta=0.60$",
)

ax.set_xlabel(r"Confidence threshold $\theta$")
ax.set_ylabel(r"Empirical offload ratio $\phi_{\mathrm{offload}}$")
ax.set_ylim(-0.03, 1.03)
ax.set_xlim(0.28, 0.92)
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.85)
ax.legend(loc="upper left", fontsize=6.5, frameon=True)

fig.tight_layout()
fig.savefig(
    OUT_DIR / "fig_theta_vs_offload_ratio.pdf",
    dpi=600,
    bbox_inches="tight",
)
fig.savefig(
    OUT_DIR / "fig_theta_vs_offload_ratio.png",
    dpi=600,
    bbox_inches="tight",
)

print("Saved:", OUT_DIR / "theta_offload_ratio_sweep.csv")
print("Saved:", OUT_DIR / "fig_theta_vs_offload_ratio.pdf")
print("Saved:", OUT_DIR / "fig_theta_vs_offload_ratio.png")
print(out.to_string(index=False))
