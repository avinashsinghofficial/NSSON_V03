from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/avi/NSSON/revision_runs_300cycle")
FIG_DIR = Path("/home/avi/NSSON/figures")
FIG_DIR.mkdir(exist_ok=True)

MODES = ["local_only", "offload_only", "adaptive_no_sdn", "nsson_full"]
COLORS = {
    "local_only": "#4c72b0",
    "offload_only": "#dd8452",
    "adaptive_no_sdn": "#55a868",
    "nsson_full": "#c44e52",
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Panel A: latency
ax = axes[0, 0]
for mode in MODES:
    df = pd.read_csv(ROOT / mode / "latency_log.csv")
    ax.plot(df["cycle"], 1000 * df["total_sec"], label=mode, color=COLORS[mode], linewidth=1.2)
ax.set_xlabel("Cycle")
ax.set_ylabel("End-to-end latency (ms)")
ax.set_title("(a) Latency per cycle")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel B: device energy
ax = axes[0, 1]
for mode in MODES:
    df = pd.read_csv(ROOT / mode / "power_log.csv")
    ax.plot(df["cycle"], df["energy_device_J"], label=mode, color=COLORS[mode], linewidth=1.2)
ax.set_xlabel("Cycle")
ax.set_ylabel("Modeled device energy (J/cycle)")
ax.set_title("(b) Modeled device-side energy per cycle")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel C: cumulative energy
ax = axes[1, 0]
for mode in MODES:
    df = pd.read_csv(ROOT / mode / "power_log.csv")
    ax.plot(df["cycle"], df["energy_device_J"].cumsum(), label=mode, color=COLORS[mode], linewidth=1.5)
ax.set_xlabel("Cycle")
ax.set_ylabel("Cumulative modeled energy (J)")
ax.set_title("(c) Cumulative modeled device-side energy")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel D: offload ratio
ax = axes[1, 1]
for mode in MODES:
    df = pd.read_csv(ROOT / mode / "power_log.csv")
    ratio = (df["decision"] == "offload").astype(int).expanding().mean()
    ax.plot(df["cycle"], 100 * ratio, label=mode, color=COLORS[mode], linewidth=1.5)
ax.set_xlabel("Cycle")
ax.set_ylabel("Cumulative offload ratio (%)")
ax.set_title("(d) Cumulative offload ratio")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = FIG_DIR / "nsson_300cycle_telemetry.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("Saved:", out)
