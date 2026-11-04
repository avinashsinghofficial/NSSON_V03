from pathlib import Path

p = Path("/home/avi/Downloads/NSSON_main_revised.tex")
text = p.read_text()

start = "Since samples are offloaded"
end = "\\revblue{For SMCC-aware operation"
a = text.find(start)
b = text.find(end)
if a == -1 or b == -1 or b <= a:
    raise SystemExit("Could not locate feedback paragraph.")

new_feedback = r"""\revblue{Since samples are offloaded when $C(s)<\theta(t)$ according to
Eq.~\eqref{eq:decision}, increasing $\theta(t)$ increases the fraction
of offloaded samples, whereas decreasing $\theta(t)$ reduces it. The
feedback gains are selected to keep the threshold within its configured
range and to avoid oscillation. The threshold applied to a given
inference is computed only from telemetry collected during earlier
completed cycles, ensuring that feedback is causal and that no update
can alter the decision for an already-completed request.}
"""
text = text[:a] + new_feedback + text[b:]

start = "Avg. latency (s) & 0.61"
end = "Offload ratio"
a = text.find(start)
b = text.find(end)
if a == -1 or b == -1 or b <= a:
    raise SystemExit("Could not locate telemetry rows.")

new_rows = r"""\revblue{Avg. latency (s) & 0.61 & 0.376458 & $38.7\%$ \\
Modeled energy per cycle & \multicolumn{2}{c}{Recomputation required} & -- \\
"""
text = text[:a] + new_rows + text[b:]

p.write_text(text)
print("Fixed remaining feedback and telemetry rows.")
print("Revision highlights:", text.count("\\revblue{"))
