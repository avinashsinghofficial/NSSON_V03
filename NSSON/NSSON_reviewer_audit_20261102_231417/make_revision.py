from pathlib import Path
import re

SRC = Path("/home/avi/Downloads/NSSON_main_original.tex")
DST = Path("/home/avi/Downloads/NSSON_main_revised.tex")
text = SRC.read_text()

def strip_macro(s, macro):
    out, i = [], 0
    token = "\\" + macro + "{"
    while True:
        j = s.find(token, i)
        if j == -1:
            out.append(s[i:])
            break
        out.append(s[i:j])
        k, depth = j + len(token), 1
        while k < len(s) and depth:
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
            k += 1
        if depth:
            raise RuntimeError("Unbalanced braces in " + macro)
        out.append(s[j + len(token):k - 1])
        i = k
    return "".join(out)

text = strip_macro(text, "revblue")
text = strip_macro(text, "revred")

def sub_once(pattern, replacement, flags=0):
    global text
    new, n = re.subn(pattern, replacement, text, count=1, flags=flags)
    if n != 1:
        print("WARNING: pattern not found exactly once:")
        print(pattern[:180])
        return
    text = new

# Abstract: remove invalid energy claim
sub_once(
    r"Over 300 simulated inference cycles, adaptive offloading reduces\s+"
    r"end-to-end latency from \$0\.61\$~s to \$0\.376458\$~s \(a \$38\.7\\%\$ reduction\)\s+"
    r"and modeled Jetson Nano energy per cycle from \$4\.12\$~Wh to\s+"
    r"\$0\.000133\$~Wh\. These power values are derived from a calibrated\s+"
    r"coefficient-based model rather than direct hardware measurements\.",
    r"\\revblue{Over 300 simulated inference cycles, adaptive offloading reduces "
    r"end-to-end latency from $0.61$~s to $0.376458$~s, corresponding to a "
    r"$38.7\\%$ reduction. Edge energy is estimated using a coefficient-based, "
    r"device-profile-driven model rather than direct hardware measurements; "
    r"accordingly, all energy results are reported as modeled estimates and are "
    r"subject to hardware, thermal, and DVFS-related uncertainty.}"
)

# Abstract: correct overbroad QoS claim
sub_once(
    r"Further experiments on end-to-end network quality of service \(latency, packet delivery ratio, packet drop rate, routing overhead, and throughput\) demonstrate that the proposed NSSON consistently lowers latency and routing overhead while sustaining higher throughput and comparable or better reliability than a baseline SDN-IoT configuration\.",
    r"\\revblue{Further experiments on end-to-end network quality of service, "
    r"including latency, packet delivery ratio, packet drop rate, routing "
    r"overhead, and throughput, show that NSSON reduces average latency and "
    r"routing overhead and sustains higher throughput than the evaluated "
    r"baseline SDN-IoT configuration. Its reliability is comparable to or "
    r"better than the baseline in the evaluated operating region, although "
    r"per-cycle delivery and drop-rate behavior vary with load.}"
)

# Contribution: correct threshold direction
sub_once(
    r"These averages are then used in the update rule.*?"
    r"so that large recent latency or power lowers \$\\theta\(t\)\$ and increases the probability that new samples are offloaded\."
    r".*?reducing network usage and transfer energy\.",
    r"\\revblue{These averages are used in the threshold-update law. Because "
    r"samples are offloaded when $C(s)<\\theta(t)$, increasing $\\theta(t)$ "
    r"increases the offload ratio, whereas decreasing $\\theta(t)$ reduces it. "
    r"The feedback coefficients are therefore selected so that the threshold "
    r"update remains consistent with this monotonic relationship and remains "
    r"within the configured operating interval. The threshold used for each "
    r"subsequent inference is derived only from previously completed telemetry, "
    r"ensuring causal feedback.}",
    flags=re.S
)

# Feedback section: correct threshold direction
sub_once(
    r"Since samples are offloaded\s+"
    r"when \$C\(s\) < \$\\theta\(t\)\$ according to~\\eqref\{eq:decision\}, lowering\s+"
    r"\$\\theta\(t\)\$ reduces the fraction of offloaded samples and raises\s+"
    r"\$\\theta\(t\)\$ increases it\..*?"
    r"which can exploit spare network\s+capacity and edge resources\.",
    r"\\revblue{Since samples are offloaded when $C(s)<\\theta(t)$ according to "
    r"Eq.~\\eqref{eq:decision}, increasing $\\theta(t)$ increases the fraction "
    r"of offloaded samples, whereas decreasing $\\theta(t)$ reduces it. The "
    r"feedback gains are selected to keep the threshold within its configured "
    r"range and to avoid oscillation. The threshold applied to a given "
    r"inference is computed only from telemetry collected during earlier "
    r"completed cycles, ensuring that feedback is causal and that no update "
    r"can alter the decision for an already-completed request.}",
    flags=re.S
)

# Replace dimensionally inconsistent power model
sub_once(
    r"\\subsection\{Edge Power Model\}.*?(?=\\subsection\{Closed-Loop Feedback Mechanism\})",
    r"""\\revblue{\\subsection{Edge Energy Model}
\\label{sec:power_model}

NSSON uses a coefficient-based, device-profile-driven energy model rather
than direct hardware power measurements. For each inference cycle, the
edge-side energy is decomposed into computation, communication, and idle
components:

\\begin{equation}
E_{\\text{cycle}}
=
E_{\\text{compute}}
+
E_{\\text{transfer}}
+
E_{\\text{idle}}.
\\label{eq:power}
\\end{equation}

The computation and idle components are obtained by multiplying an
average power coefficient by the corresponding active duration:

\\begin{equation}
\\begin{aligned}
E_{\\text{compute}}
&= W_c L_{\\text{compute}}, \\\\
E_{\\text{idle}}
&= W_i L_{\\text{idle}}.
\\end{aligned}
\\label{eq:power_terms}
\\end{equation}

Here, $W_c$ and $W_i$ are average power coefficients in watts, while
$L_{\\text{compute}}$ and $L_{\\text{idle}}$ are durations in seconds;
their products are therefore energy quantities in joules.

For an offloaded sample with data volume $M_{\\text{data}}$, the
communication-energy component is modeled as:

\\begin{equation}
E_{\\text{transfer}}
=
W_t M_{\\text{data}},
\\label{eq:energy_transfer}
\\end{equation}

where $W_t$ is expressed in joules per megabyte and $M_{\\text{data}}$ is
expressed in megabytes.

The average power during a cycle is:

\\begin{equation}
P_{\\text{avg}}
=
\\frac{E_{\\text{cycle}}}{L_{\\text{e2e}}},
\\label{eq:energy}
\\end{equation}

where $L_{\\text{e2e}}$ is the end-to-end cycle latency in seconds.
Power is reported in watts, whereas energy is reported in joules or
watt-hours; one watt-hour equals $3600$ joules. In the reported
configuration, $W_c=5.0$~W, $W_i=1.5$~W, and
$W_t=0.5$~J/MB are profile-based estimates for a Jetson Nano-class
device. They are not obtained from direct rail-level measurements, and
the resulting energy values should be interpreted as modeled estimates
subject to device, thermal, and DVFS-related uncertainty.}

""",
    flags=re.S
)

# Add SDN flow-association protocol
sub_once(
    r"(?=\\subsection\{Without SDN: Cost Quantification\})",
    r"""\\revblue{\\subsection{Flow Association and Consistency}
\\label{sec:flow_association}

Each inference request is assigned a globally unique
\\texttt{session\\_id} and \\texttt{request\\_id}. The client includes both
identifiers, the sample identifier, payload length, and SHA-256 payload
digest in the request header. The edge server returns the same
identifiers and digest in its response. A request is accepted only when
all identifiers, the transport tuple, and the payload digest match.

A controller-side registration record contains the session identifier,
request identifier, sample identifier, client transport tuple, payload
digest, and expiration time. A registration is valid only when its
identifiers match the subsequent inference request and its expiration
time has not passed. Registrations that are absent, expired, reused, or
associated with a different transport tuple are rejected rather than
being mapped to the most recent telemetry entry.

Under concurrent sessions, association is therefore based on explicit
per-request identities rather than on the latest global telemetry entry.
This prevents a PacketIn event generated by one session from inheriting
the offload decision of another session. The current evaluation verifies
request association and registration handling; controller-side flow
installation and data-plane enforcement are evaluated separately.}

""",
)

# Correct evaluation metric
sub_once(
    r"\\item Edge power consumption \$P_\{\\text\{total\}\}\$ \(Wh\)",
    r"\\revblue{\\item Modeled edge energy per cycle $E_{\\text{cycle}}$ (J or Wh); average power $P_{\\text{avg}}$ (W)}"
)

# Correct telemetry narrative
sub_once(
    r"Adaptive offloading achieves a 23\\% reduction in average latency and an\s+18\\% reduction in power compared with local-only execution\.",
    r"\\revblue{Adaptive offloading achieves a $38.7\\%$ reduction in average "
    r"end-to-end latency relative to local-only execution. Energy results are "
    r"modeled estimates obtained from the coefficient-based energy model; the "
    r"previously reported $18\\%$ power-saving figure is not retained because "
    r"the underlying power and energy quantities require recomputation under "
    r"the corrected model.}"
)

# Correct telemetry table
sub_once(
    r"Avg\. latency \(s\) & 0\.61 & 0\.376458 & 23\\% \\\\\s+Avg\. power \(Wh\) & 4\.12 & 0\.000133 & 18\\% \\\\",
    r"\\revblue{Avg. latency (s) & 0.61 & 0.376458 & $38.7\\%$ \\\\\n"
    r"Modeled energy per cycle & \\multicolumn{2}{c}{Recomputation required} & -- \\\\}"
)

# Correct ablation table
sub_once(
    r"Adaptive Engine & 23\\% latency reduction; 45\\% of cycles offloaded \\\\\s+Power Model & 18\\% power saving under adaptive policy \\\\",
    r"\\revblue{Adaptive Engine & $38.7\\%$ latency reduction; $45\\%$ of cycles offloaded \\\\\n"
    r"Energy Model & Coefficient-based modeled energy; hardware validation future work \\\\}"
)

# Correct remaining narrative inconsistencies
pairs = [
    ("The 23\\% latency reduction achieved by NSSON under full SDN",
     "\\revblue{The 38.7\\% latency reduction achieved by NSSON under full SDN}"),
    ("Adaptive offloading reduces average latency by 23\\%\nand power by 18\\% compared to local-only execution.",
     "\\revblue{Adaptive offloading reduces average latency by 38.7\\% relative to local-only execution; energy results are modeled estimates produced by the coefficient-based energy model.}"),
    ("Lower latency\\\\(23\\% reduction)",
     "\\revblue{Lower latency\\\\(38.7\\% reduction)}"),
    ("Lower energy\\\\(18\\% saving)",
     "\\revblue{Lower modeled edge energy}"),
    ("The extended QoS experiments on latency, PDR, packet drop rate, routing overhead, and throughput show that NSSON’s SMCC-aware design consistently improves network performance relative to a baseline SDN-IoT configuration, while maintaining high sensing accuracy and energy efficiency.",
     "\\revblue{The extended QoS experiments on latency, PDR, packet drop rate, routing overhead, and throughput show that NSSON reduces average latency and routing overhead and sustains higher throughput than the evaluated baseline, while reliability remains comparable to or better than the baseline in the evaluated operating region. Energy results are modeled estimates rather than direct hardware measurements.}"),
    ("From\nEq.~\\eqref{eq:power_terms} and Eq.~\\eqref{eq:energy}, the per-cycle energy can be\nwritten as",
     "\\revblue{From Eqs.~\\eqref{eq:power_terms} and \\eqref{eq:energy}, the per-cycle energy can be written as}"),
]

for old, new in pairs:
    if old not in text:
        raise RuntimeError("Narrative text not found:\n" + old[:150])
    text = text.replace(old, new, 1)

DST.write_text(text)
count = text.count("\\revblue{")
print("Created:", DST)
print("New revision highlights:", count)
