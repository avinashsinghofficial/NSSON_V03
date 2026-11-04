import re


def bandwidth_to_mbps(value, unit):
    factors = {
        "K": 1.0 / 1000.0,
        "M": 1.0,
        "G": 1000.0,
    }
    return float(value) * factors[unit.upper()]


def robust_tcp_median_mbps(output):
    active_rates = []
    pattern = re.compile(
        r"\[\s*\d+\]\s+"
        r"([0-9.]+)-([0-9.]+)\s+sec\s+"
        r".*?\s+([0-9.]+)\s+([KMG])bits/sec"
    )

    for line in output.splitlines():
        match = pattern.search(line)
        if not match:
            continue

        start_s = float(match.group(1))
        end_s = float(match.group(2))

        if end_s - start_s < 0.5:
            continue

        rate = bandwidth_to_mbps(match.group(3), match.group(4))

        if rate > 0.001:
            active_rates.append(rate)

    if not active_rates:
        return float("nan")

    active_rates.sort()
    middle = len(active_rates) // 2

    if len(active_rates) % 2:
        return active_rates[middle]

    return 0.5 * (active_rates[middle - 1] + active_rates[middle])
