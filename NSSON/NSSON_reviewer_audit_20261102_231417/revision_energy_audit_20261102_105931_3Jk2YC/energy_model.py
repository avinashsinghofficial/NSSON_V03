"""Explicit coarse-state energy model.

Inputs:
    durations_s: nonoverlapping state durations in seconds
    powers_w: explicitly supplied state-average powers in watts

Outputs:
    modeled energy in joules and watt-hours;
    modeled average power over the supplied interval.

No defaults or claims of hardware calibration are provided.
"""

import math

STATES = ("local_cnn", "local_snn", "rpc", "other")


def estimate_energy(durations_s, powers_w):
    if set(durations_s) != set(STATES):
        raise ValueError(f"Duration keys must be exactly {STATES}")
    if set(powers_w) != set(STATES):
        raise ValueError(f"Power keys must be exactly {STATES}")

    durations = {}
    powers = {}

    for state in STATES:
        duration = float(durations_s[state])
        power = float(powers_w[state])

        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f"Invalid duration for {state}: {duration}")
        if not math.isfinite(power) or power < 0:
            raise ValueError(f"Invalid power for {state}: {power}")

        durations[state] = duration
        powers[state] = power

    total_s = math.fsum(durations.values())
    if total_s <= 0:
        raise ValueError("Total interval must be positive")

    components_j = {
        state: powers[state] * durations[state]
        for state in STATES
    }

    energy_j = math.fsum(components_j.values())
    average_power_w = energy_j / total_s

    if not (
        min(powers.values()) - 1e-12
        <= average_power_w
        <= max(powers.values()) + 1e-12
    ):
        raise AssertionError("Average power outside state-power bounds")

    return {
        "component_energy_j": components_j,
        "energy_j": energy_j,
        "energy_wh": energy_j / 3600.0,
        "interval_s": total_s,
        "average_power_w": average_power_w,
        "classification": "modeled_from_explicit_power_inputs",
    }


def self_test():
    durations = dict.fromkeys(STATES, 0.0)
    durations["local_cnn"] = 2.0

    powers = dict.fromkeys(STATES, 5.0)
    result = estimate_energy(durations, powers)

    assert math.isclose(result["energy_j"], 10.0)
    assert math.isclose(result["energy_wh"], 10.0 / 3600.0)
    assert math.isclose(result["average_power_w"], 5.0)

    durations = dict.fromkeys(STATES, 1.0)
    powers = {
        "local_cnn": 2.0,
        "local_snn": 3.0,
        "rpc": 4.0,
        "other": 1.0,
    }
    result = estimate_energy(durations, powers)

    assert math.isclose(result["energy_j"], 10.0)
    assert math.isclose(result["average_power_w"], 2.5)

    invalid = dict(powers)
    invalid["rpc"] = -1.0

    try:
        estimate_energy(durations, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative power was not rejected")

    return {
        "status": "passed",
        "scope": (
            "Arithmetic, unit conversion, input validation, and "
            "average-power bounds; not hardware calibration."
        ),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2))
