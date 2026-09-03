"""Deterministic interventions for voltage/current forcing channels."""

from __future__ import annotations

import numpy as np


CONDITIONS = ("true", "zero", "shuffle", "shift")


def intervene_exogenous(
    exogenous: dict[str, np.ndarray],
    condition: str,
    *,
    seed: int,
    shift: int,
) -> dict[str, np.ndarray]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown intervention: {condition}")
    if shift < 0:
        raise ValueError("shift must be non-negative")

    output = {key: np.asarray(value).copy()
              for key, value in exogenous.items()}
    voltage = output["voltage"]
    current = output["current"]
    if voltage.shape != current.shape:
        raise ValueError("voltage and current shapes differ")

    if condition == "zero":
        voltage.fill(0)
        current.fill(0)
    elif condition == "shuffle":
        permutation = np.random.default_rng(seed).permutation(len(voltage))
        output["voltage"] = voltage[permutation]
        output["current"] = current[permutation]
    elif condition == "shift" and shift:
        delay = min(shift, len(voltage))
        shifted_voltage = np.empty_like(voltage)
        shifted_current = np.empty_like(current)
        shifted_voltage[:delay] = voltage[0]
        shifted_current[:delay] = current[0]
        shifted_voltage[delay:] = voltage[:-delay]
        shifted_current[delay:] = current[:-delay]
        output["voltage"] = shifted_voltage
        output["current"] = shifted_current
    return output
