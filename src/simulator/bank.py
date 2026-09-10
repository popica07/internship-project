"""Bank structure: business units and the weighted draws built on them.

Nothing here models a real institution. It is a parameterised structure whose
every knob lives in `config.py` (CLAUDE.md rule 7).
"""

import numpy as np

from src.simulator.config import BusinessUnit, SimulatorConfig


def weighted_choice(rng: np.random.Generator, pairs: tuple[tuple[str, float], ...], size: int) -> np.ndarray:
    """Draw `size` labels from (label, weight) pairs, weights renormalised."""
    labels = [label for label, _ in pairs]
    weights = np.array([weight for _, weight in pairs], dtype=float)
    return rng.choice(labels, size=size, p=weights / weights.sum())


def draw_business_units(rng: np.random.Generator, config: SimulatorConfig, size: int) -> np.ndarray:
    weights = np.array([bu.weight for bu in config.business_units], dtype=float)
    names = [bu.name for bu in config.business_units]
    return rng.choice(names, size=size, p=weights / weights.sum())


def business_unit_index(config: SimulatorConfig) -> dict[str, BusinessUnit]:
    return {bu.name: bu for bu in config.business_units}
