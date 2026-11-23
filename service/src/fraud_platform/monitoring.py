"""Population stability index over reference/current numerical feature distributions."""

import argparse
import json
from pathlib import Path

import numpy as np


def population_stability_index(reference: list[float], current: list[float], bins: int = 10) -> float:
    if len(reference) < bins or len(current) < bins:
        raise ValueError("Each sample must have at least 'bins' observations")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Non-finite feature values are not permitted")
    # Bin boundaries are fitted on reference data only, then frozen for comparison.
    bounds = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1))[1:-1])
    r = np.bincount(np.searchsorted(bounds, reference, side="right"), minlength=len(bounds) + 1)
    c = np.bincount(np.searchsorted(bounds, current, side="right"), minlength=len(bounds) + 1)
    p = np.clip(r / r.sum(), 1e-6, 1)
    q = np.clip(c / c.sum(), 1e-6, 1)
    return float(np.sum((q - p) * np.log(q / p)))


def main():
    parser = argparse.ArgumentParser(description="Compare JSON arrays containing numeric values")
    parser.add_argument("reference", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--warn-at", type=float, default=0.2)
    args = parser.parse_args()
    score = population_stability_index(json.loads(args.reference.read_text()),
                                       json.loads(args.current.read_text()))
    print(json.dumps({"psi": score, "warn": score > args.warn_at}))


if __name__ == "__main__":
    main()
