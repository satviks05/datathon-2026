"""The sole scoring and selection implementation used by exports and the UI."""

import numpy as np

from . import config


def score_candidates(features, strategy):
    if strategy not in config.STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    scored = features.copy()
    report_percentile = scored["signal_report_count"].rank(method="average", pct=True)
    crash_percentile = scored["harm_crash_count"].rank(method="average", pct=True)
    scored["strategy"] = strategy
    if strategy == "linear_regression":
        scored["score"] = scored["predicted_harm_crashes"]
    else:
        scored["score"] = {"crash_first": crash_percentile,
                           "complaint_first": report_percentile}[strategy]
    return scored.sort_values(["score", "target_id"], ascending=[False, True], kind="stable")


def select_spaced(scored, blocked_neighbors, budget):
    """Indices refer to the fixed candidate catalog, preserved during sorting."""
    if budget not in config.BUDGETS:
        raise ValueError(f"Unsupported budget: {budget}")
    blocked = np.zeros(len(scored), dtype=bool)
    chosen = []
    for index in scored.index:
        if blocked[index]:
            continue
        chosen.append(index)
        blocked[index] = True
        blocked[blocked_neighbors[index]] = True
        if len(chosen) == budget:
            break
    selected = scored.loc[chosen].copy()
    selected["selected_rank"] = np.arange(1, len(selected) + 1)
    selected["budget"] = budget
    return selected
