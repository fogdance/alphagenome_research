from __future__ import annotations

import pytest

from alphatrade.scripts import gen_m6_baseline_report


def test_find_baseline_experiment_uses_exp_id_not_rank_order() -> None:
    leaderboard = {
        "experiments": [
            {"exp_id": "winner", "metrics": {"primary_mean": 0.1}},
            {"exp_id": "baseline", "metrics": {"primary_mean": 0.2}},
        ]
    }

    exp = gen_m6_baseline_report.find_baseline_experiment(leaderboard, "baseline")

    assert exp["exp_id"] == "baseline"


def test_find_baseline_experiment_fails_when_missing() -> None:
    leaderboard = {"experiments": [{"exp_id": "winner"}]}

    with pytest.raises(ValueError, match="baseline experiment 'baseline' not found"):
        gen_m6_baseline_report.find_baseline_experiment(leaderboard, "baseline")
