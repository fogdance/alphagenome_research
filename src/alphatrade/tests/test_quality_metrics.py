from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alphatrade import quality_metrics
from alphatrade.scripts import eval_m4_fast


def test_target_scale_summary_and_warnings():
    y = np.array(
        [
            [0.001, 0.010],
            [-0.001, -0.010],
            [0.002, 0.020],
        ],
        dtype=np.float32,
    )

    summary = quality_metrics.target_scale_summary(y, [1, 5])

    assert summary["samples"] == 3
    assert summary["by_horizon"]["h1"]["samples"] == 3
    assert summary["by_horizon"]["h5"]["abs_p90"] > summary["by_horizon"]["h1"]["abs_p90"]
    assert summary["scale_ratio_abs_p90"] > 5.0
    assert summary["warnings"]


def test_loss_weight_report_records_pinball_residual_weights():
    report = quality_metrics.loss_weight_report(
        horizons=[1, 5],
        quantiles=[0.1, 0.5, 0.9],
        horizon_weights={1: 1.0, 5: 0.5},
        quantile_weights=[2.0, 1.0, 2.0],
        crossing_penalty_weight=0.1,
    )

    assert report["horizon_weights"] == {"h1": 1.0, "h5": 0.5}
    assert report["quantile_weights"]["q10"] == 2.0
    assert report["pinball_residual_weights"]["q90"]["under_prediction"] == 0.9
    assert np.isclose(report["pinball_residual_weights"]["q90"]["over_prediction"], 0.1)


def test_coverage_calibration_summary():
    summary = quality_metrics.coverage_calibration_summary(
        {
            "h1": {"q10": 0.2, "q50": 0.4},
            "h5": {"q10": 0.1, "q50": 0.6},
        },
        [0.1, 0.5],
    )

    assert summary["overall_mae"] > 0
    assert np.isclose(summary["by_horizon"]["h1"]["by_quantile"]["q10"]["abs_error"], 0.1)
    assert np.isclose(summary["worst"]["abs_error"], 0.1)


def test_quality_metrics_reject_invalid_weights():
    with pytest.raises(ValueError, match="horizon_weights"):
        quality_metrics.resolve_horizon_weights([-1.0, 1.0], [1, 5])
    with pytest.raises(ValueError, match="quantile_weights"):
        quality_metrics.resolve_quantile_weights([0.0, 0.0], [0.1, 0.9])


def test_coverage_calibration_records_worst_even_when_perfect():
    summary = quality_metrics.coverage_calibration_summary(
        {"h1": {"q50": 0.5}},
        [0.5],
    )

    assert summary["worst"]["horizon"] == "h1"
    assert summary["worst"]["quantile"] == "q50"
    assert summary["worst"]["abs_error"] == 0.0


def test_eval_reports_horizon_and_symbol_quality_layers():
    class Dataset:
        symbols = ["A", "B"]
        split = "val"
        x = np.zeros((2, 60, 8), dtype=np.float32)
        y = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        sample_symbols = np.array(["A", "B"])

        def __len__(self):
            return len(self.x)

    def model_apply_fn(params, state, rng, x_batch):
        batch = int(x_batch.shape[0])
        preds = np.full((batch, 1), 0.5, dtype=np.float32)
        return SimpleNamespace(log_return_quantiles={1: preds, 5: preds}), state

    metrics = eval_m4_fast.evaluate_model_batched(
        model_apply_fn,
        params=None,
        state=None,
        dataset=Dataset(),
        horizons=[1, 5],
        quantiles=[0.5],
        batch_size=1,
    )

    assert metrics["quantile_coverage"]["q50"] == 0.5
    assert np.isclose(metrics["coverage_calibration"]["overall_mae"], 0.0)
    assert set(metrics["by_horizon"]) == {"h1", "h5"}
    assert metrics["by_horizon"]["h1"]["samples"] == 2
    assert metrics["by_symbol"][0]["by_horizon"]["h1"]["samples"] == 1


def test_eval_reports_training_weighted_pinball_loss():
    class Dataset:
        symbols = ["A"]
        split = "val"
        x = np.zeros((1, 60, 8), dtype=np.float32)
        y = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        sample_symbols = np.array(["A"])

        def __len__(self):
            return len(self.x)

    def model_apply_fn(params, state, rng, x_batch):
        batch = int(x_batch.shape[0])
        preds = np.ones((batch, 2), dtype=np.float32)
        return SimpleNamespace(log_return_quantiles={1: preds}), state

    quantile_loss_report = quality_metrics.loss_weight_report(
        horizons=[1],
        quantiles=[0.1, 0.9],
        quantile_weights=[10.0, 1.0],
    )
    metrics = eval_m4_fast.evaluate_model_batched(
        model_apply_fn,
        params=None,
        state=None,
        dataset=Dataset(),
        horizons=[1],
        quantiles=[0.1, 0.9],
        batch_size=1,
        quantile_loss_report=quantile_loss_report,
    )

    assert np.isclose(metrics["pinball_loss"]["overall"], 0.5)
    assert metrics["weighted_pinball_loss"]["overall"] > metrics["pinball_loss"]["overall"]
    assert np.isclose(
        metrics["weighted_pinball_loss"]["by_horizon"]["h1"],
        (0.9 * 10.0 + 0.1) / 11.0,
    )
