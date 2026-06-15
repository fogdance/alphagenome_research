from __future__ import annotations

import numpy as np
import pandas as pd

from alphatrade import prediction_schema
from alphatrade.scripts import build_m10_prediction_eval as m10


def _write_bars(root, symbol: str, closes: list[float]):
    symbol_dir = root / symbol
    symbol_dir.mkdir(parents=True)
    bars = pd.DataFrame(
        {
            "eob": pd.date_range("2024-01-02 09:00:00", periods=len(closes), freq="1min"),
            "close": np.asarray(closes, dtype=np.float64),
        }
    )
    bars.to_parquet(symbol_dir / "bars.parquet", index=False)
    return bars


def _predictions(symbol: str, eobs, horizons=(1,), quantiles=(0.1, 0.5, 0.9)):
    n_rows = len(eobs)
    preds = {}
    for h in horizons:
        arr = np.zeros((n_rows, len(quantiles)), dtype=np.float32)
        for i, q in enumerate(quantiles):
            arr[:, i] = float(q) * 0.001
        preds[int(h)] = arr
    return prediction_schema.build_prediction_frame(
        symbols=[symbol] * n_rows,
        eobs=eobs,
        model_version="model_v1",
        predictions_by_horizon=preds,
        horizons=horizons,
        quantiles=quantiles,
    )


def test_target_alignment_on_tiny_bars(tmp_path):
    bars = _write_bars(tmp_path, "DCE.JM", [100.0, 101.0, 102.0, 103.0])
    predictions = _predictions(
        "DCE.JM",
        [bars.loc[1, "eob"], bars.loc[3, "eob"]],
        horizons=(1,),
        quantiles=(0.1, 0.5, 0.9),
    )

    rows, meta = m10.align_predictions_with_realized(
        predictions,
        data_dir=tmp_path,
        horizons=[1],
        quantiles=[0.1, 0.5, 0.9],
        rolling_window_rows=10,
        min_rolling_samples=1,
    )

    assert meta["aligned_rows"] == 1
    assert meta["dropped_unavailable_target_rows"] == 1
    assert np.isclose(rows.loc[0, "realized_h1"], np.log(102.0 / 101.0))
    assert meta["unavailable_targets_by_horizon"]["h1"] == 1


def test_rolling_historical_quantile_uses_only_past_completed_targets(tmp_path):
    bars = _write_bars(tmp_path, "DCE.JM", [100.0, 101.0, 102.0, 103.0, 1000.0])
    predictions = _predictions(
        "DCE.JM",
        [bars.loc[3, "eob"]],
        horizons=(1,),
        quantiles=(0.1, 0.5, 0.9),
    )

    rows, _ = m10.align_predictions_with_realized(
        predictions,
        data_dir=tmp_path,
        horizons=[1],
        quantiles=[0.1, 0.5, 0.9],
        rolling_window_rows=10,
        min_rolling_samples=1,
    )

    assert rows.loc[0, "rolling_h1_q90"] < 0.02
    assert rows.loc[0, "realized_h1"] > 2.0


def test_quantile_coverage_calculation():
    rows = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "A"],
            "eob": pd.date_range("2024-01-01", periods=4),
            "model_version": ["m"] * 4,
            "realized_h1": [-1.0, 0.0, 1.0, 2.0],
            "h1_q10": [-2.0, -2.0, -2.0, -2.0],
            "h1_q50": [0.5, 0.5, 0.5, 0.5],
            "h1_q90": [3.0, 3.0, 3.0, 3.0],
        }
    )

    coverage = m10.compute_quantile_coverage(rows, [1], [0.1, 0.5, 0.9])

    assert coverage["by_horizon"]["h1"]["coverage"]["q10"]["observed"] == 0.0
    assert coverage["by_horizon"]["h1"]["coverage"]["q50"]["observed"] == 0.5
    assert coverage["by_horizon"]["h1"]["coverage"]["q90"]["observed"] == 1.0


def test_quantile_crossing_calculation():
    rows = pd.DataFrame(
        {
            "h1_q10": [0.0, 1.0],
            "h1_q50": [0.5, 0.0],
            "h1_q90": [1.0, 2.0],
        }
    )

    crossing = m10.compute_quantile_crossing(rows, [1], [0.1, 0.5, 0.9])

    assert crossing["count"] == 1
    assert crossing["by_horizon"]["h1"]["pair_count"] == 4
