import numpy as np
import pandas as pd

from alphatrade import prediction_schema
from alphatrade.scripts.backtest_predictions import compute_backtest


def _prediction_frame(symbol: str, forecasts: list[float]) -> pd.DataFrame:
  n_rows = len(forecasts)
  predictions = {}
  for horizon in prediction_schema.DEFAULT_HORIZONS:
    values = np.zeros((n_rows, len(prediction_schema.DEFAULT_QUANTILES)), dtype=np.float32)
    if horizon == 1:
      values[:, 2] = forecasts
    predictions[horizon] = values
  return prediction_schema.build_prediction_frame(
      symbols=[symbol] * n_rows,
      eobs=pd.date_range("2024-01-02 09:00:00", periods=n_rows, freq="1min"),
      model_version="model_v1",
      predictions_by_horizon=predictions,
  )


def test_compute_backtest_aligns_predictions_to_future_close(tmp_path):
  symbol = "DCE.JM"
  symbol_dir = tmp_path / symbol
  symbol_dir.mkdir(parents=True)
  bars = pd.DataFrame(
      {
          "eob": pd.date_range("2024-01-02 09:00:00", periods=5, freq="1min"),
          "close": [100.0, 101.0, 102.0, 104.0, 103.0],
      }
  )
  bars.to_parquet(symbol_dir / "bars.parquet", index=False)
  predictions = _prediction_frame(symbol, [0.01, -0.01, 0.0])

  metrics, trades = compute_backtest(
      predictions,
      data_dir=tmp_path,
      horizon=1,
      quantile=0.5,
      threshold=0.0,
      cost_bps=0.0,
  )

  assert metrics["data"]["n_predictions"] == 3
  assert metrics["data"]["n_matched"] == 3
  assert metrics["model_versions"] == ["model_v1"]
  assert len(trades) == 3
  assert trades["position"].tolist() == [1.0, -1.0, 0.0]
  expected = np.log(101.0 / 100.0) - np.log(102.0 / 101.0)
  assert np.isclose(metrics["performance"]["sum_gross_log_return"], expected)


def test_compute_backtest_reports_missing_symbols(tmp_path):
  predictions = _prediction_frame("DCE.JM", [0.01])

  try:
    compute_backtest(
        predictions,
        data_dir=tmp_path,
        horizon=1,
        quantile=0.5,
        threshold=0.0,
        cost_bps=0.0,
    )
  except ValueError as exc:
    assert "zero matched samples" in str(exc)
  else:
    raise AssertionError("expected zero matched samples failure")
