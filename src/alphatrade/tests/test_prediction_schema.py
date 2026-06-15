import numpy as np
import pandas as pd

from alphatrade import prediction_schema


def test_required_prediction_columns_are_stable():
  cols = prediction_schema.required_columns()
  assert cols[:3] == ["symbol", "eob", "model_version"]
  assert cols[3] == "h1_q10"
  assert cols[-1] == "h60_q90"
  assert len(cols) == 23


def test_build_prediction_frame_validates_shape_and_types():
  n_rows = 3
  predictions = {
      h: np.zeros((n_rows, len(prediction_schema.DEFAULT_QUANTILES)), dtype=np.float32)
      for h in prediction_schema.DEFAULT_HORIZONS
  }
  df = prediction_schema.build_prediction_frame(
      symbols=["DCE.JM"] * n_rows,
      eobs=pd.date_range("2024-01-02 09:00:00", periods=n_rows, freq="1min"),
      model_version="model_v1",
      predictions_by_horizon=predictions,
  )

  assert list(df.columns) == prediction_schema.required_columns()
  assert df["h20_q50"].dtype == np.float64
  assert prediction_schema.validate_prediction_frame(df, expected_model_version="model_v1") == []


def test_validate_prediction_frame_catches_missing_column():
  n_rows = 1
  predictions = {
      h: np.zeros((n_rows, len(prediction_schema.DEFAULT_QUANTILES)), dtype=np.float32)
      for h in prediction_schema.DEFAULT_HORIZONS
  }
  df = prediction_schema.build_prediction_frame(
      symbols=["DCE.JM"],
      eobs=["2024-01-02 09:00:00"],
      model_version="model_v1",
      predictions_by_horizon=predictions,
  ).drop(columns=["h1_q10"])

  issues = prediction_schema.validate_prediction_frame(df)
  assert issues[0].field == "columns"
  assert "h1_q10" in issues[0].message


def test_validate_prediction_frame_catches_model_version_mismatch():
  n_rows = 1
  predictions = {
      h: np.zeros((n_rows, len(prediction_schema.DEFAULT_QUANTILES)), dtype=np.float32)
      for h in prediction_schema.DEFAULT_HORIZONS
  }
  df = prediction_schema.build_prediction_frame(
      symbols=["DCE.JM"],
      eobs=["2024-01-02 09:00:00"],
      model_version="model_v1",
      predictions_by_horizon=predictions,
  )

  issues = prediction_schema.validate_prediction_frame(df, expected_model_version="model_v2")
  assert any(issue.field == "model_version" for issue in issues)


def test_validate_prediction_frame_catches_duplicate_keys():
  n_rows = 2
  predictions = {
      h: np.zeros((n_rows, len(prediction_schema.DEFAULT_QUANTILES)), dtype=np.float32)
      for h in prediction_schema.DEFAULT_HORIZONS
  }
  df = prediction_schema.build_prediction_frame(
      symbols=["DCE.JM", "DCE.JM"],
      eobs=["2024-01-02 09:00:00", "2024-01-02 09:00:00"],
      model_version="model_v1",
      predictions_by_horizon=predictions,
  )

  issues = prediction_schema.validate_prediction_frame(df)
  assert any(issue.field == "duplicates" for issue in issues)


def test_validate_prediction_frame_catches_nonfinite_predictions():
  n_rows = 1
  predictions = {
      h: np.zeros((n_rows, len(prediction_schema.DEFAULT_QUANTILES)), dtype=np.float32)
      for h in prediction_schema.DEFAULT_HORIZONS
  }
  df = prediction_schema.build_prediction_frame(
      symbols=["DCE.JM"],
      eobs=["2024-01-02 09:00:00"],
      model_version="model_v1",
      predictions_by_horizon=predictions,
  )
  df.loc[0, "h1_q50"] = np.inf

  issues = prediction_schema.validate_prediction_frame(df)
  assert any(issue.field == "prediction_nonfinite" for issue in issues)


def test_validate_prediction_frame_catches_quantile_crossing():
  n_rows = 1
  predictions = {
      h: np.zeros((n_rows, len(prediction_schema.DEFAULT_QUANTILES)), dtype=np.float32)
      for h in prediction_schema.DEFAULT_HORIZONS
  }
  df = prediction_schema.build_prediction_frame(
      symbols=["DCE.JM"],
      eobs=["2024-01-02 09:00:00"],
      model_version="model_v1",
      predictions_by_horizon=predictions,
  )
  df.loc[0, "h1_q10"] = 1.0
  df.loc[0, "h1_q30"] = 0.0

  issues = prediction_schema.validate_prediction_frame(df)
  assert any(issue.field == "quantile_crossing" for issue in issues)
