import numpy as np
import pandas as pd


def _bars_frame(batch_infer_offline, n_rows=8):
  data = {
      "eob": pd.date_range("2024-01-02 09:01:00", periods=n_rows, freq="1min"),
  }
  for col in batch_infer_offline.FEATURE_COLS:
    data[col] = np.zeros(n_rows, dtype=np.float32)
  return pd.DataFrame(data)


def test_build_sliding_windows_treats_date_only_end_as_full_day(monkeypatch):
  monkeypatch.setenv("ALPHATRADE_DEVICE", "cpu")
  from alphatrade.scripts import batch_infer_offline

  bars = _bars_frame(batch_infer_offline)
  windows, eobs = batch_infer_offline.build_sliding_windows(
      bars,
      lookback=3,
      start="2024-01-02",
      end="2024-01-02",
  )

  assert windows.shape == (6, 3, batch_infer_offline.FEATURE_DIM)
  assert eobs[0] == pd.Timestamp("2024-01-02 09:03:00")


def test_build_sliding_windows_keeps_precise_end_timestamp(monkeypatch):
  monkeypatch.setenv("ALPHATRADE_DEVICE", "cpu")
  from alphatrade.scripts import batch_infer_offline

  bars = _bars_frame(batch_infer_offline)
  windows, eobs = batch_infer_offline.build_sliding_windows(
      bars,
      lookback=3,
      start="2024-01-02 09:01:00",
      end="2024-01-02 09:04:00",
  )

  assert windows.shape == (2, 3, batch_infer_offline.FEATURE_DIM)
  assert eobs[-1] == pd.Timestamp("2024-01-02 09:04:00")
