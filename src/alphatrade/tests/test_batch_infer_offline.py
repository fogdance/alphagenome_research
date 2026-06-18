import json

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


def test_smoke_limit_per_symbol_distributes_cap(monkeypatch):
  monkeypatch.setenv("ALPHATRADE_DEVICE", "cpu")
  from alphatrade.scripts import batch_infer_offline

  assert batch_infer_offline.smoke_limit_per_symbol(2, total_limit=100) == 50
  assert batch_infer_offline.smoke_limit_per_symbol(3, total_limit=100) == 33
  assert batch_infer_offline.smoke_limit_per_symbol(0, total_limit=100) == 0


def test_default_infer_batch_size_is_full_run_oriented(monkeypatch):
  monkeypatch.setenv("ALPHATRADE_DEVICE", "cpu")
  from alphatrade.scripts import batch_infer_offline

  assert batch_infer_offline._DEFAULT_INFER_BATCH_SIZE >= 2048


def test_progress_event_writes_jsonl(tmp_path, monkeypatch):
  monkeypatch.setenv("ALPHATRADE_DEVICE", "cpu")
  from alphatrade.scripts import batch_infer_offline

  path = tmp_path / "progress.jsonl"
  batch_infer_offline._write_progress_event(
      path,
      {
          "event": "batch_progress",
          "symbol": "DCE.JM",
          "total_rows": 2048,
          "rss_mb": 123.4,
      },
  )

  event = json.loads(path.read_text(encoding="utf-8").strip())
  assert event["schema_version"] == "m9_infer_progress_v1"
  assert event["event"] == "batch_progress"
  assert event["symbol"] == "DCE.JM"
  assert event["total_rows"] == 2048


def test_iter_sliding_window_batches_matches_full_builder(monkeypatch):
  monkeypatch.setenv("ALPHATRADE_DEVICE", "cpu")
  from alphatrade.scripts import batch_infer_offline

  bars = _bars_frame(batch_infer_offline, n_rows=12)
  full_windows, full_eobs = batch_infer_offline.build_sliding_windows(
      bars,
      lookback=3,
      start="2024-01-02",
      end="2024-01-02",
  )
  batches = list(
      batch_infer_offline.iter_sliding_window_batches(
          bars,
          lookback=3,
          start="2024-01-02",
          end="2024-01-02",
          batch_size=4,
      )
  )

  batch_windows = np.concatenate([b[0] for b in batches], axis=0)
  batch_eobs = [eob for _, eobs in batches for eob in eobs]
  np.testing.assert_array_equal(batch_windows, full_windows)
  assert batch_eobs == full_eobs


def test_build_sliding_windows_accepts_dynamic_feature_cols(monkeypatch):
  monkeypatch.setenv("ALPHATRADE_DEVICE", "cpu")
  from alphatrade.scripts import batch_infer_offline

  bars = _bars_frame(batch_infer_offline, n_rows=8)
  bars["chg_h1_distance"] = np.arange(len(bars), dtype=np.float32)
  feature_cols = list(batch_infer_offline.FEATURE_COLS) + ["chg_h1_distance"]

  windows, _ = batch_infer_offline.build_sliding_windows(
      bars,
      lookback=3,
      start="2024-01-02",
      end="2024-01-02",
      feature_cols=feature_cols,
  )

  assert windows.shape == (6, 3, len(feature_cols))


def test_verify_data_dir_feature_profile_matches_bundle(tmp_path, monkeypatch):
  monkeypatch.setenv("ALPHATRADE_DEVICE", "cpu")
  from alphatrade.data_pipeline.feature_profiles import (
      feature_profile_from_mapping,
      feature_profile_to_dict,
  )
  from alphatrade.scripts import batch_infer_offline

  profile = feature_profile_from_mapping(
      {
          "profile_id": "m12_chg_core",
          "feature_dim": 2,
          "feature_cols": ["a", "b"],
          "scaler_hash": "scale123",
      }
  )
  (tmp_path / "feature_manifest.json").write_text(
      json.dumps({"feature_profile": feature_profile_to_dict(profile)}),
      encoding="utf-8",
  )

  data_profile = batch_infer_offline.verify_data_dir_feature_profile(tmp_path, profile)

  assert data_profile.profile_id == "m12_chg_core"
  assert data_profile.feature_cols == ("a", "b")


def test_verify_data_dir_feature_profile_rejects_mismatch(tmp_path, monkeypatch):
  monkeypatch.setenv("ALPHATRADE_DEVICE", "cpu")
  from alphatrade.data_pipeline.feature_profiles import (
      feature_profile_from_mapping,
      feature_profile_to_dict,
  )
  from alphatrade.scripts import batch_infer_offline

  bundle_profile = feature_profile_from_mapping(
      {"profile_id": "m12_chg_core", "feature_dim": 2, "feature_cols": ["a", "b"]}
  )
  data_profile = feature_profile_from_mapping(
      {"profile_id": "m12_chg_core", "feature_dim": 2, "feature_cols": ["a", "c"]}
  )
  (tmp_path / "feature_manifest.json").write_text(
      json.dumps({"feature_profile": feature_profile_to_dict(data_profile)}),
      encoding="utf-8",
  )

  try:
    batch_infer_offline.verify_data_dir_feature_profile(tmp_path, bundle_profile)
  except ValueError as exc:
    assert "feature_profile_mismatch" in str(exc)
  else:
    raise AssertionError("expected feature_profile_mismatch")
