import pytest


def test_prepare_bundle_dir_overwrite_clears_stale_files(tmp_path):
  from alphatrade.scripts.export_model_bundle import prepare_bundle_dir

  output_dir = tmp_path / "bundles"
  bundle_dir = output_dir / "model_v1"
  bundle_dir.mkdir(parents=True)
  stale_file = bundle_dir / "stale.txt"
  stale_file.write_text("old", encoding="utf-8")

  prepared = prepare_bundle_dir(str(output_dir), "model_v1", overwrite=True)

  assert prepared == bundle_dir.resolve()
  assert prepared.exists()
  assert not stale_file.exists()
  assert list(prepared.iterdir()) == []


def test_prepare_bundle_dir_no_overwrite_fails_on_existing_bundle(tmp_path):
  from alphatrade.scripts.export_model_bundle import prepare_bundle_dir

  output_dir = tmp_path / "bundles"
  (output_dir / "model_v1").mkdir(parents=True)

  with pytest.raises(SystemExit):
    prepare_bundle_dir(str(output_dir), "model_v1", overwrite=False)


def test_validate_model_version_rejects_path_segments():
  from alphatrade.scripts.export_model_bundle import validate_model_version

  with pytest.raises(SystemExit):
    validate_model_version("../model")
  with pytest.raises(SystemExit):
    validate_model_version("")


def test_batch_infer_auto_device_normalizes_to_gpu(monkeypatch):
  monkeypatch.setenv("ALPHATRADE_DEVICE", "cpu")
  from alphatrade.scripts import batch_infer_offline

  assert batch_infer_offline._normalize_requested_device("auto") == "gpu"
  assert batch_infer_offline._normalize_requested_device("cuda") == "gpu"
  assert batch_infer_offline._normalize_requested_device("cpu") == "cpu"


def test_validator_sha256_file(tmp_path):
  from alphatrade.scripts.validate_reports_schema import sha256_file

  path = tmp_path / "payload.bin"
  path.write_bytes(b"abc")

  assert sha256_file(path) == (
      "ba7816bf8f01cfea414140de5dae2223"
      "b00361a396177a9cb410ff61f20015ad"
  )
