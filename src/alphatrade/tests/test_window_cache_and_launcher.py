from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd
import pytest

from alphatrade import gpu_launcher
from alphatrade import window_cache
from alphatrade.scripts import eval_m4_fast
from alphatrade.scripts import run_m5_sweep
from alphatrade.scripts import train_m4_alphatrade


FEATURES = ["open", "high", "low", "close", "volume", "amount", "vwap", "oi"]


def _write_symbol(root, symbol: str):
    symbol_dir = root / symbol
    symbol_dir.mkdir(parents=True)
    bars = pd.DataFrame({name: np.arange(10, dtype=np.float32) for name in FEATURES})
    bars.to_parquet(symbol_dir / "bars.parquet")
    index = pd.DataFrame(
        {
            "x_start": [0, 1],
            "x_end": [2, 3],
            "y_h1": [0.1, 0.2],
            "y_h5": [0.3, 0.4],
            "y_h20": [0.5, 0.6],
            "y_h60": [0.7, 0.8],
        }
    )
    index.to_parquet(symbol_dir / "index_val.parquet")


def test_window_cache_build_and_hit(tmp_path):
    processed = tmp_path / "processed"
    cache_dir = tmp_path / "cache"
    _write_symbol(processed, "DCE.JM")

    first = window_cache.load_or_build(
        symbols=["DCE.JM"],
        processed_root=processed,
        split="val",
        features=FEATURES,
        cache_dir=cache_dir,
        mode="auto",
    )
    assert first.cache_hit is False
    assert first.x.shape == (2, 3, len(FEATURES))
    assert first.y.shape == (2, 4)
    assert first.symbols.tolist() == ["DCE.JM", "DCE.JM"]

    second = window_cache.load_or_build(
        symbols=["DCE.JM"],
        processed_root=processed,
        split="val",
        features=FEATURES,
        cache_dir=cache_dir,
        mode="auto",
        mmap=True,
    )
    assert second.cache_hit is True
    assert second.x.shape == first.x.shape
    assert second.metadata["fingerprint"] == first.metadata["fingerprint"]


def test_gpu_launcher_env_and_cmd(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bad/cuda")
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    cmd, env = gpu_launcher.build_module_cmd(
        "train_m4_alphatrade",
        ["--max-steps", "1"],
        gpu=True,
    )
    assert cmd[:2] == [os.sys.executable, "-c"]
    assert "from alphatrade.scripts.train_m4_alphatrade import main" in cmd[2]
    assert env["JAX_PLATFORMS"] == "cuda"
    assert env["MPLCONFIGDIR"] == "/tmp/matplotlib-alphatrade"
    assert "LD_LIBRARY_PATH" not in env

    cmd_cpu, env_cpu = gpu_launcher.build_module_cmd(
        "train_m4_alphatrade",
        ["--max-steps", "1"],
        gpu=False,
    )
    assert cmd_cpu[-2:] == ["--max-steps", "1"]
    assert env_cpu is None


def test_gpu_launcher_preserves_explicit_matplotlib_dir(monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/custom-mpl")
    env = gpu_launcher.gpu_env()
    assert env["MPLCONFIGDIR"] == "/tmp/custom-mpl"


def test_m5_sweep_maps_window_cache_args():
    train_cli = run_m5_sweep._build_cli_from_map(
        run_m5_sweep._TRAIN_PARAM_MAP,
        {"window_cache": "refresh", "window_cache_dir": "/tmp/windows"},
    )
    assert "--window-cache" in train_cli
    assert train_cli[train_cli.index("--window-cache") + 1] == "refresh"
    assert "--window-cache-dir" in train_cli
    assert train_cli[train_cli.index("--window-cache-dir") + 1] == "/tmp/windows"

    eval_cli = run_m5_sweep._build_cli_from_map(run_m5_sweep._EVAL_PARAM_MAP, {})
    assert "--window-cache" in eval_cli
    assert eval_cli[eval_cli.index("--window-cache") + 1] == "auto"


def test_m5_resume_detects_train_only_state(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    train_path = reports / "m5_baseline_seed42_train_metrics.json"
    train_path.write_text(json.dumps({"run": {"run_id": "abc123"}}))

    state = run_m5_sweep.get_run_state("baseline", 42, str(reports))
    assert state["train_complete"] is True
    assert state["eval_complete"] is False
    assert state["complete"] is False
    assert state["run_id"] == "abc123"


def test_m5_config_hash_uses_effective_defaults():
    base_config = {
        "expected_seeds": [42],
        "dataset_config": "configs/dataset/m2.yaml",
        "defaults": {"max_steps": 500, "batch_size": 128},
        "experiments": [{"exp_id": "baseline", "overrides": {}}],
    }
    changed_config = {
        **base_config,
        "defaults": {"max_steps": 1000, "batch_size": 128},
    }

    base_hash = run_m5_sweep.build_run_plan(base_config)[0]["config_hash"]
    changed_hash = run_m5_sweep.build_run_plan(changed_config)[0]["config_hash"]
    smoke_hash = run_m5_sweep.build_run_plan(base_config, smoke=True)[0]["config_hash"]

    assert base_hash != changed_hash
    assert base_hash != smoke_hash


def test_m5_resume_requires_matching_sweep_metadata(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    run = {
        "exp_id": "baseline",
        "seed": 42,
        "config_hash": "abc12345",
        "dataset_config": "configs/dataset/m2.yaml",
        "overrides": {"max_steps": 500},
        "smoke": False,
    }
    metadata = run_m5_sweep.sweep_metadata(run)

    train_path = reports / "m5_baseline_seed42_train_metrics.json"
    eval_path = reports / "m5_baseline_seed42_eval_metrics.json"
    train_path.write_text(json.dumps({"run": {"run_id": "run-1", "sweep": metadata}}))
    eval_path.write_text(json.dumps({"model": {"train_run_id": "run-1", "sweep": metadata}}))

    state = run_m5_sweep.get_run_state("baseline", 42, str(reports), run)
    assert state["complete"] is True

    stale_run = {**run, "config_hash": "deadbeef"}
    stale_state = run_m5_sweep.get_run_state("baseline", 42, str(reports), stale_run)
    assert stale_state["complete"] is False
    assert stale_state["train_complete"] is False
    assert stale_state["eval_complete"] is False
    assert stale_state["stale_reasons"]
    assert "config_hash" in stale_state["stale_reasons"][0]

    with pytest.raises(run_m5_sweep.ResumeConfigMismatch, match="resume_config_mismatch"):
        run_m5_sweep.fail_on_resume_config_mismatch(stale_state, "baseline", 42)


def test_m5_resume_mismatch_checks_eval_split_and_ckpt_step(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    run = {
        "exp_id": "baseline",
        "seed": 42,
        "config_hash": "abc12345",
        "dataset_config": "configs/dataset/m2.yaml",
        "universe": "cta_top20",
        "eval_split": "val",
        "ckpt_step": "best",
        "overrides": {"max_steps": 500, "eval_split": "val", "ckpt_step": "best"},
        "smoke": False,
    }
    metadata = run_m5_sweep.sweep_metadata(run)
    train_path = reports / "m5_baseline_seed42_train_metrics.json"
    eval_path = reports / "m5_baseline_seed42_eval_metrics.json"
    train_path.write_text(json.dumps({"run": {"run_id": "run-1", "sweep": metadata}}))
    eval_path.write_text(json.dumps({"model": {"train_run_id": "run-1", "sweep": metadata}}))

    changed = {**run, "eval_split": "test", "ckpt_step": "last"}
    changed["overrides"] = {**run["overrides"], "eval_split": "test", "ckpt_step": "last"}
    stale_state = run_m5_sweep.get_run_state("baseline", 42, str(reports), changed)

    assert stale_state["stale_reasons"]
    reason = stale_state["stale_reasons"][0]
    assert "eval_split" in reason
    assert "ckpt_step" in reason


def test_create_batches_raises_prefetch_errors():
    class BadDataset:
        def __len__(self):
            return 1

        def get_batch(self, indices):
            raise RuntimeError("device transfer failed")

    with pytest.raises(RuntimeError, match="Batch prefetch failed"):
        list(train_m4_alphatrade.create_batches(BadDataset(), batch_size=1))


def test_eval_rejects_empty_dataset():
    class EmptyDataset:
        symbols = ["DCE.JM"]
        split = "val"
        x = np.empty((0, 60, len(FEATURES)), dtype=np.float32)
        y = np.empty((0, 4), dtype=np.float32)
        sample_symbols = np.empty((0,), dtype="U32")

        def __len__(self):
            return 0

    with pytest.raises(ValueError, match="No samples available"):
        eval_m4_fast.evaluate_model_batched(
            lambda *args, **kwargs: None,
            params=None,
            state=None,
            dataset=EmptyDataset(),
            horizons=[1, 5, 20, 60],
            quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
        )
