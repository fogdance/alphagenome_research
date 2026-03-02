"""Regression tests for validate_reports_schema.py (manifest + profile)."""

import json
import os
import subprocess
import sys
import textwrap

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures: minimal valid reports matching real schemas
# ---------------------------------------------------------------------------

_TRAIN_METRICS_VALID = {
    "run": {"run_id": "abc123", "git_sha": "deadbeef", "created_at": "2026-01-01T00:00:00", "device": "cpu", "seed": 42},
    "dataset": {"name": "test", "symbols": 1, "train_samples": 100, "val_samples": 10, "feature_dim": 2, "feature_cols": ["a", "b"], "lookback": 10, "horizons": [1], "quantiles": [0.5]},
    "model": {"type": "test_model", "total_params": 100, "trainable_params": 100},
    "training": {"max_steps": 10, "batch_size": 8, "learning_rate": 0.001, "grad_clip": 1.0, "optimizer": "adam"},
    "loss": {"train_last": 0.1, "train_best": 0.05, "val_last": 0.12, "val_best": 0.06, "best_step": 5, "by_horizon": {"h1": {"train": 0.1, "val": 0.12}, "h5": {"train": 0.1, "val": 0.12}, "h20": {"train": 0.1, "val": 0.12}, "h60": {"train": 0.1, "val": 0.12}}},
    "stability": {"nan_steps": 0, "inf_steps": 0, "grad_norm_pre_clip_max": 1.0, "grad_norm_post_clip_max": 0.5},
}

_EVAL_METRICS_VALID = {
    "run_id": "abc123",
    "eval_timestamp": "2026-01-01T01:00:00",
    "model": {"source": "checkpoint", "checkpoint_dir": "/tmp/ckpt", "checkpoint_step": 5, "train_run_id": "abc123"},
    "dataset": {"split": "val", "symbols": 1, "samples": 10},
    "pinball_loss": {"overall": 0.05, "by_horizon": {"h1": 0.04, "h5": 0.05, "h20": 0.06, "h60": 0.07}},
    "quantile_coverage": {"q10": 0.1, "q30": 0.3, "q50": 0.5, "q70": 0.7, "q90": 0.9},
    "quantile_crossing": {"rate": 0.0, "count": 0},
}


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _write_manifest(manifest_path, profiles):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        yaml.dump({"version": 1, "profiles": profiles}, f)


def _schemas_dir():
    """Return the real schemas dir (relative to repo root)."""
    return os.path.join(os.path.dirname(__file__), "..", "schemas")


def _run_validator(args: list[str], cwd=None) -> subprocess.CompletedProcess:
    """Run the validator as a subprocess."""
    cmd = [sys.executable, "src/alphatrade/scripts/validate_reports_schema.py"] + args
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or _repo_root())


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


# ---------------------------------------------------------------------------
# Unit tests (import the module directly)
# ---------------------------------------------------------------------------

@pytest.fixture
def schemas_dir():
    return os.path.abspath(_schemas_dir())


class TestValidateReport:
    """Test validate_report() function directly."""

    def test_valid_train_report(self, tmp_path, schemas_dir):
        from alphatrade.scripts.validate_reports_schema import validate_report

        report_path = str(tmp_path / "train.json")
        _write_json(report_path, _TRAIN_METRICS_VALID)
        schema_path = os.path.join(schemas_dir, "m2_train_metrics.schema.json")

        result = validate_report(report_path, schema_path)
        assert result["status"] == "pass"

    def test_valid_eval_report(self, tmp_path, schemas_dir):
        from alphatrade.scripts.validate_reports_schema import validate_report

        report_path = str(tmp_path / "eval.json")
        _write_json(report_path, _EVAL_METRICS_VALID)
        schema_path = os.path.join(schemas_dir, "m4_eval_metrics.schema.json")

        result = validate_report(report_path, schema_path)
        assert result["status"] == "pass"

    def test_missing_required_field_fails(self, tmp_path, schemas_dir):
        from alphatrade.scripts.validate_reports_schema import validate_report

        bad = dict(_TRAIN_METRICS_VALID)
        bad = {k: v for k, v in bad.items() if k != "stability"}
        report_path = str(tmp_path / "bad.json")
        _write_json(report_path, bad)
        schema_path = os.path.join(schemas_dir, "m2_train_metrics.schema.json")

        result = validate_report(report_path, schema_path)
        assert result["status"] == "fail"

    def test_missing_report_file(self, schemas_dir):
        from alphatrade.scripts.validate_reports_schema import validate_report

        result = validate_report("/nonexistent/path.json", os.path.join(schemas_dir, "m2_train_metrics.schema.json"))
        assert result["status"] == "missing_report"

    def test_empty_schema_checks_existence_only(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import validate_report

        report_path = str(tmp_path / "any.json")
        _write_json(report_path, {"anything": True})

        result = validate_report(report_path, "")
        assert result["status"] == "pass"


class TestManifestLoading:
    """Test manifest loading and profile extraction."""

    def test_load_manifest(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import load_manifest, get_profile_items

        manifest_path = str(tmp_path / "manifest.yaml")
        profiles = {
            "test_profile": {
                "description": "Test",
                "reports_dir": "reports",
                "items": [
                    {"name": "item1", "path": "a.json", "schema": "s.json", "required": True},
                ],
            }
        }
        _write_manifest(manifest_path, profiles)

        manifest = load_manifest(manifest_path)
        assert manifest is not None
        items = get_profile_items(manifest, "test_profile")
        assert len(items) == 1
        assert items[0]["name"] == "item1"

    def test_missing_manifest_returns_none(self):
        from alphatrade.scripts.validate_reports_schema import load_manifest

        result = load_manifest("/nonexistent/manifest.yaml")
        assert result is None

    def test_missing_profile_returns_none(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import load_manifest, get_profile_items

        manifest_path = str(tmp_path / "manifest.yaml")
        _write_manifest(manifest_path, {"only_profile": {"items": []}})

        manifest = load_manifest(manifest_path)
        assert get_profile_items(manifest, "nonexistent") is None


class TestProfileIsolation:
    """Profile isolation: legacy failures don't affect m4."""

    def test_m4_pass_despite_legacy_fail(self, tmp_path, schemas_dir):
        from alphatrade.scripts.validate_reports_schema import validate_report

        # M4 seed report is valid
        report_path = str(tmp_path / "m4_train_metrics_seed42.json")
        _write_json(report_path, _TRAIN_METRICS_VALID)
        schema_path = os.path.join(schemas_dir, "m2_train_metrics.schema.json")
        result = validate_report(report_path, schema_path)
        assert result["status"] == "pass"

        # Legacy report is broken (missing required field)
        legacy_path = str(tmp_path / "m2_train_metrics.json")
        bad = {k: v for k, v in _TRAIN_METRICS_VALID.items() if k != "stability"}
        _write_json(legacy_path, bad)
        legacy_result = validate_report(legacy_path, schema_path)
        assert legacy_result["status"] == "fail"

        # Key point: these are independent — m4 profile doesn't include legacy items


class TestSemanticChecks:
    """Test Phase 2 semantic cross-validation."""

    def test_valid_pair_passes(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import semantic_check_m4

        eval_path = str(tmp_path / "eval.json")
        train_path = str(tmp_path / "train.json")
        _write_json(eval_path, _EVAL_METRICS_VALID)
        _write_json(train_path, _TRAIN_METRICS_VALID)

        checks = semantic_check_m4(eval_path, train_path)
        assert all(c["status"] == "pass" for c in checks)

    def test_mismatched_run_id_fails(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import semantic_check_m4

        eval_data = dict(_EVAL_METRICS_VALID)
        eval_data["model"] = dict(eval_data["model"])
        eval_data["model"]["train_run_id"] = "wrong_id"

        eval_path = str(tmp_path / "eval.json")
        train_path = str(tmp_path / "train.json")
        _write_json(eval_path, eval_data)
        _write_json(train_path, _TRAIN_METRICS_VALID)

        checks = semantic_check_m4(eval_path, train_path)
        run_id_check = [c for c in checks if "train_run_id" in c["check"]][0]
        assert run_id_check["status"] == "fail"


# ---------------------------------------------------------------------------
# Integration tests (run as subprocess)
# ---------------------------------------------------------------------------

class TestIntegrationStrict:
    """Integration tests running validator as a subprocess with custom manifests."""

    def test_strict_all_present_exit_0(self, tmp_path):
        reports_dir = str(tmp_path / "reports")
        schemas_dir_abs = os.path.abspath(_schemas_dir())
        manifest_path = str(tmp_path / "manifest.yaml")

        # Write valid reports for seeds 42, 43, 44
        for seed in [42, 43, 44]:
            train = dict(_TRAIN_METRICS_VALID)
            train["run"] = dict(train["run"])
            train["run"]["seed"] = seed
            train["run"]["run_id"] = f"run_{seed}"
            _write_json(os.path.join(reports_dir, f"m4_train_metrics_seed{seed}.json"), train)

            eval_data = dict(_EVAL_METRICS_VALID)
            eval_data["run_id"] = f"run_{seed}"
            eval_data["model"] = dict(eval_data["model"])
            eval_data["model"]["train_run_id"] = f"run_{seed}"
            _write_json(os.path.join(reports_dir, f"m4_eval_metrics_seed{seed}.json"), eval_data)

        # Write manifest pointing to real schemas
        profiles = {
            "m4": {
                "description": "Test m4",
                "reports_dir": reports_dir,
                "items": [
                    {"name": f"m4_train_metrics_seed{s}", "path": f"m4_train_metrics_seed{s}.json",
                     "schema": os.path.join(schemas_dir_abs, "m2_train_metrics.schema.json"), "required": True}
                    for s in [42, 43, 44]
                ] + [
                    {"name": f"m4_eval_metrics_seed{s}", "path": f"m4_eval_metrics_seed{s}.json",
                     "schema": os.path.join(schemas_dir_abs, "m4_eval_metrics.schema.json"), "required": True}
                    for s in [42, 43, 44]
                ],
            }
        }
        _write_manifest(manifest_path, profiles)

        result = _run_validator([
            "--manifest", manifest_path,
            "--profile", "m4",
            "--strict",
            "--reports-dir", reports_dir,
            "--output-json", str(tmp_path / "out.json"),
            "--output-md", str(tmp_path / "out.md"),
        ])
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

        # Check JSON output
        with open(tmp_path / "out.json") as f:
            report = json.load(f)
        assert report["summary"]["overall"] == "pass"
        assert report["summary"]["schema_required_fail"] == 0

    def test_strict_missing_required_exit_1(self, tmp_path):
        reports_dir = str(tmp_path / "reports")
        os.makedirs(reports_dir, exist_ok=True)
        schemas_dir_abs = os.path.abspath(_schemas_dir())
        manifest_path = str(tmp_path / "manifest.yaml")

        # Only write seed42, skip 43 and 44
        _write_json(os.path.join(reports_dir, "m4_train_metrics_seed42.json"), _TRAIN_METRICS_VALID)

        profiles = {
            "m4": {
                "description": "Test m4",
                "reports_dir": reports_dir,
                "items": [
                    {"name": "m4_train_metrics_seed42", "path": "m4_train_metrics_seed42.json",
                     "schema": os.path.join(schemas_dir_abs, "m2_train_metrics.schema.json"), "required": True},
                    {"name": "m4_train_metrics_seed43", "path": "m4_train_metrics_seed43.json",
                     "schema": os.path.join(schemas_dir_abs, "m2_train_metrics.schema.json"), "required": True},
                ],
            }
        }
        _write_manifest(manifest_path, profiles)

        result = _run_validator([
            "--manifest", manifest_path,
            "--profile", "m4",
            "--strict",
            "--reports-dir", reports_dir,
            "--output-json", str(tmp_path / "out.json"),
            "--output-md", str(tmp_path / "out.md"),
        ])
        assert result.returncode == 1

    def test_legacy_missing_does_not_affect_m4(self, tmp_path):
        """Profile isolation: running m4 profile ignores legacy files entirely."""
        reports_dir = str(tmp_path / "reports")
        schemas_dir_abs = os.path.abspath(_schemas_dir())
        manifest_path = str(tmp_path / "manifest.yaml")

        # Write one valid m4 item
        _write_json(os.path.join(reports_dir, "m4_train_seed42.json"), _TRAIN_METRICS_VALID)

        profiles = {
            "m4": {
                "description": "m4",
                "reports_dir": reports_dir,
                "items": [
                    {"name": "m4_train_seed42", "path": "m4_train_seed42.json",
                     "schema": os.path.join(schemas_dir_abs, "m2_train_metrics.schema.json"), "required": True},
                ],
            },
            "legacy_m2_m3": {
                "description": "legacy",
                "reports_dir": reports_dir,
                "items": [
                    {"name": "m2_train", "path": "m2_train.json",
                     "schema": os.path.join(schemas_dir_abs, "m2_train_metrics.schema.json"), "required": False},
                ],
            },
        }
        _write_manifest(manifest_path, profiles)

        # m4 should pass even though legacy file doesn't exist
        result = _run_validator([
            "--manifest", manifest_path,
            "--profile", "m4",
            "--strict",
            "--reports-dir", reports_dir,
            "--output-json", str(tmp_path / "out.json"),
            "--output-md", str(tmp_path / "out.md"),
        ])
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
