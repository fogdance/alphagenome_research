"""Regression tests for validate_reports_schema.py (manifest + profile)."""

import copy
import json
import os
import shutil
import subprocess
import sys
import textwrap
import hashlib

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

_MG0_AUDIT_VALID = {
    "metadata": {
        "schema_version": "mg0.alphagenome_parity_audit.v1",
        "sprint": "MG0-A AlphaGenome Official Code Parity Audit",
        "generated_at": "2026-07-13T00:00:00Z",
        "audit_status": "FROZEN_WITH_NEEDS_VERIFICATION",
        "parity_verdict": "PARTIAL_PUBLISHED_CODE_PARITY",
        "gate_decision": "PASS_WITH_NEEDS_VERIFICATION",
        "official_commit": "d" * 40,
        "local_head": "e" * 40,
        "implementation_added": False,
    },
    "source_evidence": {
        "paper": "https://example.test/paper",
        "supplementary_information": "https://example.test/supplement",
        "supplementary_tables": "https://example.test/tables",
        "official_repository": "https://example.test/repository",
        "local_parity_checks": ["No source diff."],
        "inspected_files": [f"src/example_{index}.py" for index in range(10)],
        "evidence_boundaries": [
            "CODE_OBSERVED",
            "PUBLISHED_ONLY",
            "DESIGN_INFERENCE",
        ],
    },
    "component_map": [
        {
            "component": "sequence_encoder",
            "official_code_evidence": (
                "model.py:38 SequenceEncoder downsamples DNA and retains skips."
            ),
            "scientific_role": "Learn local sequence representations.",
            "market_genome_analogue": "Multi-scale causal market encoder.",
            "causal_modification": "Replace symmetric context with past-only operations.",
            "status": "CAUSALIZE",
            "risk": "Future leakage if padding or attention is non-causal.",
            "validation_test": "Prefix outputs remain invariant to future mutations.",
        }
    ],
    "non_negotiable_alpha_genome_principles": [
        "Long context with dense multi-task supervision"
    ],
    "non_isomorphic_components": ["Reverse-complement augmentation"],
    "uncertainties": [
        {
            "item": "Pretraining trainer",
            "status": "NEEDS_VERIFICATION",
            "evidence_gap": "The trainer is not published in the repository.",
            "next_verification": "Inspect official trainer source if released.",
        }
    ],
    "gate": {
        "overall": "PASS_WITH_NEEDS_VERIFICATION",
        "acceptance_checks": [
            {
                "check": "official_sources_inspected",
                "status": "PASS",
                "evidence": "The component row cites official source code.",
            }
        ],
    },
    "freeze": {
        "status": "FROZEN",
        "review_decision": "APPROVED_WITH_DOCUMENTED_GAPS",
        "frozen_at": "2026-07-13T14:10:05+08:00",
        "baseline_id": "MG0-A-v1",
        "git_tag": "mg0-a-alphagenome-parity-v1",
        "baseline_payload_sha256": (
            "bddd431775f26129a81580c5b09fd44c3dadd81a50b62d75f90ad75ed5774a83"
        ),
        "markdown_sha256": (
            "88291777fbecb555727e356bcc8a037b864688ecba12480c8ac2788b928268b0"
        ),
        "schema_sha256": (
            "6c8aa39cea89abe0afe643035796b150d46716d74159b838da6d9c42c3a14768"
        ),
        "validator_sha256": (
            "6a5a9973f97cca14a0a9643df98fa6ff1f99bee493a12063eab6d7a771417530"
        ),
        "component_count": 36,
        "uncertainty_count": 9,
        "non_negotiable_principle_count": 10,
        "non_isomorphic_component_count": 9,
        "official_commit": "232fc695d1eab27bac9e94bcd4b50499139ba4e1",
        "local_source_head": "ed738d92ab76df03b4d25ebb741e33048c9aae09",
        "official_source_tree": "27f86306d52edcbbc4152049f5856f6c1999a334",
        "change_control": "EXPLICIT_USER_APPROVAL_AND_NEW_SCHEMA_VERSION",
        "training_status": "DEFERRED_BY_USER",
        "gpu_status": "DEFERRED_UNTIL_EXPLICIT_USER_REACTIVATION",
        "gpu_revisit_not_before": "2026-07-14",
        "resume_requires_explicit_user_confirmation": True,
    },
    "execution": {
        "gpu_execution_policy": "DEFERRED_BY_USER",
        "gpu_workload_after_defer": False,
        "training_execution_policy": "DEFERRED_BY_USER",
        "training_workload_after_defer": False,
        "resume_requires_explicit_user_confirmation": True,
        "pre_defer_jax_test_attempt": "INTERRUPTED_WITHOUT_VERIFICATION",
        "full_model_execution": "DEFERRED_BY_USER",
        "checkpoint_download": "NOT_RUN",
        "dataset_download": "NOT_RUN",
        "market_genome_implementation": "NOT_STARTED",
        "partial_verification": ["Static source audit completed."],
    },
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


def _copy_mg0_canonical_reports(reports_dir):
    docs_dir = os.path.join(_repo_root(), "docs/alphaTrade/market_genome")
    shutil.copyfile(
        os.path.join(docs_dir, "MG0_ALPHAGENOME_PARITY_AUDIT.json"),
        os.path.join(reports_dir, "mg0_alphagenome_parity_audit.json"),
    )
    shutil.copyfile(
        os.path.join(docs_dir, "MG0_ALPHAGENOME_PARITY_AUDIT.md"),
        os.path.join(reports_dir, "mg0_alphagenome_parity_audit.md"),
    )


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


class TestMG0Schema:
    """MG0-A schema and manifest profile contract tests."""

    def test_schema_is_valid_draft7(self, schemas_dir):
        from jsonschema import Draft7Validator

        schema_path = os.path.join(
            schemas_dir, "mg0_alphagenome_parity_audit.schema.json"
        )
        with open(schema_path) as f:
            schema = json.load(f)

        Draft7Validator.check_schema(schema)

    def test_minimal_audit_report_validates(self, tmp_path, schemas_dir):
        from alphatrade.scripts.validate_reports_schema import validate_report

        report_path = str(tmp_path / "mg0_alphagenome_parity_audit.json")
        _write_json(report_path, _MG0_AUDIT_VALID)
        schema_path = os.path.join(
            schemas_dir, "mg0_alphagenome_parity_audit.schema.json"
        )

        result = validate_report(report_path, schema_path)

        assert result["status"] == "pass", result.get("error")

    @pytest.mark.parametrize(
        "missing_field",
        [
            "component",
            "official_code_evidence",
            "scientific_role",
            "market_genome_analogue",
            "causal_modification",
            "status",
            "risk",
            "validation_test",
        ],
    )
    def test_component_map_requires_all_contract_fields(
        self, tmp_path, schemas_dir, missing_field
    ):
        from alphatrade.scripts.validate_reports_schema import validate_report

        report = copy.deepcopy(_MG0_AUDIT_VALID)
        del report["component_map"][0][missing_field]
        report_path = str(tmp_path / f"missing_{missing_field}.json")
        _write_json(report_path, report)
        schema_path = os.path.join(
            schemas_dir, "mg0_alphagenome_parity_audit.schema.json"
        )

        result = validate_report(report_path, schema_path)

        assert result["status"] == "fail"

    def test_component_status_rejects_unknown_value(self, tmp_path, schemas_dir):
        from alphatrade.scripts.validate_reports_schema import validate_report

        report = copy.deepcopy(_MG0_AUDIT_VALID)
        report["component_map"][0]["status"] = "REUSE"
        report_path = str(tmp_path / "bad_status.json")
        _write_json(report_path, report)
        schema_path = os.path.join(
            schemas_dir, "mg0_alphagenome_parity_audit.schema.json"
        )

        result = validate_report(report_path, schema_path)

        assert result["status"] == "fail"

    def test_repository_frozen_audit_semantics_pass(self, tmp_path, schemas_dir):
        from alphatrade.scripts.validate_reports_schema import semantic_check_mg0

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        _copy_mg0_canonical_reports(str(reports_dir))

        result = semantic_check_mg0(str(reports_dir), schemas_dir)

        assert result["all_pass"], result["checks"]
        assert all(check["status"] == "pass" for check in result["checks"])

    def test_frozen_audit_payload_tamper_fails(self, tmp_path, schemas_dir):
        from alphatrade.scripts.validate_reports_schema import semantic_check_mg0

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        _copy_mg0_canonical_reports(str(reports_dir))
        audit_path = reports_dir / "mg0_alphagenome_parity_audit.json"
        audit = json.loads(audit_path.read_text())
        audit["component_map"][0]["risk"] += " tampered"
        audit_path.write_text(json.dumps(audit))

        result = semantic_check_mg0(str(reports_dir), schemas_dir)

        checks = {check["check"]: check["status"] for check in result["checks"]}
        assert not result["all_pass"]
        assert checks["baseline_payload_hash_matches"] == "fail"

    def test_frozen_audit_markdown_tamper_fails(self, tmp_path, schemas_dir):
        from alphatrade.scripts.validate_reports_schema import semantic_check_mg0

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        _copy_mg0_canonical_reports(str(reports_dir))
        markdown_path = reports_dir / "mg0_alphagenome_parity_audit.md"
        markdown_path.write_text(markdown_path.read_text() + "\ntampered\n")

        result = semantic_check_mg0(str(reports_dir), schemas_dir)

        checks = {check["check"]: check["status"] for check in result["checks"]}
        assert not result["all_pass"]
        assert checks["markdown_hash_matches"] == "fail"

    def test_coordinated_artifact_tamper_fails_tag_anchor(
        self, tmp_path, schemas_dir
    ):
        from alphatrade.scripts.validate_reports_schema import semantic_check_mg0

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        _copy_mg0_canonical_reports(str(reports_dir))
        temp_schemas = tmp_path / "schemas"
        temp_schemas.mkdir()
        schema_path = temp_schemas / "mg0_alphagenome_parity_audit.schema.json"
        shutil.copyfile(
            os.path.join(schemas_dir, schema_path.name),
            schema_path,
        )

        audit_path = reports_dir / "mg0_alphagenome_parity_audit.json"
        markdown_path = reports_dir / "mg0_alphagenome_parity_audit.md"
        audit = json.loads(audit_path.read_text())
        audit["component_map"][0]["risk"] += " coordinated tamper"
        markdown_path.write_text(markdown_path.read_text() + "\ncoordinated tamper\n")

        payload = dict(audit)
        payload.pop("freeze")
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        audit["freeze"]["baseline_payload_sha256"] = hashlib.sha256(
            canonical
        ).hexdigest()
        audit["freeze"]["markdown_sha256"] = hashlib.sha256(
            markdown_path.read_bytes()
        ).hexdigest()

        schema = json.loads(schema_path.read_text())
        freeze_properties = schema["properties"]["freeze"]["properties"]
        freeze_properties["baseline_payload_sha256"]["const"] = audit["freeze"][
            "baseline_payload_sha256"
        ]
        freeze_properties["markdown_sha256"]["const"] = audit["freeze"][
            "markdown_sha256"
        ]
        schema_path.write_text(json.dumps(schema))
        audit["freeze"]["schema_sha256"] = hashlib.sha256(
            schema_path.read_bytes()
        ).hexdigest()
        audit_path.write_text(json.dumps(audit))

        result = semantic_check_mg0(str(reports_dir), str(temp_schemas))

        checks = {check["check"]: check["status"] for check in result["checks"]}
        assert checks["baseline_payload_hash_matches"] == "pass"
        assert checks["markdown_hash_matches"] == "pass"
        assert checks["schema_hash_matches"] == "pass"
        assert checks["validator_hash_matches"] == "pass"
        assert checks["git_tag_anchor_matches"] == "fail"
        assert not result["all_pass"]

    @pytest.mark.parametrize(
        ("path", "value", "failed_check"),
        [
            (("metadata", "gate_decision"), "FAIL", "frozen_conclusion_consistent"),
            (("freeze", "frozen_at"), "not-a-date", "freeze_timestamp_valid"),
            (
                ("execution", "training_execution_policy"),
                "RUN_NOW",
                "training_and_gpu_remain_deferred",
            ),
        ],
    )
    def test_frozen_audit_semantic_invariant_fails(
        self, tmp_path, schemas_dir, path, value, failed_check
    ):
        from alphatrade.scripts.validate_reports_schema import semantic_check_mg0

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        _copy_mg0_canonical_reports(str(reports_dir))
        audit_path = reports_dir / "mg0_alphagenome_parity_audit.json"
        audit = json.loads(audit_path.read_text())
        audit[path[0]][path[1]] = value
        audit_path.write_text(json.dumps(audit))

        result = semantic_check_mg0(str(reports_dir), schemas_dir)

        checks = {check["check"]: check["status"] for check in result["checks"]}
        assert not result["all_pass"]
        assert checks[failed_check] == "fail"

    def test_repository_manifest_mg0_profile(self):
        from alphatrade.scripts.validate_reports_schema import (
            get_profile_items,
            load_manifest,
        )

        manifest_path = os.path.join(
            _repo_root(), "src/alphatrade/schemas/contracts_manifest.yaml"
        )
        manifest = load_manifest(manifest_path)

        assert manifest is not None
        items = get_profile_items(manifest, "mg0")
        assert items == [
            {
                "name": "mg0_alphagenome_parity_audit",
                "path": "mg0_alphagenome_parity_audit.json",
                "schema": "src/alphatrade/schemas/mg0_alphagenome_parity_audit.schema.json",
                "required": True,
            },
            {
                "name": "mg0_alphagenome_parity_audit_md",
                "path": "mg0_alphagenome_parity_audit.md",
                "schema": "",
                "required": True,
            },
        ]

    def test_mg0_profile_strict_cli_passes(self, tmp_path):
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        _copy_mg0_canonical_reports(str(reports_dir))

        result = _run_validator(
            [
                "--profile",
                "mg0",
                "--strict",
                "--reports-dir",
                str(reports_dir),
                "--output-json",
                str(tmp_path / "validation.json"),
                "--output-md",
                str(tmp_path / "validation.md"),
            ]
        )

        assert result.returncode == 0, (
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "Phase 2: MG0 freeze semantic checks" in result.stdout
        with open(tmp_path / "validation.json") as f:
            validation = json.load(f)
        assert validation["profile"] == "mg0"
        assert validation["summary"]["schema_total"] == 2
        assert validation["summary"]["semantic_total"] == 13
        assert validation["summary"]["semantic_passed"] == 13
        assert validation["summary"]["overall"] == "pass"

    def test_mg0_profile_strict_cli_tamper_fails(self, tmp_path):
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        _copy_mg0_canonical_reports(str(reports_dir))
        markdown_path = reports_dir / "mg0_alphagenome_parity_audit.md"
        markdown_path.write_text(markdown_path.read_text() + "\ntampered\n")
        validation_json = tmp_path / "validation.json"
        validation_md = tmp_path / "validation.md"

        result = _run_validator(
            [
                "--profile",
                "mg0",
                "--strict",
                "--reports-dir",
                str(reports_dir),
                "--output-json",
                str(validation_json),
                "--output-md",
                str(validation_md),
            ]
        )

        assert result.returncode == 1
        validation = json.loads(validation_json.read_text())
        assert validation["summary"]["overall"] == "fail"
        assert "**Overall: \u274c FAIL**" in validation_md.read_text()

    def test_mg0_profile_strict_malformed_payload_writes_failure_report(
        self, tmp_path
    ):
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        _copy_mg0_canonical_reports(str(reports_dir))
        audit_path = reports_dir / "mg0_alphagenome_parity_audit.json"
        audit = json.loads(audit_path.read_text())
        del audit["uncertainties"][0]["status"]
        audit_path.write_text(json.dumps(audit))
        validation_json = tmp_path / "validation.json"
        validation_md = tmp_path / "validation.md"

        result = _run_validator(
            [
                "--profile",
                "mg0",
                "--strict",
                "--reports-dir",
                str(reports_dir),
                "--output-json",
                str(validation_json),
                "--output-md",
                str(validation_md),
            ]
        )

        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert validation_json.exists()
        validation = json.loads(validation_json.read_text())
        assert validation["summary"]["overall"] == "fail"
        checks = {
            check["check"]: check["status"]
            for check in validation["semantic_results"]["checks"]
        }
        assert checks["uncertainties_remain_open"] == "fail"

    def test_mg0_profile_strict_non_file_artifact_writes_failure_report(
        self, tmp_path
    ):
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        _copy_mg0_canonical_reports(str(reports_dir))
        markdown_path = reports_dir / "mg0_alphagenome_parity_audit.md"
        markdown_path.unlink()
        markdown_path.mkdir()
        validation_json = tmp_path / "validation.json"
        validation_md = tmp_path / "validation.md"

        result = _run_validator(
            [
                "--profile",
                "mg0",
                "--strict",
                "--reports-dir",
                str(reports_dir),
                "--output-json",
                str(validation_json),
                "--output-md",
                str(validation_md),
            ]
        )

        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert validation_json.exists()
        validation = json.loads(validation_json.read_text())
        assert validation["summary"]["overall"] == "fail"
        checks = {
            check["check"]: check["status"]
            for check in validation["semantic_results"]["checks"]
        }
        assert checks["freeze_artifacts_exist"] == "fail"


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


# ---------------------------------------------------------------------------
# M5 semantic checks
# ---------------------------------------------------------------------------

def _make_m5_sweep_manifest(reports_dir, seeds, exp_ids=None, config_hash="cfghash_a"):
    """Create a minimal sweep manifest and its referenced run files."""
    if exp_ids is None:
        exp_ids = ["baseline"]

    runs = []
    for exp_id in exp_ids:
        for seed in seeds:
            run_id = f"{exp_id}_s{seed}"
            train_path = os.path.join(reports_dir, f"m5_{exp_id}_seed{seed}_train.json")
            eval_path = os.path.join(reports_dir, f"m5_{exp_id}_seed{seed}_eval.json")
            sweep = {
                "exp_id": exp_id,
                "seed": seed,
                "config_hash": config_hash,
                "dataset_config": "configs/dataset/m2.yaml",
                "universe": "test",
                "eval_split": "val",
                "ckpt_step": "best",
            }

            train_data = copy.deepcopy(_TRAIN_METRICS_VALID)
            train_data["run"]["seed"] = seed
            train_data["run"]["run_id"] = run_id
            train_data["run"]["sweep"] = sweep
            _write_json(train_path, train_data)

            eval_data = copy.deepcopy(_EVAL_METRICS_VALID)
            eval_data["run_id"] = run_id
            eval_data["model"]["train_run_id"] = run_id
            eval_data["model"]["sweep"] = sweep
            _write_json(eval_path, eval_data)

            runs.append({
                "exp_id": exp_id,
                "seed": seed,
                "run_id": run_id,
                "config_hash": config_hash,
                "dataset_config": "configs/dataset/m2.yaml",
                "universe": "test",
                "eval_split": "val",
                "ckpt_step": "best",
                "train_metrics_path": train_path,
                "eval_metrics_path": eval_path,
            })

    manifest = {
        "schema_version": "m5_sweep_manifest_v1",
        "generated_at": "2026-03-03T00:00:00",
        "profile": "m5",
        "git_sha": "testsha",
        "universe": "test",
        "dataset": "test_ds",
        "dataset_config": "configs/dataset/m2.yaml",
        "dataset_configs": ["configs/dataset/m2.yaml"],
        "expected_seeds": list(seeds),
        "primary_metric": "pinball_loss.overall",
        "runs": runs,
    }
    manifest_path = os.path.join(reports_dir, "m5_sweep_manifest.json")
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


class TestM5SemanticChecks:
    """Test M5 Phase 2 semantic checks."""

    def test_m5_strict_pass(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import semantic_check_m5

        reports_dir = str(tmp_path / "reports")
        schemas_dir_abs = os.path.abspath(_schemas_dir())
        _make_m5_sweep_manifest(reports_dir, seeds=[42, 43, 44])

        result = semantic_check_m5(reports_dir, schemas_dir_abs)
        assert result["all_pass"], f"Checks failed: {[c for c in result['checks'] if c['status'] != 'pass']}"

    def test_m5_missing_seed_strict_fail(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import semantic_check_m5

        reports_dir = str(tmp_path / "reports")
        schemas_dir_abs = os.path.abspath(_schemas_dir())
        # Create manifest expecting seeds [42, 43, 44] but only provide [42, 43]
        manifest_path, manifest = _make_m5_sweep_manifest(reports_dir, seeds=[42, 43])
        # Overwrite manifest to expect 3 seeds
        manifest["expected_seeds"] = [42, 43, 44]
        _write_json(manifest_path, manifest)

        result = semantic_check_m5(reports_dir, schemas_dir_abs)
        assert not result["all_pass"]
        seed_checks = [c for c in result["checks"] if "seeds_complete" in c["check"]]
        assert any(c["status"] == "fail" for c in seed_checks)

    def test_m5_missing_eval_strict_fail(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import semantic_check_m5

        reports_dir = str(tmp_path / "reports")
        schemas_dir_abs = os.path.abspath(_schemas_dir())
        _make_m5_sweep_manifest(reports_dir, seeds=[42, 43, 44])

        # Delete one eval file
        os.remove(os.path.join(reports_dir, "m5_baseline_seed44_eval.json"))

        result = semantic_check_m5(reports_dir, schemas_dir_abs)
        assert not result["all_pass"]
        eval_checks = [c for c in result["checks"] if "eval_exists" in c["check"] and c["status"] == "fail"]
        assert len(eval_checks) == 1

    def test_m5_config_hash_mismatch(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import semantic_check_m5

        reports_dir = str(tmp_path / "reports")
        schemas_dir_abs = os.path.abspath(_schemas_dir())
        manifest_path, manifest = _make_m5_sweep_manifest(reports_dir, seeds=[42, 43, 44])

        # Tamper one run's config_hash
        manifest["runs"][2]["config_hash"] = "different_hash"
        _write_json(manifest_path, manifest)

        result = semantic_check_m5(reports_dir, schemas_dir_abs)
        assert not result["all_pass"]
        hash_checks = [c for c in result["checks"] if "config_hash" in c["check"]]
        assert any(c["status"] == "fail" for c in hash_checks)

    def test_m5_artifact_sweep_metadata_mismatch_fails(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import semantic_check_m5

        reports_dir = str(tmp_path / "reports")
        schemas_dir_abs = os.path.abspath(_schemas_dir())
        _, manifest = _make_m5_sweep_manifest(reports_dir, seeds=[42, 43, 44])

        eval_path = manifest["runs"][0]["eval_metrics_path"]
        with open(eval_path) as f:
            eval_data = json.load(f)
        eval_data["model"]["sweep"]["config_hash"] = "stalehash"
        _write_json(eval_path, eval_data)

        result = semantic_check_m5(reports_dir, schemas_dir_abs)
        assert not result["all_pass"]
        metadata_checks = [
            c for c in result["checks"]
            if "eval_sweep_metadata_matches_manifest" in c["check"]
        ]
        assert any(c["status"] == "fail" and "stalehash" in c["detail"] for c in metadata_checks)

    def test_m5_leaderboard_preserves_run_provenance_fields(self, tmp_path):
        from alphatrade.scripts import build_m5_leaderboard

        reports_dir = str(tmp_path / "reports")
        _, manifest = _make_m5_sweep_manifest(reports_dir, seeds=[42, 43, 44])

        leaderboard = build_m5_leaderboard.build_leaderboard(manifest)

        exp = leaderboard["experiments"][0]
        assert exp["dataset_configs"] == ["configs/dataset/m2.yaml"]
        assert exp["universes"] == ["test"]
        assert exp["eval_splits"] == ["val"]
        assert exp["ckpt_steps"] == ["best"]
        run = exp["artifacts"]["runs"][0]
        assert run["dataset_config"] == "configs/dataset/m2.yaml"
        assert run["universe"] == "test"
        assert run["eval_split"] == "val"
        assert run["ckpt_step"] == "best"


class TestM5Integration:
    """Integration tests for M5 profile."""

    def test_m5_profile_loads(self, tmp_path):
        """subprocess running --profile m5 can identify 5 items."""
        reports_dir = str(tmp_path / "reports")
        os.makedirs(reports_dir, exist_ok=True)
        schemas_dir_abs = os.path.abspath(_schemas_dir())
        manifest_path = str(tmp_path / "manifest.yaml")

        profiles = {
            "m5": {
                "description": "M5 sweep test",
                "reports_dir": reports_dir,
                "items": [
                    {"name": "m5_sweep_manifest", "path": "m5_sweep_manifest.json",
                     "schema": os.path.join(schemas_dir_abs, "m5_sweep_manifest.schema.json"), "required": True},
                    {"name": "m5_leaderboard", "path": "m5_leaderboard.json",
                     "schema": os.path.join(schemas_dir_abs, "m5_leaderboard.schema.json"), "required": True},
                    {"name": "m5_leaderboard_md", "path": "m5_leaderboard.md", "schema": "", "required": True},
                    {"name": "m5_schema_validation", "path": "m5_schema_validation.json", "schema": "", "required": True},
                    {"name": "m5_schema_validation_md", "path": "m5_schema_validation.md", "schema": "", "required": True},
                ],
            }
        }
        _write_manifest(manifest_path, profiles)

        result = _run_validator([
            "--manifest", manifest_path,
            "--profile", "m5",
            "--reports-dir", reports_dir,
            "--schemas-dir", schemas_dir_abs,
            "--output-json", str(tmp_path / "out.json"),
            "--output-md", str(tmp_path / "out.md"),
        ])
        # Should identify 5 items (all missing, non-strict so exit 2)
        assert "5 items" in result.stdout, f"stdout:\n{result.stdout}"


# ---------------------------------------------------------------------------
# M7 semantic checks
# ---------------------------------------------------------------------------

def _make_m5_leaderboard(reports_dir, experiments_data, expected_seeds=None):
    """Create a minimal M5 leaderboard JSON.

    experiments_data: list of dicts with keys: exp_id, primary_mean, primary_std, n_runs
    """
    if expected_seeds is None:
        expected_seeds = [42, 43, 44]

    experiments = []
    for ed in experiments_data:
        experiments.append({
            "exp_id": ed["exp_id"],
            "config_hash": f"hash_{ed['exp_id']}",
            "dataset_configs": [ed.get("dataset_config", "configs/dataset/m2.yaml")],
            "universes": [ed.get("universe", "test")],
            "eval_splits": [ed.get("eval_split", "val")],
            "ckpt_steps": [str(ed.get("ckpt_step", "best"))],
            "seeds_done": expected_seeds,
            "seeds_missing": [],
            "n_runs": ed.get("n_runs", len(expected_seeds)),
            "metrics": {
                "primary_mean": ed["primary_mean"],
                "primary_std": ed.get("primary_std", 0.01),
                "primary_best": ed["primary_mean"] - 0.01,
                "best_run_id": f"{ed['exp_id']}_s42",
            },
            "artifacts": {
                "runs": [
                    {
                        "seed": s,
                        "run_id": f"{ed['exp_id']}_s{s}",
                        "dataset_config": ed.get("dataset_config", "configs/dataset/m2.yaml"),
                        "universe": ed.get("universe", "test"),
                        "eval_split": ed.get("eval_split", "val"),
                        "ckpt_step": ed.get("ckpt_step", "best"),
                        "train_metrics_path": os.path.join(reports_dir, f"m5_{ed['exp_id']}_seed{s}_train_metrics.json"),
                        "eval_metrics_path": os.path.join(reports_dir, f"m5_{ed['exp_id']}_seed{s}_eval_metrics.json"),
                    }
                    for s in expected_seeds
                ],
            },
        })

    leaderboard = {
        "schema_version": "m5_leaderboard_v1",
        "generated_at": "2026-03-03T00:00:00",
        "git_sha": "testsha",
        "primary_metric": "pinball_loss.overall",
        "expected_seeds": expected_seeds,
        "experiments": experiments,
    }
    lb_path = os.path.join(reports_dir, "m5_leaderboard.json")
    _write_json(lb_path, leaderboard)
    return lb_path


def _make_m7_regression_report(reports_dir, baseline_exp_id, comparisons, thresholds=None):
    """Create a minimal M7 regression report JSON."""
    if thresholds is None:
        thresholds = {"improvement_pct": 1.0, "regression_pct": 5.0}

    summary = {
        "total_ablations": len(comparisons),
        "improved": sum(1 for c in comparisons if c["verdict"] == "improved"),
        "neutral": sum(1 for c in comparisons if c["verdict"] == "neutral"),
        "regressed": sum(1 for c in comparisons if c["verdict"] == "regressed"),
    }

    report = {
        "schema_version": "m7_regression_report_v1",
        "generated_at": "2026-03-03T00:00:00",
        "git_sha": "testsha",
        "baseline_exp_id": baseline_exp_id,
        "primary_metric": "pinball_loss.overall",
        "thresholds": thresholds,
        "baseline": {
            "exp_id": baseline_exp_id,
            "primary_mean": 0.134,
            "primary_std": 0.011,
            "n_seeds": 3,
        },
        "comparisons": comparisons,
        "summary": summary,
    }
    rpt_path = os.path.join(reports_dir, "m7_regression_report.json")
    _write_json(rpt_path, report)
    return rpt_path


class TestM7SemanticChecks:
    """Test M7 Phase 2b regression report semantic checks."""

    def test_m7_regression_report_valid(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import semantic_check_m7_regression

        reports_dir = str(tmp_path / "reports")

        _make_m5_leaderboard(reports_dir, [
            {"exp_id": "baseline", "primary_mean": 0.134},
            {"exp_id": "no_clip", "primary_mean": 0.140},
            {"exp_id": "batch_256", "primary_mean": 0.130},
        ])

        _make_m7_regression_report(reports_dir, "baseline", [
            {"exp_id": "no_clip", "primary_mean": 0.140, "primary_std": 0.012, "n_seeds": 3,
             "delta": 0.006, "delta_pct": 4.5, "verdict": "neutral"},
            {"exp_id": "batch_256", "primary_mean": 0.130, "primary_std": 0.009, "n_seeds": 3,
             "delta": -0.004, "delta_pct": -2.99, "verdict": "improved"},
        ])

        result = semantic_check_m7_regression(reports_dir)
        assert result["all_pass"], f"Checks failed: {[c for c in result['checks'] if c['status'] != 'pass']}"

    def test_m7_regression_report_missing_baseline(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import semantic_check_m7_regression

        reports_dir = str(tmp_path / "reports")

        # Leaderboard has no "baseline" experiment
        _make_m5_leaderboard(reports_dir, [
            {"exp_id": "no_clip", "primary_mean": 0.140},
        ])

        _make_m7_regression_report(reports_dir, "baseline", [
            {"exp_id": "no_clip", "primary_mean": 0.140, "primary_std": 0.012, "n_seeds": 3,
             "delta": 0.006, "delta_pct": 4.5, "verdict": "neutral"},
        ])

        result = semantic_check_m7_regression(reports_dir)
        assert not result["all_pass"]
        baseline_checks = [c for c in result["checks"] if "baseline_exp_in_leaderboard" in c["check"]]
        assert any(c["status"] == "fail" for c in baseline_checks)

    def test_m7_regression_report_invalid_verdict(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import semantic_check_m7_regression

        reports_dir = str(tmp_path / "reports")

        _make_m5_leaderboard(reports_dir, [
            {"exp_id": "baseline", "primary_mean": 0.134},
            {"exp_id": "no_clip", "primary_mean": 0.140},
        ])

        # Write report with invalid verdict manually
        report = {
            "schema_version": "m7_regression_report_v1",
            "generated_at": "2026-03-03T00:00:00",
            "git_sha": "testsha",
            "baseline_exp_id": "baseline",
            "primary_metric": "pinball_loss.overall",
            "thresholds": {"improvement_pct": 1.0, "regression_pct": 5.0},
            "baseline": {"exp_id": "baseline", "primary_mean": 0.134, "primary_std": 0.011, "n_seeds": 3},
            "comparisons": [
                {"exp_id": "no_clip", "primary_mean": 0.140, "primary_std": 0.012, "n_seeds": 3,
                 "delta": 0.006, "delta_pct": 4.5, "verdict": "INVALID_VERDICT"},
            ],
            "summary": {"total_ablations": 1, "improved": 0, "neutral": 1, "regressed": 0},
        }
        _write_json(os.path.join(reports_dir, "m7_regression_report.json"), report)

        result = semantic_check_m7_regression(reports_dir)
        assert not result["all_pass"]
        verdict_checks = [c for c in result["checks"] if "verdict_valid" in c["check"]]
        assert any(c["status"] == "fail" for c in verdict_checks)


# ---------------------------------------------------------------------------
# M9 semantic checks
# ---------------------------------------------------------------------------

def _make_m9_predictions_parquet(path, n_rows=10, missing_col=None):
    """Create a minimal valid predictions parquet."""
    import pandas as pd
    import numpy as np

    horizons = [1, 5, 20, 60]
    quantiles = [10, 30, 50, 70, 90]

    data = {
        "symbol": ["DCE.JM"] * n_rows,
        "eob": pd.date_range("2024-01-02 09:01:00", periods=n_rows, freq="1min"),
        "model_version": ["test_v0.2"] * n_rows,
    }
    for h in horizons:
        base = np.linspace(-0.001, 0.001, n_rows, dtype=np.float64)
        for q in quantiles:
            col = f"h{h}_q{q}"
            if col != missing_col:
                data[col] = (base + q * 1e-5).astype(np.float64)

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _make_m9_bundle_manifest(reports_dir, bundle_path="/tmp/fake_bundle"):
    """Create a minimal valid M9 bundle manifest."""
    os.makedirs(bundle_path, exist_ok=True)
    feature_profile = {
        "profile_id": "m1_f8",
        "feature_dim": 8,
        "feature_cols": [
            "ret_1m",
            "hl_range",
            "co_change",
            "vol_log1p",
            "pos_log1p",
            "minute_sin",
            "minute_cos",
            "is_session_open",
        ],
        "processed_root": "data/processed/m1_f8",
        "scaler_hash": None,
        "source_schema_versions": {"alphatrade_feature_profile": "m1_f8_v1"},
        "normalization": {"policy": "precomputed_in_bars", "train_only": False},
        "fingerprint": "profile123456789",
    }
    payload_path = os.path.join(bundle_path, "model_config.json")
    payload = {
        "lookback_length": 60,
        "num_features": 8,
        "feature_profile": feature_profile,
        "feature_profile_id": "m1_f8",
        "feature_cols": feature_profile["feature_cols"],
        "horizons": [1, 5, 20, 60],
        "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9],
        "d_model": 256,
        "num_transformer_layers": 4,
    }
    _write_json(payload_path, payload)
    with open(payload_path, "rb") as f:
        payload_sha = hashlib.sha256(f.read()).hexdigest()
    manifest = {
        "schema_version": "m9_model_bundle_manifest_v1",
        "bundle_format_version": "alphatrade_model_bundle_v1",
        "prediction_schema_version": "m9_predictions_v1",
        "generated_at": "2026-03-03T00:00:00",
        "git_sha": "testsha",
        "model_version": "test_v0.2",
        "bundle_id": "bundle1234567890",
        "champion_selection": {
            "method": "leaderboard_primary_mean",
            "exp_id": "baseline",
            "run_id": "abc123",
            "seed": 42,
            "primary_mean": 0.134,
            "primary_std": 0.011,
            "primary_best": 0.123,
            "config_hash": "cfghash_a",
        },
        "feature_profile": feature_profile,
        "model_config": payload,
        "bundle_path": bundle_path,
        "bundle_files": [
            {
                "path": "model_config.json",
                "size_bytes": os.path.getsize(payload_path),
                "sha256": payload_sha,
            }
        ],
        "checkpoint_source": "/tmp/fake_ckpt",
        "dataset_config": "configs/dataset/m2.yaml",
        "universe": "test",
        "expected_seeds": [42, 43, 44],
        "leaderboard_path": "reports/m5_leaderboard.json",
    }
    manifest_path = os.path.join(reports_dir, "m9_model_bundle_manifest.json")
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def _make_m9_infer_metrics(reports_dir):
    """Create a minimal valid M9 inference metrics report."""
    report = {
        "schema_version": "m9_infer_metrics_v1",
        "generated_at": "2026-03-03T00:00:00",
        "git_sha": "testsha",
        "model_version": "test_v0.2",
        "bundle_path": os.path.join(reports_dir, "bundle"),
        "prediction_path": os.path.join(reports_dir, "m9_predictions.parquet"),
        "symbols": ["DCE.JM"],
        "start": "2024-01-02",
        "end": "2024-01-02",
        "num_rows": 10,
        "num_symbols": 1,
        "batch_size": 8,
        "jax_backend": "gpu",
        "elapsed_seconds": 1.25,
        "samples_per_second": 8.0,
        "inference": {
            "symbols": ["DCE.JM"],
            "n_symbols": 1,
            "time_range": {"start": "2024-01-02", "end": "2024-01-02"},
            "n_samples": 10,
            "n_predictions": 200,
            "batch_size": 8,
            "jax_backend": "gpu",
            "missing_symbols": [],
            "duration_seconds": 1.25,
            "samples_per_second": 8.0,
        },
        "output": {
            "predictions_path": os.path.join(reports_dir, "m9_predictions.parquet"),
            "predictions_rows": 10,
            "predictions_columns": 23,
        },
    }
    report_path = os.path.join(reports_dir, "m9_infer_metrics.json")
    _write_json(report_path, report)
    return report_path, report


def _make_m9_backtest_metrics(reports_dir, n_predictions=10, n_matched=10):
    """Create a minimal valid M9 backtest metrics report."""
    report = {
        "schema_version": "m9_backtest_metrics_v1",
        "generated_at": "2026-03-03T00:00:00",
        "git_sha": "testsha",
        "predictions_path": os.path.join(reports_dir, "m9_predictions.parquet"),
        "data_dir": "data/processed/m1_f8",
        "model_versions": ["test_v0.2"],
        "config": {
            "horizon": 20,
            "quantile": 0.5,
            "signal_column": "h20_q50",
            "threshold": 0.0,
            "cost_bps": 0.0,
        },
        "data": {
            "n_predictions": n_predictions,
            "n_matched": n_matched,
            "n_symbols": 1,
            "missing_symbols": [],
            "skipped": {
                "missing_eob": 0,
                "insufficient_future": 0,
                "invalid_close": 0,
            },
        },
        "performance": {
            "sum_gross_log_return": 0.1,
            "sum_net_log_return": 0.1,
            "mean_gross_log_return": 0.01,
            "mean_net_log_return": 0.01,
            "win_rate": 0.6,
            "direction_hit_rate": 0.6,
            "mean_turnover": 0.2,
            "sharpe_like": 1.0,
        },
        "by_symbol": [
            {
                "symbol": "DCE.JM",
                "n_samples": n_matched,
                "sum_gross_log_return": 0.1,
                "sum_net_log_return": 0.1,
                "mean_gross_log_return": 0.01,
                "mean_net_log_return": 0.01,
                "win_rate": 0.6,
                "direction_hit_rate": 0.6,
                "mean_turnover": 0.2,
                "sharpe_like": 1.0,
            }
        ],
        "output": {
            "trades_path": os.path.join(reports_dir, "m9_backtest_trades.parquet"),
            "trades_rows": n_matched,
        },
    }
    report_path = os.path.join(reports_dir, "m9_backtest_metrics.json")
    _write_json(report_path, report)
    return report_path, report


class TestM9SemanticChecks:
    """Test M9 Phase 2 semantic checks."""

    def test_m9_predictions_valid(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import semantic_check_m9

        reports_dir = str(tmp_path / "reports")
        schemas_dir_abs = os.path.abspath(_schemas_dir())

        # Create valid bundle manifest with an existing bundle_path
        bundle_dir = str(tmp_path / "bundle")
        os.makedirs(bundle_dir, exist_ok=True)
        _make_m9_bundle_manifest(reports_dir, bundle_path=bundle_dir)
        _make_m9_predictions_parquet(os.path.join(reports_dir, "m9_predictions.parquet"))
        _make_m9_backtest_metrics(reports_dir)

        result = semantic_check_m9(reports_dir, schemas_dir_abs)
        assert result["all_pass"], f"Checks failed: {[c for c in result['checks'] if c['status'] != 'pass']}"

    def test_m9_predictions_missing_column(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import semantic_check_m9

        reports_dir = str(tmp_path / "reports")
        schemas_dir_abs = os.path.abspath(_schemas_dir())

        bundle_dir = str(tmp_path / "bundle")
        os.makedirs(bundle_dir, exist_ok=True)
        _make_m9_bundle_manifest(reports_dir, bundle_path=bundle_dir)
        _make_m9_predictions_parquet(
            os.path.join(reports_dir, "m9_predictions.parquet"),
            missing_col="h1_q10",
        )
        _make_m9_backtest_metrics(reports_dir)

        result = semantic_check_m9(reports_dir, schemas_dir_abs)
        assert not result["all_pass"]
        col_checks = [c for c in result["checks"] if "columns_complete" in c["check"]]
        assert any(c["status"] == "fail" for c in col_checks)

    def test_m9_bundle_manifest_valid(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import validate_report

        reports_dir = str(tmp_path / "reports")
        schemas_dir_abs = os.path.abspath(_schemas_dir())

        bundle_dir = str(tmp_path / "bundle")
        os.makedirs(bundle_dir, exist_ok=True)
        manifest_path, _ = _make_m9_bundle_manifest(reports_dir, bundle_path=bundle_dir)
        schema_path = os.path.join(schemas_dir_abs, "m9_model_bundle_manifest.schema.json")

        result = validate_report(manifest_path, schema_path)
        assert result["status"] == "pass", f"Validation failed: {result.get('error')}"

    def test_m9_infer_metrics_valid(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import validate_report

        reports_dir = str(tmp_path / "reports")
        schemas_dir_abs = os.path.abspath(_schemas_dir())

        report_path, _ = _make_m9_infer_metrics(reports_dir)
        schema_path = os.path.join(schemas_dir_abs, "m9_infer_metrics.schema.json")

        result = validate_report(report_path, schema_path)
        assert result["status"] == "pass", f"Validation failed: {result.get('error')}"

    def test_m9_backtest_metrics_valid(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import validate_report

        reports_dir = str(tmp_path / "reports")
        schemas_dir_abs = os.path.abspath(_schemas_dir())

        report_path, _ = _make_m9_backtest_metrics(reports_dir)
        schema_path = os.path.join(schemas_dir_abs, "m9_backtest_metrics.schema.json")

        result = validate_report(report_path, schema_path)
        assert result["status"] == "pass", f"Validation failed: {result.get('error')}"


class TestM10Schemas:
    """Test M10 report schemas with minimal valid reports."""

    def test_m10_minimal_reports_validate(self, tmp_path):
        from alphatrade.scripts.validate_reports_schema import validate_report

        reports_dir = str(tmp_path / "reports")
        schemas_dir_abs = os.path.abspath(_schemas_dir())

        eval_report = {
            "schema_version": "m10_prediction_eval_metrics_v1",
            "generated_at": "2026-06-14T00:00:00",
            "inputs": {},
            "data": {},
            "pinball_loss": {},
            "quantile_coverage": {},
            "quantile_crossing": {},
            "distribution_diagnostics": {},
            "ic_metrics": {},
            "direction_metrics": {},
            "output": {},
        }
        backtest_report = {
            "schema_version": "m10_backtest_matrix_v1",
            "generated_at": "2026-06-14T00:00:00",
            "inputs": {},
            "matrix": [],
            "summary": {},
        }
        baseline_report = {
            "schema_version": "m10_baseline_comparison_v1",
            "generated_at": "2026-06-14T00:00:00",
            "inputs": {},
            "model": {},
            "baselines": {},
            "deltas": {},
        }
        cases = [
            ("m10_prediction_eval_metrics.json", "m10_prediction_eval_metrics.schema.json", eval_report),
            ("m10_backtest_matrix.json", "m10_backtest_matrix.schema.json", backtest_report),
            ("m10_baseline_comparison.json", "m10_baseline_comparison.schema.json", baseline_report),
        ]
        for report_name, schema_name, payload in cases:
            report_path = os.path.join(reports_dir, report_name)
            _write_json(report_path, payload)
            result = validate_report(report_path, os.path.join(schemas_dir_abs, schema_name))
            assert result["status"] == "pass", f"{report_name}: {result.get('error')}"


class TestM7Integration:
    """Integration tests for M7 profile."""

    def test_m7_profile_loads(self, tmp_path):
        """subprocess running --profile m7 can identify 5 items."""
        reports_dir = str(tmp_path / "reports")
        os.makedirs(reports_dir, exist_ok=True)
        schemas_dir_abs = os.path.abspath(_schemas_dir())
        manifest_path = str(tmp_path / "manifest.yaml")

        profiles = {
            "m7": {
                "description": "M7 ablation sweep test",
                "reports_dir": reports_dir,
                "items": [
                    {"name": "m5_sweep_manifest", "path": "m5_sweep_manifest.json",
                     "schema": os.path.join(schemas_dir_abs, "m5_sweep_manifest.schema.json"), "required": True},
                    {"name": "m5_leaderboard", "path": "m5_leaderboard.json",
                     "schema": os.path.join(schemas_dir_abs, "m5_leaderboard.schema.json"), "required": True},
                    {"name": "m5_leaderboard_md", "path": "m5_leaderboard.md", "schema": "", "required": True},
                    {"name": "m7_regression_report", "path": "m7_regression_report.json",
                     "schema": os.path.join(schemas_dir_abs, "m7_regression_report.schema.json"), "required": True},
                    {"name": "m7_regression_report_md", "path": "m7_regression_report.md", "schema": "", "required": True},
                ],
            }
        }
        _write_manifest(manifest_path, profiles)

        result = _run_validator([
            "--manifest", manifest_path,
            "--profile", "m7",
            "--reports-dir", reports_dir,
            "--schemas-dir", schemas_dir_abs,
            "--output-json", str(tmp_path / "out.json"),
            "--output-md", str(tmp_path / "out.md"),
        ])
        # Should identify 5 items (all missing, non-strict so exit 2)
        assert "5 items" in result.stdout, f"stdout:\n{result.stdout}"
