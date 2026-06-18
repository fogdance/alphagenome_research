"""Feature profile contract for AlphaTrade processed datasets.

M12 removes the implicit assumption that every dataset is the historical
8-feature ``m1_f8`` layout. Training, evaluation, inference, and caches should
carry this profile metadata explicitly so a wider processed dataset cannot
silently fall back to the old feature columns.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_FEATURE_PROFILE_ID = "m1_f8"
BASE_FEATURE_COLS = (
    "ret_1m",
    "hl_range",
    "co_change",
    "vol_log1p",
    "pos_log1p",
    "minute_sin",
    "minute_cos",
    "is_session_open",
)


@dataclass
class FeatureProfile:
    profile_id: str
    feature_cols: tuple[str, ...]
    feature_dim: int
    processed_root: str | None = None
    scaler_hash: str | None = None
    source_schema_versions: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        payload = {
            "profile_id": self.profile_id,
            "feature_cols": list(self.feature_cols),
            "feature_dim": self.feature_dim,
            "processed_root": self.processed_root,
            "scaler_hash": self.scaler_hash,
            "source_schema_versions": self.source_schema_versions,
            "normalization": self.normalization,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]


def base_feature_profile(processed_root: str | None = None) -> FeatureProfile:
    return FeatureProfile(
        profile_id=BASE_FEATURE_PROFILE_ID,
        feature_cols=BASE_FEATURE_COLS,
        feature_dim=len(BASE_FEATURE_COLS),
        processed_root=processed_root,
        source_schema_versions={"alphatrade_feature_profile": "m1_f8_v1"},
        normalization={"policy": "precomputed_in_bars", "train_only": False},
    )


def _coerce_feature_cols(value: Any) -> tuple[str, ...]:
    if value is None:
        raise ValueError("feature profile is missing feature_cols")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"feature_cols must be a sequence of strings, got {type(value).__name__}")
    cols = tuple(str(v) for v in value)
    if not cols:
        raise ValueError("feature_cols must not be empty")
    duplicates = sorted({c for c in cols if cols.count(c) > 1})
    if duplicates:
        raise ValueError(f"feature_cols contains duplicates: {duplicates}")
    return cols


def feature_profile_from_mapping(data: Mapping[str, Any], *, default_processed_root: str | None = None) -> FeatureProfile:
    """Create a FeatureProfile from config/report/bundle metadata."""
    raw_profile_id = data.get("profile_id") or data.get("id") or data.get("feature_profile_id")
    if raw_profile_id is None:
        raise ValueError("feature profile is missing profile_id")
    profile_id = str(raw_profile_id)
    feature_cols = _coerce_feature_cols(data.get("feature_cols") or data.get("cols"))
    feature_dim = int(data.get("feature_dim") or data.get("dim") or len(feature_cols))
    if feature_dim != len(feature_cols):
        raise ValueError(
            f"feature_dim mismatch for {profile_id}: dim={feature_dim}, cols={len(feature_cols)}"
        )
    return FeatureProfile(
        profile_id=profile_id,
        feature_cols=feature_cols,
        feature_dim=feature_dim,
        processed_root=str(data.get("processed_root") or default_processed_root)
        if (data.get("processed_root") or default_processed_root)
        else None,
        scaler_hash=data.get("scaler_hash"),
        source_schema_versions=dict(data.get("source_schema_versions") or {}),
        normalization=dict(data.get("normalization") or data.get("normalization_policy") or {}),
    )


def resolve_feature_profile(config: Mapping[str, Any]) -> FeatureProfile:
    """Resolve the active dataset feature profile from a dataset config."""
    default_processed_root = (config.get("paths") or {}).get("processed_dir")
    profile_data = config.get("feature_profile")
    if profile_data is None:
        raise ValueError("dataset config is missing required feature_profile")

    profile = feature_profile_from_mapping(
        profile_data,
        default_processed_root=str(default_processed_root) if default_processed_root else None,
    )
    if profile.processed_root is None and default_processed_root:
        profile.processed_root = str(default_processed_root)
    return profile


def feature_profile_to_dict(profile: FeatureProfile) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "feature_dim": profile.feature_dim,
        "feature_cols": list(profile.feature_cols),
        "processed_root": profile.processed_root,
        "scaler_hash": profile.scaler_hash,
        "source_schema_versions": profile.source_schema_versions,
        "normalization": profile.normalization,
        "fingerprint": profile.fingerprint,
    }


def feature_profile_from_model_metadata(metadata: Mapping[str, Any]) -> FeatureProfile:
    """Resolve profile metadata embedded in train metrics, artifacts, or bundles."""
    if "feature_profile" in metadata:
        return feature_profile_from_mapping(metadata["feature_profile"])
    if "model_config" in metadata and isinstance(metadata["model_config"], Mapping):
        model_config = metadata["model_config"]
        if "feature_profile" in model_config:
            return feature_profile_from_mapping(model_config["feature_profile"])
        if "feature_cols" in model_config:
            return feature_profile_from_mapping(model_config)
    if "feature_cols" in metadata:
        return feature_profile_from_mapping(metadata)
    raise ValueError("metadata does not contain feature profile information")


def assert_feature_profiles_match(expected: FeatureProfile, observed: FeatureProfile, *, context: str) -> None:
    errors = []
    if expected.profile_id != observed.profile_id:
        errors.append(f"profile_id expected={expected.profile_id!r} observed={observed.profile_id!r}")
    if expected.feature_dim != observed.feature_dim:
        errors.append(f"feature_dim expected={expected.feature_dim} observed={observed.feature_dim}")
    if list(expected.feature_cols) != list(observed.feature_cols):
        errors.append("feature_cols differ")
    if (expected.scaler_hash or observed.scaler_hash) and expected.scaler_hash != observed.scaler_hash:
        errors.append(f"scaler_hash expected={expected.scaler_hash!r} observed={observed.scaler_hash!r}")
    if errors:
        raise ValueError(f"feature_profile_mismatch[{context}]: " + "; ".join(errors))


def load_root_feature_manifest(processed_root: str | Path) -> dict[str, Any] | None:
    path = Path(processed_root).expanduser().resolve() / "feature_manifest.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
