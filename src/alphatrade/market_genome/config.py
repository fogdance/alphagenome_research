"""Resolved configuration contract for the MG0-B scaffold.

The model configuration is deliberately deferred. This module only makes the
current implementation boundary machine-readable and hashable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class MarketGenomeScaffoldConfig:
  """Configuration facts that are frozen for the scaffold-only milestone."""

  schema_version: str = "market_genome.scaffold_config.v1"
  milestone: str = "MG0-B"
  model_line: str = "market_genome"
  report_profile: str = "mg0b"
  implementation_scope: str = "SCAFFOLD_ONLY"
  full_backbone_implemented: bool = False
  training_enabled: bool = False
  gpu_required: bool = False
  pair_stack_status: str = "DEFERRED"
  ontology_status: str = "NOT_IMPLEMENTED"
  dense_track_schema_status: str = "NOT_IMPLEMENTED"

  def to_dict(self) -> dict[str, Any]:
    """Return the complete resolved scaffold configuration."""
    return asdict(self)

  def canonical_json(self) -> str:
    """Return the stable serialization used by report/config fingerprints."""
    return json.dumps(
        self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )

  def sha256(self) -> str:
    """Return the SHA256 of the resolved scaffold configuration."""
    return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()


def resolved_scaffold_config() -> dict[str, Any]:
  """Return the sole resolved configuration supported in MG0-B."""
  return MarketGenomeScaffoldConfig().to_dict()


def load_model_config(*_args: Any, **_kwargs: Any) -> None:
  """Reject model configuration until its later milestone contract exists."""
  raise NotImplementedError(
      "MarketGenome model configuration is not implemented in MG0-B; "
      "freeze the MG1 ontology and MG3 track contract first."
  )
