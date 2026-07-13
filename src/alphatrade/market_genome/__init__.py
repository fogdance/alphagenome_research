"""MarketGenome namespace created by the MG0-B scaffold milestone.

Importing this package intentionally does not import JAX, Haiku, AlphaTrade's
current model, or any training code.
"""

SCHEMA_VERSION = "market_genome.scaffold.v1"
IMPLEMENTATION_STATUS = "SCAFFOLD_ONLY"
FULL_BACKBONE_IMPLEMENTED = False
TRAINING_ENABLED = False

__all__ = [
    "FULL_BACKBONE_IMPLEMENTED",
    "IMPLEMENTATION_STATUS",
    "SCHEMA_VERSION",
    "TRAINING_ENABLED",
]
