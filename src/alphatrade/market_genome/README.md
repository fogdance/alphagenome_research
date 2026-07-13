# MarketGenome Scaffold

This namespace is intentionally limited to the MG0-B compatibility contract.
It does not implement a backbone, tensor data contract, ontology, training,
checkpoint restoration, inference, or evaluation pipeline.

## Current Contract

- `config.py` exposes a resolved, hashable scaffold status. It is not a model
  configuration.
- `schemas.py` defines the compatibility classifications and the independent
  `mg0b` audit validator. It does not define model tensors.
- `mg0b_compatibility_audit.schema.json` validates the compatibility report.
- Placeholder packages import safely and expose only `NOT_IMPLEMENTED` status.

The MG1 ontology and MG3 dense-track/data contracts must be frozen before a
real `MarketGenomeConfig` or model input/output schema can be added.
