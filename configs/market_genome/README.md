# MarketGenome Configuration Boundary

MG0-B contains no trainable model configuration. The only active profile is
the independent `mg0b` report contract in `contracts_manifest.yaml`.

This independent manifest is deliberate: the MG0-A freeze tag anchors the
current legacy manifest and validator byte-for-byte. Keeping MG0-B separate
preserves the frozen `mg0` 13/13 semantic gate and every M0-M12 profile.

Future resolved model configs must include their complete canonical payload and
SHA256 in reports, artifacts, checkpoints, resume checks, and bundles. They may
not inherit defaults from `AlphaTradeConfig` implicitly.
