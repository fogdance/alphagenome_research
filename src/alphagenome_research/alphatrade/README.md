# AlphaTrade

AlphaTrade is a multi-horizon financial time series predictor inspired by AlphaGenome's architecture, adapted for causal time-series modeling.

## Overview

AlphaTrade predicts future log-return distributions (quantiles) at multiple horizons from 1-minute OHLCV features. It uses a U-Net style architecture with causal convolutions and transformers to ensure strict causality (no future information leakage).

## Architecture

The model follows AlphaGenome's design principles but adapted for time-series:

1. **Stem**: Feature embedding (8 features → 128 channels)
2. **Encoder**: 6-stage causal downsampling (/128 resolution)
3. **Transformer Tower**: Causal attention with RoPE and logits soft-cap
4. **Decoder**: 6-stage causal upsampling with skip connections
5. **Readout**: Extract final timestep embedding
6. **Heads**: Multi-horizon quantile prediction

### Key Features

- **Causal Convolutions**: Left-padding ensures no future information leakage
- **Weight Standardization**: Improves training stability (from AlphaGenome)
- **RoPE**: Rotary position embeddings for better position encoding
- **Logits Soft-Cap**: Tanh capping at 5.0 for attention stability
- **Quantile Prediction**: Prevents crossing via cumulative delta approach

## Input Features

AlphaTrade expects 8 features per timestep (strictly causal):

| Index | Feature | Definition |
|-------|---------|------------|
| 0 | `lr_close` | log(C_i / C_{i-1}) |
| 1 | `hl_range` | log(H_i - L_i + ε) |
| 2 | `oc_return` | log(C_i / O_i) |
| 3 | `log_vol` | log(V_i + 1) |
| 4 | `pos_in_range` | (C_i - L_i) / (H_i - L_i + ε) |
| 5 | `time_sin` | sin(2π × minute_index / period) |
| 6 | `time_cos` | cos(2π × minute_index / period) |
| 7 | `is_session_open` | 0/1 flag for trading session |

## Output

For each horizon h ∈ {1, 5, 20, 60} minutes:
- **Quantiles**: [q10, q25, q50, q75, q90] of log(C_{t+h} / C_t)

Optional outputs:
- Range quantiles (volatility)
- Volume quantiles
- Regime classification

## Installation

```bash
# Install alphagenome_research package
pip install -e /path/to/alphagenome_research

# Dependencies: JAX, Haiku, Optax, NumPy
```

## Quick Start

### 1. Training

```python
from alphagenome_research.alphatrade import schemas, training
import jax

# Create configuration
config = schemas.AlphaTradeConfig(
    lookback_length=4096,
    horizons=[1, 5, 20, 60],
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
)

# Initialize model
rng = jax.random.PRNGKey(0)
train_state = training.create_train_state(config, rng)

# Training loop
for batch in train_loader:
    rng, step_rng = jax.random.split(rng)
    train_state, metrics = training.train_step_jit(
        train_state, batch, step_rng, config
    )
    print(f"Loss: {metrics['loss']:.6f}")
```

### 2. Inference

```python
from alphagenome_research.alphatrade import api, preprocessing
import numpy as np

# Prepare features from OHLCV data
preprocessor = preprocessing.FeaturePreprocessor()
features = preprocessor.compute_features(
    open_prices=open_prices,
    high_prices=high_prices,
    low_prices=low_prices,
    close_prices=close_prices,
    volumes=volumes,
)

# Create service
service = api.create_service_from_train_state(
    train_state=train_state,
    config=config,
)

# Make prediction
request = schemas.PredictionRequest(
    instrument_id='IF2403',
    asof_bar_end='2026-02-13T10:07:00+08:00',
    features=features,
)
response = service.predict(request)

# Access predictions
for horizon, quantiles in response.log_return_quantiles.items():
    print(f"Horizon {horizon}: {quantiles}")
```

### 3. API Service

```python
from alphagenome_research.alphatrade import api

# Create HTTP handler
service = api.create_service_from_train_state(train_state, config)
handler = api.AlphaTradeHTTPHandler(service)

# Handle prediction request
request_json = {
    'instrument_id': 'IF2403',
    'asof_bar_end': '2026-02-13T10:07:00+08:00',
    'inputs': {'features': features.tolist()},
}
response_json = handler.handle_predict(request_json)

# Health check
health = handler.handle_health()
```

## Demo

Run the complete demo suite:

```bash
python -m alphagenome_research.alphatrade.demo
```

This demonstrates:
- Feature preprocessing from OHLCV data
- Model training on dummy data
- Inference and prediction
- API service usage

## Model Configuration

Default configuration (v0.2 baseline):

```python
config = schemas.AlphaTradeConfig(
    # Input
    lookback_length=4096,
    num_features=8,

    # Architecture
    stem_channels=128,
    num_encoder_stages=6,  # Downsample to /128
    channel_increment=64,

    # Transformer
    d_model=512,
    num_transformer_layers=6,
    num_heads=8,
    head_dim=64,
    mlp_expansion=4,
    logits_soft_cap=5.0,

    # Output
    horizons=[1, 5, 20, 60],
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
)
```

## Loss Functions

### Quantile Pinball Loss

```python
L_pinball(q, y, ŷ) = max(q(y - ŷ), (q-1)(y - ŷ))
```

### Quantile Crossing Penalty

Ensures monotonicity: q_i ≤ q_{i+1}

```python
L_crossing = Σ max(0, q_i - q_{i+1})
```

### Multi-Horizon Loss

```python
L_total = Σ_h w_h × (L_pinball + λ × L_crossing)
```

Default weights: {1: 1.0, 5: 1.0, 20: 0.8, 60: 0.6}

## Evaluation Metrics

- **Pinball Loss**: Per-horizon quantile loss
- **Coverage**: Empirical coverage rate for each quantile
- **Calibration Error**: Mean absolute deviation from expected coverage
- **Crossing Rate**: Fraction of predictions with quantile violations

## API Specification

See `docs/alphaTrade/api.md` for the complete v0.1 API specification.

### Endpoints

- `POST /v0.1/predict`: Make predictions
- `GET /v0.1/health`: Health check

### Request Format

```json
{
  "instrument_id": "IF2403",
  "asof_bar_end": "2026-02-13T10:07:00+08:00",
  "lookback_L": 4096,
  "horizons": [1, 5, 20, 60],
  "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
  "inputs": {
    "features": [[...], ...]
  }
}
```

### Response Format

```json
{
  "model_version": "alphatrade_v0.2",
  "instrument_id": "IF2403",
  "asof_bar_end": "2026-02-13T10:07:00+08:00",
  "pred": {
    "log_return_quantiles": {
      "1": [-0.0008, -0.0003, 0.0001, 0.0004, 0.0009],
      "5": [-0.0019, -0.0008, 0.0002, 0.0010, 0.0021],
      ...
    }
  },
  "latency_ms": 12.7
}
```

## Design Principles

### Causality (Hard Constraint)

All operations must be strictly causal:
- Convolutions use left-padding only
- Pooling only looks at past values
- Attention uses causal masking
- No future information in features

### AlphaGenome-Inspired Engineering

Borrowed from AlphaGenome:
- Weight standardization in convolutions
- Repeat-based upsampling with residual scaling
- RMSNorm for normalization
- RoPE for position encoding
- Logits soft-cap for attention stability

### Differences from AlphaGenome

| Aspect | AlphaGenome | AlphaTrade |
|--------|-------------|------------|
| Domain | Genomics (DNA) | Finance (time-series) |
| Causality | Bidirectional | Strictly causal |
| Input | One-hot DNA (4 channels) | OHLCV features (8 channels) |
| Output | Multi-modal tracks | Quantile distributions |
| Padding | SAME (symmetric) | Causal (left-only) |
| Attention | Full attention | Causal masked |

## File Structure

```
alphagenome_research/src/alphagenome_research/alphatrade/
├── __init__.py              # Package initialization
├── causal_layers.py         # Causal convolution layers
├── causal_attention.py      # Causal transformer layers
├── model.py                 # Main AlphaTrade model
├── schemas.py               # Data schemas and configs
├── losses.py                # Loss functions
├── training.py              # Training utilities
├── preprocessing.py         # Feature preprocessing
├── api.py                   # API service
├── demo.py                  # Demo and examples
└── README.md                # This file
```

## References

- AlphaGenome paper: [Nature 2026](https://doi.org/10.1038/s41586-025-10014-0)
- AlphaGenome code: [GitHub](https://github.com/google-deepmind/alphagenome_research)
- Model specification: `docs/alphaTrade/mode.md`
- API specification: `docs/alphaTrade/api.md`

## License

Copyright 2026 Google LLC. Licensed under Apache 2.0.
