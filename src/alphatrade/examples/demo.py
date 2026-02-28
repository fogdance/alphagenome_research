# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example usage and demo for AlphaTrade."""

import jax
import jax.numpy as jnp
import numpy as np

from alphatrade import api
from alphatrade.core import model as model_lib
from alphatrade.core import preprocessing
from alphatrade.core import schemas
from alphatrade.training import training


def create_dummy_data(
    batch_size: int = 4,
    lookback_length: int = 4096,
    horizons: list[int] = None,
) -> schemas.TrainingBatch:
  """Creates dummy training data for testing.

  Args:
    batch_size: Number of samples in batch
    lookback_length: Length of lookback window
    horizons: List of prediction horizons

  Returns:
    TrainingBatch with dummy data.
  """
  if horizons is None:
    horizons = [1, 5, 20, 60]

  # Generate dummy OHLCV data
  rng = np.random.RandomState(42)

  # Simulate price movements
  base_price = 100.0
  returns = rng.randn(batch_size, lookback_length) * 0.001
  close_prices = base_price * np.exp(np.cumsum(returns, axis=1))

  # Generate OHLC from close
  open_prices = close_prices * (1 + rng.randn(batch_size, lookback_length) * 0.0005)
  high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(rng.randn(batch_size, lookback_length)) * 0.001)
  low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(rng.randn(batch_size, lookback_length)) * 0.001)
  volumes = rng.lognormal(10, 1, size=(batch_size, lookback_length))

  # Compute features for each sample
  preprocessor = preprocessing.FeaturePreprocessor()
  features_list = []

  for i in range(batch_size):
    features = preprocessor.compute_features(
        open_prices=jnp.array(open_prices[i]),
        high_prices=jnp.array(high_prices[i]),
        low_prices=jnp.array(low_prices[i]),
        close_prices=jnp.array(close_prices[i]),
        volumes=jnp.array(volumes[i]),
    )
    features_list.append(features)

  features_batch = jnp.stack(features_list, axis=0)

  # Generate dummy targets
  log_returns = {}
  for h in horizons:
    # Simulate future returns
    future_returns = rng.randn(batch_size) * 0.01 * np.sqrt(h)
    log_returns[h] = jnp.array(future_returns)

  inputs = schemas.AlphaTradeInput(features=features_batch)
  targets = schemas.AlphaTradeTargets(log_returns=log_returns)

  return schemas.TrainingBatch(inputs=inputs, targets=targets)


def demo_training():
  """Demonstrates model training on dummy data."""
  print("=" * 60)
  print("AlphaTrade Training Demo")
  print("=" * 60)

  # Create configuration
  config = schemas.AlphaTradeConfig(
      lookback_length=512,  # Smaller for demo
      stem_channels=64,
      num_encoder_stages=4,  # Downsample to /16 instead of /128
      d_model=256,
      num_transformer_layers=2,
      num_heads=4,
      horizons=[1, 5, 20],
  )

  print(f"\nModel Configuration:")
  print(f"  Lookback length: {config.lookback_length}")
  print(f"  Encoder stages: {config.num_encoder_stages}")
  print(f"  Transformer layers: {config.num_transformer_layers}")
  print(f"  d_model: {config.d_model}")
  print(f"  Horizons: {config.horizons}")

  # Create dummy data
  print("\nGenerating dummy training data...")
  batch = create_dummy_data(
      batch_size=2,
      lookback_length=config.lookback_length,
      horizons=config.horizons,
  )
  print(f"  Batch features shape: {batch.inputs.features.shape}")

  # Initialize training state
  print("\nInitializing model...")
  rng = jax.random.PRNGKey(0)
  train_state = training.create_train_state(
      config=config,
      rng=rng,
      learning_rate=1e-4,
      sample_batch=batch,
  )
  print("  Model initialized successfully")

  # Perform a few training steps
  print("\nTraining for 3 steps...")
  for step in range(3):
    rng, step_rng = jax.random.split(rng)
    train_state, metrics = training.train_step(
        train_state, batch, step_rng, config
    )
    print(f"  Step {step + 1}: loss={float(metrics['loss']):.6f}, "
          f"grad_norm={float(metrics['grad_norm']):.6f}")

  print("\nTraining demo completed!")
  return train_state, config


def demo_inference():
  """Demonstrates model inference."""
  print("\n" + "=" * 60)
  print("AlphaTrade Inference Demo")
  print("=" * 60)

  # Train a small model first
  train_state, config = demo_training()

  # Create test data
  print("\nGenerating test data...")
  test_batch = create_dummy_data(
      batch_size=1,
      lookback_length=config.lookback_length,
      horizons=config.horizons,
  )

  # Make prediction
  print("\nMaking prediction...")
  rng = jax.random.PRNGKey(42)
  predictions = training.predict(
      train_state,
      test_batch.inputs.features,
      rng,
  )

  print("\nPredictions:")
  for horizon, quantiles in predictions.log_return_quantiles.items():
    print(f"  Horizon {horizon}:")
    q_values = quantiles[0]  # First sample
    for i, q_level in enumerate(config.quantiles):
      print(f"    q{int(q_level*100):02d} = {float(q_values[i]):+.6f}")

  print("\nInference demo completed!")
  return train_state, config


def demo_api_service():
  """Demonstrates API service usage."""
  print("\n" + "=" * 60)
  print("AlphaTrade API Service Demo")
  print("=" * 60)

  # Get trained model
  train_state, config = demo_training()

  # Create service
  print("\nCreating API service...")
  service = api.create_service_from_train_state(
      train_state=train_state,
      config=config,
      model_version='alphatrade_v0.2_demo',
  )
  print("  Service created successfully")

  # Health check
  print("\nHealth check:")
  health = service.health_check()
  print(f"  Status: {health['status']}")
  print(f"  Model version: {health['model_version']}")

  # Create prediction request
  print("\nCreating prediction request...")
  test_batch = create_dummy_data(
      batch_size=1,
      lookback_length=config.lookback_length,
      horizons=config.horizons,
  )

  request = schemas.PredictionRequest(
      instrument_id='TEST_INSTRUMENT',
      asof_bar_end='2026-02-13T10:00:00+08:00',
      features=np.array(test_batch.inputs.features[0]),
      request_id='demo-request-001',
  )

  # Make prediction
  print("\nMaking API prediction...")
  response = service.predict(request)

  print(f"\nResponse:")
  print(f"  Model version: {response.model_version}")
  print(f"  Instrument: {response.instrument_id}")
  print(f"  Latency: {response.latency_ms:.2f} ms")
  print(f"\n  Predictions:")
  for horizon_str, quantiles in response.log_return_quantiles.items():
    print(f"    Horizon {horizon_str}: {[f'{q:+.6f}' for q in quantiles]}")

  print("\nAPI service demo completed!")


def demo_preprocessing():
  """Demonstrates feature preprocessing."""
  print("\n" + "=" * 60)
  print("AlphaTrade Preprocessing Demo")
  print("=" * 60)

  # Generate dummy OHLCV data
  print("\nGenerating dummy OHLCV data...")
  L = 100
  rng = np.random.RandomState(42)

  base_price = 100.0
  returns = rng.randn(L) * 0.001
  close_prices = base_price * np.exp(np.cumsum(returns))
  open_prices = close_prices * (1 + rng.randn(L) * 0.0005)
  high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(rng.randn(L)) * 0.001)
  low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(rng.randn(L)) * 0.001)
  volumes = rng.lognormal(10, 1, size=L)

  print(f"  Generated {L} bars")
  print(f"  Price range: [{low_prices.min():.2f}, {high_prices.max():.2f}]")

  # Compute features
  print("\nComputing features...")
  preprocessor = preprocessing.FeaturePreprocessor()
  features = preprocessor.compute_features(
      open_prices=jnp.array(open_prices),
      high_prices=jnp.array(high_prices),
      low_prices=jnp.array(low_prices),
      close_prices=jnp.array(close_prices),
      volumes=jnp.array(volumes),
  )

  print(f"  Features shape: {features.shape}")
  print(f"\n  Feature statistics:")
  feature_names = [
      'lr_close', 'hl_range', 'oc_return', 'log_vol',
      'pos_in_range', 'time_sin', 'time_cos', 'is_session_open'
  ]
  for i, name in enumerate(feature_names):
    mean = float(jnp.mean(features[:, i]))
    std = float(jnp.std(features[:, i]))
    print(f"    {name:15s}: mean={mean:+.6f}, std={std:.6f}")

  # Normalize features
  print("\nNormalizing features...")
  scaler = preprocessing.RobustScaler()
  normalized = scaler.fit_transform(np.array(features))

  print(f"  Normalized features shape: {normalized.shape}")
  print(f"  Scaler fitted: {scaler.fitted_}")

  # Validate
  print("\nValidating features...")
  try:
    preprocessing.validate_features(jnp.array(normalized), normalized=True)
    print("  ✓ Features are valid")
  except ValueError as e:
    print(f"  ✗ Validation failed: {e}")

  print("\nPreprocessing demo completed!")


def main():
  """Runs all demos."""
  print("\n" + "=" * 60)
  print("AlphaTrade Complete Demo Suite")
  print("=" * 60)

  # Run demos
  demo_preprocessing()
  demo_training()
  demo_inference()
  demo_api_service()

  print("\n" + "=" * 60)
  print("All demos completed successfully!")
  print("=" * 60)


if __name__ == '__main__':
  main()
