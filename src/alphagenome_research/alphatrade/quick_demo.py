"""Quick demo for AlphaTrade - runs faster than full demo."""

import jax
import jax.numpy as jnp
import numpy as np

from alphagenome_research.alphatrade import model as model_lib
from alphagenome_research.alphatrade import preprocessing
from alphagenome_research.alphatrade import schemas
from alphagenome_research.alphatrade import training


def quick_demo():
  """Quick demonstration of AlphaTrade functionality."""
  print("=" * 60)
  print("AlphaTrade Quick Demo")
  print("=" * 60)

  # Small configuration for fast testing
  config = schemas.AlphaTradeConfig(
      lookback_length=128,
      stem_channels=32,
      num_encoder_stages=3,  # Downsample to /8
      channel_increment=32,
      d_model=128,
      num_transformer_layers=2,
      num_heads=4,
      horizons=[1, 5],
      quantiles=[0.25, 0.5, 0.75],
  )

  print(f"\nConfiguration:")
  print(f"  Lookback: {config.lookback_length}")
  print(f"  Horizons: {config.horizons}")
  print(f"  Quantiles: {config.quantiles}")

  # Generate dummy data
  print("\nGenerating dummy data...")
  rng = np.random.RandomState(42)
  L = config.lookback_length

  close_prices = 100.0 * np.exp(np.cumsum(rng.randn(L) * 0.001))
  open_prices = close_prices * (1 + rng.randn(L) * 0.0005)
  high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(rng.randn(L)) * 0.001)
  low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(rng.randn(L)) * 0.001)
  volumes = rng.lognormal(10, 1, size=L)

  # Compute features
  preprocessor = preprocessing.FeaturePreprocessor()
  features = preprocessor.compute_features(
      open_prices=jnp.array(open_prices),
      high_prices=jnp.array(high_prices),
      low_prices=jnp.array(low_prices),
      close_prices=jnp.array(close_prices),
      volumes=jnp.array(volumes),
  )
  features_batch = features[jnp.newaxis, ...]  # Add batch dim

  print(f"  Features shape: {features_batch.shape}")

  # Initialize model
  print("\nInitializing model...")
  rng_key = jax.random.PRNGKey(0)
  train_state = training.create_train_state(
      config=config,
      rng=rng_key,
      learning_rate=1e-4,
  )
  print("  ✓ Model initialized")

  # Make prediction
  print("\nMaking prediction...")
  predictions = training.predict(
      train_state,
      features_batch,
      rng_key,
  )

  print("\nPredictions:")
  for horizon, quantiles in predictions.log_return_quantiles.items():
    q_values = quantiles[0]
    print(f"  Horizon {horizon}:")
    for i, q_level in enumerate(config.quantiles):
      print(f"    q{int(q_level*100):02d} = {float(q_values[i]):+.6f}")

  print("\n" + "=" * 60)
  print("Quick demo completed successfully!")
  print("=" * 60)


if __name__ == '__main__':
  quick_demo()
