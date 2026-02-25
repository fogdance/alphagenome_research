#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AlphaTrade v0.2 minimal-but-production training loop.

Features:
  - Gradient accumulation (micro-batch) => effective batch_size
  - Optional bf16 mixed precision (params fp32 + compute bf16 if Haiku supports)
  - Full-dataset eval traversal (stable averages)
  - Save params/state/opt_state + scaler params
  - Simple quantile calibration report (coverage + ECE)

Expected NPZ format (train/val):
  - X: [N, L, 8]  (or "features")
  - y_h{h}: [N] for each horizon h (e.g., y_h1, y_h5, y_h20, y_h60)
    (also accepts: y_return_h{h}, y_returns_h{h}, y_logret_h{h})
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from alphagenome_research.alphatrade import model as model_lib
from alphagenome_research.alphatrade import schemas


# -----------------------------
# Utilities
# -----------------------------

def _now() -> str:
  return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def ensure_dir(p: Path) -> None:
  p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def atomic_pickle_dump(path: Path, obj) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with tmp.open("wb") as f:
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
  tmp.replace(path)


def _find_feature_array(npz: np.lib.npyio.NpzFile) -> np.ndarray:
  if "X" in npz.files:
    return npz["X"]
  if "features" in npz.files:
    return npz["features"]
  # Fallback: find the first 3D array with last dim 8
  for k in npz.files:
    arr = npz[k]
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 8:
      return arr
  raise ValueError(f"Cannot find feature array in npz. Keys={npz.files}")


def _parse_targets(npz: np.lib.npyio.NpzFile, horizons_arg: List[int] | None) -> Dict[int, np.ndarray]:
  keys = npz.files
  y: Dict[int, np.ndarray] = {}

  def try_match(prefixes: List[str], k: str) -> int | None:
    for p in prefixes:
      if k.startswith(p):
        suffix = k[len(p):]
        if suffix.isdigit():
          return int(suffix)
    return None

  prefixes = ["y_h", "y_return_h", "y_returns_h", "y_logret_h"]
  for k in keys:
    h = try_match(prefixes, k)
    if h is not None:
      y[h] = npz[k]

  # Alternative: a single "y" matrix [N, H] with provided horizons
  if not y and "y" in keys:
    if horizons_arg is None or len(horizons_arg) == 0:
      raise ValueError("NPZ has key 'y' but --horizons not provided. "
                       "Provide --horizons like 1,5,20,60.")
    ym = npz["y"]
    if ym.ndim != 2 or ym.shape[1] != len(horizons_arg):
      raise ValueError(f"'y' must be [N, len(horizons)] but got {ym.shape}. "
                       f"horizons={horizons_arg}")
    for i, h in enumerate(horizons_arg):
      y[h] = ym[:, i]

  if not y:
    raise ValueError(
        "Cannot find targets in npz. Expect keys like y_h1/y_h5/... or 'y' with --horizons. "
        f"Keys={keys}"
    )

  # Validate shapes
  n = None
  for h, arr in y.items():
    if arr.ndim != 1:
      raise ValueError(f"Target for horizon {h} must be 1D [N], got shape={arr.shape}")
    n = arr.shape[0] if n is None else n
    if arr.shape[0] != n:
      raise ValueError(f"Target lengths mismatch: horizon {h} has {arr.shape[0]} vs {n}")

  return dict(sorted(y.items(), key=lambda kv: kv[0]))


def load_dataset(npz_path: str, horizons_arg: List[int] | None) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
  npz = np.load(npz_path, allow_pickle=True)
  X = _find_feature_array(npz)
  y = _parse_targets(npz, horizons_arg)
  return X, y


# -----------------------------
# Robust scaler (sample-based)
# -----------------------------

def fit_robust_scaler_from_samples(
    X: np.ndarray,
    sample_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
  """
  Approx robust scaler by sampling points from X flattened over (N,L).

  Returns:
    medians: [8]
    iqrs: [8]
  """
  assert X.ndim == 3 and X.shape[-1] == 8, X.shape
  N, L, F = X.shape
  rng = np.random.default_rng(seed)

  total = N * L
  sample_points = int(min(sample_points, total))
  idx = rng.integers(0, total, size=(sample_points,), endpoint=False)
  n = idx // L
  t = idx % L
  sample = X[n, t, :].astype(np.float32)  # [S, 8]

  med = np.median(sample, axis=0)
  q75 = np.percentile(sample, 75, axis=0)
  q25 = np.percentile(sample, 25, axis=0)
  iqr = q75 - q25
  iqr = np.where(iqr < 1e-6, 1.0, iqr)
  return med.astype(np.float32), iqr.astype(np.float32)


def apply_robust_scaler(x: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
  return (x - med[None, None, :]) / iqr[None, None, :]


# -----------------------------
# Mixed precision helper
# -----------------------------

def enable_bf16_mixed_precision_if_possible() -> bool:
  """
  Prefer: params fp32 + compute bf16 via Haiku mixed_precision policies.
  If unavailable, return False and caller can fallback to casting inputs to bf16.
  """
  try:
    from haiku import mixed_precision as mp  # type: ignore

    policy = mp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.bfloat16,
    )

    # Apply policy to major modules. (Safe even if some modules ignore it.)
    mp.set_policy(model_lib.AlphaTrade, policy)
    mp.set_policy(model_lib.TemporalEncoder, policy)
    mp.set_policy(model_lib.TemporalDecoder, policy)
    mp.set_policy(model_lib.QuantileHead, policy)
    mp.set_policy(model_lib.RegimeHead, policy)

    # These imports exist in v0.2
    from alphagenome_research.alphatrade import causal_layers, causal_attention  # noqa
    mp.set_policy(causal_layers.CausalStandardizedConv1D, policy)
    mp.set_policy(causal_layers.CausalConvBlock, policy)
    mp.set_policy(causal_layers.CausalDownResBlock, policy)
    mp.set_policy(causal_layers.CausalUpResBlock, policy)
    mp.set_policy(causal_layers.FeatureEmbedder, policy)

    mp.set_policy(causal_attention.CausalMHABlock, policy)
    mp.set_policy(causal_attention.CausalMLPBlock, policy)
    mp.set_policy(causal_attention.CausalTransformerLayer, policy)
    mp.set_policy(causal_attention.CausalTransformerTower, policy)

    return True
  except Exception:
    return False


# -----------------------------
# Model transforms
# -----------------------------

def build_transforms(config: schemas.AlphaTradeConfig):
  # Loss transform: (features, log_returns_dict) -> (loss, metrics_dict)
  def loss_forward(features: jax.Array, log_returns: Dict[int, jax.Array]):
    model = model_lib.AlphaTrade(config)
    batch = schemas.TrainingBatch(
        inputs=schemas.AlphaTradeInput(features=features),
        targets=schemas.AlphaTradeTargets(log_returns=log_returns),
    )
    return model.loss(batch)  # (loss, metrics_dict)

  loss_t = hk.transform_with_state(loss_forward)

  # Predict transform: features -> AlphaTradeOutput
  def pred_forward(features: jax.Array):
    model = model_lib.AlphaTrade(config)
    return model(features)

  pred_t = hk.transform_with_state(pred_forward)
  return loss_t, pred_t


# -----------------------------
# Training step with grad accumulation
# -----------------------------

def make_train_step(loss_t, optimizer: optax.GradientTransformation, accum_steps: int):
  """
  Returns a jitted function that takes a global batch of size (micro_batch*accum_steps),
  splits into micro-batches, accumulates grads, applies optimizer update once.
  """

  @jax.jit
  def train_step(params, state, opt_state, rng, xb, yb_dict):
    # xb: [B, L, 8], yb_dict[h]: [B]
    B = xb.shape[0]
    if B % accum_steps != 0:
      raise ValueError(f"Global batch {B} must be divisible by accum_steps={accum_steps}")
    micro_bs = B // accum_steps

    xb_m = xb.reshape((accum_steps, micro_bs) + xb.shape[1:])
    yb_m = {h: yb_dict[h].reshape((accum_steps, micro_bs)) for h in yb_dict}

    grads0 = jax.tree_util.tree_map(jnp.zeros_like, params)
    loss0 = jnp.zeros((), dtype=jnp.float32)

    def micro_body(carry, inp):
      grads_sum, state_c, rng_c, loss_sum = carry
      x_i, y_i = inp
      rng_c, rng_i = jax.random.split(rng_c)

      def _loss_fn(p):
        (loss, metrics), new_state = loss_t.apply(p, state_c, rng_i, x_i, y_i)
        # Only backprop through loss; carry new_state (no grad)
        return loss, (metrics, new_state)

      (loss, (metrics, new_state)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(params)

      grads_sum = jax.tree_util.tree_map(lambda a, b: a + b, grads_sum, grads)
      loss_sum = loss_sum + loss.astype(jnp.float32)
      return (grads_sum, new_state, rng_c, loss_sum), metrics

    (grads_sum, state_out, rng_out, loss_sum), metrics_mb = jax.lax.scan(
        micro_body,
        (grads0, state, rng, loss0),
        (xb_m, yb_m),
    )

    grads_mean = jax.tree_util.tree_map(lambda g: g / accum_steps, grads_sum)
    updates, opt_state_out = optimizer.update(grads_mean, opt_state, params)
    params_out = optax.apply_updates(params, updates)

    loss_mean = loss_sum / accum_steps
    grad_norm = optax.global_norm(grads_mean)

    # metrics: keep small + stable
    metrics_out = {
        "loss": loss_mean,
        "grad_norm": grad_norm,
    }
    return params_out, state_out, opt_state_out, rng_out, metrics_out

  return train_step



# -----------------------------
# Eval (full traversal) + calibration
# -----------------------------
def make_eval_fns(loss_t, pred_t, quantiles: List[float]):
  quantiles_np = np.asarray(quantiles, dtype=np.float32)  # [Q]
  Q = int(len(quantiles_np))

  # JIT-friendly predict: only return dict[int -> (B,Q)] of quantiles
  def _pred_apply_quantiles(params, state, rng, features_bls8):
    out, _new_state = pred_t.apply(params, state, rng, features_bls8)
    return out.log_return_quantiles  # dict[int, jax.Array [B,Q]]

  pred_apply_jit = jax.jit(_pred_apply_quantiles)

  def eval_full(
      *,
      params,
      state,
      X_val: np.ndarray,                 # [N,L,8] (unscaled)
      y_val: Dict[int, np.ndarray],      # dict[h] -> [N]
      med: np.ndarray,                   # [8]
      iqr: np.ndarray,                   # [8]
      batch_size: int,
      rng_key,
      bf16_inputs: bool = False,
  ):
    horizons = sorted(list(y_val.keys()))
    N = X_val.shape[0]

    pinball_sum = {h: 0.0 for h in horizons}
    crossing_sum = {h: 0.0 for h in horizons}
    n_sum = {h: 0 for h in horizons}

    below_counts = {h: np.zeros((Q,), dtype=np.float64) for h in horizons}
    total_counts = {h: 0 for h in horizons}

    for start in range(0, N, batch_size):
      end = min(start + batch_size, N)
      xb = X_val[start:end].astype(np.float32)  # [B,L,8]
      B = xb.shape[0]

      # scale on-the-fly (avoid extra val copy)
      xb = (xb - med[None, None, :]) / iqr[None, None, :]
      xb_j = jnp.asarray(xb)
      if bf16_inputs:
        xb_j = xb_j.astype(jnp.bfloat16)

      rng_key, rk = jax.random.split(rng_key)

      lrq = pred_apply_jit(params, state, rk, xb_j)  # dict[h] -> [B,Q]
      lrq = jax.device_get(lrq)                      # to CPU

      for h in horizons:
        y_true = np.asarray(y_val[h][start:end], dtype=np.float32)  # [B]
        y_pred = np.asarray(lrq[h], dtype=np.float32)              # [B,Q]

        resid = y_true[:, None] - y_pred
        q = quantiles_np[None, :]
        loss = np.where(resid >= 0, q * resid, (q - 1.0) * resid)  # [B,Q]
        pinball = float(loss.mean())

        diffs = y_pred[:, 1:] - y_pred[:, :-1]  # [B,Q-1]
        crossing = float(np.maximum(0.0, -diffs).mean())

        pinball_sum[h] += pinball * B
        crossing_sum[h] += crossing * B
        n_sum[h] += B

        below = (y_true[:, None] <= y_pred)  # [B,Q]
        below_counts[h] += below.sum(axis=0).astype(np.float64)
        total_counts[h] += B

    # finalize
    ev_metrics = {}
    calib_by_h = {}

    totals = []
    for h in horizons:
      if n_sum[h] == 0:
        continue
      pinball_mean = pinball_sum[h] / n_sum[h]
      crossing_mean = crossing_sum[h] / n_sum[h]
      total_h = pinball_mean + crossing_mean
      totals.append(total_h)

      ev_metrics[f"eval_pinball_h{h}"] = float(pinball_mean)
      ev_metrics[f"eval_crossing_h{h}"] = float(crossing_mean)
      ev_metrics[f"eval_total_h{h}"] = float(total_h)

      cov = (below_counts[h] / max(1, total_counts[h])).astype(np.float64)  # [Q]
      ece = float(np.mean(np.abs(cov - quantiles_np)))

      # match your printing style: coverage_q10/coverage_q25/...
      d = {"ece": ece}
      for i, qv in enumerate(quantiles_np):
        d[f"coverage_q{int(qv*100):02d}"] = float(cov[i])
      calib_by_h[h] = d

    ev_metrics["eval_loss"] = float(np.mean(totals)) if totals else float("nan")
    return ev_metrics, calib_by_h

  return eval_full



# -----------------------------
# Main
# -----------------------------

def parse_args():
  ap = argparse.ArgumentParser()

  ap.add_argument("--train_npz", type=str, required=True)
  ap.add_argument("--val_npz", type=str, required=True)
  ap.add_argument("--workdir", type=str, required=True)

  ap.add_argument("--steps", type=int, default=20000)
  ap.add_argument("--lr", type=float, default=1e-4)
  ap.add_argument("--seed", type=int, default=0)

  # Effective batch size
  ap.add_argument("--batch_size", type=int, default=32, help="Effective/global batch size")
  ap.add_argument("--micro_batch", type=int, default=4, help="Micro-batch per step (GPU resident)")
  ap.add_argument("--accum_steps", type=int, default=0, help="If 0, inferred as batch_size//micro_batch")

  ap.add_argument("--eval_every", type=int, default=500)
  ap.add_argument("--ckpt_every", type=int, default=1000)
  ap.add_argument("--eval_batch", type=int, default=8, help="Eval traversal batch size")

  # Scaler
  ap.add_argument("--scaler_fit_samples", type=int, default=2_000_000,
                  help="How many (N,L) points to sample for robust scaler fit.")
  ap.add_argument("--scaler_seed", type=int, default=123)

  # Quantiles / horizons
  ap.add_argument("--quantiles", type=str, default="0.1,0.25,0.5,0.75,0.9")
  ap.add_argument("--horizons", type=str, default="", help="Optional: override horizons e.g. 1,5,20,60")

  # Model overrides (optional)
  ap.add_argument("--d_model", type=int, default=512)
  ap.add_argument("--num_layers", type=int, default=6)
  ap.add_argument("--num_heads", type=int, default=8)
  ap.add_argument("--head_dim", type=int, default=64)
  ap.add_argument("--mlp_expansion", type=int, default=4)
  ap.add_argument("--stem_channels", type=int, default=128)
  ap.add_argument("--num_encoder_stages", type=int, default=6)
  ap.add_argument("--channel_increment", type=int, default=64)
  ap.add_argument("--logits_soft_cap", type=float, default=5.0)
  ap.add_argument("--max_position", type=int, default=8192)

  # bf16
  ap.add_argument("--bf16", action="store_true",
                  help="Enable bf16 to save memory (prefer Haiku mixed_precision; fallback to input cast).")

  # Misc stability
  ap.add_argument("--grad_clip", type=float, default=1.0, help="Global norm clip (0 to disable).")

  # --- Production checkpoint policy ---
  ap.add_argument(
      "--keep_last_ckpts",
      type=int,
      default=3,
      help=("Keep last K periodic ckpt_*.pkl files. "
            "0 means keep none (only latest + best). "
            "-1 means keep all (no pruning)."),
  )
  ap.add_argument(
      "--save_ckpt_every_eval",
      action="store_true",
      help="Also save periodic/latest right after each eval (in addition to ckpt_every).",
  )

  # --- Early stopping (based on eval_loss) ---
  ap.add_argument(
      "--early_stop",
      action="store_true",
      help="Enable early stopping based on eval_loss improvements.",
  )
  ap.add_argument(
      "--early_stop_patience",
      type=int,
      default=10,
      help="Number of evals with no improvement before stopping.",
  )
  ap.add_argument(
      "--early_stop_min_delta",
      type=float,
      default=0.0,
      help="Require improvement by at least this delta to reset patience.",
  )
  ap.add_argument(
      "--early_stop_warmup_evals",
      type=int,
      default=2,
      help="Ignore early stopping for first N evals (still track best).",
  )

  return ap.parse_args()


def main():
  args = parse_args()
  workdir = Path(args.workdir)
  ensure_dir(workdir)

  print(f"[{_now()}] Tip env: export XLA_PYTHON_CLIENT_PREALLOCATE=false ; export TF_GPU_ALLOCATOR=cuda_malloc_async")

  # -----------------------------
  # Helpers: ckpt pruning + saving
  # -----------------------------
  def _prune_old_ckpts(keep_last: int):
    """Prune ckpt_*.pkl, keeping the last K (by filename order)."""
    if keep_last < 0:
      return  # keep all
    ckpts = sorted(workdir.glob("ckpt_*.pkl"))
    if keep_last == 0:
      for p in ckpts:
        try:
          p.unlink()
        except OSError:
          pass
      return
    if len(ckpts) <= keep_last:
      return
    for p in ckpts[:-keep_last]:
      try:
        p.unlink()
      except OSError:
        pass

  def _make_ckpt_payload(step: int, best_eval_loss: float, best_step: int):
    return {
        "step": step,
        "time": _now(),
        "params": jax.device_get(params),
        "state": jax.device_get(state),
        "opt_state": jax.device_get(opt_state),
        "config": dataclasses.asdict(config),
        "scaler": {"medians": med.tolist(), "iqrs": iqr.tolist()},
        "best": {"best_eval_loss": best_eval_loss, "best_step": best_step},
    }

  def save_periodic_ckpt(step: int, best_eval_loss: float, best_step: int, reason: str):
    """Always write latest.pkl; optionally write ckpt_XXXXXX.pkl and prune."""
    payload = _make_ckpt_payload(step, best_eval_loss, best_step)

    atomic_pickle_dump(workdir / "latest.pkl", payload)

    if int(args.keep_last_ckpts) != 0:
      ckpt_path = workdir / f"ckpt_{step:06d}.pkl"
      atomic_pickle_dump(ckpt_path, payload)
      _prune_old_ckpts(int(args.keep_last_ckpts))
      print(f"[{_now()}] [ckpt] ({reason}) saved {ckpt_path.name} + latest.pkl (keep_last_ckpts={args.keep_last_ckpts})")
    else:
      _prune_old_ckpts(0)  # ensure no old ckpt_*.pkl remain
      print(f"[{_now()}] [ckpt] ({reason}) saved latest.pkl (no ckpt_*.pkl kept)")

  def save_best_ckpt(step: int, best_eval_loss: float, best_step: int):
    """Only called when eval improves."""
    payload = _make_ckpt_payload(step, best_eval_loss, best_step)
    atomic_pickle_dump(workdir / "best.pkl", payload)
    print(f"[{_now()}] [best] saved best.pkl (best_eval_loss={best_eval_loss:.6f} at step={best_step})")

  # If user wants "keep none", proactively clean old ckpts.
  _prune_old_ckpts(int(args.keep_last_ckpts))

  # -----------------------------
  # Parse quantiles/horizons
  # -----------------------------
  quantiles = [float(x) for x in args.quantiles.split(",") if x.strip()]
  horizons_arg = [int(x) for x in args.horizons.split(",") if x.strip()] if args.horizons.strip() else None

  # -----------------------------
  # Load data
  # -----------------------------
  X_train, y_train = load_dataset(args.train_npz, horizons_arg)
  X_val, y_val = load_dataset(args.val_npz, horizons_arg)

  assert X_train.shape[-1] == 8 and X_val.shape[-1] == 8
  L = X_train.shape[1]
  if X_val.shape[1] != L:
    raise ValueError(f"Train/Val lookback length mismatch: train L={L} val L={X_val.shape[1]}")

  horizons = sorted(list(y_train.keys()))
  if sorted(list(y_val.keys())) != horizons:
    raise ValueError(f"Train/Val horizons mismatch: train={sorted(y_train.keys())} val={sorted(y_val.keys())}")

  print(f"[data] train: {X_train.shape} val: {X_val.shape} horizons={horizons} quantiles={quantiles}")

  # -----------------------------
  # Effective batch + accumulation
  # -----------------------------
  if args.accum_steps and args.accum_steps > 0:
    accum_steps = int(args.accum_steps)
    micro_batch = int(args.micro_batch)
    batch_size = micro_batch * accum_steps
    if batch_size != args.batch_size:
      print(f"[warn] You set --accum_steps, overriding effective batch_size to {batch_size} "
            f"(micro_batch={micro_batch} * accum_steps={accum_steps}).")
  else:
    micro_batch = int(args.micro_batch)
    if args.batch_size % micro_batch != 0:
      raise ValueError(f"--batch_size={args.batch_size} must be divisible by --micro_batch={micro_batch}")
    batch_size = int(args.batch_size)
    accum_steps = batch_size // micro_batch

  print(f"[batch] effective batch_size={batch_size} (micro_batch={micro_batch} accum_steps={accum_steps})")

  # -----------------------------
  # Robust scaler
  # -----------------------------
  scaler_path = workdir / "scaler_params.json"
  if scaler_path.exists():
    sp = json.loads(scaler_path.read_text(encoding="utf-8"))
    med = np.array(sp["medians"], dtype=np.float32)
    iqr = np.array(sp["iqrs"], dtype=np.float32)
    print("[scaler] loaded existing scaler_params.json")
  else:
    med, iqr = fit_robust_scaler_from_samples(
        X_train,
        sample_points=args.scaler_fit_samples,
        seed=args.scaler_seed,
    )
    save_json(scaler_path, {
        "medians": med.tolist(),
        "iqrs": iqr.tolist(),
        "fit_samples": int(args.scaler_fit_samples),
        "seed": int(args.scaler_seed),
    })
    print(f"[scaler] fitted (sample={args.scaler_fit_samples}) and saved scaler_params.json")

  # -----------------------------
  # bf16 settings
  # -----------------------------
  bf16_inputs = False
  if args.bf16:
    ok = enable_bf16_mixed_precision_if_possible()
    if ok:
      print("[bf16] Haiku mixed_precision enabled: params fp32 + compute bf16")
      bf16_inputs = False
    else:
      print("[bf16] mixed_precision not available; fallback to casting inputs to bf16.")
      bf16_inputs = True

  # -----------------------------
  # Build config
  # -----------------------------
  config = schemas.AlphaTradeConfig(
      lookback_length=int(L),
      num_features=8,
      stem_channels=int(args.stem_channels),
      num_encoder_stages=int(args.num_encoder_stages),
      channel_increment=int(args.channel_increment),
      d_model=int(args.d_model),
      num_transformer_layers=int(args.num_layers),
      num_heads=int(args.num_heads),
      head_dim=int(args.head_dim),
      mlp_expansion=int(args.mlp_expansion),
      logits_soft_cap=float(args.logits_soft_cap),
      max_position=int(args.max_position),
      horizons=list(horizons),
      quantiles=list(quantiles),
  )

  # -----------------------------
  # Transforms + init
  # -----------------------------
  loss_t, pred_t = build_transforms(config)

  rng = jax.random.PRNGKey(args.seed)
  rng, init_rng = jax.random.split(rng)

  dummy_x = jnp.zeros((1, L, 8), dtype=jnp.float32)
  dummy_y = {h: jnp.zeros((1,), dtype=jnp.float32) for h in horizons}
  params, state = loss_t.init(init_rng, dummy_x, dummy_y)

  # -----------------------------
  # Optimizer
  # -----------------------------
  lr = float(args.lr)
  tx_chain = []
  if args.grad_clip and args.grad_clip > 0:
    tx_chain.append(optax.clip_by_global_norm(args.grad_clip))
  tx_chain.append(optax.adam(lr))
  optimizer = optax.chain(*tx_chain)
  opt_state = optimizer.init(params)

  # Train step
  train_step = make_train_step(loss_t, optimizer, accum_steps)

  # Eval
  eval_full = make_eval_fns(loss_t, pred_t, quantiles)

  # -----------------------------
  # Sampler
  # -----------------------------
  rng_np = np.random.default_rng(args.seed + 999)
  N_train = X_train.shape[0]

  def sample_global_batch():
    idx = rng_np.integers(0, N_train, size=(batch_size,), endpoint=False)
    xb = X_train[idx].astype(np.float32)  # [B,L,8]
    xb = apply_robust_scaler(xb, med, iqr).astype(np.float32)
    yb = {h: y_train[h][idx].astype(np.float32) for h in horizons}
    return xb, yb

  # -----------------------------
  # Early stopping state
  # -----------------------------
  best_eval_loss = float("inf")
  best_step = -1
  num_evals = 0
  bad_evals = 0  # evals since last improvement

  # -----------------------------
  # Train loop
  # -----------------------------
  print(f"[{_now()}] start training: steps={args.steps} eval_every={args.eval_every} ckpt_every={args.ckpt_every}")
  t0 = time.time()

  for step in range(1, args.steps + 1):
    xb_np, yb_np = sample_global_batch()
    xb = jnp.asarray(xb_np)
    if bf16_inputs:
      xb = xb.astype(jnp.bfloat16)
    yb = {h: jnp.asarray(yb_np[h]) for h in horizons}

    rng, step_rng = jax.random.split(rng)
    params, state, opt_state, rng, tr_metrics = train_step(params, state, opt_state, step_rng, xb, yb)

    if step == 1 or step % 50 == 0:
      dt = time.time() - t0
      it_s = step / max(dt, 1e-6)
      loss_v = float(jax.device_get(tr_metrics["loss"]))
      gn_v = float(jax.device_get(tr_metrics["grad_norm"]))
      print(f"[{_now()}] step={step:6d} loss={loss_v:.6f} grad_norm={gn_v:.6f} ({it_s:.2f} it/s)")

    # -----------------------------
    # Eval
    # -----------------------------
    if step % args.eval_every == 0:
      num_evals += 1
      rng, ev_rng = jax.random.split(rng)

      ev_metrics, calib_by_h = eval_full(
          params=params,
          state=state,
          X_val=X_val,
          y_val=y_val,
          med=med,
          iqr=iqr,
          batch_size=int(args.eval_batch),
          rng_key=ev_rng,
          bf16_inputs=bf16_inputs,
      )

      eval_loss = float(ev_metrics["eval_loss"])
      print(f"[{_now()}] [eval] step={step:6d} eval_loss={eval_loss:.6f}")

      for h in horizons:
        ece = calib_by_h[h]["ece"]
        cov_str = ", ".join([
            f"q{int(q*100):02d}={calib_by_h[h][f'coverage_q{int(q*100):02d}']:.3f}"
            for q in quantiles
        ])
        print(f"         [calib] h={h:>3d} ece={ece:.4f}  {cov_str}")

      # Save eval artifacts (small)
      save_json(workdir / f"eval_step{step:06d}.json", {"step": step, "time": _now(), **ev_metrics})
      save_json(workdir / f"calibration_step{step:06d}.json", {
          "step": step,
          "time": _now(),
          "quantiles": quantiles,
          "by_horizon": {str(h): calib_by_h[h] for h in horizons},
      })

      # best: save ONLY if improved
      improved = (eval_loss < (best_eval_loss - float(args.early_stop_min_delta)))
      if improved:
        best_eval_loss = eval_loss
        best_step = step
        bad_evals = 0
        save_best_ckpt(step, best_eval_loss, best_step)
      else:
        bad_evals += 1

      # optional: also save periodic/latest after eval
      if args.save_ckpt_every_eval:
        save_periodic_ckpt(step, best_eval_loss, best_step, reason="eval")

      # early stopping decision
      if args.early_stop:
        if num_evals <= int(args.early_stop_warmup_evals):
          pass
        else:
          if bad_evals >= int(args.early_stop_patience):
            print(f"[{_now()}] [early_stop] stop at step={step} "
                  f"(no improvement for {bad_evals} evals; best={best_eval_loss:.6f} at step={best_step})")
            # Always save a final latest/periodic snapshot on stop
            save_periodic_ckpt(step, best_eval_loss, best_step, reason="early_stop_final")
            break

    # -----------------------------
    # Periodic/latest ckpt for resume
    # -----------------------------
    if step % args.ckpt_every == 0:
      save_periodic_ckpt(step, best_eval_loss, best_step, reason="periodic")

    # End-of-run safety snapshot
    if step == args.steps:
      save_periodic_ckpt(step, best_eval_loss, best_step, reason="final")

  print(f"[{_now()}] done. best_eval_loss={best_eval_loss:.6f} at step={best_step}")


# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export TF_GPU_ALLOCATOR=cuda_malloc_async
#
# python train_loop_minimal.py \
#   --train_npz ./alphatrade_ds_jm/train.npz \
#   --val_npz   ./alphatrade_ds_jm/val.npz \
#   --workdir   ./runs/jm_v02 \
#   --steps 20000 \
#   --batch_size 32 \
#   --micro_batch 4 \
#   --bf16 \
#   --lr 1e-4 \
#   --eval_every 500 \
#   --ckpt_every 1000

if __name__ == "__main__":
  main()
