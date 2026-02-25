#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# 用你现有实现，保证特征一致
from alphagenome_research.alphatrade import preprocessing


# ---- 交易时段（可选，仅用于 fill_missing_minutes + schedule 模式） ----
# DCE JM 常见：09:00–10:15, 10:30–11:30, 13:30–15:00, 21:00–23:00
DEFAULT_SESSIONS = [("09:00", "10:15"), ("10:30", "11:30"), ("13:30", "15:00"), ("21:00", "23:00")]


def _hhmm_to_minute(hhmm: str) -> int:
  h, m = hhmm.split(":")
  return int(h) * 60 + int(m)


def load_ohlcv_csv(csv_path: str) -> pd.DataFrame:
  df = pd.read_csv(csv_path)
  if "Date" not in df.columns:
    raise ValueError("CSV must have 'Date' column")

  # 兼容你示例：Date,Open,High,Low,Close,Volume,OpenInterest
  need = ["Open", "High", "Low", "Close", "Volume"]
  for c in need:
    if c not in df.columns:
      raise ValueError(f"CSV missing required column: {c}")

  df["Date"] = pd.to_datetime(df["Date"])
  df = df.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
  # 强制 float
  for c in ["Open", "High", "Low", "Close", "Volume"]:
    df[c] = df[c].astype(np.float32)

  # OpenInterest 可选
  if "OpenInterest" in df.columns:
    df["OpenInterest"] = df["OpenInterest"].astype(np.float32)

  return df


def fill_missing_minutes(
    df: pd.DataFrame,
    *,
    freq: str = "1min",
    method: str = "by_presence",
    sessions: List[Tuple[str, str]] = DEFAULT_SESSIONS,
) -> pd.DataFrame:
  """
  把分钟补齐（会变大很多，慎用）。

  Warning:
    对补出来的非交易分钟，我们会令 O=H=L=C=prev_close, Volume=0。
    这会导致 hl_range=log(eps)（约 -20.x）等特征出现极端值；虽然 is_session_open=0
    可以让模型学会忽略，但也可能让模型"过度依赖极端值识别休市"。

  - method="by_presence": 原CSV里存在=开盘(1)，补出来=收盘(0)（推荐，更稳，自动处理周末/节假日）
  - method="by_schedule": 按 sessions + 周一到周五 判定开盘（遇到节假日会误判）
  """
  full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
  df_full = df.reindex(full_index)

  present = ~df_full["Close"].isna()

  # 价格缺失：用前值 close 填，OHLC 全部设为 close；量设 0
  close_ffill = df_full["Close"].ffill()
  for c in ["Close"]:
    df_full[c] = close_ffill
  for c in ["Open", "High", "Low"]:
    df_full[c] = df_full[c].where(present, close_ffill)

  df_full["Volume"] = df_full["Volume"].where(present, 0.0).astype(np.float32)
  if "OpenInterest" in df_full.columns:
    df_full["OpenInterest"] = df_full["OpenInterest"].ffill().astype(np.float32)

  if method == "by_presence":
    df_full["is_session_open"] = present.astype(np.float32)
  elif method == "by_schedule":
    sess_min = [(_hhmm_to_minute(a), _hhmm_to_minute(b)) for a, b in sessions]
    minute_of_day = df_full.index.hour * 60 + df_full.index.minute
    weekday = df_full.index.weekday  # 0=Mon
    open_flag = np.zeros(len(df_full), dtype=np.float32)
    is_weekday = (weekday >= 0) & (weekday <= 4)
    for (s, e) in sess_min:
      open_flag |= (is_weekday & (minute_of_day >= s) & (minute_of_day < e)).astype(np.float32)
    df_full["is_session_open"] = open_flag.astype(np.float32)
  else:
    raise ValueError("method must be 'by_presence' or 'by_schedule'")

  return df_full


def compute_features(
    df: pd.DataFrame,
    *,
    session_period_minutes: int = 1440,
    trading_day_start: str = "21:00",
    epsilon: float = 1e-9,
) -> np.ndarray:
  """
  计算 [T, 8] 特征。
  minute_indices：用“交易日起点”做相位对齐（默认 21:00），让夜盘在同一周期里更自然。
  is_session_open：如果 df 有 is_session_open 列就用；否则全 1（表示这些行都是开盘分钟）。
  """
  pre = preprocessing.FeaturePreprocessor(epsilon=epsilon, session_period_minutes=session_period_minutes)

  o = df["Open"].to_numpy(np.float32)
  h = df["High"].to_numpy(np.float32)
  l = df["Low"].to_numpy(np.float32)
  c = df["Close"].to_numpy(np.float32)
  v = df["Volume"].to_numpy(np.float32)

  # minute index in [0, 1439] but aligned by trading_day_start
  start_minute = _hhmm_to_minute(trading_day_start)
  minute_of_day = (df.index.hour * 60 + df.index.minute).to_numpy(np.int32)
  minute_idx = ((minute_of_day - start_minute) % session_period_minutes).astype(np.float32)

  if "is_session_open" in df.columns:
    is_open = df["is_session_open"].to_numpy(np.float32)
  else:
    is_open = np.ones(len(df), dtype=np.float32)

  feats = pre.compute_features(
      open_prices=o,
      high_prices=h,
      low_prices=l,
      close_prices=c,
      volumes=v,
      minute_indices=minute_idx,
      is_session_open=is_open,
  )
  # jnp -> np
  return np.asarray(feats, dtype=np.float32)


def make_windows_and_targets(
    features: np.ndarray,
    close: np.ndarray,
    asof_indices: np.ndarray,
    lookback: int,
    horizons: List[int],
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
  """
  X: [N, L, 8]
  y[h]: [N]  where y[h] = log(C_{t+h} / C_t)
  """
  T = features.shape[0]
  max_h = max(horizons)
  valid = asof_indices[(asof_indices >= (lookback - 1)) & (asof_indices + max_h < T)]
  N = len(valid)

  X = np.empty((N, lookback, 8), dtype=np.float32)
  y: Dict[int, np.ndarray] = {h: np.empty((N,), dtype=np.float32) for h in horizons}

  for i, t in enumerate(valid):
    X[i] = features[t - lookback + 1 : t + 1]
    c0 = close[t]
    for h in horizons:
      y[h][i] = np.log(close[t + h] / (c0 + 1e-12))

  return X, y


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--csv", required=True, help="path to 1m OHLCV csv")
  ap.add_argument("--out_dir", required=True, help="output directory")
  ap.add_argument("--lookback", type=int, default=4096)
  ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20, 60])
  ap.add_argument("--stride", type=int, default=60, help="sample one asof every N bars to control dataset size")
  ap.add_argument("--max_train", type=int, default=12000, help="cap train samples (after stride); 0=unlimited")
  ap.add_argument("--max_val", type=int, default=3000, help="cap val samples; 0=unlimited")
  ap.add_argument("--val_start", type=str, default="2025-01-01", help="YYYY-MM-DD time split by asof timestamp")
  ap.add_argument("--trading_day_start", type=str, default="21:00", help="phase align time_sin/cos")
  ap.add_argument("--fill_minutes", action="store_true", help="reindex to full 1-min grid (can be huge)")
  ap.add_argument("--fill_method", type=str, default="by_presence", choices=["by_presence", "by_schedule"])
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)

  df = load_ohlcv_csv(args.csv)
  if args.fill_minutes:
    df = fill_missing_minutes(df, method=args.fill_method)

  features = compute_features(df, trading_day_start=args.trading_day_start)
  close = df["Close"].to_numpy(np.float32)

  T = len(df)
  max_h = max(args.horizons)
  # 以“bar index”为时间轴：每 stride 个 bar 取一个 asof
  asof_all = np.arange(args.lookback - 1, T - max_h, args.stride, dtype=np.int32)
  asof_ts = df.index[asof_all]

  val_start = pd.Timestamp(args.val_start)
  train_mask = asof_ts < val_start
  train_asof = asof_all[train_mask]
  val_asof = asof_all[~train_mask]

  # 下采样/截断，避免爆内存
  rng = np.random.default_rng(42)
  if args.max_train and len(train_asof) > args.max_train:
    train_asof = np.sort(rng.choice(train_asof, size=args.max_train, replace=False))
  if args.max_val and len(val_asof) > args.max_val:
    val_asof = np.sort(rng.choice(val_asof, size=args.max_val, replace=False))

  X_train, y_train = make_windows_and_targets(features, close, train_asof, args.lookback, args.horizons)
  X_val, y_val = make_windows_and_targets(features, close, val_asof, args.lookback, args.horizons)

  train_path = os.path.join(args.out_dir, "train.npz")
  val_path = os.path.join(args.out_dir, "val.npz")

  train_dict = {"features": X_train}
  for h in args.horizons:
    train_dict[f"y_h{h}"] = y_train[h]
  np.savez(train_path, **train_dict)

  val_dict = {"features": X_val}
  for h in args.horizons:
    val_dict[f"y_h{h}"] = y_val[h]
  np.savez(val_path, **val_dict)

  meta = {
      "csv": os.path.abspath(args.csv),
      "lookback": args.lookback,
      "horizons": args.horizons,
      "stride": args.stride,
      "val_start": args.val_start,
      "trading_day_start": args.trading_day_start,
      "fill_minutes": args.fill_minutes,
      "fill_method": args.fill_method,
      "train_samples": int(X_train.shape[0]),
      "val_samples": int(X_val.shape[0]),
      "T_bars": int(T),
  }
  with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
    import json
    json.dump(meta, f, ensure_ascii=False, indent=2)

  print("Saved:")
  print(" ", train_path, X_train.shape)
  print(" ", val_path, X_val.shape)
  print("Meta:", os.path.join(args.out_dir, "meta.json"))

# python csv_to_alphatrade_npz.py \
#   --csv ./data/8Y_DCE_JM2601_1m.csv \
#   --out_dir ./alphatrade_ds_jm \
#   --lookback 4096 \
#   --horizons 1 5 20 60 \
#   --stride 60 \
#   --max_train 12000 --max_val 3000 \
#   --val_start 2025-01-01 \
#   --trading_day_start 21:00

if __name__ == "__main__":
  main()
