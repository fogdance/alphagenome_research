#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan multiple checkpoints and rank them by:
- corr_mean(q50)
- cov_abs_err_mean
- scale_ratio closeness to 1.0
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
from pathlib import Path


def _run_eval(eval_py: str, ckpt: str, npz: str, batch: int) -> str:
    cmd = [
        "python", eval_py,
        "--ckpt", ckpt,
        "--val_npz", npz,
        "--batch", str(batch),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    return out


def _parse_summary(stdout: str) -> dict:
    # Parse lines like:
    # corr_mean(q50): +0.0189  |  cov_abs_err_mean: 0.2138  |  crossing_mean: 0.0000
    m = re.search(
        r"corr_mean\(q50\):\s*([+\-]?\d+\.\d+)\s*\|\s*"
        r"cov_abs_err_mean:\s*(\d+\.\d+)\s*\|\s*"
        r"crossing_mean:\s*(\d+\.\d+)",
        stdout,
    )
    if not m:
        return {}
    return {
        "corr_mean": float(m.group(1)),
        "cov_err_mean": float(m.group(2)),
        "cross_mean": float(m.group(3)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. src/.../runs/jm_v03")
    ap.add_argument("--npz", required=True, help="val.npz or train.npz")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument(
        "--eval_py",
        default="src/alphagenome_research/alphatrade/evaluate_ckpt_calibration.py",
    )
    ap.add_argument("--limit", type=int, default=20, help="print top K")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpts = []
    for name in ["best.pkl", "latest.pkl"]:
        p = run_dir / name
        if p.exists():
            ckpts.append(str(p))

    ckpts += sorted(glob.glob(str(run_dir / "ckpt_*.pkl")))

    rows = []
    for ckpt in ckpts:
        try:
            out = _run_eval(args.eval_py, ckpt, args.npz, args.batch)
            s = _parse_summary(out)
            if not s:
                continue
            rows.append((ckpt, s["corr_mean"], s["cov_err_mean"], s["cross_mean"]))
            print(f"[ok] {os.path.basename(ckpt)} corr={s['corr_mean']:+.4f} cov_err={s['cov_err_mean']:.4f}")
        except subprocess.CalledProcessError as e:
            print(f"[fail] {ckpt}\n{e.output}")

    # rank: prioritize corr high, then cov_err low
    rows.sort(key=lambda x: (-x[1], x[2]))

    print("\n" + "=" * 80)
    print(f"TOP {min(args.limit, len(rows))} (rank by corr desc, then cov_err asc) | npz={args.npz}")
    print("=" * 80)
    for ckpt, corr, cov, cross in rows[: args.limit]:
        print(f"{os.path.basename(ckpt):>16}  corr_mean={corr:+.4f}  cov_err_mean={cov:.4f}  cross_mean={cross:.4f}")

# # 扫 val
# python src/alphagenome_research/alphatrade/scan_ckpts.py \
#   --run_dir src/alphagenome_research/alphatrade/runs/jm_v03 \
#   --npz src/alphagenome_research/alphatrade/alphatrade_ds_jm/val.npz \
#   --batch 64
#
# # 扫 train（用于判断过拟合）
# python src/alphagenome_research/alphatrade/scan_ckpts.py \
#   --run_dir src/alphagenome_research/alphatrade/runs/jm_v03 \
#   --npz src/alphagenome_research/alphatrade/alphatrade_ds_jm/train.npz \
#   --batch 64
if __name__ == "__main__":
    main()