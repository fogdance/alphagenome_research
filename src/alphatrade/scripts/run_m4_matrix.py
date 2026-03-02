#!/usr/bin/env python3
"""
M4 Matrix Runner

Runs train→eval pairs for multiple seeds and generates a single-page comparison summary.

Usage:
    python src/alphatrade/scripts/run_m4_matrix.py \
        --config configs/dataset/m2.yaml \
        --seeds 42 43 44 \
        --max-steps 500 \
        --smoke
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="M4 Matrix Runner (train→eval paired)")
    parser.add_argument("--config", type=str, default="configs/dataset/m2.yaml",
                        help="Dataset config file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Random seeds to run (default: 42 43 44)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max training steps (default: 500)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--smoke", action="store_true",
                        help="Use smoke test symbols")
    parser.add_argument("--jit", type=int, default=0,
                        help="Use JIT compilation (0/1, default: 0)")
    parser.add_argument("--eval-split", type=str, default="val",
                        help="Evaluation split (default: val)")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="Run on GPU (default; uses python -c workaround for JAX 0.9 CUDA bug)")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu",
                        help="Force CPU mode")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# GPU workaround helpers (see docs/gpu_jit_issue.md)
# ---------------------------------------------------------------------------

_GPU_ENV = {
    **os.environ,
    "JAX_PLATFORMS": "cuda",
    "XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer=",
}


def _build_cmd(module: str, cli_args: List[str], gpu: bool) -> tuple:
    """Return (cmd, env) for subprocess.

    When gpu=True, use ``python -c`` import trick to avoid JAX 0.9 CUDA
    plugin init-order crash (docs/gpu_jit_issue.md).
    """
    if gpu:
        argv_str = json.dumps(["run"] + cli_args)
        code = (
            f"import sys; sys.argv = {argv_str}; "
            f"from alphatrade.scripts.{module} import main; main()"
        )
        return ["python", "-c", code], _GPU_ENV
    else:
        script = f"src/alphatrade/scripts/{module}.py"
        return ["python", script] + cli_args, None   # inherit env


# ---------------------------------------------------------------------------
# Train / Eval runners
# ---------------------------------------------------------------------------

def run_training(config: str, seed: int, max_steps: int, batch_size: int,
                 smoke: bool, jit: int, gpu: bool = False) -> Optional[Dict]:
    """Run single training experiment. Returns result dict or None on failure."""

    print(f"\n{'='*60}")
    print(f"Training seed={seed}")
    print(f"{'='*60}\n")

    ckpt_dir = f"checkpoints/m4/matrix_seed{seed}"

    cli_args = [
        "--config", config,
        "--seed", str(seed),
        "--max-steps", str(max_steps),
        "--batch-size", str(batch_size),
        "--jit", str(jit),
        "--ckpt-dir", ckpt_dir,
    ]
    if smoke:
        cli_args.append("--smoke")

    cmd, env = _build_cmd("train_m4_alphatrade", cli_args, gpu)
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        print(f"  Training FAILED for seed={seed}")
        return None

    # Load & rename metrics
    src_path = "reports/m4_train_metrics.json"
    if not os.path.exists(src_path):
        print(f"  Metrics file not found: {src_path}")
        return None

    with open(src_path, 'r') as f:
        metrics = json.load(f)

    dst_path = f"reports/m4_train_metrics_seed{seed}.json"
    os.rename(src_path, dst_path)

    md_src = "reports/m4_train_run.md"
    if os.path.exists(md_src):
        os.rename(md_src, f"reports/m4_train_run_seed{seed}.md")

    print(f"  Training complete: {dst_path}")
    return {
        "seed": seed,
        "metrics": metrics,
        "metrics_path": dst_path,
        "ckpt_dir": str(Path(ckpt_dir).resolve()),
    }


def run_evaluation(config: str, seed: int, split: str,
                   smoke: bool = False, gpu: bool = False) -> Optional[Dict]:
    """Run evaluation for a trained model. Returns result dict or None on failure."""

    print(f"\n{'='*60}")
    print(f"Evaluating seed={seed}")
    print(f"{'='*60}\n")

    train_metrics_path = f"reports/m4_train_metrics_seed{seed}.json"

    cli_args = [
        "--train-metrics", train_metrics_path,
        "--dataset-config", config,
        "--split", split,
        "--batch-size", "128",
        "--ckpt-step", "best",
    ]
    if smoke:
        cli_args.append("--smoke")

    cmd, env = _build_cmd("eval_m4_fast", cli_args, gpu)
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        print(f"  Evaluation FAILED for seed={seed}")
        return None

    src_path = "reports/m4_eval_metrics_fast.json"
    if not os.path.exists(src_path):
        print(f"  Eval metrics not found: {src_path}")
        return None

    with open(src_path, 'r') as f:
        eval_metrics = json.load(f)

    dst_path = f"reports/m4_eval_metrics_seed{seed}.json"
    os.rename(src_path, dst_path)

    md_src = "reports/m4_eval_run_fast.md"
    if os.path.exists(md_src):
        os.rename(md_src, f"reports/m4_eval_run_seed{seed}.md")

    print(f"  Evaluation complete: {dst_path}")
    return {
        "seed": seed,
        "eval_metrics": eval_metrics,
        "eval_path": dst_path,
    }


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary_json(pairs: List[Dict], args) -> Dict:
    """Build machine-readable summary dict."""

    seeds_data = []
    for p in pairs:
        entry = {"seed": p["seed"]}

        # --- train ---
        tr = p.get("train")
        if tr:
            m = tr["metrics"]
            entry["train"] = {
                "metrics_path": tr["metrics_path"],
                "ckpt_dir": tr["ckpt_dir"],
                "train_loss_last": m["loss"]["train_last"],
                "val_loss_last": m["loss"]["val_last"],
                "best_step": m["loss"]["best_step"],
                "nan_steps": m["stability"]["nan_steps"],
                "inf_steps": m["stability"]["inf_steps"],
                "grad_norm_pre_clip_max": m["stability"]["grad_norm_pre_clip_max"],
                "grad_norm_post_clip_max": m["stability"]["grad_norm_post_clip_max"],
                "by_horizon_train": {
                    h: m["loss"]["by_horizon"][h]["train"]
                    for h in ["h1", "h5", "h20", "h60"]
                },
            }
        else:
            entry["train"] = None

        # --- eval ---
        ev = p.get("eval")
        if ev:
            em = ev["eval_metrics"]
            entry["eval"] = {
                "metrics_path": ev["eval_path"],
                "ckpt_step": em["model"]["checkpoint_step"],
                "pinball_loss_overall": em["pinball_loss"]["overall"],
                "pinball_by_horizon": em["pinball_loss"]["by_horizon"],
                "ic": em["ic_metrics"]["ic"],
                "rank_ic": em["ic_metrics"]["rank_ic"],
                "quantile_coverage": em["quantile_coverage"],
                "crossing_rate": em["quantile_crossing"]["rate"],
                "sanity_warnings": em.get("sanity_warnings", []),
            }
        else:
            entry["eval"] = None

        seeds_data.append(entry)

    return {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "dataset_config": args.config,
            "seeds": args.seeds,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "jit": bool(args.jit),
            "smoke": args.smoke,
            "gpu": args.gpu,
            "eval_split": args.eval_split,
        },
        "seeds": seeds_data,
    }


def generate_summary_md(summary_json: Dict) -> str:
    """Render the JSON summary into a single-page markdown report."""

    seeds = summary_json["seeds"]
    cfg = summary_json["config"]
    ok_train = [s for s in seeds if s["train"] is not None]
    ok_eval = [s for s in seeds if s["eval"] is not None]

    lines = []
    w = lines.append

    w("# M4 Matrix Summary\n")
    w(f"**生成时间**: {summary_json['generated_at']}\n")
    w("---\n")

    # ── Config ──
    w("\n## 配置\n")
    w(f"- Dataset config: `{cfg['dataset_config']}`")
    w(f"- Seeds: {cfg['seeds']}")
    w(f"- Max steps: {cfg['max_steps']}")
    w(f"- Batch size: {cfg['batch_size']}")
    w(f"- JIT: {'enabled' if cfg['jit'] else 'disabled'}")
    w(f"- Smoke: {'yes' if cfg['smoke'] else 'no'}")
    w(f"- GPU: {'yes' if cfg.get('gpu') else 'no'}")
    w(f"- Eval split: {cfg['eval_split']}")
    w("")

    # ── Combined table ──
    w("## Train + Eval 对比\n")
    w("| Seed | Train Loss | Val Loss | Best Step | NaN/Inf | Grad Max | Pinball | IC | Rank IC | Crossing |")
    w("|------|-----------|----------|-----------|---------|----------|---------|------|---------|----------|")

    for s in seeds:
        seed = s["seed"]
        tr = s["train"]
        ev = s["eval"]

        if tr:
            tl = f"{tr['train_loss_last']:.6f}"
            vl = f"{tr['val_loss_last']:.6f}"
            bs = str(tr['best_step'])
            ni = f"{tr['nan_steps']}/{tr['inf_steps']}"
            gm = f"{tr['grad_norm_pre_clip_max']:.4f}"
        else:
            tl = vl = bs = ni = gm = "FAIL"

        if ev:
            pb = f"{ev['pinball_loss_overall']:.6f}"
            ic = f"{ev['ic']:.4f}"
            ric = f"{ev['rank_ic']:.4f}"
            cr = f"{ev['crossing_rate']:.4f}"
        else:
            pb = ic = ric = cr = "FAIL"

        w(f"| {seed} | {tl} | {vl} | {bs} | {ni} | {gm} | {pb} | {ic} | {ric} | {cr} |")

    w("")

    # ── By-horizon train ──
    if ok_train:
        w("### By-Horizon 训练损失\n")
        w("| Seed | h1 | h5 | h20 | h60 |")
        w("|------|----|----|-----|-----|")
        for s in ok_train:
            tr = s["train"]
            bh = tr["by_horizon_train"]
            w(f"| {s['seed']} | {bh['h1']:.6f} | {bh['h5']:.6f} | {bh['h20']:.6f} | {bh['h60']:.6f} |")
        w("")

    # ── By-horizon eval ──
    if ok_eval:
        w("### By-Horizon 评估损失\n")
        w("| Seed | h1 | h5 | h20 | h60 |")
        w("|------|----|----|-----|-----|")
        for s in ok_eval:
            bh = s["eval"]["pinball_by_horizon"]
            w(f"| {s['seed']} | {bh['h1']:.6f} | {bh['h5']:.6f} | {bh['h20']:.6f} | {bh['h60']:.6f} |")
        w("")

    # ── Quantile coverage ──
    if ok_eval:
        w("### Quantile Coverage\n")
        w("| Seed | q10 | q30 | q50 | q70 | q90 |")
        w("|------|-----|-----|-----|-----|-----|")
        for s in ok_eval:
            qc = s["eval"]["quantile_coverage"]
            w(f"| {s['seed']} | {qc['q10']:.4f} | {qc['q30']:.4f} | {qc['q50']:.4f} | {qc['q70']:.4f} | {qc['q90']:.4f} |")
        w("")

    # ── Statistics ──
    w("## 统计分析\n")

    if ok_train:
        tl_arr = [s["train"]["train_loss_last"] for s in ok_train]
        vl_arr = [s["train"]["val_loss_last"] for s in ok_train]
        best_idx = int(np.argmin(vl_arr))
        w("### 训练\n")
        w(f"- Train loss: {np.mean(tl_arr):.6f} +/- {np.std(tl_arr):.6f}")
        w(f"- Val loss:   {np.mean(vl_arr):.6f} +/- {np.std(vl_arr):.6f}")
        w(f"- Best val:   {vl_arr[best_idx]:.6f} (seed={ok_train[best_idx]['seed']})")
        w("")

    if ok_eval:
        pb_arr = [s["eval"]["pinball_loss_overall"] for s in ok_eval]
        ic_arr = [s["eval"]["ic"] for s in ok_eval]
        best_ic_idx = int(np.argmax(ic_arr))
        w("### 评估\n")
        w(f"- Pinball loss: {np.mean(pb_arr):.6f} +/- {np.std(pb_arr):.6f}")
        w(f"- IC:           {np.mean(ic_arr):.4f} +/- {np.std(ic_arr):.4f}")
        w(f"- Best IC:      {ic_arr[best_ic_idx]:.4f} (seed={ok_eval[best_ic_idx]['seed']})")
        w("")

    # ── Traceability ──
    w("## 文件追溯\n")
    w("| Seed | train_metrics | eval_metrics | ckpt_dir |")
    w("|------|---------------|--------------|----------|")
    for s in seeds:
        tr_path = s["train"]["metrics_path"] if s["train"] else "N/A"
        ev_path = s["eval"]["metrics_path"] if s["eval"] else "N/A"
        ckpt = s["train"]["ckpt_dir"] if s["train"] else "N/A"
        w(f"| {s['seed']} | `{tr_path}` | `{ev_path}` | `{ckpt}` |")
    w("")

    # ── Footer ──
    ok_count = len(ok_train)
    total = len(seeds)
    w("---\n")
    w(f"**总计**: {ok_count}/{total} seeds 完成 train+eval")
    w("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"M4 Matrix Runner (train→eval paired)")
    print(f"{'='*60}")
    print(f"Seeds: {args.seeds}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"JIT: {'enabled' if args.jit else 'disabled'}")
    print(f"GPU: {'yes' if args.gpu else 'no'}")
    print(f"Eval split: {args.eval_split}")
    print(f"{'='*60}\n")

    pairs = []

    for seed in args.seeds:
        pair = {"seed": seed, "train": None, "eval": None}

        # Train
        train_result = run_training(
            args.config, seed, args.max_steps, args.batch_size,
            args.smoke, args.jit, gpu=args.gpu,
        )
        pair["train"] = train_result

        # Eval (only if train succeeded)
        if train_result is not None:
            eval_result = run_evaluation(
                args.config, seed, args.eval_split,
                smoke=args.smoke, gpu=args.gpu,
            )
            pair["eval"] = eval_result

        pairs.append(pair)

    # Build summaries
    summary_json = generate_summary_json(pairs, args)
    summary_md = generate_summary_md(summary_json)

    # Write outputs
    os.makedirs("reports", exist_ok=True)

    json_path = "reports/m4_matrix_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary_json, f, indent=2)

    md_path = "reports/m4_matrix_summary.md"
    with open(md_path, 'w') as f:
        f.write(summary_md)

    ok_count = len([p for p in pairs if p["train"] and p["eval"]])
    print(f"\n{'='*60}")
    print(f"Matrix Run Complete")
    print(f"{'='*60}")
    print(f"  {ok_count}/{len(args.seeds)} seeds completed train+eval")
    print(f"  Summary (md):   {md_path}")
    print(f"  Summary (json): {json_path}")
    print()


if __name__ == "__main__":
    main()
