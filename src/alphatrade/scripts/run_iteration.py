#!/usr/bin/env python3
"""
M8 Iteration Loop Runner

End-to-end pipeline: sweep → regression report → schema/semantic gate.

Usage:
    python src/alphatrade/scripts/run_iteration.py \
      --sweep-config configs/sweep/m7.yaml \
      --profile m7 \
      [--smoke] [--resume] [--strict] [--dry-run]
"""

import argparse
import subprocess
import sys

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="M8 Iteration Loop Runner")
    parser.add_argument("--sweep-config", type=str, required=True,
                        help="Path to sweep config YAML")
    parser.add_argument("--profile", type=str, required=True,
                        help="Validator profile (e.g. m5, m7)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test mode (3 symbols, 3 steps)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs whose output files already exist")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 on required missing/fail in validator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit (forwarded to sweep)")
    return parser.parse_args()


def load_baseline_exp_id(sweep_config_path: str) -> str:
    """Read baseline_exp_id from sweep config, default to 'baseline'."""
    with open(sweep_config_path) as f:
        config = yaml.safe_load(f)
    return config.get("baseline_exp_id", "baseline")


def run_step(label: str, cmd: list[str]) -> int:
    """Run a subprocess, print header, return exit code."""
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    args = parse_args()

    baseline_exp_id = load_baseline_exp_id(args.sweep_config)

    print(f"\n{'='*60}")
    print("M8 Iteration Loop")
    print(f"{'='*60}")
    print(f"Sweep config:    {args.sweep_config}")
    print(f"Profile:         {args.profile}")
    print(f"Baseline exp:    {baseline_exp_id}")
    print(f"Smoke:           {'yes' if args.smoke else 'no'}")
    print(f"Resume:          {'yes' if args.resume else 'no'}")
    print(f"Strict:          {'yes' if args.strict else 'no'}")
    print(f"Dry-run:         {'yes' if args.dry_run else 'no'}")
    print(f"{'='*60}\n")

    # ── Step 1: Sweep ──
    sweep_cmd = [
        sys.executable, "src/alphatrade/scripts/run_m5_sweep.py",
        "--sweep-config", args.sweep_config,
    ]
    if args.smoke:
        sweep_cmd.append("--smoke")
    if args.resume:
        sweep_cmd.append("--resume")
    if args.dry_run:
        sweep_cmd.append("--dry-run")

    rc = run_step("Step 1/3: Sweep", sweep_cmd)
    if rc != 0:
        print(f"\n[FAIL] Sweep exited with code {rc}")
        sys.exit(rc)

    if args.dry_run:
        print("\n[DRY-RUN] Stopping after sweep plan.")
        return

    # ── Step 2: Regression report ──
    regression_cmd = [
        sys.executable, "src/alphatrade/scripts/build_m7_regression_report.py",
        "--baseline-exp", baseline_exp_id,
    ]

    rc = run_step("Step 2/3: Regression Report", regression_cmd)
    if rc != 0:
        print(f"\n[FAIL] Regression report exited with code {rc}")
        sys.exit(rc)

    # ── Step 3: Validate ──
    validate_cmd = [
        sys.executable, "src/alphatrade/scripts/validate_reports_schema.py",
        "--profile", args.profile,
    ]
    if args.strict:
        validate_cmd.append("--strict")

    rc = run_step("Step 3/3: Validate", validate_cmd)

    # ── Verdict ──
    print(f"\n{'='*60}")
    print("M8 Iteration Loop — Verdict")
    print(f"{'='*60}")
    if rc == 0:
        print("  PASS")
    else:
        print(f"  FAIL (validator exit code {rc})")
    print(f"{'='*60}\n")

    sys.exit(rc)


if __name__ == "__main__":
    main()
