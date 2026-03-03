#!/usr/bin/env python3
"""
Build M7 Regression Report

Reads the M5 leaderboard (which contains baseline + ablation experiments),
compares each non-baseline experiment against the baseline, and outputs
a regression report (JSON + Markdown).

Usage:
  python src/alphatrade/scripts/build_m7_regression_report.py \
    [--leaderboard reports/m5_leaderboard.json] \
    [--baseline-exp baseline] \
    [--improvement-pct 1.0] \
    [--regression-pct 5.0] \
    [--output-json reports/m7_regression_report.json] \
    [--output-md reports/m7_regression_report.md]
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Build M7 regression report")
    parser.add_argument("--leaderboard", type=str, default="reports/m5_leaderboard.json",
                        help="Path to M5 leaderboard JSON")
    parser.add_argument("--baseline-exp", type=str, default="baseline",
                        help="Experiment ID to use as baseline")
    parser.add_argument("--improvement-pct", type=float, default=1.0,
                        help="Minimum %% decrease in primary metric to count as improved")
    parser.add_argument("--regression-pct", type=float, default=5.0,
                        help="Minimum %% increase in primary metric to count as regressed")
    parser.add_argument("--output-json", type=str, default="reports/m7_regression_report.json",
                        help="Output JSON path")
    parser.add_argument("--output-md", type=str, default="reports/m7_regression_report.md",
                        help="Output Markdown path")
    return parser.parse_args()


def get_git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def compute_verdict(delta_pct, improvement_pct, regression_pct):
    """Determine verdict based on delta percentage.

    Lower primary metric is better, so:
    - delta_pct <= -improvement_pct → improved (metric decreased)
    - delta_pct >= regression_pct → regressed (metric increased)
    - otherwise → neutral
    """
    if delta_pct <= -improvement_pct:
        return "improved"
    elif delta_pct >= regression_pct:
        return "regressed"
    else:
        return "neutral"


def build_report(leaderboard, baseline_exp_id, improvement_pct, regression_pct):
    """Build the regression report from leaderboard data."""
    experiments = leaderboard["experiments"]

    # Find baseline
    baseline = None
    for exp in experiments:
        if exp["exp_id"] == baseline_exp_id:
            baseline = exp
            break

    if baseline is None:
        print(f"ERROR: baseline experiment '{baseline_exp_id}' not found in leaderboard")
        print(f"Available experiments: {[e['exp_id'] for e in experiments]}")
        sys.exit(1)

    baseline_mean = baseline["metrics"]["primary_mean"]
    baseline_std = baseline["metrics"]["primary_std"]
    baseline_n_seeds = baseline["n_runs"]

    comparisons = []
    for exp in experiments:
        if exp["exp_id"] == baseline_exp_id:
            continue

        exp_mean = exp["metrics"]["primary_mean"]
        exp_std = exp["metrics"]["primary_std"]
        exp_n_seeds = exp["n_runs"]

        delta = exp_mean - baseline_mean
        delta_pct = (delta / baseline_mean * 100) if baseline_mean != 0 else 0.0
        verdict = compute_verdict(delta_pct, improvement_pct, regression_pct)

        comparisons.append({
            "exp_id": exp["exp_id"],
            "primary_mean": round(exp_mean, 6),
            "primary_std": round(exp_std, 6),
            "n_seeds": exp_n_seeds,
            "delta": round(delta, 6),
            "delta_pct": round(delta_pct, 2),
            "verdict": verdict,
        })

    summary = {
        "total_ablations": len(comparisons),
        "improved": sum(1 for c in comparisons if c["verdict"] == "improved"),
        "neutral": sum(1 for c in comparisons if c["verdict"] == "neutral"),
        "regressed": sum(1 for c in comparisons if c["verdict"] == "regressed"),
    }

    report = {
        "schema_version": "m7_regression_report_v1",
        "generated_at": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "baseline_exp_id": baseline_exp_id,
        "primary_metric": leaderboard.get("primary_metric", "pinball_loss.overall"),
        "thresholds": {
            "improvement_pct": improvement_pct,
            "regression_pct": regression_pct,
        },
        "baseline": {
            "exp_id": baseline_exp_id,
            "primary_mean": round(baseline_mean, 6),
            "primary_std": round(baseline_std, 6),
            "n_seeds": baseline_n_seeds,
        },
        "comparisons": comparisons,
        "summary": summary,
    }

    return report


_VERDICT_ICON = {
    "improved": "\u2705",
    "neutral": "\u2796",
    "regressed": "\u274c",
}


def generate_markdown(report):
    """Generate markdown table from regression report."""
    lines = []
    w = lines.append

    w("# M7 Ablation Regression Report\n")
    w(f"Generated: {report['generated_at']}")
    w(f"Git SHA: `{report['git_sha']}`")
    w(f"Primary metric: `{report['primary_metric']}` (lower is better)\n")

    w("## Thresholds\n")
    w(f"- Improved: decrease >= {report['thresholds']['improvement_pct']}%")
    w(f"- Regressed: increase >= {report['thresholds']['regression_pct']}%\n")

    w("## Baseline\n")
    b = report["baseline"]
    w(f"- Experiment: `{b['exp_id']}`")
    w(f"- Primary mean: {b['primary_mean']:.6f} +/- {b['primary_std']:.6f}")
    w(f"- Seeds: {b['n_seeds']}\n")

    w("## Comparisons\n")
    w("| Experiment | Mean | Std | Delta | Delta % | Verdict |")
    w("|------------|------|-----|-------|---------|---------|")
    for c in report["comparisons"]:
        icon = _VERDICT_ICON.get(c["verdict"], "?")
        sign = "+" if c["delta"] >= 0 else ""
        w(f"| {c['exp_id']} | {c['primary_mean']:.6f} | {c['primary_std']:.6f} "
          f"| {sign}{c['delta']:.6f} | {sign}{c['delta_pct']:.2f}% | {icon} {c['verdict']} |")
    w("")

    w("## Summary\n")
    s = report["summary"]
    w(f"- Total ablations: {s['total_ablations']}")
    w(f"- Improved: {s['improved']}")
    w(f"- Neutral: {s['neutral']}")
    w(f"- Regressed: {s['regressed']}")
    w("")

    return "\n".join(lines)


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("M7 Regression Report Builder")
    print(f"{'='*60}")
    print(f"Leaderboard: {args.leaderboard}")
    print(f"Baseline:    {args.baseline_exp}")
    print(f"Improvement: >= {args.improvement_pct}% decrease")
    print(f"Regression:  >= {args.regression_pct}% increase")
    print(f"{'='*60}\n")

    if not os.path.exists(args.leaderboard):
        print(f"ERROR: leaderboard not found: {args.leaderboard}")
        sys.exit(1)

    with open(args.leaderboard) as f:
        leaderboard = json.load(f)

    report = build_report(leaderboard, args.baseline_exp, args.improvement_pct, args.regression_pct)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report:     {args.output_json}")

    with open(args.output_md, "w") as f:
        f.write(generate_markdown(report))
    print(f"Markdown report: {args.output_md}")

    print(f"\n{'='*60}")
    s = report["summary"]
    print(f"Results: {s['improved']} improved, {s['neutral']} neutral, {s['regressed']} regressed")
    for c in report["comparisons"]:
        icon = _VERDICT_ICON.get(c["verdict"], "?")
        sign = "+" if c["delta_pct"] >= 0 else ""
        print(f"  {icon} {c['exp_id']}: {sign}{c['delta_pct']:.2f}% → {c['verdict']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
