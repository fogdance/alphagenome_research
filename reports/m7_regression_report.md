# M7 Ablation Regression Report

Generated: 2026-03-03T14:03:06.353229
Git SHA: `8892e72`
Primary metric: `pinball_loss.overall` (lower is better)

## Thresholds

- Improved: decrease >= 1.0%
- Regressed: increase >= 5.0%

## Baseline

- Experiment: `baseline`
- Primary mean: 0.134178 +/- 0.010631
- Seeds: 3

## Comparisons

| Experiment | Mean | Std | Delta | Delta % | Verdict |
|------------|------|-----|-------|---------|---------|
| batch_256 | 0.133383 | 0.011707 | -0.000795 | -0.59% | ➖ neutral |
| steps_1000 | 0.134178 | 0.010631 | +0.000000 | +0.00% | ➖ neutral |
| no_clip | 0.135976 | 0.010437 | +0.001798 | +1.34% | ➖ neutral |

## Summary

- Total ablations: 3
- Improved: 0
- Neutral: 3
- Regressed: 0
