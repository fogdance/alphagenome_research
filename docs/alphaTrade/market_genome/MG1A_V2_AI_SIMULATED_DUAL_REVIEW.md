# MG1-A.1 AI-Simulated Dual Review and Adjudication

## Status

This package records two independent AI-simulated reviews and a separately
rubric-bound AI-simulated adjudication of all 70 MG1-A.1 assumptions. It is a
technical pre-review artifact, not a human review, human approval, or ontology
freeze.

- Review target commit: `0d378954d49b6467d05fdef152c522e94b660978`
- Target v2 file SHA-256:
  `cd569c19a73768b19b8f9ded2b570dc3fc93796a87b97ecd27becdd32ba5978e`
- Target resolved-contract SHA-256:
  `ad3228a608998cc0823c440af0013b12658c9f8dcda4b858ea719be879389ebd`
- Result: `REVISION_AND_EXTERNAL_EVIDENCE_REQUIRED_BEFORE_HUMAN_FREEZE`
- MG1-B: `BLOCKED`
- GPU, JAX, training, detector, labeler, and trading execution: not performed

The machine-readable package is in
`configs/market_genome/reviews/mg1a_v2_ai_simulated/`. Its `bundle.json`
binds the original reviewer outputs, Reviewer B attestation, pre-committed
adjudication rubric, final adjudication, target hashes, and canonical decision
hash.

## Independent Reviews

The two reviewer roles worked from the same 61 inherited and 9 new assumption
IDs without reading each other's decisions. Reviewer B's original output did
not contain a commit and per-file hash binding, so a separate attestation was
added without changing the original bytes or any decision.

| Role | ACCEPT | REVISE | DEFER | REJECT |
|---|---:|---:|---:|---:|
| AI-simulated Reviewer A | 21 | 45 | 3 | 1 |
| AI-simulated Reviewer B | 19 | 39 | 12 | 0 |
| AI-simulated adjudicator | 13 | 53 | 4 | 0 |

Reviewer A and B agreed on 42 assumptions and disagreed on 28. The adjudicator
aligned with both on 41, only A on 15, only B on 13, and neither on 1. The
adjudication therefore did not use majority voting.

The one override of reviewer consensus was `MG1A.RANGE.PHASE.001`: both
reviewers selected `ACCEPT`, while adjudication selected `REVISE`. Replacing
`RANGE_TERMINATION_RECORDED` with an arbitrary administrative placeholder
still produced zero validator errors, which demonstrated that the event
semantics were not closed by the current contract.

## Accepted Technical Propositions

The following 13 propositions were accepted at symbolic technical scope:

1. `MG1A.COMMON.PARTIAL_BAR.001`
2. `MG1A.TREND_TRANSITION.STRUCTURE.001`
3. `MG1A.TREND_TRANSITION.PHASE.001`
4. `MG1A.DOUBLE_TOP.PHASE.001`
5. `MG1A.DOUBLE_BOTTOM.PHASE.001`
6. `MG1A.HEAD_SHOULDERS_TOP.PHASE.001`
7. `MG1A.INVERSE_HEAD_SHOULDERS.PHASE.001`
8. `MG1A.FAILED_BREAKOUT.STRUCTURE.001`
9. `MG1A.SUPPORT_RESISTANCE_CONVERSION.STRUCTURE.001`
10. `MG1A.SUPPORT_RESISTANCE_CONVERSION.PHASE.001`
11. `MG1A1.COMMON.OBJECT_MODEL.001`
12. `MG1A1.COMMON.MORPHOLOGY_EFFECT.001`
13. `MG1A1.COMMON.LIFECYCLE_MAPPING.001`

These are non-binding simulated `ACCEPT` decisions. Every record still has
`human_review_required: true` and `freeze_eligible: false`.
Reviewer A's raw summary calls its 21 `ACCEPT` decisions `freeze_ready`; the
bundle explicitly constrains that legacy label to mean technical acceptance
inside that simulated review only. It does not mean human freeze eligibility.

## Deferred Evidence

Four propositions require evidence that is intentionally absent from the
current contract-only, no-training scope:

| Assumption | Evidence required to resume |
|---|---|
| `MG1A.COMMON.SCALE.001` | Versioned scale/base-timeframe registries, causal statistics, and later train-selected values |
| `MG1A.COMMON.NORMALIZATION.001` | The train-partition-selected normalizer with hash-bound fit metadata |
| `MG1A.COMMON.SESSION.001` | Versioned calendar registry with session, timezone, holiday, early-close, and unsupported-instrument rules |
| `MG1A1.COMMON.SCALE_SPEC.001` | Content-addressed scale, session, and causal-prominence registries |

No GPU is required to review these artifacts once they exist. Training remains
deferred only where the proposition explicitly requires a train-fitted value or
artifact.

## Revision Themes

The 53 `REVISE` decisions are recorded individually in `adjudication.json`.
Their recurring blockers are:

- close event and predicate vocabularies with versioned registries rather than
  free symbolic strings;
- define tri-state confirmation truth tables and optional-filter behavior;
- complete invalidation, expiry, recross, and reentry coverage with deterministic
  event precedence;
- bind pivot, anchor, zone, relation, and formula references to exact identities
  and lifecycle rules;
- define append-only, hash-bound reviewer assignment, correction, amendment,
  and adjudicator-independence records;
- keep numeric thresholds and fitted policies train-partition-only rather than
  inventing values in the ontology review.

## Governance Boundary

The review protocol in v2 states
`model_assisted_independent_truth_allowed: false`. Accordingly:

- actual human reviewer count is zero;
- no simulated reviewer or adjudicator has freeze authority;
- `pattern_ontology_v2.yaml` remains `DRAFT_NEEDS_HUMAN_FREEZE` with
  `human_review.status: PENDING` and `frozen: false`;
- all 70 assumptions still require two independent real-human decisions and
  hash-bound adjudication before a human freeze can be recorded;
- MG1-B remains blocked.

## Verification

The bundle validator performs duplicate-key-safe JSON parsing, closed-shape
schema validation, target and Git-object hash checks, v2 structural validation,
source-evidence and attestation checks, exact 70-ID/order validation, v1 blocker
coverage, reviewer-to-adjudication traceability, canonical decision hashing,
summary and disagreement-matrix recomputation, non-human freeze enforcement,
and input snapshot checks.

Run it without GPU visibility:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m alphatrade.market_genome.ontology.simulated_review --strict
```

The original adjudication also recorded 53 passing CPU-only v2 contract tests.
The dedicated bundle test module covers valid input plus hash, provenance,
human-freeze, decision, blocker, duplicate-key, and snapshot tampering failures.
