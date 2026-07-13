# MG1 Pattern Ontology v2 Review

## Decision

| Item | Decision |
|---|---|
| v1 domain-review outcome | `DOMAIN_REVIEW_COMPLETED_WITH_REQUIRED_REVISIONS` |
| v1 draft assessment | `APPROVED_AS_DRAFT` |
| v1 freeze assessment | `REVISION_REQUIRED_BEFORE_FREEZE` |
| v2 status | `DRAFT_NEEDS_HUMAN_FREEZE` |
| MG1-B | `BLOCKED` |
| Frozen assumptions | `0/70` (`0/61` inherited and `0/9` new) |
| GPU, JAX, model training | Not executed |
| Detector, labeler | Not implemented |
| Real labels, synthetic samples | Not created |

The v2 ontology is a review draft, not a frozen contract. It incorporates the
domain-review direction into a more explicit object, phase, anchor, relation,
formula, and review model, but it does not satisfy the human-freeze gate.
`MG1-B` remains blocked until every inherited and new assumption has a recorded
human decision and the resulting review is bound to the content-addressed
artifacts.

## Evidence Basis

This review uses the following evidence:

- `configs/market_genome/pattern_ontology_v1.yaml`: immutable v1 baseline. Its
  observed SHA-256 is
  `3ab4ce88dd63dbb5ec484b6ca420ce9a3ab06579f63b53776252216a10012b90`.
- `configs/market_genome/pattern_ontology_v1_review.yaml`: review ledger with
  61 reviewed assumptions, 51 `REVISE`, 10 `DEFER`, 0 `ACCEPT`, 0 freeze-ready,
  114 unresolved parameter references, and decision
  `DOMAIN_REVIEW_COMPLETED_WITH_REQUIRED_REVISIONS`.
- `docs/alphaTrade/market_genome/MG1_ASSUMPTION_REVIEW.md`: human-readable
  rendering of the v1 review ledger.
- `configs/market_genome/pattern_ontology_v2.yaml`: current v2 review draft.
  Its canonical resolved-contract SHA-256 is
  `ad3228a608998cc0823c440af0013b12658c9f8dcda4b858ea719be879389ebd`.
- The externally pasted domain review: design input for terminology, object
  classes, family phases, P0 family corrections, and review protocol. It is not
  a versioned or signed repository artifact and therefore is not freeze proof.

The v1 ontology is byte-identical to the tracked baseline: its working-tree
blob and `HEAD:configs/market_genome/pattern_ontology_v1.yaml` both resolve to
Git blob `c56c517488a79597cd5dee6e332a91dfc3e38dd9`, and its SHA-256 matches the v1
review ledger. All 7 files listed by v2 under `v1_artifact_hashes` also match
their recorded SHA-256 values and have no tracked diff. v2 is additive; it does
not mutate v1.

## Draft Contract Review

### Object model

The authoritative v2 class vocabulary and family partition are:

| Object class | Families |
|---|---|
| `STRUCTURAL_STATE` | `trend_up`, `trend_down`, `range` |
| `MORPHOLOGY` | `double_top`, `double_bottom`, `head_shoulders_top`, `inverse_head_shoulders` |
| `LEVEL_EVENT` | `trend_transition`, `breakout`, `failed_breakout`, `retest` |
| `STRUCTURAL_RELATION` | `support_resistance_conversion` |

`retest` is a `LEVEL_EVENT`, not a relation-class object. It preserves its own
identity and connects to its source through `RETESTS` and `ORIGINATES_FROM`.
Likewise, cross-family and cross-scale objects remain distinct; relations do
not destructively merge their identities.

The separation between `structural_orientation`, `semantic_role`, and future
outcome is directionally correct. Morphology is not allowed to acquire a
bullish or bearish future effect from later returns. A classic reversal role
requires preceding structural context and causal confirmation. Future-visible
effects belong only in the `future_outcome` namespace.

### Lifecycle and family phases

The common lifecycle is reduced to the coarse statuses `INACTIVE`, `ACTIVE`,
`RESOLVED`, and `INVALIDATED`. Each family has a variable-length phase graph
with a many-to-one common mapping and explicit retest/continuation branches.
Terminal objects are intended to be immutable, and a later formation requires
a new `pattern_id`.

This model addresses the v1 problem of forcing all 12 families through one
eight-phase sequence. It also distinguishes persistent structural states from
finite events and morphologies. The transition graphs are structurally encoded;
their assumption-level meaning still requires human review.

### Scale, anchors, and formulas

The draft adds a structured `ScaleSpec` contract: closed, session-aligned bars;
versioned session alignment and causal prominence references; explicit
formation spans; optional scale octave and encoder metadata; topology that is
scale-invariant; and train-partition-only selection of scale-specific policy.

The anchor model adds stable identity, causal event and availability times,
closed-bar state, lifecycle state, revision lineage, `price_basis`,
`confirmation_basis`, scale reference, source method/version, and assumption
lineage. The five draft anchor states are `PROVISIONAL`, `CONFIRMED`, `REVISED`,
`SUPERSEDED`, and `REVOKED`. Committed history is append-only, and provisional
partial-bar anchors are excluded from committed ontology history.

Formula entries name their numerator, denominator, sign, unit, missingness,
clipping policy, and fit boundary. Absolute-price denominators and numeric
detector thresholds are prohibited in this draft. Unknown or inapplicable
inputs remain masked. No numeric threshold is selected or frozen by this
review.

### Relations and review namespaces

The v2 relation vocabulary is `NESTED_IN`, `CONTAINS`, `CONFIRMED_BY`,
`INVALIDATED_BY`, `RETESTS`, `ORIGINATES_FROM`, `CONVERTS_ROLE_OF`, and
`SAME_STRUCTURE_DIFFERENT_SCALE`. Relation instances carry their own identity,
causal timestamps, provenance, and assumption lineage.

The namespaces `online_causal_phase`, `retrospective_completion`, and
`future_outcome` remain separate. The online namespace is prefix-only, but its
`runtime_input_allowed` flag is also `false`: none of the three label namespaces
is authorized as a runtime model input by this contract. Retrospective and
outcome fields are additionally future-visible. The proposed review modes are
`ONLINE_BLINDED`, `RETROSPECTIVE`, and `OUTCOME`, with two independent reviewers
and adjudication of disagreement.

## Family-Specific Review

| Family group | v2 draft correction | Review status |
|---|---|---|
| `trend_up`, `trend_down` | Persistent state, sequence-readiness phase, explicit pullback/resumption/weakening, and structure break leading to termination instead of resetting the same object | Draft contract encoded; human decision not recorded |
| `trend_transition` | Starts at the shared defending-pivot structure-break event, preserves prior-trend identity, then records counter-pivot/sequence evidence and delayed orientation binding | Draft contract encoded; human decision not recorded |
| `double_top`, `double_bottom` | Morphology separated from future effect; one intervening pivot defines a horizontal neckline zone; sloped-neckline formula removed | Draft contract encoded; human decision not recorded |
| `head_shoulders_top`, `inverse_head_shoulders` | Left and right neckline anchors define a dynamic neckline; break, retest, and continuation are distinct causal anchors and branches | Draft contract encoded; human decision not recorded |
| `breakout` | Outward cross attempt separated from accepted outside hold; a versioned reference zone is required and post-confirmation retest is explicit | Draft contract encoded; human decision not recorded |
| `failed_breakout` | Object creation occurs at the shared boundary reentry; the earlier attempt is linked through `ORIGINATES_FROM`; attempt, reentry, and resolution directions are not collapsed | Draft contract encoded; human decision not recorded |
| `retest` | Independent `LEVEL_EVENT`; source event/pattern/zone, ordinal, hold, penetrate-reclaim, full recross, and continuation are separated | Draft contract encoded; human decision not recorded |
| `range` | Persistent horizontal zone pair with alternating touches; `ACCEPTED_BREAK` remains active until `TERMINATED` resolves the object | Draft contract encoded; human decision not recorded |
| `support_resistance_conversion` | Independent structural relation; only upward resistance-to-support and downward support-to-resistance conversions are admitted | Draft contract encoded; human decision not recorded |

## Resolved Structural Review Findings

The current v2 draft closes the structural P0 findings found during its initial
read-only audit:

1. All 12 family graphs now declare explicit causal
   `invalidation_transitions`; every declared `INVALIDATED` phase is reachable.
2. Terminal phases have no outgoing edges. In particular, `range` now maps
   `ACCEPTED_BREAK` to `ACTIVE` and only `TERMINATED` to `RESOLVED`.
3. The relation graph now requires `source_ref`, `target_ref`,
   `relation_event_id`, and `source_event_ids`, matching its source-event
   identity invariant.
4. The v1 review-record hash and v2 resolved-contract hash are content-addressed
   values rather than hash placeholders.

These corrections make the draft internally reviewable; they do not constitute
human acceptance of its assumptions.

## Remaining Freeze Blockers

1. **No human decisions are recorded.** `human_review.status` is `PENDING`,
   `frozen` is `false`, `reviewer_decisions_recorded` is `false`, and all 9 new
   assumptions have `proposed_default: null` and `reviewer_decision: null`.
2. **The inherited v1 blockers are not discharged by copying their IDs.** All
   61 inherited assumptions remain `NEEDS_HUMAN_REVIEW`; the v1 ledger records
   0 freeze-ready assumptions and 114 unresolved parameter references. Each
   proposed v2 resolution must be mapped back to the relevant v1 blocker and
   explicitly accepted, revised, deferred, or rejected by human reviewers.

The design is therefore structurally drafted but has 70 open human decisions.
The correct current status remains `DRAFT_NEEDS_HUMAN_FREEZE`.

## Assumption Freeze Gate

The v2 registry contains 70 unfrozen assumptions:

- 61 inherited `MG1A.*` assumptions from the v1 ledger, all with status
  `NEEDS_HUMAN_REVIEW` in v2.
- 9 new `MG1A1.*` assumptions covering object model, morphology/effect
  separation, lifecycle mapping, `ScaleSpec`, anchor revision, relation graph,
  formula contract, review protocol, and family-specific revisions.

Human freeze requires all of the following:

1. Review each v2 proposal against its inherited v1 required changes and
   blocking items; do not infer approval from assumption-ID inheritance.
2. Record two independent reviewer decisions for all 70 assumptions, then
   adjudicate every disagreement and ambiguity.
3. Record the approver, approval time, final file hashes, resolved-contract
   hash, review-record hash, and reviewer/adjudication provenance.
4. Re-run structural validation against the reviewed, hash-bound artifact.
5. Verify the final v2 artifact without modifying v1, and only then change its
   decision/freeze fields and reconsider the `MG1-B` gate.

## Execution Boundary

This sprint is contract-and-validation-only. It does not implement detectors
or trading rules, create labels or synthetic samples, execute JAX, train a
model, or use a GPU. Those activities remain deferred and are not prerequisites
for reviewing or freezing the ontology text itself.
