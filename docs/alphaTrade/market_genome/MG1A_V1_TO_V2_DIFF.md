# MG1-A Pattern Ontology v1 to v2 Diff

## Scope and Compatibility Statement

v2 is an additive review artifact at milestone `MG1-A.1`. It does not edit or
replace the v1 artifact. The observed v1 file is byte-identical to the tracked
baseline:

| Evidence | Value |
|---|---|
| Path | `configs/market_genome/pattern_ontology_v1.yaml` |
| SHA-256 | `3ab4ce88dd63dbb5ec484b6ca420ce9a3ab06579f63b53776252216a10012b90` |
| Working-tree Git blob | `c56c517488a79597cd5dee6e332a91dfc3e38dd9` |
| `HEAD` Git blob | `c56c517488a79597cd5dee6e332a91dfc3e38dd9` |
| `git diff --exit-code` | Clean for the v1 path |
| v2 canonical resolved-contract SHA-256 | `ad3228a608998cc0823c440af0013b12658c9f8dcda4b858ea719be879389ebd` |

All 7 `v1_artifact_hashes` entries in the v2 draft were independently re-hashed
and match their recorded SHA-256 values; none has a tracked working-tree diff.
The byte-identical statement therefore covers the full recorded v1 artifact
set, not only the ontology YAML.

The v1 review conclusion is unchanged:

- Decision: `DOMAIN_REVIEW_COMPLETED_WITH_REQUIRED_REVISIONS`
- Draft assessment: `APPROVED_AS_DRAFT`
- Freeze assessment: `REVISION_REQUIRED_BEFORE_FREEZE`
- MG1-B: `BLOCKED`

The v2 conclusion is `DRAFT_NEEDS_HUMAN_FREEZE`. It is not a frozen replacement
for v1 and does not unblock MG1-B.

## Contract-Level Changes

| Area | v1 | v2 review draft | Compatibility impact |
|---|---|---|---|
| Governance | Pending human freeze, parameterized review draft | Carries v1 review outcome, explicit human-review record, new assumption registry, and content-addressed contract/review evidence | Additive governance; still not frozen |
| Object classification | Family plus overloaded bullish/bearish/neutral `side` | Four object classes with exact family mapping | Breaking for consumers that treat every family as the same object kind |
| Meaning | `side` mixes geometry, direction, and expected effect | `structural_orientation`, `semantic_role`, and `future_outcome` are separate | Breaking field semantics; prevents future leakage |
| Common lifecycle | One fixed eight-phase sequence for every family | Four coarse statuses with variable family phase graphs and many-to-one mapping | Breaking lifecycle representation |
| Terminal policy | `COMPLETED` and `INVALIDATED` may reset the same object to `INACTIVE` | Terminal objects are immutable; reactivation needs a new `pattern_id` | Breaking identity semantics |
| Retest/continuation | Combined `RETEST_OR_CONTINUATION` phase | Explicit branches and, where applicable, independent retest object | Breaking phase semantics |
| Scale | `scale_id`, `base_timeframe`, shared topology, symbolic scale-specific refs | Structured `ScaleSpec`, session alignment, formation spans, causal prominence, optional encoder scale | Additive fields with stricter provenance |
| Anchors | Role, times, price, source, closed state, coarse state | Stable ID, five states, revision/supersession lineage, price and confirmation bases, source method version | Breaking required-field expansion |
| Relations | Overlap and source identity mostly embedded in family fields | Eight typed relation kinds with independent identity, time, provenance, and assumption lineage | New graph contract |
| Geometry | Lists of normalized geometry/duration/price feature names | Per-formula numerator, denominator, sign, unit, missingness, clipping, and fit boundary | Breaking formula schema; no numeric thresholds added |
| Labels | Online, retrospective, and future namespaces exist | Runtime permission, future visibility, censoring, and review modes are explicit | Stronger anti-leakage contract |
| Review protocol | Human review pending without an executable reviewer protocol | Blinded, retrospective, and outcome modes; two reviewers and adjudication | New review governance |

## Object-Class Mapping

The v2 mapping is normative for the review draft:

| Family | v2 object class |
|---|---|
| `trend_up` | `STRUCTURAL_STATE` |
| `trend_down` | `STRUCTURAL_STATE` |
| `range` | `STRUCTURAL_STATE` |
| `double_top` | `MORPHOLOGY` |
| `double_bottom` | `MORPHOLOGY` |
| `head_shoulders_top` | `MORPHOLOGY` |
| `inverse_head_shoulders` | `MORPHOLOGY` |
| `trend_transition` | `LEVEL_EVENT` |
| `breakout` | `LEVEL_EVENT` |
| `failed_breakout` | `LEVEL_EVENT` |
| `retest` | `LEVEL_EVENT` |
| `support_resistance_conversion` | `STRUCTURAL_RELATION` |

`retest` remains a first-class `LEVEL_EVENT` and links through `RETESTS` and
`ORIGINATES_FROM`; it is not reclassified as `STRUCTURAL_RELATION`.

## Family-Level Changes

| Family | Material v1-to-v2 change |
|---|---|
| `trend_up` | Becomes a persistent `STRUCTURAL_STATE`; replaces the generic phase sequence with impulse, sequence formation, establishment, pullback, resumption, weakening, structure break, and termination phases. |
| `trend_down` | Independent downward persistent-state contract with mirrored geometry but independently reviewable evidence and parameters. |
| `trend_transition` | Becomes a `LEVEL_EVENT` created from the shared defending-pivot structure-break event; explicitly links the prior trend, counter-pivot/sequence, orientation binding, resolution, and expiry. |
| `double_top` | Becomes morphology rather than a bearish effect label. The single intervening trough defines a horizontal neckline zone; the unsupported `neckline_slope_norm` field is removed. |
| `double_bottom` | Independent mirrored morphology. The single intervening peak defines a horizontal neckline zone; the unsupported slope field is removed. |
| `head_shoulders_top` | Splits the neckline into explicit left/right neck anchors and a derived dynamic segment; break, neckline retest, and continuation have separate post-break branches. |
| `inverse_head_shoulders` | Mirrors the top morphology structurally while retaining independent parameter and review decisions. |
| `breakout` | Separates outward cross attempt, closed-bar cross, outside hold/acceptance, post-break retest, continuation, reentry failure, and expiry. |
| `failed_breakout` | Creates the object at shared boundary reentry, links the historical attempt with `ORIGINATES_FROM`, and keeps attempt, reentry, and resolution directions separate. |
| `retest` | Adds source event/pattern/zone identity and retest ordinal; distinguishes new-side hold, penetration then reclaim, full recross failure, continuation, and expiry. |
| `range` | Becomes a persistent horizontal-zone-pair state with alternating touch sequence, boundary tests, false breaks, accepted breaks, and termination. |
| `support_resistance_conversion` | Becomes a `STRUCTURAL_RELATION`; only upward resistance-to-support and downward support-to-resistance mappings are valid, with source breakout, retest, and zone lineage. |

## Assumption Migration

v1 contains 61 assumptions: 13 common assumptions plus four assumptions for each
of the 12 families. The v1 ledger reviewed all 61 but froze none:

| v1 ledger measure | Count |
|---|---:|
| Reviewed | 61 |
| `REVISE` | 51 |
| `DEFER` | 10 |
| `ACCEPT` | 0 |
| Freeze-ready | 0 |
| Freeze-blocked | 61 |
| Unresolved parameter references | 114 |

v2 inherits all 61 IDs with status `NEEDS_HUMAN_REVIEW`; inheritance does not
convert a v1 expert recommendation into a frozen decision. v2 adds these 9
unfrozen assumptions:

1. `MG1A1.COMMON.OBJECT_MODEL.001`
2. `MG1A1.COMMON.MORPHOLOGY_EFFECT.001`
3. `MG1A1.COMMON.LIFECYCLE_MAPPING.001`
4. `MG1A1.COMMON.SCALE_SPEC.001`
5. `MG1A1.COMMON.ANCHOR_REVISION.001`
6. `MG1A1.COMMON.RELATION_GRAPH.001`
7. `MG1A1.COMMON.FORMULA_CONTRACT.001`
8. `MG1A1.COMMON.REVIEW_PROTOCOL.001`
9. `MG1A1.COMMON.FAMILY_REVISIONS.001`

The migration total is therefore 70 assumptions, all unfrozen. The required
state is `0/61` inherited assumptions frozen and `0/9` new assumptions frozen,
not an implicit `61/70` approval.

## Structural Review Delta

The current v2 draft incorporates the initial structural-audit corrections:

1. All 12 family graphs have an explicit causal path to `INVALIDATED`, and all
   declared phases are reachable.
2. Terminal phases have no outgoing edges. `range.ACCEPTED_BREAK` is `ACTIVE`;
   `range.TERMINATED` is the `RESOLVED` terminal.
3. Relation required fields now include `source_ref`, `target_ref`,
   `relation_event_id`, and `source_event_ids`.
4. Review-record and resolved-contract hashes are concrete content hashes, not
   hash placeholders.
5. `failed_breakout` creation is bound to boundary reentry, `trend_transition`
   starts from the shared structure-break event, and both H&S variants use
   separate left/right neckline anchors.

The remaining gap is governance: human review is pending, all 70 assumptions
are open, and no approver, approval time, or frozen hash is recorded. This alone
preserves the v2 status `DRAFT_NEEDS_HUMAN_FREEZE` and MG1-B status `BLOCKED`.
No numeric detector threshold or train-selected artifact is introduced by v2.

## Execution Impact

This diff is contract-and-validation-only. It does not run a detector or
labeler, create real-data or synthetic labels, execute JAX, train a model, or
use a GPU. The v2 draft itself records all of those execution flags as `false`.
GPU- and training-dependent work remains deferred.

## Required Consumer Migration After Freeze

No consumer should migrate while v2 is unfrozen. After human freeze, a consumer
must at minimum:

1. Read `object_class`, `structural_orientation`, and `semantic_role` instead of
   treating v1 `side` as a single semantic axis.
2. Process family-specific phase graphs and their common-status mapping instead
   of assuming the v1 eight-phase sequence.
3. Preserve anchor revisions and relation objects as append-only lineage.
4. Enforce causal timestamps, closed-bar commitments, masked missingness, and
   namespace runtime permissions.
5. Keep all three label namespaces out of runtime model inputs; the v2 draft
   sets `runtime_input_allowed: false` for online, retrospective, and outcome
   namespaces.

This migration plan is descriptive only. It does not authorize MG1-B
implementation before the v2 freeze gate is satisfied.
