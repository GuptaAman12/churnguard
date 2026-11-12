# ADR 0002 — Aliases (not stages) and gated promotion

- **Status:** Accepted
- **Date:** 2026-11-12

## Context

MLflow's legacy `Staging` / `Production` stages are deprecated. Teams also
routinely "promote" by overwriting whatever alias exists, which makes a
regression indistinguishable from an improvement after the fact.

## Decision

1. Use the mutable alias `champion` as the single production pointer.
2. Promotion requires **both**:
   - `candidate_auc >= min_roc_auc` (absolute floor), and
   - `candidate_auc >= champion_auc - tolerance` (no material regression).
3. Every version — promoted or rejected — receives tags: `git_sha`,
   `data_fingerprint`, `decision_threshold`, `promotion_reason`.
4. Rollback is `set_registered_model_alias(..., "champion", "<old version>")`;
   the API hot-reloads within one refresh interval.

## Consequences

- The registry is an audit log of decisions, not just artifacts.
- A rejected model still consumes a version number, which is intentional:
  version numbers are immutable history, not a quality ranking.
- If the champion's metric is missing (e.g. manually registered model), the
  system falls back to the absolute floor and records that in the reason.
