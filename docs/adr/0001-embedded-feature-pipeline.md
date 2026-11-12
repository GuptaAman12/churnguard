# ADR 0001 — One artifact, features embedded in the pipeline

- **Status:** Accepted
- **Date:** 2026-11-12

## Context

Churn models typically ship as "model file + separate feature code". That split
is the most common cause of training/serving skew: the notebook computes
`avg_monthly_spend` from `TotalCharges / tenure`, the service computes it from a
different denominator, and offline AUC silently overstates online quality.

## Decision

All deterministic feature engineering lives in `engineer_features` and is
embedded as the **first step** of the fitted scikit-learn `Pipeline`
(`FunctionTransformer`). The serialized artifact is therefore the complete
transformation + model, and the API receives raw customer fields only.

## Consequences

- Serving cannot diverge from training for these features.
- The artifact is larger and tied to the project's Python package (pickle).
- Any change to feature code requires a new model version — which is desirable,
  because it forces re-evaluation and re-promotion.
- Non-deterministic features (target encodings, time-window aggregates) would
  need a separate feature store; they are deliberately out of scope.
