# Model Card — ChurnGuard churn classifier

**Generated:** 2026-11-12T07:30:09.022731+00:00  
**Algorithm:** XGBoost (750 trees, depth 3)  
**Data fingerprint:** `903c00e5fdb62320`  
**Decision threshold:** 0.5800

## Intended use
Estimate the probability that a telecom customer will churn so that retention
teams can prioritise outreach. Not intended for automated adverse decisions
(e.g. denying service) without human review.

## Test-set performance
| Metric | Value |
|---|---|
| ROC-AUC | 0.8475 |
| PR-AUC | 0.6625 |
| F1 | 0.6458 |
| Precision | 0.5694 |
| Recall | 0.7460 |
| Log loss | 0.4852 |
| Brier score | 0.1630 |

## Disaggregated performance
| Group | ROC-AUC | Recall | n |
|---|---|---|---|
| `gender=Male` | 0.8488 | 0.7624 | 722 |
| `gender=Female` | 0.8470 | 0.7306 | 687 |
| `SeniorCitizen=0` | 0.8489 | 0.7065 | 1187 |
| `SeniorCitizen=1` | 0.7901 | 0.8571 | 222 |
| `Contract=Two year` | 0.7700 | 0.0000 | 336 |
| `Contract=Month-to-month` | 0.7561 | 0.8419 | 773 |
| `Contract=One year` | 0.7522 | 0.0556 | 300 |

## Most important features
| Feature | Importance |
|---|---|
| `is_month_to_month` | 0.2379 |
| `Contract_Month-to-month` | 0.2237 |
| `OnlineSecurity_No` | 0.0507 |
| `TechSupport_No` | 0.0506 |
| `InternetService_Fiber optic` | 0.0449 |
| `OnlineSecurity_No internet service` | 0.0292 |
| `Contract_Two year` | 0.0269 |
| `PaymentMethod_Electronic check` | 0.0217 |

## Limitations
- Trained on a single public IBM telecom dataset; geography and product mix differ
  from production populations.
- Churn label reflects historical outcomes and may encode selection bias.
- Drift detection must be paired with periodic retraining and human review.
- Probabilities are not calibrated post-hoc; use rank ordering for prioritisation.
