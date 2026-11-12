# Model Card — ChurnGuard churn classifier

**Generated:** 2026-11-12T11:43:21.897135+00:00  
**Algorithm:** XGBoost (650 trees, depth 4)  
**Data fingerprint:** `903c00e5fdb62320`  
**Decision threshold:** 0.2700  
**Calibration:** isotonic

## Intended use
Estimate the probability that a telecom customer will churn so that retention
teams can prioritise outreach. Not intended for automated adverse decisions
(e.g. denying service) without human review.

## Test-set performance
| Metric | Value |
|---|---|
| ROC-AUC | 0.8463 |
| PR-AUC | 0.6457 |
| F1 | 0.6314 |
| Precision | 0.5299 |
| Recall | 0.7807 |
| Log loss | 0.4157 |
| Brier score | 0.1354 |

## Calibration
Isotonic calibration applied (fitted on out-of-fold predictions).

| Metric | Raw | Served |
|---|---|---|
| Log loss | 0.4808 | 0.4157 |
| Brier score | 0.1615 | 0.1354 |
| ECE (10 bins) | 0.1404 | 0.0174 |

## Disaggregated performance
| Group | ROC-AUC | Recall | n |
|---|---|---|---|
| `gender=Male` | 0.8480 | 0.8011 | 722 |
| `gender=Female` | 0.8465 | 0.7617 | 687 |
| `SeniorCitizen=0` | 0.8483 | 0.7464 | 1187 |
| `SeniorCitizen=1` | 0.7839 | 0.8776 | 222 |
| `Contract=Two year` | 0.7329 | 0.0000 | 336 |
| `Contract=Month-to-month` | 0.7550 | 0.8784 | 773 |
| `Contract=One year` | 0.7449 | 0.0833 | 300 |

## Most important features
| Feature | Importance |
|---|---|
| `is_month_to_month` | 0.2946 |
| `Contract_Month-to-month` | 0.2623 |
| `InternetService_Fiber optic` | 0.0383 |
| `OnlineSecurity_No` | 0.0334 |
| `TechSupport_No` | 0.0272 |
| `InternetService_No` | 0.0270 |
| `Contract_Two year` | 0.0249 |
| `OnlineBackup_No internet service` | 0.0222 |

## Explanations
Per-prediction TreeSHAP values are exposed at `POST /explain`. Contributions are
in log-odds space and identify which features push a customer toward or away
from churn.

## Limitations
- Trained on a single public IBM telecom dataset; geography and product mix differ
  from production populations.
- Churn label reflects historical outcomes and may encode selection bias.
- Drift detection must be paired with periodic retraining and human review.
- Calibration is fitted on historical data; monitor reliability after drift.
