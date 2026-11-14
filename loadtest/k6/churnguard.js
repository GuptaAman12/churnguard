// ChurnGuard load test.
//
// Two scenarios run concurrently against the prediction API:
//   * predict_ramp    - arrival-rate ramp to a sustained target, exercises /predict
//   * explain_steady  - low, steady rate against /explain (TreeSHAP is ~5x costlier)
//
// Everything is environment-driven so CI, a laptop and a staging cluster can run
// the same file:
//   BASE_URL, API_KEY, PREDICT_RATE
//
// Example:
//   k6 run loadtest/k6/churnguard.js
//   k6 run -e BASE_URL=https://churn.example.com -e API_KEY=... loadtest/k6/churnguard.js

import http from 'k6/http';
import { check } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

const BASE_URL = (__ENV.BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const API_KEY = __ENV.API_KEY || '';
const PREDICT_RATE = Number(__ENV.PREDICT_RATE || 25);
const EXPLAIN_RATE = Number(__ENV.EXPLAIN_RATE || 3);

const predictDuration = new Trend('predict_duration', true);
const explainDuration = new Trend('explain_duration', true);
const contractViolations = new Rate('contract_violations');
const predictions = new Counter('predictions_total');
const predictionsHighRisk = new Counter('predictions_high_risk');

// A small, deliberately varied pool so the model sees more than one row shape.
// Values are real records from the Telco churn dataset.
const CUSTOMERS = [
  {
    gender: 'Female', SeniorCitizen: 0, Partner: 'Yes', Dependents: 'No', tenure: 3,
    PhoneService: 'Yes', MultipleLines: 'No', InternetService: 'Fiber optic',
    OnlineSecurity: 'No', OnlineBackup: 'No', DeviceProtection: 'No', TechSupport: 'No',
    StreamingTV: 'Yes', StreamingMovies: 'No', Contract: 'Month-to-month',
    PaperlessBilling: 'Yes', PaymentMethod: 'Electronic check',
    MonthlyCharges: 95.5, TotalCharges: 286.5,
  },
  {
    gender: 'Male', SeniorCitizen: 1, Partner: 'No', Dependents: 'No', tenure: 1,
    PhoneService: 'Yes', MultipleLines: 'Yes', InternetService: 'Fiber optic',
    OnlineSecurity: 'No', OnlineBackup: 'No', DeviceProtection: 'No', TechSupport: 'No',
    StreamingTV: 'No', StreamingMovies: 'No', Contract: 'Month-to-month',
    PaperlessBilling: 'Yes', PaymentMethod: 'Electronic check',
    MonthlyCharges: 85.15, TotalCharges: 85.15,
  },
  {
    gender: 'Female', SeniorCitizen: 0, Partner: 'No', Dependents: 'No', tenure: 24,
    PhoneService: 'Yes', MultipleLines: 'Yes', InternetService: 'DSL',
    OnlineSecurity: 'Yes', OnlineBackup: 'No', DeviceProtection: 'Yes', TechSupport: 'No',
    StreamingTV: 'No', StreamingMovies: 'Yes', Contract: 'One year',
    PaperlessBilling: 'No', PaymentMethod: 'Credit card (automatic)',
    MonthlyCharges: 61.4, TotalCharges: 1481.7,
  },
  {
    gender: 'Male', SeniorCitizen: 0, Partner: 'Yes', Dependents: 'Yes', tenure: 58,
    PhoneService: 'Yes', MultipleLines: 'Yes', InternetService: 'DSL',
    OnlineSecurity: 'Yes', OnlineBackup: 'Yes', DeviceProtection: 'Yes', TechSupport: 'Yes',
    StreamingTV: 'No', StreamingMovies: 'No', Contract: 'Two year',
    PaperlessBilling: 'No', PaymentMethod: 'Bank transfer (automatic)',
    MonthlyCharges: 45.9, TotalCharges: 2673.4,
  },
  {
    gender: 'Female', SeniorCitizen: 1, Partner: 'No', Dependents: 'No', tenure: 12,
    PhoneService: 'No', MultipleLines: 'No phone service', InternetService: 'Fiber optic',
    OnlineSecurity: 'No', OnlineBackup: 'Yes', DeviceProtection: 'No', TechSupport: 'No',
    StreamingTV: 'Yes', StreamingMovies: 'Yes', Contract: 'Month-to-month',
    PaperlessBilling: 'Yes', PaymentMethod: 'Credit card (automatic)',
    MonthlyCharges: 84.0, TotalCharges: 1010.5,
  },
  {
    gender: 'Male', SeniorCitizen: 0, Partner: 'Yes', Dependents: 'No', tenure: 36,
    PhoneService: 'Yes', MultipleLines: 'No', InternetService: 'Fiber optic',
    OnlineSecurity: 'Yes', OnlineBackup: 'No', DeviceProtection: 'Yes', TechSupport: 'Yes',
    StreamingTV: 'Yes', StreamingMovies: 'Yes', Contract: 'One year',
    PaperlessBilling: 'Yes', PaymentMethod: 'Electronic check',
    MonthlyCharges: 105.25, TotalCharges: 3799.0,
  },
];

const HEADERS = {
  'Content-Type': 'application/json',
  ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
};

function customer() {
  return JSON.stringify(CUSTOMERS[__ITER % CUSTOMERS.length]);
}

export const options = {
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  scenarios: buildScenarios(),
  thresholds: {
    // Calibrated against the local baseline in docs/benchmarks.md; they are a
    // regression guard, not a service-level objective for production hardware.
    http_req_failed: ['rate<0.01'],
    contract_violations: ['rate<0.001'],
    predict_duration: ['p(95)<120', 'p(99)<250'],
    explain_duration: ['p(95)<250', 'p(99)<500'],
    checks: ['rate>0.99'],
  },
};

function buildScenarios() {
  const scenarios = {
    predict_ramp: {
      executor: 'ramping-arrival-rate',
      exec: 'predictScenario',
      startRate: Number(__ENV.PREDICT_START_RATE || 5),
      timeUnit: '1s',
      preAllocatedVUs: 60,
      maxVUs: 400,
      stages: [
        { target: PREDICT_RATE, duration: __ENV.RAMP_UP || '20s' },
        { target: PREDICT_RATE, duration: __ENV.HOLD || '45s' },
        { target: 0, duration: __ENV.RAMP_DOWN || '10s' },
      ],
      tags: { scenario: 'predict' },
    },
  };
  // EXPLAIN_RATE=0 runs a predict-only capacity sweep.
  if (EXPLAIN_RATE > 0) {
    scenarios.explain_steady = {
      executor: 'constant-arrival-rate',
      exec: 'explainScenario',
      rate: EXPLAIN_RATE,
      timeUnit: '1s',
      duration: __ENV.EXPLAIN_DURATION || '60s',
      startTime: __ENV.EXPLAIN_START || '20s',
      preAllocatedVUs: 10,
      maxVUs: 80,
      tags: { scenario: 'explain' },
    };
  }
  return scenarios;
}

export function predictScenario() {
  const response = http.post(`${BASE_URL}/predict`, customer(), { headers: HEADERS });

  predictDuration.add(response.timings.duration);
  check(response, {
    'predict: status 200': (r) => r.status === 200,
    'predict: probability in [0,1]': (r) => {
      const body = safeJson(r);
      return body !== null && body.churn_probability >= 0 && body.churn_probability <= 1;
    },
    'predict: model version present': (r) => {
      const body = safeJson(r);
      return body !== null && body.model !== undefined && body.model.version !== undefined;
    },
  });

  const body = safeJson(response);
  contractViolations.add(body === null || typeof body.risk_tier !== 'string');
  if (body !== null) {
    predictions.add(1);
    if (body.risk_tier === 'high') {
      predictionsHighRisk.add(1);
    }
  }
}

export function explainScenario() {
  const response = http.post(`${BASE_URL}/explain`, customer(), { headers: HEADERS });

  explainDuration.add(response.timings.duration);
  check(response, {
    'explain: status 200': (r) => r.status === 200,
    'explain: contributions returned': (r) => {
      const body = safeJson(r);
      return body !== null && Array.isArray(body.top_features) && body.top_features.length > 0;
    },
    'explain: shrunk probability present': (r) => {
      const body = safeJson(r);
      return (
        body !== null &&
        body.prediction !== undefined &&
        typeof body.prediction.churn_probability === 'number'
      );
    },
  });
}

function safeJson(response) {
  try {
    return response.json();
  } catch (error) {
    return null;
  }
}

function row(name, values, indent) {
  const pad = ' '.repeat(indent);
  return `${pad}${name.padEnd(26)}${values.map((v) => String(v).padStart(11)).join('')}`;
}

export function handleSummary(data) {
  const m = data.metrics;
  const lines = [];
  lines.push('');
  lines.push('ChurnGuard load test');
  lines.push(`  base url   ${BASE_URL}`);
  lines.push(`  target     ${PREDICT_RATE} rps /predict, ${EXPLAIN_RATE} rps /explain`);
  lines.push('');
  lines.push(row('metric', ['p50', 'p95', 'p99', 'max'], 2) + '    count');
  for (const [key, label] of [
    ['predict_duration', '/predict'],
    ['explain_duration', '/explain'],
    ['http_req_duration', 'all requests'],
  ]) {
    const metric = m[key];
    if (!metric) {
      continue;
    }
    const v = metric.values;
    lines.push(
      row(label, [ms(v.med), ms(v['p(95)']), ms(v['p(99)']), ms(v.max)], 2) +
        String(v.count ?? '').padStart(9),
    );
  }
  lines.push('');
  if (m.http_reqs) {
    lines.push(`  throughput ${m.http_reqs.values.rate.toFixed(1)} req/s`);
  }
  if (m.http_req_failed) {
    lines.push(`  error rate ${(m.http_req_failed.values.rate * 100).toFixed(3)}%`);
  }
  if (m.contract_violations) {
    lines.push(`  contract   ${(m.contract_violations.values.rate * 100).toFixed(3)}% invalid`);
  }
  if (m.predictions_total) {
    lines.push(
      `  risk mix   ${m.predictions_total.values.count} predictions, ` +
        `${m.predictions_high_risk.values.count} high risk`,
    );
  }
  const failed = data.metrics.checks ? data.metrics.checks.values.rate < 1 : false;
  lines.push('');
  lines.push(failed ? '  FAIL  one or more checks failed' : '  PASS  all checks and thresholds met');
  lines.push('');

  const out = __ENV.SUMMARY_PATH || 'reports/loadtest/summary.json';
  return {
    stdout: lines.join('\n'),
    [out]: JSON.stringify(data, null, 2),
  };
}

function ms(value) {
  return value === undefined ? '-' : `${value.toFixed(1)} ms`;
}
