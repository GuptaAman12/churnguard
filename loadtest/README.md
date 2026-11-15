# Load testing

[k6](https://k6.io) drives the prediction API with two concurrent scenarios:

| Scenario | Executor | Exercises |
|---|---|---|
| `predict_ramp` | `ramping-arrival-rate` | `/predict`, ramps 5 → target rps |
| `explain_steady` | `constant-arrival-rate` | `/explain` (TreeSHAP), low steady rate |

The executor is open-loop: k6 offers a fixed request rate regardless of how fast
the service answers, so a saturated API shows up as growing latency instead of
silently throttled load. `EXPLAIN_RATE=0` turns the explanation scenario off for
a predict-only capacity sweep.

## Run it

```bash
make loadtest                          # 25 rps /predict, 3 rps /explain
make loadtest LOADTEST_RATE=50         # higher target
```

Or directly, without `make`:

```bash
docker run --rm --network churnguard_default \
  -e BASE_URL=http://api:8000 -e PREDICT_RATE=50 \
  -e SUMMARY_PATH=/reports/summary.json \
  -v "$PWD/loadtest:/scripts:ro" -v "$PWD/reports/loadtest:/reports" \
  grafana/k6:2.4.0 run /scripts/k6/churnguard.js
```

Against a remote deployment, drop the `--network` flag and point `BASE_URL` at
it; add `-e API_KEY=...` when the API requires `X-API-Key`.

## What it checks

- `/predict` returns 200, a probability in `[0, 1]` and a model version
- `/explain` returns 200, a non-empty `top_features` list and a nested prediction
- every response has a string `risk_tier` (custom `contract_violations` rate)
- thresholds: error rate < 1%, p95 < 120 ms and p99 < 250 ms for `/predict`
  (these are regression guards calibrated to the numbers in
  [docs/benchmarks.md](../docs/benchmarks.md), not a production SLO)

Every run writes the full k6 summary to `reports/loadtest/summary.json`
(`SUMMARY_PATH` overrides it), which is what the benchmark table is built from.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `BASE_URL` | `http://127.0.0.1:8000` | API under test |
| `API_KEY` | unset | sent as `X-API-Key` |
| `PREDICT_RATE` | `25` | sustained `/predict` rps |
| `EXPLAIN_RATE` | `3` | sustained `/explain` rps; `0` disables the scenario |
| `PREDICT_START_RATE` | `5` | initial arrival rate |
| `RAMP_UP` / `HOLD` / `RAMP_DOWN` | `20s` / `45s` / `10s` | ramp schedule |
| `EXPLAIN_DURATION` / `EXPLAIN_START` | `60s` / `20s` | explanation window |
| `SUMMARY_PATH` | `reports/loadtest/summary.json` | where the JSON summary is written |

## Reading the output

```
  metric                            p50        p95        p99        max    count
  /predict                      15.8 ms    37.5 ms    49.1 ms   189.5 ms
  /explain                      53.5 ms    68.0 ms    78.3 ms   206.3 ms
  all requests                  16.3 ms    54.5 ms    66.9 ms   206.3 ms

  throughput 21.6 req/s
  error rate 0.000%
  contract   0.000% invalid
  risk mix   1549 predictions, 1069 high risk
```

Latency is client-observed wall time and includes HTTP, so it is what a caller
feels. `throughput` is the mean over the whole run, including the ramp-down, so
it sits below the sustained target by design.
