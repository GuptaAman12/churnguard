# Benchmarks

Numbers from the local Compose stack, measured with
[k6 2.4.0](https://k6.io) (see [`loadtest/`](../loadtest)). Every table below is
reproducible with `make loadtest`.

The point of this page is not "the API is fast" — it is a single container on a
laptop — but that the platform has a measured latency budget, a known saturation
point, and two real defects that the measurements exposed.

## Environment

| | |
|---|---|
| Host | Windows 11, Docker Desktop (WSL2 backend), 12 vCPU / 7.9 GB for the VM |
| Stack | API, MLflow, Postgres, Prefect server + worker, Pushgateway, Prometheus, Grafana |
| Serving | 1 uvicorn worker, champion v3 (calibrated XGBoost, isotonic) |
| Load generator | k6 2.4.0, container on the Compose network (no host port proxy in the path) |
| Dataset shape | one customer per request, 19 raw fields → 49 model features |

`/predict` and `/explain` are open-loop, arrival-rate scenarios: k6 offers a
fixed request rate and lets latency grow if the service cannot keep up, so
saturation is visible instead of being hidden by a closed-loop virtual-user cap.

## Sustained mixed workload

25 rps `/predict` plus 3 rps `/explain` (TreeSHAP), 80 s per run, three runs
after a warm start. This is the documented baseline; the k6 thresholds in
`loadtest/k6/churnguard.js` are calibrated against it.

| Endpoint | p50 | p95 | p99 | max |
|---|---|---|---|---|
| `POST /predict` | 14.7 ms | 34.8 ms | 47.3 ms | 166 ms |
| `POST /explain` | 49.1 ms | 68.0 ms | 78.3 ms | 189 ms |
| all requests | 15.0 ms | 50.6 ms | 66.9 ms | 189 ms |

Median of the three runs (p50 spread on `/predict`: 14.2–15.8 ms, p95 33.6–37.5 ms).

- **1,730 requests**, **0 errors**, **0 contract violations**, 1,549 predictions
  (1,069 of them high risk, because the pool deliberately includes at-risk rows).
- Latency is client-observed, so it includes HTTP, JSON parsing on the server and
  the whole request path — what a caller actually experiences.
- `/explain` costs ~3.3× `/predict` at p50: TreeSHAP traverses the trees per
  feature, where `/predict` only reads the forest.

## Where the time goes

Single-request profile inside the container (100 iterations each, warm):

| Stage | Cost |
|---|---|
| `engineer_features` (pandas) | 2.9 ms |
| `ColumnTransformer` (impute + one-hot + scale) | 4.3 ms |
| XGBoost `predict_proba` (650 trees, depth 4) | **0.31 ms** |
| Isotonic calibration | < 0.1 ms |
| Prediction log append (async, after the fix below) | ~0 ms on the request path |

The model is not the bottleneck — fixed per-call pandas/scikit-learn overhead is.
That is the expected shape for batch-trained tabular models served one row at a
time, and it explains why the batch endpoint (`/predict/batch`, ≤ 1,000 rows)
exists: amortising that overhead is worth an order of magnitude in throughput.

## Capacity

1 worker, predict-only, 50 s per run (10 s ramp up, 30 s hold, 10 s down). The
`achieved` column is the mean rate over the whole run, so a run that keeps up
lands slightly below the offered rate because of the ramps.

| Offered (hold) | Achieved | p50 | p95 | Verdict |
|---|---|---|---|---|
| 60 rps | 48.5 rps | 14.0 ms | 18.7 ms | keeps up |
| 100 rps | 69.7 rps | 2.9 s | 5.5 s | saturated |
| 150 rps | 62.6 rps | 4.1 s | 14.4 s | overloaded |

Throughput peaks at **~70 rps per worker** and then falls back to ~63 rps at 2.5×
overload, while the error rate stays at 0%: requests are not rejected, they just
queue until the client gives up. A single Python process is the limit —
`predict_proba` releases the GIL, but request handling, Pydantic validation and
the pandas pipeline do not, so extra in-process concurrency only adds overhead.

Scaling out is therefore horizontal. Four workers behind the same container
saturate at ~170–180 rps on this host (~2.7×, not 4×, because 12 vCPUs are shared
with MLflow, Postgres, Prometheus, Grafana, k6 itself and the Windows host).
The Helm chart sets replicas and an HPA on CPU for exactly this reason; the
Compose file stays at one worker to keep the laptop footprint small.

## Two defects the benchmarks exposed

### 1. Model hot-reload blocked inference

`ModelService.load()` held the same lock `predict_proba` read under, for the whole
duration of a registry download. Polling the `champion` alias every 30 s was
cheap only while the alias never moved — the moment a new version was promoted,
every in-flight prediction queued behind the download.

Fixed by building the bundle outside the serving lock and swapping the reference
in a short critical section, with a separate lock serialising reloads.
`tests/test_model_loader.py::test_reload_does_not_block_inference` fails on the
old code and passes on the new one.

### 2. Synchronous logging cost 8 ms per request

Every `/predict` appended a JSONL line to `./logs` — a Windows bind mount — under
a global lock. Measured cost of that append: **8.2 ms** through virtiofs versus
**0.04 ms** on the container's own filesystem.

Replaced with a bounded queue and a single writer thread that appends in batches
and rotates on size. The request path now serialises the record and hands it off;
records are dropped and counted rather than allowed to block or grow memory.

Effect at the 25 rps baseline:

| | before | after |
|---|---|---|
| `/predict` p50 | 19.2 ms | **14.7 ms** |
| `/predict` p95 | 50.6 ms | **34.8 ms** |
| `/predict` p99 | 57.5 ms | **47.3 ms** |
| saturation, 1 worker | ~44 rps | **~70 rps** |

The remaining ~15 ms p50 on Windows still includes filesystem and scheduler
overhead that a Linux host does not pay; the log write itself is no longer part
of it.

## Reproducing

```bash
docker compose up -d          # warm stack
make loadtest                 # 25 rps /predict + 3 rps /explain
make loadtest LOADTEST_RATE=60

# predict-only capacity sweep
docker run --rm --network churnguard_default \
  -e BASE_URL=http://api:8000 -e PREDICT_RATE=100 -e EXPLAIN_RATE=0 \
  -v "$PWD/loadtest:/scripts:ro" \
  grafana/k6:2.4.0 run /scripts/k6/churnguard.js
```

Absolute numbers depend on the host. The shape — flat p50 until ~70 rps per
worker, then unbounded queueing with a 0% error rate — is what to compare
against on real hardware.
