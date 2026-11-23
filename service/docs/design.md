# Product requirements and operating notes

## User decision and opportunity

A review team has limited capacity. This project trains a risk model, chooses a review threshold within a validation budget, packages an identifiable release, serves scores and exercises rollback.

| Stakeholder | Priority and milestone | Acceptance criterion |
|---|---|---|
| Risk operations | P0 · feasible queue | Fit the decision threshold with at most 10% of validation rows flagged. Report held-out recall, review volume and the always-allow comparator. |
| Platform operator | P0 · recoverable serving | Check artifact identity before switching. A corrupt candidate must leave the previous release active. |
| Model owner | P1 · drift diagnosis | Compare numeric input distributions and join delayed labels in a future pilot before authorizing retraining. |

## Metric definitions and validation

Split 4,000 seeded independent observations 60/20/20 with stratification. Fit a scaler and logistic regression on training. Search validation score thresholds plus the always-allow threshold, minimizing €500 per miss and €20 per false alert subject to a 10% validation review cap. Evaluate the fixed rule once on test. The manifest records split counts, standardized coefficients and error counts. The 10% constraint is a calibration-set rule, so live arrival bursts may exceed capacity.

On an 800-transaction synthetic holdout, the service flagged 59 transactions. Four were simulated frauds, 55 were false positives and 18 frauds were missed. Average precision was 0.070 against a prevalence baseline of 0.0275. The simulated cost was €10,100 versus €11,000 for always allowing. **The low recall is central to the release discussion.**

## Architecture decision record

**Accepted for this local workload.** Scikit-learn shares the same feature mapper between training and API. FastAPI enforces the transaction contract. A filesystem registry demonstrates immutable releases and promotion on one host. Prometheus counters and latency histograms expose service behavior.

Contracts.py maps bounded transaction fields into the six-element vector. Train.py fits the pipeline, selects a threshold and writes a model card as JSON. Registry.py stages a release, checks its hash, and atomically replaces a small active pointer. At startup the API validates the selected release. Each score reports its release ID, score and threshold. Monitoring.py compares reference and current feature histograms independently from scoring.

The implementation lives in [the Python package](../src/fraud_platform). The boundaries are exercised by [the test suite](../tests).

## Operating procedure

For versioned service startup in PowerShell, use `$version = fraud-registry register --rows 4000 --seed 42`, then `fraud-registry promote $version`, then `$env:MODEL_REGISTRY = "artifacts/registry"`. Restart the service after changing the active version. Keep the previous release ID. To roll back, promote that ID and restart. Readiness returns 503 before a model loads. Score requests and readiness are exercised over loopback in the verification log.

1. Reproduce the baseline with `python scripts/demo.py`. Compare the decision fields with the committed example.
2. Run the regression suite before changing input definitions or thresholds. Use the persistent example in the README to inspect intermediate state.
3. Run `python scripts/http_smoke.py` to test the API over TCP. The server exits at the end of the check.

### Container example

From the repository root:

```shell
docker compose up --build
```

For Compose scoring, first run `fraud-train --output artifacts` on the host. The service mounts this folder read-only. A training artifact is required for readiness.
This configuration awaits Docker execution. The same demo was exercised with the installed Python package.

## Constraints, reliability and delivery

Synthetic labels encode a fixed stochastic rule. The holdout contains only 22 positive cases. A checksum verifies integrity relative to the manifest. Restrict write access to both artifacts and manifests because joblib deserializes trusted Python objects.

Establish the cost comparator, lock the feature contract, exercise promotion and rollback, then run a labeled shadow trial. Before a pilot, replace the independent split with a time- and entity-aware evaluation and rehearse a capacity alert.

Local demo data and code are independent of former employers. Cost scenarios use declared assumptions. Operational targets describe a proposed pilot.
