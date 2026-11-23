# Interview preparation · Fraud model delivery and monitoring

Use first-person wording after you have run the demo and can explain or change the code yourself. Describe this as a portfolio implementation. Quantities below refer to the synthetic fixture.

## 30-second introduction

I extended a fraud-modeling portfolio topic into a release and review decision. I constrained the validation review queue, added an immutable local registry, scored through an API and tested rollback. On 800 synthetic test transactions it flagged 59 and caught four of 22 frauds. I would present both the operational controls and the weak recall before discussing deployment.

## Two-minute technical explanation

Contracts.py maps bounded transaction fields into the six-element vector. Train.py fits the pipeline, selects a threshold and writes a model card as JSON. Registry.py stages a release, checks its hash, and atomically replaces a small active pointer. At startup the API validates the selected release. Each score reports its release ID, score and threshold. Monitoring.py compares reference and current feature histograms independently from scoring.

Split 4,000 seeded independent observations 60/20/20 with stratification. Fit a scaler and logistic regression on training. Search validation score thresholds plus the always-allow threshold, minimizing €500 per miss and €20 per false alert subject to a 10% validation review cap. Evaluate the fixed rule once on test. The manifest records split counts, standardized coefficients and error counts. The 10% constraint is a calibration-set rule, so live arrival bursts may exceed capacity.

On an 800-transaction synthetic holdout, the service flagged 59 transactions. Four were simulated frauds, 55 were false positives and 18 frauds were missed. Average precision was 0.070 against a prevalence baseline of 0.0275. The simulated cost was €10,100 versus €11,000 for always allowing. **The low recall is central to the release discussion.**

Synthetic labels encode a fixed stochastic rule. The holdout contains only 22 positive cases. A checksum verifies integrity relative to the manifest. Restrict write access to both artifacts and manifests because joblib deserializes trusted Python objects.

## Architecture walkthrough

Open the files in this order and follow one record through the system.

[contracts.py](../src/fraud_platform/contracts.py) · [train.py](../src/fraud_platform/train.py) · [registry.py](../src/fraud_platform/registry.py) · [api.py](../src/fraud_platform/api.py)

Contracts.py maps bounded transaction fields into the six-element vector. Train.py fits the pipeline, selects a threshold and writes a model card as JSON. Registry.py stages a release, checks its hash, and atomically replaces a small active pointer. At startup the API validates the selected release. Each score reports its release ID, score and threshold. Monitoring.py compares reference and current feature histograms independently from scoring.

## Five technical questions

**1. Why prefer average precision to accuracy here?**

Always allowing already classifies most records correctly. Average precision measures ranking of scarce positives relative to prevalence.

**2. Where is the threshold chosen?**

On validation predictions, under an empirical capacity limit. Test labels are reserved for final evaluation.

**3. What makes a release identifiable?**

A digest of the model bytes is included in a manifest-derived ID. It incorporates decision settings and excludes the training timestamp.

**4. What happens if promotion is interrupted?**

The pointer is replaced atomically after verification. Staging is separate. Operator coordination and storage durability still matter for multiple hosts.

**5. Does a drift alert prove a model is worse?**

A population stability index compares feature distributions. Outcome quality requires matured labels, cohort analysis and calibration monitoring.

## Five product and business questions

**1. Would you deploy this model?**

The observed recall is 4/22 and the data is synthetic. I would keep it in a shadow evaluation.

**2. Who owns the threshold?**

Risk operations sets capacity and cost assumptions. The model owner evaluates candidate policies against them.

**3. What is the false-positive burden?**

55 of 59 reviewed test transactions were legitimate under the simulator.

**4. What is the cost comparison worth?**

€900 lower loss under one fixed cost matrix. Actual capacity, recovery value and investigation cost require operational evidence.

**5. What should trigger rollback?**

An invalid release, readiness failure, unacceptable queue growth or a monitored deterioration against pre-agreed thresholds.

## What to demonstrate live

Run `python scripts/demo.py`. Point to one input, the relevant transformation and the output that supports your decision. Tighten the 10% validation cap to 5% in a working copy. Compare the threshold, false negatives and review volume against the reference run.

## Main trade-off and next test

Scikit-learn shares the same feature mapper between training and API. FastAPI enforces the transaction contract. A filesystem registry demonstrates immutable releases and promotion on one host. Prometheus counters and latency histograms expose service behavior.

Synthetic labels encode a fixed stochastic rule. The holdout contains only 22 positive cases. A checksum verifies integrity relative to the manifest. Restrict write access to both artifacts and manifests because joblib deserializes trusted Python objects.

## CV wording for a portfolio section

- Implemented a local fraud scoring and model-release workflow with constrained threshold selection, versioned artifacts, HTTP inference, drift diagnostics and tested rollback on synthetic data.
