import json

import numpy as np
import os
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fraud_platform.api import app, store
from fraud_platform.contracts import Transaction, vector
from fraud_platform.monitoring import population_stability_index
from fraud_platform.synthetic import generate
from fraud_platform.train import expected_cost, train


def sample():
    return Transaction(transaction_id="synthetic-1", amount_eur=125, merchant_risk=0.4,
                       hour_utc=3, card_age_days=200, transactions_24h=4,
                       ip_country_matches_billing=False)


class PlatformTests(TestCase):
    def test_feature_contract_and_rejection(self):
        assert len(vector(sample())) == 6
        with self.assertRaises(ValidationError):
            Transaction(**{**sample().model_dump(), "amount_eur": -1})
        with self.assertRaises(ValidationError):
            Transaction(**{**sample().model_dump(), "card_pan": "secret"})

    def test_synthetic_reproducibility(self):
        assert generate(200, seed=17) == generate(200, seed=17)

    def test_threshold_cost(self):
        y = np.array([1, 0])
        p = np.array([0.2, 0.8])
        assert expected_cost(y, p, 0.5, 500, 20) == 520
        assert expected_cost(y, p, 0.1, 500, 20) == 20

    def test_train_and_serving(self):
        tmp_path = Path(self.enterContext(TemporaryDirectory()))
        report = train(tmp_path, rows=1200, seed=17)
        assert report["split_counts"] == {"train": 720, "validation": 240, "test": 240}
        assert 0 <= report["test_average_precision"] <= 1
        assert (tmp_path / "model.joblib").exists()
        assert json.loads((tmp_path / "manifest.json").read_text())["feature_version"]
        self.enterContext(patch.dict(os.environ, MODEL_DIR=str(tmp_path)))
        with TestClient(app) as client:
            response = client.post("/v1/score", json=sample().model_dump())
            assert response.status_code == 200
            assert response.json()["decision"] in {"allow", "review"}
            assert client.get("/health/ready").status_code == 200
        store.model = None
        store.manifest = None

    def test_unavailable_model_returns_503(self):
        store.model = None
        with TestClient(app) as client:
            assert client.post("/v1/score", json=sample().model_dump()).status_code == 503
            assert client.get("/health/ready").status_code == 503

    def test_drift(self):
        x = list(range(1000))
        assert population_stability_index(x, x) < 1e-9
        assert population_stability_index(x, [t + 200 for t in x]) > 0.1
        with self.assertRaises(ValueError):
            population_stability_index([1, 2], [1, 2])
