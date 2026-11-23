"""Low-latency inference API with readiness and Prometheus instrumentation."""

import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

from fraud_platform.registry import checked_release, release_id
from fraud_platform.contracts import FEATURE_VERSION, Transaction, vector

REQUESTS = Counter("fraud_predictions_total", "Completed scoring requests", ["decision"])
LATENCY = Histogram("fraud_inference_duration_seconds", "Inference request duration")


class Score(BaseModel):
    transaction_id: str
    probability: float
    decision: str
    threshold: float
    model_version: str
    release_id: str
    feature_version: str


class ModelStore:
    """Explicit reload supports controlled deployment and avoids importing untrusted models."""

    def __init__(self):
        self.model = None
        self.manifest = None

    def load(self, location: Path):
        manifest = json.loads((location / "manifest.json").read_text())
        if manifest["feature_version"] != FEATURE_VERSION:
            raise ValueError("Training/serving feature contract mismatch")
        # joblib uses pickle. Only load artifacts from a trusted training workflow.
        actual_hash = hashlib.sha256((location / "model.joblib").read_bytes()).hexdigest()
        if actual_hash != manifest["model_sha256"]:
            raise ValueError("Model checksum does not match manifest")
        model = joblib.load(location / "model.joblib")
        self.model, self.manifest = model, manifest


store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    target = Path(os.getenv("MODEL_DIR", "artifacts"))
    if os.getenv("MODEL_REGISTRY"):
        root = Path(os.environ["MODEL_REGISTRY"])
        target = checked_release(root, (root / "active").read_text().strip())
    store.model, store.manifest = None, None
    if (target / "manifest.json").exists() and (target / "model.joblib").exists():
        store.load(target)
    try:
        yield
    finally:
        store.model, store.manifest = None, None


app = FastAPI(title="Fraud Risk Reference Service", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    if store.model is None:
        raise HTTPException(503, "model artifact not loaded")
    return {"status": "ready", "model_version": store.manifest["model_version"]}


@app.post("/v1/score", response_model=Score)
def score(transaction: Transaction):
    if store.model is None:
        raise HTTPException(503, "model artifact not loaded")
    start = time.perf_counter()
    try:
        probability = float(store.model.predict_proba(np.asarray([vector(transaction)]))[:, 1][0])
        threshold = float(store.manifest["decision_threshold"])
        decision = "review" if probability >= threshold else "allow"
        REQUESTS.labels(decision=decision).inc()
        return Score(transaction_id=transaction.transaction_id, probability=probability,
                     decision=decision, threshold=threshold, release_id=release_id(store.manifest),
                     model_version=store.manifest["model_version"], feature_version=FEATURE_VERSION)
    finally:
        LATENCY.observe(time.perf_counter() - start)
