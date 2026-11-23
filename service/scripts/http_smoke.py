"""Exercise the actual ASGI server over a loopback TCP connection."""
import json
import os
import socket
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
import uvicorn

from fraud_platform.api import app as APP
from fraud_platform.registry import promote, register


def prepare(folder: Path) -> dict:
    promote(folder, register(folder, 4000, 42))
    return {"MODEL_REGISTRY": str(folder)}


def verify(client: httpx.Client) -> None:
    ready = client.get("/health/ready")
    payload = dict(transaction_id="demo-1", amount_eur=251.5, merchant_risk=0.8, hour_utc=2,
                   card_age_days=30, transactions_24h=9, ip_country_matches_billing=False)
    valid = client.post("/v1/score", json=payload)
    invalid = client.post("/v1/score", json=dict(payload, amount_eur=-1))
    metrics = client.get("/metrics/")
    assert (ready.status_code, valid.status_code, invalid.status_code,
            metrics.status_code) == (200, 200, 422, 200)
    assert "fraud_predictions_total" in metrics.text and 0 <= valid.json()["probability"] <= 1
    latencies = []
    for _ in range(50):
        start = time.perf_counter()
        reply = client.post("/v1/score",json=payload)
        reply.raise_for_status()
        latencies.append((time.perf_counter()-start)*1000)
    latencies.sort()
    print(json.dumps({"sequential_requests":50,"loopback_p50_ms":round(latencies[24],2),
                      "loopback_p95_ms":round(latencies[47],2)}))
    print(json.dumps(dict(ready=ready.status_code, score=valid.json(),
          invalid=invalid.status_code, metrics=200)))


def main() -> None:
    with TemporaryDirectory() as folder, socket.socket() as listener:
        settings = prepare(Path(folder))
        listener.bind(("127.0.0.1", 0))
        listener.listen(16)
        port = listener.getsockname()[1]
        with patch.dict(os.environ, settings):
            server = uvicorn.Server(uvicorn.Config(APP, log_level="error"))
            thread = threading.Thread(target=server.run, kwargs={
                                      "sockets": [listener]}, daemon=True)
            thread.start()
            try:
                deadline = time.monotonic()+10
                while not server.started and thread.is_alive() and time.monotonic() < deadline:
                    time.sleep(0.02)
                if not server.started:
                    raise RuntimeError("Server failed to start")
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=5) as client:
                    verify(client)
            finally:
                server.should_exit = True
                thread.join(timeout=10)
                if thread.is_alive():
                    raise RuntimeError("Server failed to stop")


if __name__ == "__main__":
    main()
