import pytest
from fastapi.testclient import TestClient

import app as app_module
import config
import security
import service

client = TestClient(app_module.app)
BODY = {"query": "rag poisoning attacks defense"}


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(service, "run_audit", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(security, "limiter", security.RateLimiter(1000, 1000))
    monkeypatch.setattr(config, "API_KEY", "secret-key")


def test_missing_key_is_rejected():
    r = client.post("/audit", json=BODY)
    assert r.status_code == 401


def test_wrong_key_is_rejected():
    r = client.post("/audit", json=BODY, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_correct_key_is_accepted():
    r = client.post("/audit", json=BODY, headers={"X-API-Key": "secret-key"})
    assert r.status_code == 200


def test_health_stays_open():
    assert client.get("/health").status_code == 200


def test_auth_is_off_when_no_key_configured(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "")
    assert client.post("/audit", json=BODY).status_code == 200


def test_rate_limit_returns_429_with_retry_after(monkeypatch):
    monkeypatch.setattr(security, "limiter", security.RateLimiter(2, 100))
    h = {"X-API-Key": "secret-key"}
    assert client.post("/audit", json=BODY, headers=h).status_code == 200
    assert client.post("/audit", json=BODY, headers=h).status_code == 200
    r = client.post("/audit", json=BODY, headers=h)
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1


def test_wrong_keys_do_not_use_up_the_real_allowance(monkeypatch):
    monkeypatch.setattr(security, "limiter", security.RateLimiter(2, 100))
    for _ in range(5):
        assert client.post("/audit", json=BODY, headers={"X-API-Key": "bad"}).status_code == 401
    ok = client.post("/audit", json=BODY, headers={"X-API-Key": "secret-key"})
    assert ok.status_code == 200


def test_hourly_limit_applies(monkeypatch):
    monkeypatch.setattr(security, "limiter", security.RateLimiter(100, 1))
    h = {"X-API-Key": "secret-key"}
    assert client.post("/audit", json=BODY, headers=h).status_code == 200
    assert client.post("/audit", json=BODY, headers=h).status_code == 429


def test_docs_show_authorize_button():
    schemes = client.get("/openapi.json").json()["components"]["securitySchemes"]
    assert any(s.get("name") == "X-API-Key" for s in schemes.values())
