from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from backend.app.main import create_app
from backend.app.services import SessionStore


class FakePredictor:
    version = "test-only"
    scope = "test_fixture"
    metrics = {"scope": "test_fixture", "models": {}, "split": "val"}

    def predict(self, prompt, images):
        return [{"score": 3., "cosine": 0.2} for _ in images]


def png():
    stream = BytesIO()
    Image.new("RGB", (8, 8), "red").save(stream, format="PNG")
    return stream.getvalue()


def client():
    return TestClient(create_app(FakePredictor()))


def test_health_unconfigured_and_metrics():
    empty = TestClient(create_app())
    assert empty.get("/api/health").json()["data"]["model_ready"] is False
    assert empty.get("/api/metrics").status_code == 503
    assert client().get("/api/metrics").json()["data"]["scope"] == "test_fixture"


def test_evaluate_and_compare_ties():
    c = client()
    response = c.post("/api/evaluate", data={"prompt": "red"}, files={"image": ("../../a.png", png())})
    assert response.status_code == 200
    assert response.json()["data"]["items"][0]["score"] == 3
    response = c.post("/api/compare", data={"prompt": "red"}, files=[("images", ("a.png", png()))] * 2)
    assert [r["rank"] for r in response.json()["data"]["items"]] == [1, 1]


@pytest.mark.parametrize("prompt,payload,status", [(" ", b"bad", 422), ("red", b"bad", 415),
    ("x" * 1001, b"bad", 422), ("red", b"0" * (10 * 1024 * 1024 + 1), 413)],
    ids=["empty-prompt", "invalid-image", "long-prompt", "large-file"])
def test_bad_inputs(prompt, payload, status):
    assert client().post("/api/evaluate", data={"prompt": prompt}, files={"image": ("a.png", payload)}).status_code == status


def test_missing_and_wrong_count():
    c = client()
    assert c.post("/api/evaluate", data={"prompt": "red"}).status_code == 422
    for count in (1, 7):
        assert c.post("/api/compare", data={"prompt": "red"}, files=[("images", ("a.png", png()))] * count).status_code == 422


def test_session_isolation_followup_and_expiry():
    now = [0]
    store = SessionStore(ttl=5, clock=lambda: now[0])
    c = TestClient(create_app(FakePredictor(), store))
    first, second = [c.post("/api/sessions").json()["data"]["session_id"] for _ in range(2)]
    ids = c.post("/api/uploads", data={"session_id": first}, files=[("files", ("a.png", png()))] * 2).json()["data"]["image_ids"]
    body = {"session_id": first, "action": "compare", "prompt": "red", "image_ids": ids}
    assert c.post("/api/chat", json=body).status_code == 200
    assert c.post("/api/chat", json={**body, "session_id": second}).status_code == 404
    follow = c.post("/api/chat", json={"session_id": first, "text": "ảnh thứ hai thì sao"})
    assert follow.status_code == 200
    assert follow.json()["data"]["prompt"] == "red"
    assert follow.json()["data"]["items"][0]["index"] == 1
    now[0] = 6
    assert c.post("/api/chat", json=body).status_code == 404
    assert not store.items


def test_exception_is_sanitized():
    class Broken(FakePredictor):
        def predict(self, *args):
            raise RuntimeError("SECRET-MUST-NOT-LEAK")
    c = TestClient(create_app(Broken()))
    response = c.post("/api/evaluate", data={"prompt": "red"}, files={"image": ("a.png", png())})
    assert response.status_code == 500
    assert "SECRET" not in response.text
    assert response.json()["request_id"]


def test_pixel_limit(monkeypatch):
    monkeypatch.setattr("backend.app.services.MAX_PIXELS", 4)
    assert client().post("/api/evaluate", data={"prompt": "red"}, files={"image": ("a.png", png())}).status_code == 413


def test_chunked_body_limit():
    import asyncio
    from backend.app.limits import BodyLimitMiddleware
    called, messages = [], []
    async def app(*args):
        called.append(True)
    chunks = iter([{"type": "http.request", "body": b"123", "more_body": True},
                   {"type": "http.request", "body": b"456", "more_body": False}])
    async def receive():
        return next(chunks)
    async def send(message):
        messages.append(message)
    asyncio.run(BodyLimitMiddleware(app, limit=5)({"type": "http", "method": "POST"}, receive, send))
    assert not called
    assert messages[0]["status"] == 413
