from io import BytesIO
import math
import os
import time
import uuid
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .services import SessionStore, decode_image, MAX_BYTES
from .limits import BodyLimitMiddleware


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)
    text: str = Field(default="", max_length=1000)
    prompt: str | None = Field(default=None, max_length=1000)
    image_ids: list[str] = Field(default_factory=list, max_length=6)
    action: Literal["evaluate", "compare", "metrics", "help", "auto"] = "auto"


def create_app(predictor=None, store=None):
    app = FastAPI(title="PromptAlign AI", version="0.1.0")
    app.state.predictor = predictor
    app.state.store = store or SessionStore()

    def envelope(request, kind, data):
        current = app.state.predictor
        return {"request_id": request.state.request_id, "type": kind, "data": data,
                "model_version": current.version if current else None,
                "elapsed_ms": round((time.perf_counter() - request.state.started) * 1000, 2),
                "warnings": ["Mô hình thử nghiệm; chưa đánh giá trên GenAI-Bench."] if current else []}

    @app.middleware("http")
    async def request_context(request, call_next):
        request.state.request_id = uuid.uuid4().hex
        request.state.started = time.perf_counter()
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > 61 * 1024 * 1024):
            return JSONResponse(envelope(request, "error", {"code": "REQUEST_TOO_LARGE", "message": "Yêu cầu quá lớn."}), status_code=413)
        try:
            return await call_next(request)
        except Exception:
            return JSONResponse(envelope(request, "error", {"code": "INTERNAL_ERROR", "message": "Không thể xử lý yêu cầu."}), status_code=500)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(envelope(request, "error", {"code": f"HTTP_{exc.status_code}", "message": exc.detail}), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(envelope(request, "error", {"code": "INVALID_INPUT", "message": "Kiểm tra số ảnh, prompt và các trường bắt buộc."}), status_code=422)

    def required_model():
        if app.state.predictor is None:
            raise HTTPException(503, "Chưa có mô hình được cấu hình. Không có điểm giả được trả về.")
        return app.state.predictor

    def score(prompt, payloads, compare=False):
        prompt = prompt.strip()
        if not 1 <= len(prompt) <= 1000:
            raise HTTPException(422, "Prompt cần từ 1 đến 1000 ký tự.")
        if not (2 <= len(payloads) <= 6 if compare else len(payloads) == 1):
            raise HTTPException(422, "Chấm điểm cần 1 ảnh; so sánh cần 2–6 ảnh.")
        images = []
        try:
            for payload in payloads:
                images.append(decode_image(payload))
            results = required_model().predict(prompt, images)
            if len(results) != len(images) or any(not math.isfinite(r["score"]) or not 1 <= r["score"] <= 5
                    or not math.isfinite(r["cosine"]) or not -1 <= r["cosine"] <= 1 for r in results):
                raise RuntimeError("invalid predictor output")
            values = sorted({r["score"] for r in results}, reverse=True)
            return [{**r, "index": i, "rank": values.index(r["score"]) + 1} for i, r in enumerate(results)]
        finally:
            for image in images:
                image.close()

    async def read_files(files):
        if not 1 <= len(files) <= 6:
            raise HTTPException(422, "Chọn từ 1 đến 6 ảnh.")
        payloads = []
        for file in files:
            try:
                payload = await file.read(MAX_BYTES + 1)
                if len(payload) > MAX_BYTES:
                    raise HTTPException(413, "Ảnh vượt giới hạn 10 MB.")
                payloads.append(payload)
            finally:
                await file.close()
        return payloads

    @app.get("/api/health")
    def health(request: Request):
        p = app.state.predictor
        return envelope(request, "health", {"model_ready": p is not None, "scope": p.scope if p else "not_configured"})

    @app.post("/api/sessions")
    def sessions(request: Request):
        return envelope(request, "session", {"session_id": app.state.store.create(), "ttl_seconds": app.state.store.ttl})

    @app.post("/api/uploads")
    async def uploads(request: Request, session_id: str = Form(...), files: list[UploadFile] = File(...)):
        payloads = await read_files(files)
        for payload in payloads:
            decode_image(payload).close()
        ids = app.state.store.upload(session_id, payloads)
        return envelope(request, "uploads", {"image_ids": ids})

    @app.post("/api/evaluate")
    async def evaluate(request: Request, prompt: str = Form(...), image: UploadFile = File(...)):
        results = await run_in_threadpool(score, prompt, await read_files([image]))
        return envelope(request, "score", {"items": results})

    @app.post("/api/compare")
    async def compare(request: Request, prompt: str = Form(...), images: list[UploadFile] = File(...)):
        results = await run_in_threadpool(score, prompt, await read_files(images), True)
        return envelope(request, "ranking", {"items": results})

    @app.get("/api/metrics")
    def metrics(request: Request):
        return envelope(request, "metrics", required_model().metrics)

    @app.post("/api/chat")
    def chat(request: Request, body: ChatRequest):
        with app.state.store.lock:
            app.state.store.get(body.session_id)
        action = body.action
        text = body.text.strip().casefold()
        if action == "auto":
            action = "metrics" if any(w in text for w in ("biểu đồ", "metric")) else "compare" if "so sánh" in text else "evaluate" if any(w in text for w in ("chấm", "ảnh thứ")) else "help"
        if action == "metrics":
            return envelope(request, "metrics", required_model().metrics)
        if action == "help":
            return envelope(request, "text", {"text": "Nhập prompt, tải ảnh rồi chọn Chấm ảnh hoặc So sánh. Bạn cũng có thể xem kết quả đánh giá."})
        ids, prompt, payloads = app.state.store.resolve(body.session_id, body.image_ids, body.prompt)
        second_image = "ảnh thứ hai" in text and not body.image_ids
        if second_image:
            if len(ids) < 2:
                raise HTTPException(422, "Phiên chưa có ảnh thứ hai.")
            payloads = [payloads[1]]
        results = score(prompt, payloads, action == "compare")
        if second_image:
            results[0]["index"] = 1
        app.state.store.remember(body.session_id, ids, prompt)
        return envelope(request, "ranking" if action == "compare" else "score", {"items": results, "prompt": prompt})

    app.add_middleware(BodyLimitMiddleware)
    return app


def configured_app():
    model_path, training_path = os.getenv("PROMPTALIGN_CLIP"), os.getenv("PROMPTALIGN_TRAINING")
    predictor = None
    if model_path and training_path:
        from src.inference.predictor import Predictor
        predictor = Predictor(model_path, training_path, os.getenv("PROMPTALIGN_ALLOW_DEVELOPMENT") == "1")
    return create_app(predictor)
