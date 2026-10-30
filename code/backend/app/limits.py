"""Bound request bodies before multipart parsing, including chunked uploads."""

import uuid
from starlette.responses import JSONResponse


class BodyLimitMiddleware:
    def __init__(self, app, limit=61 * 1024 * 1024):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > self.limit:
                response = JSONResponse({"request_id": uuid.uuid4().hex, "type": "error",
                    "data": {"code": "REQUEST_TOO_LARGE", "message": "Yêu cầu quá lớn."},
                    "model_version": None, "elapsed_ms": 0, "warnings": []}, status_code=413)
                return await response(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(chunks)

        async def replay():
            return next(iterator, None) or await receive()

        await self.app(scope, replay, send)
